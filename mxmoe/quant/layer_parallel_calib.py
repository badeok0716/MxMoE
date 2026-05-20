"""Layer-parallel calibration for online GPTQ/GPTQ-HAD layer_out_norm.

This path keeps the intended GPTQ calibration topology: for layer L, every
(expert, block) observes activations produced from the full-precision prefix
0..L-1.  GPUs own contiguous layer ranges.  A worker replays the full-precision
prefix for its first layer, then calibrates its assigned layers sequentially.
"""

import gc
import json
import logging
import os
import shutil
import sys
import threading
import traceback
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
from typing import Literal

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch import Tensor

from mxmoe.quant.moe_utils import (
    MOE_MLP_NAME_MAP,
    MOE_WEIGHT_NAME_MAP,
    is_non_moe_layer,
)


logger = logging.getLogger("quant-calib")
_GPTQ_QUANT_LOCK = threading.Lock()


def _to_device(x, device: torch.device):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, tuple):
        return tuple(_to_device(v, device) for v in x)
    if isinstance(x, list):
        return [_to_device(v, device) for v in x]
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


def _share_cpu_(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        assert x.device.type == "cpu", f"shared tensor must be CPU, got {x.device}"
        return x.share_memory_()
    if isinstance(x, tuple):
        return tuple(_share_cpu_(v) for v in x)
    if isinstance(x, list):
        return [_share_cpu_(v) for v in x]
    if isinstance(x, dict):
        return {k: _share_cpu_(v) for k, v in x.items()}
    return x


def _cleanup_cuda(device: torch.device | None = None, *, collect_ipc: bool = False):
    gc.collect()
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return

    def _empty():
        torch.cuda.empty_cache()
        if collect_ipc:
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass

    if device is None or torch.device(device).type != "cuda":
        _empty()
        return

    with torch.cuda.device(device):
        _empty()


def _ensure_spawn_picklable_class_globals(model: nn.Module) -> list[str]:
    """Repair HuggingFace remote-code class globals before mp spawn pickling.

    Some trust_remote_code models can leave an instance whose class object is not
    identical to the class object currently stored under the same
    ``sys.modules[cls.__module__].<cls.__name__>`` global.  Python's spawn
    pickler rejects that state before child processes even start.  Rebinding the
    module global to the live class keeps spawn semantics unchanged while
    avoiding the identity mismatch.
    """
    fixed: list[str] = []
    seen: set[type] = set()

    candidates = [
        model,
        getattr(model, "config", None),
        getattr(model, "generation_config", None),
    ]
    candidates.extend(model.modules())

    for obj in candidates:
        if obj is None:
            continue
        cls = obj.__class__
        if cls in seen:
            continue
        seen.add(cls)

        module = sys.modules.get(cls.__module__)
        if module is None:
            continue
        if "." in cls.__qualname__:
            continue

        current = getattr(module, cls.__name__, None)
        if current is not cls:
            setattr(module, cls.__name__, cls)
            fixed.append(f"{cls.__module__}.{cls.__name__}")

    return fixed


def _write_layer_result(parts_dir: str, layer_idx: int, result: list[list[float]]):
    with open(os.path.join(parts_dir, f"layer_{layer_idx:04d}.json"), "w") as f:
        json.dump({str(e): result[e] for e in range(len(result))}, f)


def _merge_layer_result_parts(
    *,
    parts_dir: str,
    save_path: str,
    num_layers: int,
) -> list[list[list[float]]]:
    layer_loss: list[list[list[float]]] = [[] for _ in range(num_layers)]
    layer_loss_save: dict[int, dict[str, list[float]]] = {}

    for filename in sorted(os.listdir(parts_dir)):
        if not filename.startswith("layer_") or not filename.endswith(".json"):
            continue
        layer_idx = int(filename[len("layer_") : -len(".json")])
        with open(os.path.join(parts_dir, filename)) as f:
            saved = json.load(f)
        result = [saved[str(e)] for e in range(len(saved))]
        layer_loss[layer_idx] = result
        layer_loss_save[layer_idx] = saved

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(layer_loss_save, f)
    return layer_loss


def _configure_worker_logging(parts_dir: str, worker_idx: int):
    os.makedirs(parts_dir, exist_ok=True)
    handler = logging.FileHandler(
        os.path.join(parts_dir, f"worker_{worker_idx}.log"),
        mode="w",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [worker-%(process)d] %(message)s"
        )
    )
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _layer_ranges(num_layers: int, n_gpus: int) -> list[tuple[int, int]]:
    n_gpus = min(n_gpus, num_layers)
    base, extra = divmod(num_layers, n_gpus)
    ranges = []
    start = 0
    for i in range(n_gpus):
        width = base + (1 if i < extra else 0)
        ranges.append((start, start + width))
        start += width
    return ranges


def _get_layer_experts(
    model_id: str,
    model_type: str,
    layer: nn.Module,
    layer_idx: int,
) -> list[nn.Module]:
    if model_type == "deepseek_v2":
        if layer_idx == 0:
            return [layer.mlp]
        return [*layer.mlp.experts, layer.mlp.shared_experts]
    if model_type == "qwen2_moe":
        return [*layer.mlp.experts, layer.mlp.shared_expert]
    if model_type == "mixtral":
        return list(layer.block_sparse_moe.experts)
    raise NotImplementedError(f"Unsupported model type: {model_type}")


def _get_block_linear(
    model_id: str,
    layer: nn.Module,
    layer_idx: int,
    expert_idx: int,
    linear_block: str,
) -> nn.Linear:
    mlp_block_name = MOE_MLP_NAME_MAP[model_id]
    linear_attr = MOE_WEIGHT_NAME_MAP[model_id][linear_block]
    mlp = getattr(layer, mlp_block_name)
    if is_non_moe_layer(model_id, layer_idx):
        return getattr(mlp, linear_attr)

    real_experts = mlp.experts
    if expert_idx < len(real_experts):
        return getattr(real_experts[expert_idx], linear_attr)
    if model_id == "ds2":
        return getattr(mlp.shared_experts, linear_attr)
    if model_id in ("qwen2_moe", "qwen2_moe_57b"):
        return getattr(mlp.shared_expert, linear_attr)
    raise ValueError(
        f"expert_idx={expert_idx} is outside real experts for model_id={model_id!r}"
    )


def _forward_layer(
    layer: nn.Module,
    inps: Tensor,
    attention_mask,
    position_ids,
    pos_emb,
    batch_size: int,
    out: Tensor | None = None,
) -> Tensor:
    out = torch.empty_like(inps) if out is None else out
    pe_kw = {} if pos_emb is None else {"position_embeddings": pos_emb}
    with torch.inference_mode():
        for start in range(0, inps.shape[0], batch_size):
            end = min(start + batch_size, inps.shape[0])
            out[start:end] = layer(
                inps[start:end],
                attention_mask=attention_mask,
                position_ids=position_ids,
                **pe_kw,
            )[0]
    return out


def _diff_norm_fp64(a: Tensor, b: Tensor, chunk: int | None = None) -> float:
    if chunk is None:
        chunk = int(os.environ.get("MXMOE_DIFF_NORM_CHUNK", "16"))
    chunk = max(1, chunk)
    sq = 0.0
    for start in range(0, a.shape[0], chunk):
        end = min(start + chunk, a.shape[0])
        diff = a[start:end].to(torch.float64) - b[start:end].to(torch.float64)
        sq += diff.pow(2).sum().item()
    return sq ** 0.5


def _collect_gptq_and_quantize(
    *,
    layer: nn.Module,
    target_linear: nn.Linear,
    qcfg,
    gptq_inps: Tensor,
    attention_mask,
    position_ids,
    pos_emb,
    percdamp: float,
):
    from mxmoe.quant.gptq import GPTQ, Quantizer as GPTQuantizer

    gptq = GPTQ(target_linear)
    # GPTQ's Triton graph backend uses process-global CUDA capture state.  Let
    # Slurm smoke jobs choose `legacy` while we keep the default graph backend
    # available for smaller runs.
    gptq.triton_backend = os.environ.get(
        "MXMOE_LAYER_PARALLEL_GPTQ_BACKEND",
        "triton_graph",
    )
    gptq.quantizer = GPTQuantizer()
    gptq.quantizer.configure(qcfg.w_bits, perchannel=True, sym=qcfg.w_sym, mse=False)

    handle = target_linear.register_forward_hook(
        lambda _, inp, out: gptq.add_batch(inp[0].data, out.data)
    )
    pe_kw = {} if pos_emb is None else {"position_embeddings": pos_emb}
    try:
        with torch.inference_mode():
            # Keep this loop at B=1.  MoE expert Linear hooks receive 2D token
            # tensors; batching here changes GPTQ.add_batch's sample accounting.
            for j in range(gptq_inps.shape[0]):
                layer(
                    gptq_inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **pe_kw,
                )
        device = target_linear.weight.device
        needs_lock = gptq.triton_backend in ("triton", "triton_graph", "triton_nograph")
        lock_ctx = _GPTQ_QUANT_LOCK if needs_lock else nullcontext()
        with lock_ctx, torch.cuda.device(device):
            gptq.fasterquant(
                percdamp=percdamp,
                groupsize=qcfg.w_gsize,
                actorder=True,
                static_groups=True,
            )
            torch.cuda.synchronize(device)
    finally:
        handle.remove()
        device = target_linear.weight.device
        needs_lock = gptq.triton_backend in ("triton", "triton_graph", "triton_nograph")
        lock_ctx = _GPTQ_QUANT_LOCK if needs_lock else nullcontext()
        with lock_ctx, torch.cuda.device(device):
            gptq.free()
            del gptq
            _cleanup_cuda(device)


def _eval_one_block(
    *,
    qmethod,
    model_id: str,
    layer: nn.Module,
    layer_idx: int,
    expert_idx: int,
    linear_block: str,
    qcfg,
    loss_inps: Tensor,
    full_precision_outs: Tensor,
    loss_attention_mask,
    loss_position_ids,
    loss_pos_emb,
    gptq_inps: Tensor,
    gptq_attention_mask,
    gptq_position_ids,
    gptq_pos_emb,
    gptq_percdamp: float,
    batch_size: int,
    quantized_outs: Tensor,
) -> float:
    from mxmoe.quant.quant import (
        QMethod,
        Quantizer as RTNQuantizer,
        mlp_inp_quant_hook,
    )

    target_linear = _get_block_linear(
        model_id, layer, layer_idx, expert_idx, linear_block
    )
    original_weight = target_linear.weight.detach().clone()
    act_handle = None

    try:
        if qcfg.w_bits < 16:
            if qmethod in (QMethod.GPTQ, QMethod.GPTQ_HAD):
                _collect_gptq_and_quantize(
                    layer=layer,
                    target_linear=target_linear,
                    qcfg=qcfg,
                    gptq_inps=gptq_inps,
                    attention_mask=gptq_attention_mask,
                    position_ids=gptq_position_ids,
                    pos_emb=gptq_pos_emb,
                    percdamp=gptq_percdamp,
                )
            else:
                weight_quantizer = RTNQuantizer(
                    qcfg.w_bits, qcfg.w_sym, qcfg.w_gsize, qcfg.w_clip
                )
                target_linear.weight.data.copy_(
                    weight_quantizer.fake_quant(target_linear.weight.data)
                )

        if qcfg.a_bits < 16:
            act_quantizer = RTNQuantizer(
                qcfg.a_bits, qcfg.a_sym, qcfg.a_gsize, qcfg.a_clip
            )
            act_handle = target_linear.register_forward_pre_hook(
                partial_mlp_inp_quant_hook(mlp_inp_quant_hook, act_quantizer)
            )

        _forward_layer(
            layer,
            loss_inps,
            loss_attention_mask,
            loss_position_ids,
            loss_pos_emb,
            batch_size,
            out=quantized_outs,
        )
        return _diff_norm_fp64(full_precision_outs, quantized_outs)
    finally:
        target_linear.weight.data.copy_(original_weight)
        del original_weight
        if act_handle is not None:
            act_handle.remove()
        _cleanup_cuda(target_linear.weight.device)


def partial_mlp_inp_quant_hook(hook_fn, quantizer):
    def _hook(module, inp):
        return hook_fn(module, inp, quantizer=quantizer)

    return _hook


def _process_layer(
    *,
    quantizer,
    layer: nn.Module,
    layer_idx: int,
    loss_inps: Tensor,
    loss_attention_mask,
    loss_position_ids,
    loss_pos_emb,
    gptq_inps: Tensor,
    gptq_attention_mask,
    gptq_position_ids,
    gptq_pos_emb,
    granularity: Literal["expert", "linear"],
    moe_bits_alloc,
    attn_bits_alloc,
) -> tuple[list[list[float]], Tensor, Tensor]:
    from mxmoe.quant.quant import enumerate_expert_qconfig

    full_precision_outs = _forward_layer(
        layer,
        loss_inps,
        loss_attention_mask,
        loss_position_ids,
        loss_pos_emb,
        quantizer.batch_size,
    )
    quantized_outs = torch.empty_like(loss_inps)

    layer_experts = _get_layer_experts(
        quantizer.model_id, quantizer.model_type, layer, layer_idx
    )
    layer_loss: list[list[float]] = []
    for exp_id in range(len(layer_experts)):
        qlayer_cfgs = enumerate_expert_qconfig(
            moe_bits_alloc,
            quantizer.num_layers,
            len(layer_experts),
            exp_id,
            granularity,
        )
        expert_err: list[float] = []
        for linear_block, qlayer_cfg in zip(["gate", "up", "down"], qlayer_cfgs):
            qcfg = getattr(qlayer_cfg.experts[str(exp_id)], linear_block)
            quant_err = _eval_one_block(
                qmethod=quantizer.qmethod,
                model_id=quantizer.model_id,
                layer=layer,
                layer_idx=layer_idx,
                expert_idx=exp_id,
                linear_block=linear_block,
                qcfg=qcfg,
                loss_inps=loss_inps,
                full_precision_outs=full_precision_outs,
                loss_attention_mask=loss_attention_mask,
                loss_position_ids=loss_position_ids,
                loss_pos_emb=loss_pos_emb,
                gptq_inps=gptq_inps,
                gptq_attention_mask=gptq_attention_mask,
                gptq_position_ids=gptq_position_ids,
                gptq_pos_emb=gptq_pos_emb,
                gptq_percdamp=quantizer.gptq_percdamp,
                batch_size=quantizer.batch_size,
                quantized_outs=quantized_outs,
            )
            expert_err.append(quant_err)
            _cleanup_cuda(loss_inps.device)
        layer_loss.append(expert_err)
        logger.info(
            f"{quantizer.model_id} L{layer_idx}-E{exp_id} {moe_bits_alloc} "
            f"(layer_out_norm): {expert_err}"
        )

    del quantized_outs
    _cleanup_cuda(loss_inps.device)
    next_gptq_inps = _forward_layer(
        layer,
        gptq_inps,
        gptq_attention_mask,
        gptq_position_ids,
        gptq_pos_emb,
        batch_size=1,
    )
    return layer_loss, full_precision_outs, next_gptq_inps


def _advance_prefix_layer(
    *,
    layer: nn.Module,
    loss_inps: Tensor,
    loss_attention_mask,
    loss_position_ids,
    loss_pos_emb,
    gptq_inps: Tensor,
    gptq_attention_mask,
    gptq_position_ids,
    gptq_pos_emb,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    next_loss_inps = _forward_layer(
        layer,
        loss_inps,
        loss_attention_mask,
        loss_position_ids,
        loss_pos_emb,
        batch_size=batch_size,
    )
    next_gptq_inps = _forward_layer(
        layer,
        gptq_inps,
        gptq_attention_mask,
        gptq_position_ids,
        gptq_pos_emb,
        batch_size=1,
    )
    return next_loss_inps, next_gptq_inps


def _spawn_layer_range_worker(
    *,
    worker_idx: int,
    device_idx: int,
    start_layer: int,
    end_layer: int,
    model: nn.Module,
    quantizer_state,
    loss_bundle: dict,
    gptq_bundle: dict,
    granularity: Literal["expert", "linear"],
    moe_bits_alloc,
    attn_bits_alloc,
    parts_dir: str,
):
    _configure_worker_logging(parts_dir, worker_idx)
    try:
        device = torch.device(f"cuda:{device_idx}")
        torch.cuda.set_device(device)
        model.config.use_cache = False
        if getattr(quantizer_state, "online_had", False):
            from mxmoe.quant.rotation import ModelRotator

            ModelRotator(model, "hadamard", dev=device).plug_online_had_hook()
        logger.info(
            f">>> process worker {worker_idx} uses cuda:{device_idx}, "
            f"layers [{start_layer}, {end_layer})"
        )

        loss_inps = loss_bundle["inps"].to(device)
        loss_attn = _to_device(loss_bundle["attention_mask"], device)
        loss_pos_ids = _to_device(loss_bundle["position_ids"], device)
        loss_pos_emb = _to_device(loss_bundle["position_embeddings"], device)

        gptq_inps = gptq_bundle["inps"].to(device)
        gptq_attn = _to_device(gptq_bundle["attention_mask"], device)
        gptq_pos_ids = _to_device(gptq_bundle["position_ids"], device)
        gptq_pos_emb = _to_device(gptq_bundle["position_embeddings"], device)

        layers = model.model.layers
        for layer_idx in range(end_layer):
            layer = layers[layer_idx].to(device).eval()
            if layer_idx < start_layer:
                loss_inps, gptq_inps = _advance_prefix_layer(
                    layer=layer,
                    loss_inps=loss_inps,
                    loss_attention_mask=loss_attn,
                    loss_position_ids=loss_pos_ids,
                    loss_pos_emb=loss_pos_emb,
                    gptq_inps=gptq_inps,
                    gptq_attention_mask=gptq_attn,
                    gptq_position_ids=gptq_pos_ids,
                    gptq_pos_emb=gptq_pos_emb,
                    batch_size=quantizer_state.batch_size,
                )
            else:
                result, loss_inps, gptq_inps = _process_layer(
                    quantizer=quantizer_state,
                    layer=layer,
                    layer_idx=layer_idx,
                    loss_inps=loss_inps,
                    loss_attention_mask=loss_attn,
                    loss_position_ids=loss_pos_ids,
                    loss_pos_emb=loss_pos_emb,
                    gptq_inps=gptq_inps,
                    gptq_attention_mask=gptq_attn,
                    gptq_position_ids=gptq_pos_ids,
                    gptq_pos_emb=gptq_pos_emb,
                    granularity=granularity,
                    moe_bits_alloc=moe_bits_alloc,
                    attn_bits_alloc=attn_bits_alloc,
                )
                _write_layer_result(parts_dir, layer_idx, result)
                logger.info(
                    f"Layer-{layer_idx} quant error(layer_out_norm):\n{result}"
                )

            layers[layer_idx] = nn.Identity()
            del layer
            _cleanup_cuda(device, collect_ipc=True)

        torch.cuda.synchronize(device)
    except BaseException:
        with open(os.path.join(parts_dir, f"worker_{worker_idx}.error"), "w") as f:
            f.write(traceback.format_exc())
        raise


def _run_layer_parallel_layer_out_norm_thread(
    quantizer,
    dataloader: list[Tensor],
    granularity: Literal["expert", "linear"],
    save_path: str,
    moe_bits_alloc,
    attn_bits_alloc=None,
):
    from mxmoe.quant.quant import QMethod, prepare_inps

    self = quantizer
    assert self.qmethod in (QMethod.GPTQ, QMethod.GPTQ_HAD)
    assert self.pre_quantized_weight is None
    assert getattr(self, "gptq_layer_state", None) is not None

    n_gpus = min(int(self.n_gpus), torch.cuda.device_count())
    num_layers = min(self.num_layers, getattr(self, "max_layers", None) or self.num_layers)
    assert n_gpus >= 2, f"layer-parallel calib requires at least 2 GPUs, got {n_gpus}"
    devices = [torch.device(f"cuda:{i}") for i in range(n_gpus)]

    for device in devices:
        torch.cuda.set_device(device)
        eye = torch.eye(4, device=device, dtype=torch.float32)
        torch.linalg.cholesky(eye + eye * 1e-3)
        torch.cholesky_inverse(eye)
    torch.cuda.set_device(devices[0])

    model_use_cache = self.ori_model.config.use_cache
    self.ori_model.config.use_cache = False

    logger.info(
        f">>> Layer-parallel online GPTQ calib: {num_layers} layers over {n_gpus} GPUs"
    )

    try:
        loss_inps0, loss_attn, loss_pos_ids, loss_pos_emb = prepare_inps(
            self.ori_model, dataloader
        )
        loss_inps0 = loss_inps0.to("cpu")
        loss_attn = _to_device(loss_attn, torch.device("cpu"))
        loss_pos_ids = _to_device(loss_pos_ids, torch.device("cpu"))
        loss_pos_emb = _to_device(loss_pos_emb, torch.device("cpu"))

        gptq_inps0 = self.gptq_layer_state.to("cpu")
        gptq_attn = _to_device(self.gptq_attention_mask, torch.device("cpu"))
        gptq_pos_ids = _to_device(self.gptq_position_ids, torch.device("cpu"))
        gptq_pos_emb = _to_device(self.gptq_pos_emb, torch.device("cpu"))
        self.gptq_layer_state = gptq_inps0
        self.gptq_attention_mask = gptq_attn
        self.gptq_position_ids = gptq_pos_ids
        self.gptq_pos_emb = gptq_pos_emb
        _cleanup_cuda(torch.device("cuda:0"), collect_ipc=True)

        layer_loss: list[list[list[float]]] = [[] for _ in range(self.num_layers)]
        layer_loss_save: dict[int, dict[int, list[float]]] = {}
        result_lock = threading.Lock()
        clone_lock = threading.Lock()
        layers = self.ori_model.model.layers
        ranges = _layer_ranges(num_layers, n_gpus)

        def save_layer_result(layer_idx: int, result: list[list[float]]):
            with result_lock:
                layer_loss[layer_idx] = result
                layer_loss_save[layer_idx] = {
                    e: result[e] for e in range(len(result))
                }
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w") as f:
                    json.dump(layer_loss_save, f)
                logger.info(
                    f"Layer-{layer_idx} quant error(layer_out_norm):\n{result}"
                )

        def worker(worker_idx: int, start_layer: int, end_layer: int):
            device = devices[worker_idx]
            torch.cuda.set_device(device)
            logger.info(
                f">>> GPU {worker_idx} owns layers [{start_layer}, {end_layer})"
            )

            loss_inps = loss_inps0.to(device)
            gptq_inps = gptq_inps0.to(device)
            local_loss_attn = _to_device(loss_attn, device)
            local_loss_pos_ids = _to_device(loss_pos_ids, device)
            local_loss_pos_emb = _to_device(loss_pos_emb, device)
            local_gptq_attn = _to_device(gptq_attn, device)
            local_gptq_pos_ids = _to_device(gptq_pos_ids, device)
            local_gptq_pos_emb = _to_device(gptq_pos_emb, device)

            for layer_idx in range(end_layer):
                with clone_lock:
                    layer = deepcopy(layers[layer_idx]).to(device).eval()

                if layer_idx < start_layer:
                    loss_inps, gptq_inps = _advance_prefix_layer(
                        layer=layer,
                        loss_inps=loss_inps,
                        loss_attention_mask=local_loss_attn,
                        loss_position_ids=local_loss_pos_ids,
                        loss_pos_emb=local_loss_pos_emb,
                        gptq_inps=gptq_inps,
                        gptq_attention_mask=local_gptq_attn,
                        gptq_position_ids=local_gptq_pos_ids,
                        gptq_pos_emb=local_gptq_pos_emb,
                        batch_size=self.batch_size,
                    )
                else:
                    result, loss_inps, gptq_inps = _process_layer(
                        quantizer=self,
                        layer=layer,
                        layer_idx=layer_idx,
                        loss_inps=loss_inps,
                        loss_attention_mask=local_loss_attn,
                        loss_position_ids=local_loss_pos_ids,
                        loss_pos_emb=local_loss_pos_emb,
                        gptq_inps=gptq_inps,
                        gptq_attention_mask=local_gptq_attn,
                        gptq_position_ids=local_gptq_pos_ids,
                        gptq_pos_emb=local_gptq_pos_emb,
                        granularity=granularity,
                        moe_bits_alloc=moe_bits_alloc,
                        attn_bits_alloc=attn_bits_alloc,
                    )
                    save_layer_result(layer_idx, result)

                del layer
                _cleanup_cuda(device, collect_ipc=True)

        with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
            futures = [
                executor.submit(worker, worker_idx, start, end)
                for worker_idx, (start, end) in enumerate(ranges)
                if start < end
            ]
            for future in futures:
                future.result()

        return layer_loss
    finally:
        self.ori_model.config.use_cache = model_use_cache


def run_layer_parallel_layer_out_norm(
    quantizer,
    dataloader: list[Tensor],
    granularity: Literal["expert", "linear"],
    save_path: str,
    moe_bits_alloc,
    attn_bits_alloc=None,
):
    if os.environ.get("MXMOE_LAYER_PARALLEL_BACKEND") == "thread":
        logger.warning(
            ">>> MXMOE_LAYER_PARALLEL_BACKEND=thread selected. This backend is "
            "kept only for debugging; CUDA/Triton thread sharing is unstable."
        )
        return _run_layer_parallel_layer_out_norm_thread(
            quantizer,
            dataloader,
            granularity,
            save_path,
            moe_bits_alloc,
            attn_bits_alloc,
        )

    from mxmoe.quant.quant import QMethod, prepare_inps

    self = quantizer
    assert self.qmethod in (QMethod.GPTQ, QMethod.GPTQ_HAD)
    assert self.pre_quantized_weight is None
    assert getattr(self, "gptq_layer_state", None) is not None

    num_layers = min(
        self.num_layers, getattr(self, "max_layers", None) or self.num_layers
    )
    n_gpus = min(int(self.n_gpus), torch.cuda.device_count(), num_layers)
    assert n_gpus >= 2, f"layer-parallel calib requires at least 2 GPUs, got {n_gpus}"

    model_use_cache = self.ori_model.config.use_cache
    self.ori_model.config.use_cache = False
    parts_dir = f"{save_path}.parts_{os.getpid()}"
    online_had_enabled = bool(getattr(self, "online_had_hooks", []))

    logger.info(
        f">>> Spawn layer-parallel online GPTQ calib: {num_layers} layers over "
        f"{n_gpus} processes"
    )
    logger.info(f">>> Layer-parallel worker outputs: {parts_dir}")

    try:
        try:
            mp.set_sharing_strategy(
                os.environ.get("MXMOE_MP_SHARING_STRATEGY", "file_system")
            )
        except RuntimeError:
            # PyTorch allows setting this only before CPU tensor sharing starts.
            pass

        if os.path.exists(parts_dir):
            shutil.rmtree(parts_dir)
        os.makedirs(parts_dir, exist_ok=True)

        loss_inps0, loss_attn, loss_pos_ids, loss_pos_emb = prepare_inps(
            self.ori_model, dataloader
        )
        loss_bundle = {
            "inps": loss_inps0.to("cpu"),
            "attention_mask": _to_device(loss_attn, torch.device("cpu")),
            "position_ids": _to_device(loss_pos_ids, torch.device("cpu")),
            "position_embeddings": _to_device(loss_pos_emb, torch.device("cpu")),
        }
        gptq_bundle = {
            "inps": self.gptq_layer_state.to("cpu"),
            "attention_mask": _to_device(
                self.gptq_attention_mask, torch.device("cpu")
            ),
            "position_ids": _to_device(self.gptq_position_ids, torch.device("cpu")),
            "position_embeddings": _to_device(
                self.gptq_pos_emb, torch.device("cpu")
            ),
        }
        self.gptq_layer_state = gptq_bundle["inps"]
        self.gptq_attention_mask = gptq_bundle["attention_mask"]
        self.gptq_position_ids = gptq_bundle["position_ids"]
        self.gptq_pos_emb = gptq_bundle["position_embeddings"]

        del loss_inps0, loss_attn, loss_pos_ids, loss_pos_emb
        self.ori_model.to("cpu")
        _cleanup_cuda(torch.device("cuda:0"), collect_ipc=True)

        self.ori_model.share_memory()
        loss_bundle = _share_cpu_(loss_bundle)
        gptq_bundle = _share_cpu_(gptq_bundle)

        if online_had_enabled:
            for handle in self.online_had_hooks:
                handle.remove()
            self.online_had_hooks = []

        fixed_pickle_globals = _ensure_spawn_picklable_class_globals(self.ori_model)
        if fixed_pickle_globals:
            logger.info(
                ">>> Rebound remote-code classes for spawn pickling: "
                + ", ".join(fixed_pickle_globals[:8])
                + (" ..." if len(fixed_pickle_globals) > 8 else "")
            )

        quantizer_state = SimpleNamespace(
            model_id=self.model_id,
            model_type=self.model_type,
            num_layers=self.num_layers,
            qmethod=self.qmethod,
            batch_size=self.batch_size,
            gptq_percdamp=self.gptq_percdamp,
            online_had=online_had_enabled,
        )

        ctx = mp.get_context("spawn")
        ranges = _layer_ranges(num_layers, n_gpus)
        procs: list[mp.Process] = []
        for worker_idx, (start_layer, end_layer) in enumerate(ranges):
            if start_layer >= end_layer:
                continue
            proc = ctx.Process(
                target=_spawn_layer_range_worker,
                kwargs=dict(
                    worker_idx=worker_idx,
                    device_idx=worker_idx,
                    start_layer=start_layer,
                    end_layer=end_layer,
                    model=self.ori_model,
                    quantizer_state=quantizer_state,
                    loss_bundle=loss_bundle,
                    gptq_bundle=gptq_bundle,
                    granularity=granularity,
                    moe_bits_alloc=moe_bits_alloc,
                    attn_bits_alloc=attn_bits_alloc,
                    parts_dir=parts_dir,
                ),
            )
            proc.start()
            procs.append(proc)

        failed: list[tuple[int, int]] = []
        for worker_idx, proc in enumerate(procs):
            proc.join()
            if proc.exitcode != 0:
                failed.append((worker_idx, proc.exitcode))

        if failed:
            details = []
            for worker_idx, exitcode in failed:
                err_path = os.path.join(parts_dir, f"worker_{worker_idx}.error")
                if os.path.exists(err_path):
                    with open(err_path) as f:
                        err = f.read().strip()
                    details.append(f"worker {worker_idx} exit={exitcode}\n{err}")
                else:
                    details.append(f"worker {worker_idx} exit={exitcode}")
            raise RuntimeError(
                "Layer-parallel spawn workers failed:\n" + "\n\n".join(details)
            )

        layer_loss = _merge_layer_result_parts(
            parts_dir=parts_dir,
            save_path=save_path,
            num_layers=self.num_layers,
        )
        for layer_idx, result in enumerate(layer_loss):
            if result:
                logger.info(
                    f"Layer-{layer_idx} quant error(layer_out_norm):\n{result}"
                )
        return layer_loss
    finally:
        self.ori_model.config.use_cache = model_use_cache
        if online_had_enabled and not getattr(self, "online_had_hooks", []):
            from mxmoe.quant.rotation import ModelRotator

            self.online_had_hooks = ModelRotator(
                self.ori_model, "hadamard"
            ).plug_online_had_hook()
