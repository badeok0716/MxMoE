"""Multi-GPU parallel calibration for layer_out_norm metric.

Dispatch the (expert, block) iteration inside one MoE layer across N CUDA
devices. Each device gets a deepcopy of the layer (post-Hadamard rotation if
applicable) and its own replica of the layer's input activations. Threads —
not processes — because every hot-path op is CUDA (forward, copy, diff-norm)
and releases the GIL, so a thread-per-device is enough to drive parallel work
without IPC overhead.

Determinism: every worker starts from the same layer weights and the same
inps; CUDA arithmetic with `allow_bf16_reduced_precision_reduction = False`
(set in mxmoe.quant.quant top-level) is device-pure for identical inputs on
matching hardware. The single-GPU path is byte-identical to the legacy
`get_model_quant_error` implementation since `n_gpus == 1` short-circuits
before this module is even imported.

Scope: `metric == "layer_out_norm"` only — the dispatch in
`MoeModelQuantizer.get_model_quant_error` falls through to the legacy code
for other metrics.
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from mxmoe.quant.moe_utils import (
    MOE_MLP_NAME_MAP, MOE_WEIGHT_NAME_MAP,
    get_expert_linears, get_linear_block_weight,
    is_non_moe_layer, offload_moe_weights,
)


logger = logging.getLogger("quant-calib")


@dataclass
class _GPTQContext:
    """Per-device GPTQ Hessian-collection state.

    All tensors live on the device that consumes them. `layer_state` is the
    drift-free 256-sample (or `nsamples`-sample) calibration buffer that stays
    constant for every (expert, block) iter inside a single layer; the master
    advances it once per layer via `cur_layer(state)` after the per-expert
    loop completes, then re-replicates to the other devices.
    """
    nsamples: int
    percdamp: float
    layer_state: Tensor
    attn_mask: object
    position_ids: object
    pos_emb: object


def _get_block_linear(
    model_id: str, layer: nn.Module, layer_idx: int,
    expert_idx: int, linear_block: str,
) -> nn.Linear:
    """Locate the nn.Linear inside `layer` corresponding to (expert_idx, linear_block).

    Layer-local equivalent of `moe_utils.get_linear_block_weight`: avoids the
    `model.layers.X.…` string key indirection so workers can operate on a
    deepcopy'd layer without a model wrapper.
    """
    mlp_block_name = MOE_MLP_NAME_MAP[model_id]                   # "mlp" | "block_sparse_moe"
    linear_attr = MOE_WEIGHT_NAME_MAP[model_id][linear_block]     # e.g. "gate_proj" | "w1"
    mlp = getattr(layer, mlp_block_name)
    if is_non_moe_layer(model_id, layer_idx):
        return getattr(mlp, linear_attr)
    real_experts = mlp.experts
    if expert_idx < len(real_experts):
        return getattr(real_experts[expert_idx], linear_attr)
    # shared-expert tail position (qwen2_moe / ds2)
    if model_id == "ds2":
        return getattr(mlp.shared_experts, linear_attr)
    if model_id in ("qwen2_moe", "qwen2_moe_57b"):
        return getattr(mlp.shared_expert, linear_attr)
    raise ValueError(
        f"expert_idx={expert_idx} >= num_real_experts={len(real_experts)} but "
        f"no shared-expert lookup path defined for model_id={model_id!r}"
    )


def _to_device(x, device: torch.device):
    """Recursively move tensors in `x` to `device`. Tuple/list/dict/None pass through."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, tuple):
        return tuple(_to_device(e, device) for e in x)
    if isinstance(x, list):
        return [_to_device(e, device) for e in x]
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


def _eval_one_task(
    *,
    layer: nn.Module,
    device: torch.device,
    full_precision_outs: Tensor,
    inps: Tensor,
    attn_mask,
    position_ids,
    pos_emb,
    model_id: str,
    qmethod,
    layer_idx: int,
    expert_idx: int,
    linear_block: str,
    qcfg,
    pre_quantized_weight,
    ori_weight_cpu: Tensor,
    batch_size: int,
    num_samples: int,
    quantized_outs_buf: Tensor,
    gptq_ctx: Optional[_GPTQContext] = None,
    norm_chunk: int = 16,
) -> float:
    """Quantize one (expert_idx, linear_block) on `layer`, forward, compute norm, restore.

    Returns the FP64-chunked diff-norm of (fp_outs - quantized_outs).
    """
    from mxmoe.quant.quant import (
        QMethod, Quantizer as RTNQuantizer, mlp_inp_quant_hook,
    )
    from mxmoe.quant.gptq import GPTQ as _GPTQ, Quantizer as _GPTQuantizer

    target_linear = _get_block_linear(model_id, layer, layer_idx, expert_idx, linear_block)

    # 1. Apply weight quantization to the target Linear (in place).
    if pre_quantized_weight is not None:
        pre_q_w = get_linear_block_weight(
            model_id, pre_quantized_weight, layer_idx, expert_idx, linear_block,
        ).to(device=device, dtype=target_linear.weight.dtype, non_blocking=True)
        target_linear.weight.data.copy_(pre_q_w)
    elif qmethod in (QMethod.GPTQ, QMethod.GPTQ_HAD) and qcfg.w_bits < 16:
        # Online GPTQ: collect Hessian by hooking target_linear, run
        # gptq_ctx.nsamples forwards through `layer` using the SHARED
        # device-local layer_state (drift-free — never mutated here; the master
        # advances it once at layer end and re-replicates).
        assert gptq_ctx is not None, (
            "GPTQ online path in parallel_calib requires gptq_ctx; "
            "MoeModelQuantizer.__init__ should have populated gptq_layer_state."
        )
        gptq = _GPTQ(target_linear)
        gptq.quantizer = _GPTQuantizer()
        gptq.quantizer.configure(qcfg.w_bits, perchannel=True, sym=qcfg.w_sym, mse=False)
        h_handle = target_linear.register_forward_hook(
            lambda _, inp, out: gptq.add_batch(inp[0].data, out.data)
        )
        _pe_kw_g = (
            {} if gptq_ctx.pos_emb is None
            else {'position_embeddings': gptq_ctx.pos_emb}
        )
        # Single-sample loop, mirroring the legacy comment about MoE 2D linear
        # input scaling H by tmp=1.
        for j in range(gptq_ctx.nsamples):
            layer(
                gptq_ctx.layer_state[j].unsqueeze(0),
                attention_mask=gptq_ctx.attn_mask,
                position_ids=gptq_ctx.position_ids,
                **_pe_kw_g,
            )
        h_handle.remove()
        gptq.fasterquant(
            percdamp=gptq_ctx.percdamp, groupsize=qcfg.w_gsize,
            actorder=True, static_groups=True,
        )
        gptq.free()
    else:
        if qcfg.w_bits < 16:
            rtn = RTNQuantizer(qcfg.w_bits, qcfg.w_sym, qcfg.w_gsize, qcfg.w_clip)
            target_linear.weight.data.copy_(rtn.fake_quant(target_linear.weight.data))

    # 2. Plug forward_pre_hook for act quant if needed.
    handle = None
    if qcfg.a_bits < 16:
        a_q = RTNQuantizer(qcfg.a_bits, qcfg.a_sym, qcfg.a_gsize, qcfg.a_clip)
        if qmethod not in (QMethod.WxAy_NAIVE, QMethod.GPTQ, QMethod.GPTQ_HAD, QMethod.RTN_HAD):
            # SMOOTH_QUANT / AWQ need per-channel calib data threaded through —
            # outside the immediate target set for the multi-GPU rollout.
            raise NotImplementedError(
                f"Parallel calib path does not yet support qmethod={qmethod!r} with a_bits<16"
            )
        hook = partial(mlp_inp_quant_hook, quantizer=a_q)
        handle = target_linear.register_forward_pre_hook(hook)

    # 3. Forward through `layer` in batches of `batch_size`.
    _pe_kw = {} if pos_emb is None else {'position_embeddings': pos_emb}
    with torch.inference_mode():
        for _s in range(0, num_samples, batch_size):
            _e = min(_s + batch_size, num_samples)
            quantized_outs_buf[_s:_e] = layer(
                inps[_s:_e],
                attention_mask=attn_mask,
                position_ids=position_ids,
                **_pe_kw,
            )[0]

    # 4. Chunked FP64 diff norm — mirror the legacy path's accumulation (chunk=16).
    _sq = 0.0
    for _s in range(0, num_samples, norm_chunk):
        _e = min(_s + norm_chunk, num_samples)
        _d = (
            full_precision_outs[_s:_e].to(torch.float64)
            - quantized_outs_buf[_s:_e].to(torch.float64)
        )
        _sq += _d.pow(2).sum().item()
    quant_err = _sq ** 0.5

    # 5. Restore weight + drop the hook.
    target_linear.weight.data.copy_(
        ori_weight_cpu.to(device=device, dtype=target_linear.weight.dtype, non_blocking=True)
    )
    if handle is not None:
        handle.remove()

    return quant_err


def run_parallel_layer_out_norm(
    quantizer,                                  # MoeModelQuantizer
    dataloader: list[Tensor],
    granularity: Literal["expert", "linear"],
    save_path: str,
    moe_bits_alloc,
    attn_bits_alloc=None,
):
    """Multi-GPU dispatch of the layer_out_norm calibration loop.

    Per layer:
      1. Move the layer to cuda:0 (master), snapshot original weights to CPU.
      2. Compute full_precision_outs on cuda:0 once.
      3. Replicate the layer (deepcopy) and fp_outs to cuda:1..N-1.
      4. Round-robin (expert, block) tasks into N device bins; run one thread
         per device sequentially through its bin.
      5. Gather results into `layer_loss[layer_idx]` in the legacy [expert][block]
         shape.
      6. Carry fp_outs forward as next-layer inps on every device (no extra copy).
    """
    from mxmoe.quant.quant import QMethod, enumerate_expert_qconfig, prepare_inps

    self = quantizer
    n_gpus = self.n_gpus
    visible = torch.cuda.device_count()
    assert 2 <= n_gpus <= visible, (
        f"--n-gpus={n_gpus} but only {visible} CUDA devices visible. "
        f"Set CUDA_VISIBLE_DEVICES or request more GPUs via Slurm."
    )
    # Expert granularity has a subtle restore asymmetry in the legacy path
    # (quantizes all blocks but restores only "gate") that we don't replicate
    # here. Linear granularity is the main calib target.
    assert granularity == "linear", (
        f"parallel calib path currently supports --gran linear only "
        f"(got --gran {granularity!r}). Re-run with --n-gpus 1 for expert gran."
    )
    devices = [torch.device(f"cuda:{i}") for i in range(n_gpus)]
    master_dev = devices[0]

    # Pre-warm `torch.linalg.cholesky` on each device. Two threads racing on a
    # cold lazy-init wrapper triggers `RuntimeError: lazy wrapper should be
    # called at most once`. A 4×4 SPD matrix on each device pays the init cost
    # once on the main thread; subsequent worker calls are reentrant.
    for d in devices:
        with torch.inference_mode():
            _spd = torch.eye(4, device=d, dtype=torch.float32) + torch.eye(4, device=d, dtype=torch.float32) * 1e-3
            torch.linalg.cholesky(_spd)
            torch.linalg.cholesky(_spd, upper=True)
            torch.cholesky_inverse(_spd)
        del _spd

    ori_dev = self.ori_model.device
    assert ori_dev.type in ("cpu", "cuda")
    assert self.quantized_model is None, "Quantized model should not be initialized"

    model_use_cache = self.ori_model.config.use_cache
    self.ori_model.config.use_cache = False

    num_samples = len(dataloader)

    # ----- 1) prepare_inps once on cuda:0 -----
    inps_master, attn_mask, position_ids, pos_emb = prepare_inps(self.ori_model, dataloader)

    # Replicate calib inputs to every device (shape never changes across layers).
    inps_per_dev: dict[torch.device, Tensor] = {master_dev: inps_master}
    for d in devices[1:]:
        inps_per_dev[d] = inps_master.to(d)
    attn_per_dev = {d: _to_device(attn_mask, d) for d in devices}
    pos_ids_per_dev = {d: _to_device(position_ids, d) for d in devices}
    pos_emb_per_dev = {d: _to_device(pos_emb, d) for d in devices}

    # ----- 1b) Replicate GPTQ calibration state (online GPTQ path only) -----
    # `self.gptq_layer_state` is populated by MoeModelQuantizer.__init__ from a
    # separate 256-sample wikitext loader when qmethod is GPTQ/GPTQ_HAD AND no
    # pre-quantized weight was loaded. With drift-free semantics this buffer is
    # read-only within a layer; the master advances it once at layer end.
    gptq_online = (
        self.qmethod in (QMethod.GPTQ, QMethod.GPTQ_HAD)
        and self.pre_quantized_weight is None
        and getattr(self, "gptq_layer_state", None) is not None
    )
    gptq_ctx_per_dev: dict[torch.device, _GPTQContext] = {}
    if gptq_online:
        master_state = self.gptq_layer_state.to(master_dev)
        # Update self.gptq_layer_state in place so the layer-end advance writes
        # to a master copy on the right device (mirrors legacy behavior).
        self.gptq_layer_state = master_state
        master_attn = _to_device(self.gptq_attention_mask, master_dev)
        master_pos_ids = _to_device(self.gptq_position_ids, master_dev)
        master_pos_emb = _to_device(self.gptq_pos_emb, master_dev)
        gptq_ctx_per_dev[master_dev] = _GPTQContext(
            nsamples=self.gptq_nsamples,
            percdamp=self.gptq_percdamp,
            layer_state=master_state,
            attn_mask=master_attn,
            position_ids=master_pos_ids,
            pos_emb=master_pos_emb,
        )
        for d in devices[1:]:
            gptq_ctx_per_dev[d] = _GPTQContext(
                nsamples=self.gptq_nsamples,
                percdamp=self.gptq_percdamp,
                layer_state=master_state.to(d),
                attn_mask=_to_device(self.gptq_attention_mask, d),
                position_ids=_to_device(self.gptq_position_ids, d),
                pos_emb=_to_device(self.gptq_pos_emb, d),
            )

    layers = self.ori_model.model.layers
    num_layers = self.num_layers
    layer_loss: list[list[list[float]]] = [[] for _ in range(num_layers)]
    layer_loss_save: dict = {}

    for layer_idx in tqdm(range(num_layers), desc=f"parallel calib ({n_gpus} GPUs)"):
        # ----- 2) master layer → cuda:0; snapshot CPU originals -----
        self.ori_model.model.layers[layer_idx] = layers[layer_idx].to(master_dev)
        cpu_weights_copy = offload_moe_weights(self.model_id, self.ori_model, layer_idx)
        master_layer = self.ori_model.model.layers[layer_idx]

        # ----- 3) full_precision_outs on cuda:0 (batched, same as legacy) -----
        fp_outs_master = torch.zeros_like(inps_per_dev[master_dev])
        with torch.inference_mode():
            _pe_kw = (
                {} if pos_emb_per_dev[master_dev] is None
                else {'position_embeddings': pos_emb_per_dev[master_dev]}
            )
            _B = self.batch_size
            for _s in range(0, num_samples, _B):
                _e = min(_s + _B, num_samples)
                fp_outs_master[_s:_e] = master_layer(
                    inps_per_dev[master_dev][_s:_e],
                    attention_mask=attn_per_dev[master_dev],
                    position_ids=pos_ids_per_dev[master_dev],
                    **_pe_kw,
                )[0]

        # ----- 4) Replicate layer + fp_outs to other devices -----
        layer_per_dev: dict[torch.device, nn.Module] = {master_dev: master_layer}
        fp_outs_per_dev: dict[torch.device, Tensor] = {master_dev: fp_outs_master}
        for d in devices[1:]:
            layer_per_dev[d] = deepcopy(master_layer).to(d)
            fp_outs_per_dev[d] = fp_outs_master.to(d)

        # Pre-warm a single-sample forward on every non-master device. Flash-
        # attention CUDA kernels do per-device lazy init that has raced under
        # ThreadPoolExecutor (manifests as `illegal memory access` from a prior
        # async kernel). Running one warmup forward per device on the main
        # thread initializes the kernel cache for that device before workers
        # touch it.
        with torch.inference_mode():
            for d in devices[1:]:
                _pe_kw_w = (
                    {} if pos_emb_per_dev[d] is None
                    else {'position_embeddings': pos_emb_per_dev[d]}
                )
                layer_per_dev[d](
                    inps_per_dev[d][:1],
                    attention_mask=attn_per_dev[d],
                    position_ids=pos_ids_per_dev[d],
                    **_pe_kw_w,
                )

        # ----- 5) Build (expert, block) task list (order matches legacy) -----
        num_layer_experts = len(get_expert_linears(
            self.ori_model, layer_idx, exclude_non_moe_layer=False,
        ))

        tasks: list[tuple[int, str, object]] = []  # (expert_idx, linear_block, qcfg)
        for exp_id in range(num_layer_experts):
            qlayer_cfgs = enumerate_expert_qconfig(
                moe_bits_alloc, num_layers, num_layer_experts, exp_id, granularity,
            )
            for linear_block, qlayer_cfg in zip(["gate", "up", "down"], qlayer_cfgs):
                qcfg = getattr(qlayer_cfg.experts[str(exp_id)], linear_block)
                tasks.append((exp_id, linear_block, qcfg))

        # One quantized_outs buffer per device, reused across all of that device's
        # tasks (sequential within the device's thread). Saves ~hundreds × 2 GB
        # allocation churn over the per-layer task set.
        quantized_buf_per_dev = {
            d: torch.empty_like(inps_per_dev[d]) for d in devices
        }

        # Static device assignment by task index — keeps each thread bound to
        # exactly one device's layer copy, so no two threads ever touch the same
        # layer concurrently.
        tasks_per_dev: list[list[tuple[int, tuple]]] = [[] for _ in range(n_gpus)]
        for ti, task in enumerate(tasks):
            tasks_per_dev[ti % n_gpus].append((ti, task))

        results: list[float | None] = [None] * len(tasks)

        def _device_worker(dev_idx: int):
            d = devices[dev_idx]
            torch.cuda.set_device(d)
            local_layer = layer_per_dev[d]
            local_fp = fp_outs_per_dev[d]
            local_inps = inps_per_dev[d]
            local_attn = attn_per_dev[d]
            local_pos_ids = pos_ids_per_dev[d]
            local_pos_emb = pos_emb_per_dev[d]
            local_buf = quantized_buf_per_dev[d]
            local_gptq_ctx = gptq_ctx_per_dev.get(d)
            for ti, (exp_id, linear_block, qcfg) in tasks_per_dev[dev_idx]:
                ori_weight_cpu = get_linear_block_weight(
                    self.model_id, cpu_weights_copy, layer_idx, exp_id, linear_block,
                )
                err = _eval_one_task(
                    layer=local_layer, device=d,
                    full_precision_outs=local_fp,
                    inps=local_inps,
                    attn_mask=local_attn,
                    position_ids=local_pos_ids,
                    pos_emb=local_pos_emb,
                    model_id=self.model_id,
                    qmethod=self.qmethod,
                    layer_idx=layer_idx,
                    expert_idx=exp_id,
                    linear_block=linear_block,
                    qcfg=qcfg,
                    pre_quantized_weight=self.pre_quantized_weight,
                    ori_weight_cpu=ori_weight_cpu,
                    batch_size=self.batch_size,
                    num_samples=num_samples,
                    quantized_outs_buf=local_buf,
                    gptq_ctx=local_gptq_ctx,
                )
                results[ti] = err

        with ThreadPoolExecutor(max_workers=n_gpus) as ex:
            futs = [ex.submit(_device_worker, i) for i in range(n_gpus)]
            for f in futs:
                f.result()  # propagate worker exceptions

        # ----- 6) Re-assemble into per-expert [block_err] shape -----
        expert_errs: list[list[float]] = [[] for _ in range(num_layer_experts)]
        for ti, (exp_id, linear_block, qcfg) in enumerate(tasks):
            assert results[ti] is not None, f"task {ti} produced no result"
            expert_errs[exp_id].append(results[ti])

        for exp_id in range(num_layer_experts):
            layer_loss[layer_idx].append(expert_errs[exp_id])
            logger.info(
                f"{self.model_id} L{layer_idx}-E{exp_id} {moe_bits_alloc} "
                f"(layer_out_norm): {expert_errs[exp_id]}"
            )

        # ----- 6b) Online GPTQ: advance gptq_layer_state once on master, then
        # re-replicate to other devices. Mirrors the legacy single-advance step
        # added in get_model_quant_error after the per-expert loop.
        if gptq_online:
            with torch.inference_mode():
                _pe_kw_g = (
                    {} if gptq_ctx_per_dev[master_dev].pos_emb is None
                    else {'position_embeddings': gptq_ctx_per_dev[master_dev].pos_emb}
                )
                m_state = gptq_ctx_per_dev[master_dev].layer_state
                for j in range(self.gptq_nsamples):
                    m_state[j] = master_layer(
                        m_state[j].unsqueeze(0),
                        attention_mask=gptq_ctx_per_dev[master_dev].attn_mask,
                        position_ids=gptq_ctx_per_dev[master_dev].position_ids,
                        **_pe_kw_g,
                    )[0]
            # Re-replicate advanced state to non-master devices.
            for d in devices[1:]:
                gptq_ctx_per_dev[d].layer_state.copy_(m_state.to(d))
            # Keep self.gptq_layer_state pointing at the master copy (for any
            # downstream code that reads it, mirroring the legacy behavior).
            self.gptq_layer_state = m_state

        # ----- 7) Move master layer back to ori_dev (CPU); drop replicas -----
        self.ori_model.model.layers[layer_idx] = layers[layer_idx].to(ori_dev)
        for d in devices[1:]:
            layer_per_dev.pop(d, None)
        del cpu_weights_copy, quantized_buf_per_dev

        # ----- 8) Serialize after each layer (resume-friendly, matches legacy) -----
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            layer_loss_save[layer_idx] = {
                e: layer_loss[layer_idx][e] for e in range(len(layer_loss[layer_idx]))
            }
            json.dump(layer_loss_save, f)
        logger.info(f"Layer-{layer_idx} quant error(layer_out_norm):\n{layer_loss[layer_idx]}")

        # ----- 9) Promote fp_outs → next-layer inps on every device -----
        inps_per_dev = fp_outs_per_dev
        fp_outs_per_dev = None  # release strong ref so prior inps_per_dev can be GC'd
        torch.cuda.empty_cache()

    self.ori_model.config.use_cache = model_use_cache
    return layer_loss
