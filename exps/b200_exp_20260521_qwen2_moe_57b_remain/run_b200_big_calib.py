"""Run one qwen2_moe_57b online GPTQ-HAD calibration job on B200."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mxmoe.quant.quant as quant_mod
from logger_utils import setup_logger
from mxmoe.kernels.qconfig import QLinearConfig
from mxmoe.quant.data_utils import get_wikitext2
from mxmoe.quant.quant import MoeModelQuantizer, QMethod, seed_everything
from project_config import ID2NAME


QCFG_MAP = {
    "w1_g128_asym": QLinearConfig(w_bits=1, w_sym=False, w_gsize=128),
    "w2_g128_asym": QLinearConfig(w_bits=2, w_sym=False, w_gsize=128),
    "w3_g128_asym": QLinearConfig(w_bits=3, w_sym=False, w_gsize=128),
    "w4_g128_asym": QLinearConfig(w_bits=4, w_sym=False, w_gsize=128),
    "w4_g-1_asym": QLinearConfig(w_bits=4, w_sym=False, w_gsize=-1),
    "w8_g-1_asym": QLinearConfig(w_bits=8, w_sym=False, w_gsize=-1),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2_moe_57b", choices=["qwen2_moe_57b"])
    parser.add_argument("--qcfg", required=True, choices=sorted(QCFG_MAP))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-gpus", type=int, default=4)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--gptq-nsamples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--gptq-backend", default=None)
    args = parser.parse_args()

    if args.gptq_backend is not None:
        os.environ["MXMOE_LAYER_PARALLEL_GPTQ_BACKEND"] = args.gptq_backend

    model_id = args.model
    seqlen = 4096
    metric = "layer_out_norm"
    granularity = "linear"
    qcfg = QCFG_MAP[args.qcfg]
    attn_qcfg = QLinearConfig(w_bits=16)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quant_mod.logger = setup_logger(
        "quant-calib",
        log_file=str(out_dir / f"{args.tag}.runner.log"),
    )

    seed_everything(42)
    model_name = ID2NAME[model_id]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="sdpa",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cpu",
    )
    trainloader, _ = get_wikitext2(
        args.nsamples, 42, seqlen, tokenizer, model_id=model_id, test_only=False
    )

    quant_mod.logger.info(
        "b200-calib tag=%s model=%s qcfg=%s n_gpus=%s nsamples=%s "
        "gptq_nsamples=%s batch_size=%s max_layers=%s backend=%s",
        args.tag,
        model_id,
        args.qcfg,
        args.n_gpus,
        args.nsamples,
        args.gptq_nsamples,
        args.batch_size,
        args.max_layers,
        os.environ.get("MXMOE_LAYER_PARALLEL_GPTQ_BACKEND", "triton_graph"),
    )
    start = time.monotonic()
    quantizer = MoeModelQuantizer(
        model,
        model_id,
        QMethod.GPTQ_HAD,
        pre_quantized_weight=None,
        online_had=False,
        batch_size=args.batch_size,
        gptq_nsamples=args.gptq_nsamples,
    )
    quantizer.n_gpus = args.n_gpus
    quantizer.max_layers = args.max_layers

    save_path = out_dir / f"{args.tag}.json"
    quantizer.get_model_quant_error(
        trainloader,
        metric,
        granularity,
        str(save_path),
        qcfg,
        attn_qcfg,
    )
    meta = {
        "tag": args.tag,
        "model": model_id,
        "method": "gptq-had",
        "metric": metric,
        "qcfg": args.qcfg,
        "qcfg_repr": str(qcfg),
        "granularity": granularity,
        "nsamples": args.nsamples,
        "gptq_nsamples": args.gptq_nsamples,
        "batch_size": args.batch_size,
        "n_gpus": args.n_gpus,
        "max_layers": args.max_layers,
        "gptq_backend": os.environ.get(
            "MXMOE_LAYER_PARALLEL_GPTQ_BACKEND", "triton_graph"
        ),
        "elapsed_sec": time.monotonic() - start,
        "save_path": str(save_path),
    }
    (out_dir / f"{args.tag}.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
