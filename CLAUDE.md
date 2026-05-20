# CLAUDE.md (MxMoE)

MxMoE-specific rules. Common behavior rules (§1–§7) and execution conventions
(Slurm / B200 / `exps/`) are auto-merged from the parent
[`../CLAUDE.md`](../CLAUDE.md) and [`../EXECUTION.md`](../EXECUTION.md) — not
repeated here.

## Project

Fork of upstream MxMoE (arXiv 2505.05799) — accuracy/performance co-design
for MoE mixed-precision quantization (DeepSeek-V2-Lite, Qwen1.5-MoE,
Qwen2-MoE-57B, Mixtral-8x7B). We use it for **quality experiments** (LP-based
bit assignment + GPTQ-HAD calibration); the upstream CUDA kernels in
`mxmoe/kernels/` are not needed for our workflow.

- Source dirs: `mxmoe/quant/` (`quant.py`, `gptq.py`, `bits_solver.py`,
  `moe_utils.py`, `moe_tracer.py`, `rotation.py`, `evaluator.py`, …)
- Entry CLIs (run from repo root):
  - `python -m mxmoe.quant.quant calib …` — per-(model, qcfg) sensitivity
  - `python -m mxmoe.quant.bits_solver …` — LP bit assignment
  - `python -m mxmoe.quant.quant eval …` — PPL + downstream eval
  - `python -m mxmoe.quant.moe_tracer --trace_gate …` — MoE-gate frequency
- Environment: `.venv/` (uv-managed, `pyproject.toml` at repo root)
- Run outputs: `exps/baseline_reprd/` (paper reproduction + verification
  smokes) — see [`exps/README.md`](exps/README.md) and parent
  [`../EXECUTION.md`](../EXECUTION.md) §Output paths.
- All work-in-progress under `exps/baseline_reprd/tmp_YYYYMMDD_<slug>/`.

## Environment (uv venv + CUDA 12.4)

`.venv` — torch 2.6.0+cu124, Python 3.11. `fast-hadamard-transform` and
`flash-attn` are built from source (one-time, ~3 min on a GPU node) and
installed in the venv. Subsequent `uv sync` is fast.

### Setup (one-time per venv. check before setup.)

```bash
uv sync                                                  # base deps
# flash_attn + fast_hadamard_transform need a GPU node + nvcc:
srun --gres=gpu:1 --time=2:00:00 --mem=60G --partition=… bash -c '
  export CUDA_HOME=/storage/deokjae/.local/cuda/cuda-12.4
  export PATH=$CUDA_HOME/bin:$PATH
  uv pip install flash-attn==2.7.4.post1 --no-build-isolation
  git submodule update --init mxmoe/3rdparty/fast-hadamard-transform
  uv pip install -e mxmoe/3rdparty/fast-hadamard-transform --no-build-isolation
'
```

**NB**: `flash-attn` and `fast-hadamard-transform` are NOT in `pyproject.toml`
(build-from-source). A subsequent `uv sync` will remove them from the venv —
re-run the GPU step above, or use `uv sync --inexact` to preserve.

`gurobipy` (used by `bits_solver`) needs an academic license at
`~/gurobi.lic`. WLS license OK on multiple nodes — also a one-time setup.

### Runtime env (every job)

```bash
export CUDA_HOME=/storage/deokjae/.local/cuda/cuda-12.4    # env CUDA_HOME default
export PATH=$CUDA_HOME/bin:$PATH                            # is wrong (empty)
export HF_HOME=/storage/deokjae/.cache                      # shared HF model cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # reduces frag
```

## GPTQ Triton kernel (default)

`GPTQ.fasterquant` → Triton + CUDA-Graph kernel in
[`mxmoe/quant/gptq_triton.py`](mxmoe/quant/gptq_triton.py) by default.
~2-3× end-to-end vs legacy PyTorch loop. Spike + verification:
[`exps/baseline_reprd/tmp_20260520_triton_gptq_spike/`](exps/baseline_reprd/tmp_20260520_triton_gptq_spike/).

- Override: `--gptq-backend {legacy,triton}` (CLI) or
  `MXMOE_GPTQ_TRITON_BACKEND={legacy,triton,triton_graph,triton_nograph}` (env).
- Auto-falls-through to legacy when unsupported: `actorder + !static_groups`,
  `quantizer.mse=True`, `quantizer.maxq<0` (trits).
- Calib floor: MoE needs `gptq_nsamples ≥ 64-128, seqlen=4096`. Below that,
  both backends produce broken PPL (Hessian rank-deficient per expert).

## Patches applied to upstream

Upstream MxMoE has author-local assumptions that break in our env. Each patch
is documented inline (rationale + verification) — see the surrounding comments
when editing. Summary:

- `moe_utils.load_hf_model` — resolve HF id → local snapshot before `load_checkpoint_and_dispatch`.
- `moe_utils.get_device_map` — single-GPU returns `{"": "cpu"}` (enables per-layer GPU swap).
- `quant.py` top — `allow_bf16_reduced_precision_reduction = False` (FP32-accumulate BF16 matmul; B=1 vs B=32 outputs effectively bit-identical at ~2% cost).
- `quant.py:get_model_quant_error` — `assert ori_dev.type in ("cpu", "cuda")`.
- `quant.py:Catcher.forward` + 3 layer calls — `position_embeddings` optional for DS2.
- `quant.py:get_model_quant_error` — BF16 storage + chunked FP64 diff-norm (`_NORM_CHUNK=16`).
- `quant.py` measurement loops — batched forward with `_B = self.batch_size` (default 32). **NB**: GPTQ Hessian loop intentionally kept at `B=1` — batching it scales `H` by ~B× via `GPTQ.add_batch`'s `tmp=inp.shape[0]=1` for MoE 2D input. Verified in `exps/baseline_reprd/tmp_20260520_batch_verify/`.

## Per-partition `--batch-size` / `--mem` (verified models only)

Verified on **`qwen2_moe` (Qwen1.5-MoE-A2.7B, 14.3 B)** and **`ds2`
(DeepSeek-V2-Lite, 15.7 B)**. Bigger models (`mixtral`, `qwen2_moe_57b`)
need re-verification separately.

| partition | `--batch-size` | `--mem` |
|---|---:|---:|
| rtx3090 | 16 | 120G |
| ada     | 32 | 120G |
| a100    | 32 | 120G |
| h100    | 32 | 120G |

`--batch-size 32` on rtx3090 24 GB OOMs at GPTQ Hessian `inp.float()` cast
for `ds2` specifically; 16 is safe. Treat 120G as the safe floor.
