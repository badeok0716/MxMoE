# b200_exp_20260521_qwen2_moe_57b_remain

Qwen2-MoE-57B remaining online GPTQ-HAD calibration on B200.

This experiment is intentionally B200-only. B200 has no NFS with the gateway,
so the flow is:

1. Push the exact MxMoE commit to the B200 remote.
2. Clone/check out that commit on B200.
3. Build the Python/CUDA environment and download the HF model on B200.
4. Submit one B200 job per qconfig.
5. Pull logs/results back to this directory.

Do not create a PR for this workflow. Use direct git remote + commit SHA only.

## Scope

Already done elsewhere:

```text
qwen2_moe_57b / w1_g128_asym  completed
```

Run on B200:

```text
qwen2_moe_57b / w2_g128_asym  retry; previous Slurm/Ada run OOMed
qwen2_moe_57b / w3_g128_asym  not submitted yet
qwen2_moe_57b / w4_g128_asym  not submitted yet
qwen2_moe_57b / w4_g-1_asym  not submitted yet
qwen2_moe_57b / w8_g-1_asym  not submitted yet
```

## Git Remote

Use this remote for B200 work:

```bash
cd /data_fast/home/deokjae/QUANT_works/MxMoE
git remote add b200 https://github.com/badeok0716/MxMoE.git 2>/dev/null || \
git remote set-url b200 https://github.com/badeok0716/MxMoE.git
```

Commit explicit files only, then push the branch/commit you want B200 to run:

```bash
git add .gitignore CLAUDE.md project_config.py \
    mxmoe/quant/gptq.py mxmoe/quant/gptq_triton.py \
    mxmoe/quant/layer_parallel_calib.py mxmoe/quant/parallel_calib.py \
    mxmoe/quant/moe_utils.py \
    mxmoe/quant/quant.py \
    exps/b200_exp_20260521_qwen2_moe_57b_remain/
git commit -m "B200 qwen2_moe_57b remaining calib setup"
git push b200 HEAD:<branch>
SHA=$(git rev-parse HEAD)
```

No `gh pr`, no pull request.

Do not edit the gateway/root `pyproject.toml` for B200. B200-specific package
changes live in `pyproject_b200.toml` under this experiment directory only.

## Phase 1: Bootstrap B200

This stages `b200_setup.sh` with sftp, then submits it as a B200 job. It clones
`https://github.com/badeok0716/MxMoE.git` to `$B200_ROOT/MxMoE`, checks out the
given SHA, copies this experiment's `pyproject_b200.toml` to the B200 checkout
as `pyproject.toml`, builds the uv env, installs `flash-attn` and
`fast-hadamard-transform`, and downloads `Qwen/Qwen2-57B-A14B-Instruct`.

The B200-only pyproject switches PyTorch to the cu128 PyTorch index and selects
a CUDA 12.8 toolkit before building CUDA extensions. This is intentionally
confined to the B200 checkout; the gateway/root environment remains unchanged.
If CUDA 12.8 is not installed on the B200 image, setup falls back to the first
available CUDA toolkit and logs the selected `CUDA_HOME`. The setup also patches
the B200 checkout's `fast-hadamard-transform/setup.py` to build `sm_100` only,
because CUDA 13.x rejects the upstream hard-coded `sm_70` arch flag.

```bash
bash exps/b200_exp_20260521_qwen2_moe_57b_remain/gateway_bootstrap.sh "$SHA"
get_b200_queue.sh
```

## Phase 2: Submit Remaining Qconfigs

After bootstrap finishes:

```bash
bash exps/b200_exp_20260521_qwen2_moe_57b_remain/gateway_submit_remaining.sh "$SHA"
get_b200_queue.sh
```

The submit script launches five independent B200 jobs:

```text
w2_g128_asym
w3_g128_asym
w4_g128_asym
w4_g-1_asym
w8_g-1_asym
```

Each job uses:

```text
ngpus=4
ncpus=40
batch_size=16
nsamples=128
gptq_nsamples=256
```

The calibration jobs do not run `git checkout` or `uv sync`; they only verify
that `$B200_ROOT/MxMoE` is already at the requested SHA. This avoids concurrent
jobs racing on the same checkout or shared `.venv`.

## Phase 3: Pull Results

After the jobs finish:

```bash
bash exps/b200_exp_20260521_qwen2_moe_57b_remain/gateway_pull.sh
```

Local landing paths:

```text
exps/b200_exp_20260521_qwen2_moe_57b_remain/logs/
exps/b200_exp_20260521_qwen2_moe_57b_remain/results/
```

## B200 Paths

```text
B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
REPO=$B200_ROOT/MxMoE
EXP=$REPO/exps/b200_exp_20260521_qwen2_moe_57b_remain
HF_HOME=$B200_ROOT/hf_cache
```

## Files

```text
b200_setup.sh              runs on B200; clone/env/model setup
b200_calib_one.sh          runs on B200; one qconfig calibration
gateway_bootstrap.sh       runs on gateway; uploads/submits setup
gateway_submit_remaining.sh runs on gateway; submits five qconfig jobs
gateway_pull.sh            runs on gateway; pulls logs/results back
pyproject_b200.toml        B200-only uv project; copied on B200 during setup
run_b200_big_calib.py      exp-local qwen2_moe_57b calibration runner
```
