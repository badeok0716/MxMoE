#!/bin/bash
# Runs ON B200. Clone MxMoE from the B200 remote, check out a gateway-pushed
# SHA, build the env, install CUDA extension deps, and download Qwen2-MoE-57B.

set -euo pipefail

SHA="${1:-}"

B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
REPO_URL=https://github.com/badeok0716/MxMoE.git
REPO=$B200_ROOT/MxMoE
EXP_REL=exps/b200_exp_20260521_qwen2_moe_57b_remain
MODEL_ID=Qwen/Qwen2-57B-A14B-Instruct

export HF_HOME=$B200_ROOT/hf_cache
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-16}"

LOG=$B200_ROOT/mxmoe_qwen57b_setup_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "=== node: $(hostname) ==="
echo "=== start: $(date -Iseconds) ==="
echo "=== B200_ROOT: $B200_ROOT ==="
echo "=== REPO_URL: $REPO_URL ==="
echo "=== HF_HOME: $HF_HOME ==="
nvidia-smi -L || true
df -h "$B200_ROOT" || true
SECONDS=0

mkdir -p "$B200_ROOT" "$HF_HOME"

if [[ ! -d "$REPO/.git" ]]; then
    git clone "$REPO_URL" "$REPO"
fi

cd "$REPO"
git remote set-url origin "$REPO_URL"
git fetch origin
git checkout -- uv.lock 2>/dev/null || true
if [[ -n "$SHA" ]]; then
    git checkout "$SHA"
fi
echo "=== checked out: $(git rev-parse HEAD) ==="

EXP=$REPO/$EXP_REL
mkdir -p "$EXP/logs" "$EXP/results" "$EXP/diag"

uv python install 3.11
uv venv --python 3.11 --python-preference only-managed
uv sync --inexact

uv pip install flash-attn==2.7.4.post1 --no-build-isolation
git submodule update --init --recursive mxmoe/3rdparty/fast-hadamard-transform
uv pip install -e mxmoe/3rdparty/fast-hadamard-transform --no-build-isolation

uv run python - <<'PY'
import torch
import flash_attn
import fast_hadamard_transform
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda devices", torch.cuda.device_count())
print("flash_attn ok", flash_attn.__version__)
print("fast_hadamard_transform ok")
PY

uv run huggingface-cli whoami || true
uv run huggingface-cli download "$MODEL_ID"

echo "=== model cache ==="
du -sh "$HF_HOME"/hub/models--Qwen--Qwen2-57B-A14B-Instruct 2>/dev/null || true
echo "=== end: $(date -Iseconds) elapsed=${SECONDS}s ==="
echo "=== setup log: $LOG ==="
