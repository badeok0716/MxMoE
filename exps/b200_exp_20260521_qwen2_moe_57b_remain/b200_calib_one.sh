#!/bin/bash
# Runs ON B200. Run one qwen2_moe_57b online GPTQ-HAD calibration qconfig.
#
# Usage:
#   bash b200_calib_one.sh <SHA> <qcfg> [batch_size]

set -euo pipefail

SHA="${1:?missing git SHA}"
QCFG="${2:?missing qcfg}"
BATCH_SIZE="${3:-16}"

B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
REPO_URL=https://github.com/badeok0716/MxMoE.git
REPO=$B200_ROOT/MxMoE
EXP_REL=exps/b200_exp_20260521_qwen2_moe_57b_remain
EXP=$REPO/$EXP_REL

select_cuda_home() {
    local wanted="${1:?missing CUDA version}"
    local candidates=()

    if [[ -n "${CUDA_HOME:-}" ]]; then
        candidates+=("$CUDA_HOME")
    fi

    if [[ -f /etc/profile.d/modules.sh ]]; then
        # shellcheck source=/dev/null
        source /etc/profile.d/modules.sh || true
    fi
    if command -v module >/dev/null 2>&1; then
        module load "cuda/$wanted" >/dev/null 2>&1 || \
        module load "cuda/${wanted}.0" >/dev/null 2>&1 || true
        if [[ -n "${CUDA_HOME:-}" ]]; then
            candidates+=("$CUDA_HOME")
        fi
    fi

    candidates+=(
        "/usr/local/cuda-$wanted"
        "/usr/local/cuda-${wanted}.0"
        "/opt/cuda-$wanted"
        "/opt/cuda-${wanted}.0"
        /usr/local/cuda
    )

    local cuda_home
    local nvcc_version
    for cuda_home in "${candidates[@]}"; do
        if [[ -x "$cuda_home/bin/nvcc" ]]; then
            nvcc_version="$("$cuda_home/bin/nvcc" --version 2>/dev/null || true)"
            if grep -q "release $wanted" <<<"$nvcc_version"; then
                echo "$cuda_home"
                return 0
            fi
        fi
    done

    echo "ERROR: could not find CUDA toolkit $wanted for B200 torch cu128 builds." >&2
    echo "Checked candidates:" >&2
    printf '  %s\n' "${candidates[@]}" >&2
    echo "Available /usr/local CUDA dirs:" >&2
    ls -ld /usr/local/cuda* 2>/dev/null >&2 || true
    return 1
}

export HF_HOME=$B200_ROOT/hf_cache
export MXMOE_B200_CUDA_VERSION="${MXMOE_B200_CUDA_VERSION:-12.8}"
export CUDA_HOME="$(select_cuda_home "$MXMOE_B200_CUDA_VERSION")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export MXMOE_DIFF_NORM_CHUNK="${MXMOE_DIFF_NORM_CHUNK:-16}"
export MXMOE_LAYER_PARALLEL_GPTQ_BACKEND="${MXMOE_LAYER_PARALLEL_GPTQ_BACKEND:-triton_graph}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"

LOG=$EXP/logs/calib_${QCFG}_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$EXP/logs" "$EXP/results" "$EXP/diag"
exec > >(tee -a "$LOG") 2>&1

echo "=== node: $(hostname) ==="
echo "=== start: $(date -Iseconds) ==="
echo "=== SHA: $SHA ==="
echo "=== QCFG: $QCFG ==="
echo "=== batch_size: $BATCH_SIZE ==="
echo "=== backend: $MXMOE_LAYER_PARALLEL_GPTQ_BACKEND ==="
echo "=== CUDA_HOME: $CUDA_HOME ==="
echo "=== TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST ==="
"$CUDA_HOME/bin/nvcc" --version || true
nvidia-smi -L || true
df -h "$B200_ROOT" || true
SECONDS=0

cd "$REPO"
CURRENT_SHA="$(git rev-parse HEAD)"
if [[ "$CURRENT_SHA" != "$SHA" ]]; then
    echo "ERROR: B200 checkout is $CURRENT_SHA, expected $SHA."
    echo "Run gateway_bootstrap.sh for this SHA before submitting calibration jobs."
    exit 1
fi
echo "=== checked out: $CURRENT_SHA ==="

uv run --no-sync python - <<'PY'
import torch
import flash_attn
import fast_hadamard_transform
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda devices", torch.cuda.device_count())
print("flash_attn ok", flash_attn.__version__)
print("fast_hadamard_transform ok")
PY

TAG=qwen2_moe_57b_${QCFG}_b200
uv run --no-sync python "$EXP/run_b200_big_calib.py" \
    --model qwen2_moe_57b \
    --qcfg "$QCFG" \
    --tag "$TAG" \
    --out-dir "$EXP/results" \
    --n-gpus 4 \
    --nsamples 128 \
    --gptq-nsamples 256 \
    --batch-size "$BATCH_SIZE"

echo "=== end: $(date -Iseconds) elapsed=${SECONDS}s ==="
ls -lh "$EXP/results/${TAG}.json" "$EXP/results/${TAG}.meta.json"
