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
export MAX_JOBS="${MAX_JOBS:-16}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"

LOG=$B200_ROOT/mxmoe_qwen57b_setup_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "=== node: $(hostname) ==="
echo "=== start: $(date -Iseconds) ==="
echo "=== B200_ROOT: $B200_ROOT ==="
echo "=== REPO_URL: $REPO_URL ==="
echo "=== HF_HOME: $HF_HOME ==="
echo "=== CUDA_HOME: $CUDA_HOME ==="
echo "=== TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST ==="
"$CUDA_HOME/bin/nvcc" --version || true
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
git checkout -- pyproject.toml uv.lock 2>/dev/null || true
if [[ -n "$SHA" ]]; then
    git checkout "$SHA"
fi
echo "=== checked out: $(git rev-parse HEAD) ==="

EXP=$REPO/$EXP_REL
mkdir -p "$EXP/logs" "$EXP/results" "$EXP/diag"
test -f "$EXP/pyproject_b200.toml" || {
    echo "ERROR: missing B200 pyproject: $EXP/pyproject_b200.toml"
    exit 1
}
cp "$EXP/pyproject_b200.toml" pyproject.toml
rm -f uv.lock
echo "=== using B200-only pyproject: $EXP/pyproject_b200.toml ==="
git status --short pyproject.toml uv.lock || true

uv python install 3.11
uv venv --python 3.11 --python-preference only-managed
uv sync --inexact

uv run python - <<'PY'
import torch
print("torch after sync", torch.__version__, "cuda", torch.version.cuda)
assert torch.version.cuda == "12.8", torch.version.cuda
PY

uv pip install --force-reinstall flash-attn==2.7.4.post1 --no-build-isolation
git submodule update --init --recursive mxmoe/3rdparty/fast-hadamard-transform
rm -rf mxmoe/3rdparty/fast-hadamard-transform/build
uv pip install --force-reinstall -e mxmoe/3rdparty/fast-hadamard-transform --no-build-isolation

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
