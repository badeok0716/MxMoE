#!/bin/bash
# Runs on GATEWAY. Submit the remaining qwen2_moe_57b qconfigs to B200.

set -euo pipefail

SHA="${1:-$(git rev-parse HEAD)}"
shift || true

REPO_LOCAL=/data_fast/home/deokjae/QUANT_works/MxMoE
EXP_REL=exps/b200_exp_20260521_qwen2_moe_57b_remain
B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
REMOTE_EXP=$B200_ROOT/MxMoE/$EXP_REL

cd "$REPO_LOCAL"

git remote add b200 https://github.com/badeok0716/MxMoE.git 2>/dev/null || \
git remote set-url b200 https://github.com/badeok0716/MxMoE.git

mkdir -p "$EXP_REL/logs" "$EXP_REL/results" "$EXP_REL/diag"
chmod 777 "$EXP_REL" "$EXP_REL/logs" "$EXP_REL/results" "$EXP_REL/diag"

if [[ "$#" -gt 0 ]]; then
    QCFGS=("$@")
else
    QCFGS=(
        w2_g128_asym
        w3_g128_asym
        w4_g128_asym
        w4_g-1_asym
        w8_g-1_asym
    )
fi

echo "=== using SHA: $SHA ==="
echo "=== remote b200: $(git remote get-url b200) ==="
echo "=== qcfgs: ${QCFGS[*]} ==="

for qcfg in "${QCFGS[@]}"; do
    submit_b200.sh --user "$USER" --ngpus 4 --ncpus 40 \
        --command "bash $REMOTE_EXP/b200_calib_one.sh $SHA $qcfg"
done
