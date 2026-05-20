#!/bin/bash
# Runs on GATEWAY. Upload b200_setup.sh to B200_ROOT and submit it.

set -euo pipefail

SHA="${1:-$(git rev-parse HEAD)}"

REPO_LOCAL=/data_fast/home/deokjae/QUANT_works/MxMoE
EXP_REL=exps/b200_exp_20260521_qwen2_moe_57b_remain
B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
REMOTE_SETUP=$B200_ROOT/mxmoe_qwen57b_setup.sh

cd "$REPO_LOCAL"

git remote add b200 https://github.com/badeok0716/MxMoE.git 2>/dev/null || \
git remote set-url b200 https://github.com/badeok0716/MxMoE.git

mkdir -p "$EXP_REL/logs" "$EXP_REL/results" "$EXP_REL/diag"
chmod 777 "$EXP_REL" "$EXP_REL/logs" "$EXP_REL/results" "$EXP_REL/diag"

echo "=== using SHA: $SHA ==="
echo "=== remote b200: $(git remote get-url b200) ==="
echo "Make sure this SHA has been pushed to the b200 remote before setup runs."

connect_sftp_b200.sh <<EOF
cd $B200_ROOT
put $EXP_REL/b200_setup.sh mxmoe_qwen57b_setup.sh
chmod 755 mxmoe_qwen57b_setup.sh
bye
EOF

submit_b200.sh --user "$USER" --ngpus 1 --ncpus 20 \
    --command "bash $REMOTE_SETUP $SHA"
