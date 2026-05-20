#!/bin/bash
# Runs on GATEWAY. Pull B200 logs/results back into this experiment directory.

set -euo pipefail

REPO_LOCAL=/data_fast/home/deokjae/QUANT_works/MxMoE
EXP_REL=exps/b200_exp_20260521_qwen2_moe_57b_remain
B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae

cd "$REPO_LOCAL"
mkdir -p "$EXP_REL/logs" "$EXP_REL/results" "$EXP_REL/diag"
chmod 777 "$EXP_REL" "$EXP_REL/logs" "$EXP_REL/results" "$EXP_REL/diag"

connect_sftp_b200.sh <<EOF
cd $B200_ROOT/MxMoE/$EXP_REL
lcd $REPO_LOCAL/$EXP_REL
get -r logs
get -r results
get -r diag
bye
EOF

echo "=== local result files ==="
find "$EXP_REL/results" -maxdepth 1 -type f -name '*.json' -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort
