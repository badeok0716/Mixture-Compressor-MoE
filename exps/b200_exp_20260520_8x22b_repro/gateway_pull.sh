#!/bin/bash
# Runs on GATEWAY. Pulls B200-produced artifacts (factor pkls, bit-selection pkls,
# logs) back into this exp dir so we have local copies to feed gateway-side
# precision_solver / main.py if desired.

set -euo pipefail

EXP_REL=exps/b200_exp_20260520_8x22b_repro
REPO_LOCAL=/data_fast/home/deokjae/QUANT_works/Mixture-Compressor-MoE
B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae

cd "$REPO_LOCAL"

# Receiving dir must be world-writable (sftp runs as user `hosking`).
chmod 777 "$EXP_REL" "$EXP_REL/logs" "$EXP_REL/pkls" 2>/dev/null || true

connect_sftp_b200.sh <<EOF
cd $B200_ROOT/MxMoE/$EXP_REL
lcd $REPO_LOCAL/$EXP_REL
get experts_act_frequency.pkl    pkls/experts_act_frequency.pkl
get experts_act_weight.pkl       pkls/experts_act_weight.pkl
get experts_quant_loss.pkl       pkls/experts_quant_loss.pkl
get -r experts_mixture_bit_selection
get -r logs
bye
EOF

echo "=== local artifacts ==="
ls -la "$REPO_LOCAL/$EXP_REL/pkls/" "$REPO_LOCAL/$EXP_REL/experts_mixture_bit_selection/" 2>/dev/null || true
