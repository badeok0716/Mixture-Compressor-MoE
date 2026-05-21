#!/bin/bash
# Run on gateway. Pull logs/ + results/ from B200 into this exp dir.
#
# Requires `connect_sftp_b200.sh` from the user's umbrella (see EXECUTION.md).
# The B200 side writes into:
#   $B200_ROOT/Mixture-Compressor-MoE/exps/b200_exp_20260522_8x22b_correct/{logs,results}/

set -euo pipefail
EXP=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$EXP/logs" "$EXP/results"

connect_sftp_b200.sh <<EOF
cd /NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae/Mixture-Compressor-MoE/exps/b200_exp_20260522_8x22b_correct
lcd $EXP
get -r logs
get -r results
bye
EOF

echo "=== pull done ==="
ls -la "$EXP/logs" "$EXP/results" 2>/dev/null | head -40 || true
