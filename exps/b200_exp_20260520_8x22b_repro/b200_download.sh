#!/bin/bash
# Runs ON B200. Downloads Mixtral-8x22B-v0.1 into $HF_HOME (cloud-local, ~282 GB).
# Submit from gateway with:
#   submit_b200.sh --user "$USER" --ngpus 1 --ncpus 4 \
#       --command "bash /NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae/MxMoE/exps/b200_exp_20260520_8x22b_repro/b200_download.sh"

set -euo pipefail

B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
export HF_HOME=$B200_ROOT/hf_cache

REPO=$B200_ROOT/MxMoE
EXP=$REPO/exps/b200_exp_20260520_8x22b_repro
LOG=$EXP/logs/download_$(date +%Y%m%d_%H%M%S).log

mkdir -p "$EXP/logs"
exec > >(tee -a "$LOG") 2>&1

echo "=== node: $(hostname) ==="
echo "=== HF_HOME: $HF_HOME ==="
echo "=== free disk on B200_ROOT ===" && df -h "$B200_ROOT" | tail -2
echo "=== start: $(date -Iseconds) ==="
SECONDS=0

cd "$REPO"
# Skip legacy consolidated PyTorch checkpoints (~280 GB extra).
uv run huggingface-cli download mistralai/Mixtral-8x22B-v0.1 \
    --exclude "consolidated*" "*.pt"

echo "=== end: $(date -Iseconds) elapsed=${SECONDS}s ==="
du -sh "$HF_HOME/hub/models--mistralai--Mixtral-8x22B-v0.1" || true
