#!/bin/bash
# Runs ON B200. fp16 (no quantization) baseline for Mixtral-8x22B-v0.1.
# Computes wikitext2 + c4 PPL with chunked forward then runs the same
# 8-task lm-eval set as the quantized runs. Needs 4 H200 (device_map='auto'
# shards 282 GB weights across them).
#
# Submit from gateway with:
#   B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
#   EXP=$B200_ROOT/Mixture-Compressor-MoE/exps/b200_exp_20260522_8x22b_correct
#   submit_b200.sh --user "$USER" --ngpus 4 --ncpus 32 \
#       --command "bash $EXP/b200_fp16_eval.sh"

set -euo pipefail

B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
export HF_HOME=$B200_ROOT/hf_cache

REPO=$B200_ROOT/Mixture-Compressor-MoE
EXP=$REPO/exps/b200_exp_20260522_8x22b_correct
MODEL="mistralai/Mixtral-8x22B-v0.1"

RESULT="$EXP/results/fp16.json"
LOG="$EXP/logs/fp16_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$EXP/logs" "$EXP/results"
exec > >(tee -a "$LOG") 2>&1

echo "=== node: $(hostname) ==="
nvidia-smi -L || true
echo "=== model: $MODEL  result: $RESULT ==="
echo "=== start: $(date -Iseconds) ==="
SECONDS=0

cd "$REPO"

uv run python "$EXP"/run_fp16_full.py \
    --model "$MODEL" \
    --output_path "$RESULT" \
    --batch_size 4

echo "=== end: $(date -Iseconds) elapsed=${SECONDS}s ==="
