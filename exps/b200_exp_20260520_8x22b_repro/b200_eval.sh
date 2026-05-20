#!/bin/bash
# Runs ON B200. Fake-quant mixed-precision PPL eval (wikitext2 + c4) on Mixtral-8x22B-v0.1.
#   $1  bitwidth bucket — one of {12,13,14,15,16,17,18,19,20}, picks the matching
#       experts_mixture_bitwidth_combination_<N>bit.pkl from this exp dir.
#
# Submit from gateway with:
#   for B in 12 13 14 15 16 17 18 19 20; do
#       submit_b200.sh --user "$USER" --ngpus 1 --ncpus 8 \
#           --command "bash /NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae/MxMoE/exps/b200_exp_20260520_8x22b_repro/b200_eval.sh $B"
#   done

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <bitwidth 12..20>"
    exit 1
fi
B="$1"

B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
export HF_HOME=$B200_ROOT/hf_cache

REPO=$B200_ROOT/MxMoE
EXP=$REPO/exps/b200_exp_20260520_8x22b_repro
PKL=$EXP/experts_mixture_bit_selection/experts_mixture_bitwidth_combination_${B}bit.pkl
LOG=$EXP/logs/eval_${B}bit_$(date +%Y%m%d_%H%M%S).log

mkdir -p "$EXP/logs"
exec > >(tee -a "$LOG") 2>&1

echo "=== node: $(hostname) ==="
nvidia-smi -L || true
echo "=== bit_budget: $B  pkl: $PKL ==="
test -f "$PKL" || { echo "ERROR: $PKL missing (run b200_solver.sh first)"; exit 1; }
echo "=== start: $(date -Iseconds) ==="
SECONDS=0

cd "$REPO"

# main.py loads model with device_map='cpu' then moves layer-by-layer to GPU.
# Single GPU is enough; activation cache lives in host RAM.
uv run python "$REPO"/main.py mistralai/Mixtral-8x22B-v0.1 \
    --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 \
    --eval_ppl --mixed_type mixed --precisions "$PKL"

echo "=== end: $(date -Iseconds) elapsed=${SECONDS}s ==="
