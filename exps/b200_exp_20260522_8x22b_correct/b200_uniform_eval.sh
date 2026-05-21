#!/bin/bash
# Runs ON B200. Uniform-W eval for Mixtral-8x22B-v0.1.
#   $1  W ∈ {1, 2, 3, 4}   uniform expert weight bit; attn fixed at 4-bit.
#
# Uses the paper-faithful convention: --mixed_type mixed with a 56-block
# uniform pkl (all experts at bit=2). main.py then dispatches every expert
# through the "else" branch → args.wbits (which is W). Attention stays at
# args.attn_bits=4. (Matches gateway exp_20260521_uniform_eval_8x7b_base.)
#
# Submit from gateway with:
#   B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
#   EXP=$B200_ROOT/Mixture-Compressor-MoE/exps/b200_exp_20260522_8x22b_correct
#   for W in 1 2 3 4; do
#       submit_b200.sh --user "$USER" --ngpus 1 --ncpus 8 \
#           --command "bash $EXP/b200_uniform_eval.sh $W"
#   done

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <W>  (1, 2, 3, or 4)"; exit 1
fi
W="$1"
case "$W" in 1|2|3|4) ;; *) echo "ERROR: W must be 1/2/3/4 (got $W)"; exit 1 ;; esac

B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
export HF_HOME=$B200_ROOT/hf_cache

REPO=$B200_ROOT/Mixture-Compressor-MoE
EXP=$REPO/exps/b200_exp_20260522_8x22b_correct
MODEL="mistralai/Mixtral-8x22B-v0.1"
MODEL_SHORT="Mixtral-8x22B-v0.1"
SAVING_ROOT=$B200_ROOT/mcmoe_checkpoint_uniform

# 56-block all-bit-2 pkl committed in bit_selection_correct/Mixtral-8x22B-v0.1/.
PRECISIONS="$REPO/bit_selection_correct/$MODEL_SHORT/experts_mixture_bitwidth_uniform.pkl"
SAVE_DIR="$SAVING_ROOT/${MODEL_SHORT}-atten_4-e_${W}.0"
RESULT="$EXP/results/uniform_W${W}.json"
LOG="$EXP/logs/uniform_W${W}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$EXP/logs" "$EXP/results" "$SAVING_ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "=== node: $(hostname) ==="
nvidia-smi -L || true
echo "=== model: $MODEL  W=${W}bit  precisions: $PRECISIONS ==="
test -f "$PRECISIONS" || { echo "ERROR: missing $PRECISIONS"; exit 1; }
echo "=== save_dir: $SAVE_DIR  result: $RESULT ==="
echo "=== start: $(date -Iseconds) ==="
SECONDS=0

cd "$REPO"

echo "[step1] main.py — uniform W=${W}bit quant + PPL + save"
uv run python "$REPO"/main.py "$MODEL" \
    --wbits "${W}bit" --attn_bits 4bit --dataset wikitext2 --groupsize 128 \
    --eval_ppl --mixed_type mixed --precisions "$PRECISIONS" \
    --save --saving_path "$SAVING_ROOT"

test -f "$SAVE_DIR/qmodel.pt" || { echo "ERROR: qmodel.pt missing"; exit 1; }

echo
echo "[step2] run_lmeval_full.py — 6 zero-shot + mmlu/gsm8k 5-shot"
uv run python "$EXP"/run_lmeval_full.py \
    --save_dir "$SAVE_DIR" \
    --output_path "$RESULT" \
    --batch_size 8

echo "=== end: $(date -Iseconds) elapsed=${SECONDS}s ==="
