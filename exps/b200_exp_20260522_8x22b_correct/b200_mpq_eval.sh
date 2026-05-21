#!/bin/bash
# Runs ON B200. MPQ eval for Mixtral-8x22B-v0.1 with a bit_selection_correct
# pkl at one target T.
#
# Pipeline (same as the gateway 8x7B exp):
#   1. main.py --mixed_type mixed --precisions effavg_<T>.pkl --eval_ppl --save
#        → wikitext2 + c4 PPL (stdout, 3 "Perplexity:" lines: calib, wiki2, c4)
#        → packed checkpoint under $B200_ROOT/mcmoe_checkpoint_correct/...
#   2. run_lmeval_full.py
#        → 6 zero-shot + mmlu/gsm8k 5-shot merged JSON
#
# Submit from gateway with:
#   B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
#   EXP=$B200_ROOT/Mixture-Compressor-MoE/exps/b200_exp_20260522_8x22b_correct
#   for T in 1.500 1.625 1.750 1.875 2.000 2.125 2.250 2.375 2.500 2.625 2.750; do
#       submit_b200.sh --user "$USER" --ngpus 1 --ncpus 8 \
#           --command "bash $EXP/b200_mpq_eval.sh $T"
#   done

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <T> (e.g. 1.500 1.625 ... 2.750)"; exit 1
fi
T="$1"
T_STR=$(python3 -c "print(f'{float($T):.3f}')")

B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
export HF_HOME=$B200_ROOT/hf_cache

REPO=$B200_ROOT/Mixture-Compressor-MoE
EXP=$REPO/exps/b200_exp_20260522_8x22b_correct
MODEL="mistralai/Mixtral-8x22B-v0.1"
MODEL_SHORT="Mixtral-8x22B-v0.1"
SAVING_ROOT=$B200_ROOT/mcmoe_checkpoint_correct

PRECISIONS="$REPO/bit_selection_correct/$MODEL_SHORT/experts_mixture_bitwidth_effavg_${T_STR}.pkl"
SAVE_DIR="$SAVING_ROOT/${MODEL_SHORT}-atten_4-e_${T}"
RESULT="$EXP/results/mpq_T_${T_STR}.json"
LOG="$EXP/logs/mpq_T_${T_STR}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$EXP/logs" "$EXP/results" "$SAVING_ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "=== node: $(hostname) ==="
nvidia-smi -L || true
echo "=== model: $MODEL  T=$T  precisions: $PRECISIONS ==="
test -f "$PRECISIONS" || { echo "ERROR: missing $PRECISIONS"; exit 1; }
echo "=== save_dir: $SAVE_DIR  result: $RESULT ==="
echo "=== start: $(date -Iseconds) ==="
SECONDS=0

cd "$REPO"

echo "[step1] main.py — quant + PPL + save"
uv run python "$REPO"/main.py "$MODEL" \
    --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 \
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
