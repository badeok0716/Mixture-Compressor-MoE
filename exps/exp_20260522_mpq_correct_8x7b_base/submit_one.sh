#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --partition=a100,h100,ada

# One job per target T ∈ {1.500, 1.625, ..., 2.750}. Two-step pipeline:
#   1. main.py --mixed_type mixed --precisions <effavg_T.pkl> --eval_ppl --save
#        → quant + wiki2/c4 PPL (stdout)
#        → /storage/deokjae/mcmoe_checkpoint_correct/Mixtral-8x7B-v0.1-atten_4-e_<T>/
#   2. run_lmeval_full.py
#        → results/T_<T>.json  (6 zero-shot + mmlu/gsm8k 5-shot)
#
# Usage (via submit_all.sh):
#   sbatch --export=ALL,T=<float> --job-name=mpq_correct_8x7b_T<T> \
#          --output=<exp>/logs/T_<T>.out --error=<exp>/logs/T_<T>.err submit_one.sh

set -euo pipefail

REPO=/data_fast/home/deokjae/QUANT_works/Mixture-Compressor-MoE
EXP=$REPO/exps/exp_20260522_mpq_correct_8x7b_base
MODEL="mistralai/Mixtral-8x7B-v0.1"
MODEL_SHORT="Mixtral-8x7B-v0.1"
SAVING_ROOT=/storage/deokjae/mcmoe_checkpoint_correct

if [[ -z "${T:-}" ]]; then
    echo "ERROR: T env var required (one of 1.500..2.750 step 0.125)"; exit 1
fi

# Three-decimal target string used in pkl name + main.py path.
T_STR=$(python3 -c "print(f'{float($T):.3f}')")
PRECISIONS="$REPO/bit_selection_correct/$MODEL_SHORT/experts_mixture_bitwidth_effavg_${T_STR}.pkl"
SAVE_DIR="$SAVING_ROOT/${MODEL_SHORT}-atten_4-e_${T}"
RESULT="$EXP/results/T_${T_STR}.json"

test -f "$PRECISIONS" || { echo "ERROR: missing $PRECISIONS"; exit 1; }

export HF_HOME=/storage/deokjae/.cache
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1

echo "=== node: $(hostname) ==="
nvidia-smi -L || true
echo "=== model: $MODEL  T=$T  precisions: $PRECISIONS ==="
echo "=== save_dir: $SAVE_DIR  result: $RESULT ==="
echo "=== start: $(date -Iseconds) ==="
SECONDS=0

mkdir -p "$SAVING_ROOT" "$EXP/results"
cd "$REPO"

# ---- Step 1: quantize + PPL + save ----
echo "[step1] main.py — quant + PPL + save"
"$REPO"/.venv/bin/python "$REPO"/main.py "$MODEL" \
    --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 \
    --eval_ppl --mixed_type mixed --precisions "$PRECISIONS" \
    --save --saving_path "$SAVING_ROOT"

test -f "$SAVE_DIR/qmodel.pt" || { echo "ERROR: qmodel.pt missing after main.py"; exit 1; }

# ---- Step 2: lm_eval (6 zero-shot + mmlu/gsm8k 5-shot) ----
echo
echo "[step2] run_lmeval_full.py — 6 zero-shot + mmlu/gsm8k 5-shot"
"$REPO"/.venv/bin/python "$EXP"/run_lmeval_full.py \
    --save_dir "$SAVE_DIR" \
    --output_path "$RESULT" \
    --batch_size 32

echo "=== end: $(date -Iseconds) elapsed=${SECONDS}s ==="
echo "=== saved: $SAVE_DIR ==="
echo "=== result: $RESULT ==="
