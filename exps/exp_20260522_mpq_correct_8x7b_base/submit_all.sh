#!/bin/bash
# Launch 11 jobs for Mixtral-8x7B-v0.1 base with bit_selection_correct MPQ
# at T ∈ {1.500, 1.625, ..., 2.750}. Each job runs:
#     main.py (quant + PPL + save) → run_lmeval_full.py (8 lm-eval tasks)

set -euo pipefail
EXP=$(cd "$(dirname "$0")" && pwd)

for T in 1.500 1.625 1.750 1.875 2.000 2.125 2.250 2.375 2.500 2.625 2.750; do
    PKL=/data_fast/home/deokjae/QUANT_works/Mixture-Compressor-MoE/bit_selection_correct/Mixtral-8x7B-v0.1/experts_mixture_bitwidth_effavg_${T}.pkl
    if [[ ! -f "$PKL" ]]; then
        echo "SKIP T=$T: precisions pkl missing ($PKL)"
        continue
    fi
    JID=$(sbatch --parsable \
        --job-name="mpq_correct_8x7b_T${T}" \
        --output="$EXP/logs/T_${T}.out" --error="$EXP/logs/T_${T}.err" \
        --export=ALL,T=$T \
        "$EXP/submit_one.sh")
    echo "T=$T  JOB=$JID  pkl=$PKL"
done
