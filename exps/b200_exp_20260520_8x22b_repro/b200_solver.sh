#!/bin/bash
# Runs ON B200 (CPU only, ~minute). Solves ILP for 9 total-bit budgets (12..20) using
# the factor pkls that b200_awareness.sh emitted into this exp dir.
# Optional — solver runs equally well on gateway; cheaper to do it there in fact.
#
# Submit from gateway with:
#   submit_b200.sh --user "$USER" --ngpus 1 --ncpus 2 \
#       --command "bash /NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae/MxMoE/exps/b200_exp_20260520_8x22b_repro/b200_solver.sh"

set -euo pipefail

B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae

REPO=$B200_ROOT/MxMoE
EXP=$REPO/exps/b200_exp_20260520_8x22b_repro
LOG=$EXP/logs/solver_$(date +%Y%m%d_%H%M%S).log

mkdir -p "$EXP/logs"
exec > >(tee -a "$LOG") 2>&1

echo "=== node: $(hostname) ==="
echo "=== start: $(date -Iseconds) ==="

cd "$EXP"
test -f experts_act_frequency.pkl || { echo "ERROR: run b200_awareness.sh first"; exit 1; }

uv run --project "$REPO" python -u "$REPO"/precision_solver.py \
    --actnum_path "$EXP"/experts_act_frequency.pkl \
    --weight_path "$EXP"/experts_act_weight.pkl \
    --quant_loss_path "$EXP"/experts_quant_loss.pkl \
    --save_path "$EXP"/experts_mixture_bit_selection

echo "=== end: $(date -Iseconds) ==="
ls -la "$EXP"/experts_mixture_bit_selection/
