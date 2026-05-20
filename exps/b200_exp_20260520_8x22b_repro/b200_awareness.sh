#!/bin/bash
# Runs ON B200. awareness step for Mixtral-8x22B-v0.1.
#   $1  optional git SHA to check out before running (default: leave working tree as-is)
#
# Submit from gateway with:
#   SHA=$(git rev-parse HEAD)        # from MxMoE checkout on gateway, after push
#   submit_b200.sh --user "$USER" --ngpus 4 --ncpus 40 \
#       --command "bash /NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae/MxMoE/exps/b200_exp_20260520_8x22b_repro/b200_awareness.sh $SHA"
#
# Output factor pkls land in this exp dir (under exps/.../) — pull them back via gateway_pull.sh.

set -euo pipefail

B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
export HF_HOME=$B200_ROOT/hf_cache

REPO=$B200_ROOT/MxMoE
EXP=$REPO/exps/b200_exp_20260520_8x22b_repro
LOG=$EXP/logs/awareness_$(date +%Y%m%d_%H%M%S).log

mkdir -p "$EXP/logs"
exec > >(tee -a "$LOG") 2>&1

echo "=== node: $(hostname) ==="
nvidia-smi -L || true
echo "=== HF_HOME: $HF_HOME ==="
echo "=== start: $(date -Iseconds) ==="
SECONDS=0

cd "$REPO"

# Pin code to a specific gateway-pushed commit if requested.
# EXECUTION.md Constraint 5a: clear uv.lock dirt from prior `uv sync` before checkout.
if [[ $# -ge 1 && -n "${1:-}" ]]; then
    SHA="$1"
    git fetch origin
    git checkout -- uv.lock 2>/dev/null || true
    git checkout "$SHA"
    echo "=== checked out SHA: $(git rev-parse HEAD) ==="
fi

# Sync python deps to whatever is now on disk.
uv sync

# Verify the c4 calibration shard is present on B200 (not in git — sftp'd once during setup).
test -f data/c4-train.00000-of-01024.json || {
    echo "ERROR: data/c4-train.00000-of-01024.json missing on B200."
    echo "  sftp it once from gateway: connect_sftp_b200.sh -> put data/c4-train.00000-of-01024.json $REPO/data/"
    exit 1
}

# awareness.py writes 3 pkls into cwd. Run from the exp dir so pkls land there
# instead of polluting repo root.
cd "$EXP"
test -L data || ln -s ../../data data

uv run --project "$REPO" python "$REPO"/awareness.py mistralai/Mixtral-8x22B-v0.1 --calibration c4

echo "=== end: $(date -Iseconds) elapsed=${SECONDS}s ==="
ls -la "$EXP"/experts_act_frequency.pkl "$EXP"/experts_act_weight.pkl "$EXP"/experts_quant_loss.pkl

# Sanity check: 8x22B has 56 layers — pkls should have 56 keys.
uv run --project "$REPO" python - <<'PY'
import pickle
for f in ("experts_act_frequency.pkl", "experts_act_weight.pkl", "experts_quant_loss.pkl"):
    with open(f, "rb") as fh:
        d = pickle.load(fh)
    print(f"{f}: {len(d)} blocks  (expect 56 for 8x22B)")
PY
