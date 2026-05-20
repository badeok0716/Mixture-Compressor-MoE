# b200_exp_20260520_8x22b_repro

Mixtral-8x22B MCMoE reproduction on B200 (cloud node, 4× H200-class GPU). The
8x22B model (~282 GB fp16) does not fit Slurm partitions on the gateway; B200 is
the only path. See [`../../../EXECUTION.md` §B200](../../../EXECUTION.md) for
the umbrella conventions referenced below.

Pipeline (same 3 steps as the paper):

1. `awareness.py` (c4 calib) → `experts_act_frequency.pkl`,
   `experts_act_weight.pkl`, `experts_quant_loss.pkl` — **on B200** (only step
   that truly benefits from 4× H200).
2. `precision_solver.py` (ILP) → 9 `experts_mixture_bitwidth_combination_*bit.pkl`
   — on B200 (CPU minute job) **or** on gateway (`exps/exp_*_step2_solver_8x7b/`
   pattern), either works.
3. `main.py` (GPTQ + PPL eval) → wikitext2 / c4 numbers — on B200 (1 GPU
   sufficient; main.py uses CPU device_map + layer-by-layer GPU shuttling).

All B200 wrapper scripts live in this folder and are committed in the
repo — after the gateway-side push to `badeok0716/MxMoE` and a `git fetch &&
git checkout <SHA>` on B200, they are present on B200 disk for
`submit_b200.sh` to invoke.

---

## Phase 0 — gateway: push code to the MxMoE remote

The user's fork (assumed already added as a `mxmoe` remote on gateway, **or**
swap to `origin` of a fresh `git clone`):

```bash
cd /data_fast/home/deokjae/QUANT_works/Mixture-Compressor-MoE

# One-time: add the fork as a remote (skip if already done).
git remote add mxmoe https://github.com/badeok0716/MxMoE.git

# Commit your local edits (notably the expert_weight.py device-split patch
# that this 8x22B repro depends on) and push.
git add expert_weight.py exps/b200_exp_20260520_8x22b_repro/
git commit -m "8x22B B200 repro setup"
git push mxmoe HEAD:main      # or whichever branch you want B200 to track
SHA=$(git rev-parse HEAD); echo "$SHA"
```

Save the `$SHA` — every B200 `submit_b200.sh` invocation below pins to it so
the on-B200 working tree matches gateway intent.

---

## Phase 1 — B200: one-time bootstrap (manual SSH)

B200 has no NFS to gateway; the repo and Python env must exist on B200 first.
`connect_ssh_b200.sh` is the only path for this kind of one-shot setup
(submit_b200.sh expects already-staged scripts to run).

```bash
connect_ssh_b200.sh
# --- inside B200 SSH session ---
B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
cd $B200_ROOT

git clone https://github.com/badeok0716/MxMoE.git
cd MxMoE
uv python install 3.10
uv venv --python 3.10 --python-preference only-managed
uv sync

# Sanity check: token already seeded; gated repos should auth.
export HF_HOME=$B200_ROOT/hf_cache
uv run huggingface-cli whoami
exit
```

---

## Phase 2 — gateway → B200: stage c4 calibration data

`data/c4-train.00000-of-01024.json` (~820 MB) is *not* tracked in git. Push it
to B200 once via sftp (also stage the c4 validation shard while at it, used by
the `--eval_ppl c4` branch in main.py).

```bash
# from gateway
cd /data_fast/home/deokjae/QUANT_works/Mixture-Compressor-MoE
connect_sftp_b200.sh <<EOF
cd /NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae/MxMoE/data
put data/c4-train.00000-of-01024.json      c4-train.00000-of-01024.json
put data/c4-validation.00000-of-00008.json c4-validation.00000-of-00008.json
bye
EOF
```

---

## Phase 3 — B200 job: download Mixtral-8x22B-v0.1 (~282 GB)

One-shot, queued via `submit_b200.sh`. Wrapper [`b200_download.sh`](b200_download.sh)
runs `huggingface-cli download` excluding the legacy consolidated checkpoints
(~280 GB of redundant `.pt` shards).

```bash
B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
EXP=$B200_ROOT/MxMoE/exps/b200_exp_20260520_8x22b_repro

submit_b200.sh --user "$USER" --ngpus 1 --ncpus 4 \
    --command "bash $EXP/b200_download.sh"

# Watch:
get_b200_queue.sh
# Log: $EXP/logs/download_<timestamp>.log on B200 (sftp tail or pull at end).
```

---

## Phase 4 — B200 job: awareness (4× H200)

Wrapper [`b200_awareness.sh`](b200_awareness.sh) checks out `$SHA`, syncs deps,
runs `awareness.py mistralai/Mixtral-8x22B-v0.1 --calibration c4`. Output pkls
land in this exp dir on B200.

Resource note: 8x22B in fp16 ≈ 282 GB. `device_map='auto'` shards across 4
H200 (4 × 180 GB = 720 GB). Activation cache (`CacheDataset` for Xs / Zs)
lives on CPU and reaches ~360 GB during step 2 (layerwise_quant) — B200 host
RAM must be ≥ that. The `expert_weight.py` device-split patch (committed in
`$SHA`) is what makes this work for 56-layer Mixtral; do not skip Phase 0.

```bash
B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
EXP=$B200_ROOT/MxMoE/exps/b200_exp_20260520_8x22b_repro
SHA=<paste from Phase 0>

submit_b200.sh --user "$USER" --ngpus 4 --ncpus 40 \
    --command "bash $EXP/b200_awareness.sh $SHA"

get_b200_queue.sh
```

Expected runtime: hours (proportional to 8x7B awareness × ~5 for per-layer
size × ~1.75 for 56/32 layer count).

---

## Phase 5 — solver (cheap; either side)

Option A — on B200, in this exp dir:

```bash
submit_b200.sh --user "$USER" --ngpus 1 --ncpus 2 \
    --command "bash $EXP/b200_solver.sh"
```

Option B — on gateway after pulling the factor pkls back (Phase 7), reusing
the existing `exps/exp_*_step2_solver_*` scaffold. Gateway has a Gurobi
license file at `$HOME/gurobi.lic` already; B200 relies on the pip-bundled
trial license, which is sufficient for this 24-binary-var ILP per block but
not guaranteed across Gurobi versions.

---

## Phase 6 — eval sweep (mixed-precision fake-quant PPL)

Wrapper [`b200_eval.sh`](b200_eval.sh) takes a bit-budget (12..20) and runs
`main.py` with the corresponding bit-selection pkl. Submit one job per
budget — `submit_b200.sh` has no array mode, so a shell loop is the pattern.

```bash
for B in 12 13 14 15 16 17 18 19 20; do
    submit_b200.sh --user "$USER" --ngpus 1 --ncpus 8 \
        --command "bash $EXP/b200_eval.sh $B"
done
get_b200_queue.sh
```

Each job logs to `$EXP/logs/eval_<B>bit_<timestamp>.log`.

---

## Phase 7 — gateway: pull artifacts back

[`gateway_pull.sh`](gateway_pull.sh) sftp's the 3 factor pkls, the 9 bit-selection
pkls, and the `logs/` directory back into this exp dir on gateway.

```bash
cd /data_fast/home/deokjae/QUANT_works/Mixture-Compressor-MoE
bash exps/b200_exp_20260520_8x22b_repro/gateway_pull.sh
```

PPL numbers themselves are in `logs/eval_<B>bit_*.log` (search for
`wikitext2` / `c4` blocks).

---

## File layout summary

```
exps/b200_exp_20260520_8x22b_repro/
├── README.md             # this file
├── b200_download.sh      # B200: huggingface-cli download Mixtral-8x22B
├── b200_awareness.sh     # B200: awareness.py (4× H200)
├── b200_solver.sh        # B200: precision_solver.py  (optional — see Phase 5)
├── b200_eval.sh          # B200: main.py mixed PPL eval, arg = bit budget
├── gateway_pull.sh       # Gateway: sftp results back
├── logs/                 # local landing for sftp'd B200 logs (chmod 777)
└── pkls/                 # local landing for sftp'd factor pkls (chmod 777)
```

## Gotchas (recap from EXECUTION.md §B200)

- `submit_b200.sh --command` is not a real shell. **Always** invoke a single
  pre-staged wrapper (`bash <abs path>`); inline `&&` / heredocs silently
  fail with no log, no queue entry.
- `chmod 777` the exp dir + `logs/` + `pkls/` on gateway so the `hosking`-as
  sftp user can write.
- Before `git checkout <SHA>` on B200, run `git checkout -- uv.lock`
  (uv sync in a prior run dirties the lock; `b200_awareness.sh` does this).
- Use **`mxmoe` (or whichever remote name)** for the fork; do not push to
  upstream `Aaronhuang-778/Mixture-Compressor-MoE`.
