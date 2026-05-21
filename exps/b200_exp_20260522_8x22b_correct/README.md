# b200_exp_20260522_8x22b_correct

Mixtral-8x22B-v0.1 evaluation on **B200**, covering:

| group | settings | source pkl |
|---|---|---|
| fp16 (no quant) | 1 row | — |
| uniform | W ∈ {1, 2, 3, 4} | `bit_selection_correct/Mixtral-8x22B-v0.1/experts_mixture_bitwidth_uniform.pkl` (56-block all-bit-2) |
| MPQ correct | T ∈ {1.500, 1.625, …, 2.750} (11 rows) | `bit_selection_correct/Mixtral-8x22B-v0.1/experts_mixture_bitwidth_effavg_<T>.pkl` |

Each row contributes 3 measurements:
- **wikitext2 PPL** (test split)
- **c4 PPL**         (validation 0)
- **lm-eval 0.4.5** with 8 tasks:
    - 6 zero-shot (`arc_challenge, arc_easy, boolq, hellaswag, piqa, winogrande`)
    - 2 5-shot   (`mmlu, gsm8k`)

Everything is collated into `result.md` by `aggregate.py`.

This dir follows the same B200 wrapper-script discipline as
[`../b200_exp_20260520_8x22b_repro`](../b200_exp_20260520_8x22b_repro/): each
script is a single bash entry point that `submit_b200.sh` can invoke; no
inline `&&`/heredoc/chained commands.

---

## Phase 0 — gateway: prepare the artifacts and push

Already done on this branch:

- `precision_solver.py` extended with `--mode global_eff` (global ILP with
  effective bitwidths 1.125/2.25/3.25, no per-block diversity constraint).
- `bit_selection_correct/Mixtral-8x22B-v0.1/` populated with 11
  `experts_mixture_bitwidth_effavg_<T>.pkl` files + the 56-block
  `experts_mixture_bitwidth_uniform.pkl`.
- `main.py` updated to derive `average_bits` from `effavg_<T>.pkl` filenames
  (so `--save` writes to `…-e_<T>/` correctly).

To regenerate the bit_selection_correct pkls from the calib factor pkls
(`calib/Mixtral-8x22B-v0.1/experts_act_*.pkl`, already on gateway):

```bash
cd /data_fast/home/deokjae/QUANT_works/Mixture-Compressor-MoE
.venv/bin/python precision_solver.py \
    --actnum_path     calib/Mixtral-8x22B-v0.1/experts_act_frequency.pkl \
    --weight_path     calib/Mixtral-8x22B-v0.1/experts_act_weight.pkl \
    --quant_loss_path calib/Mixtral-8x22B-v0.1/experts_quant_loss.pkl \
    --save_path       bit_selection_correct/Mixtral-8x22B-v0.1 \
    --mode global_eff
```

Then push everything to the fork (same mechanics as
`b200_exp_20260520_8x22b_repro/README.md` Phase 0):

```bash
git add precision_solver.py main.py \
        bit_selection_correct/Mixtral-8x22B-v0.1/ \
        bit_selection_correct/Mixtral-8x7B-v0.1/ \
        exps/b200_exp_20260522_8x22b_correct/
git commit -m "8x22B B200 effective-bitwidth eval scaffold"
git push mxmoe HEAD:main
SHA=$(git rev-parse HEAD); echo "$SHA"
```

Save `$SHA` — every B200 wrapper script does a `git fetch && git checkout $SHA`
implicitly via the umbrella `b200_*.sh` convention (or you can checkout manually
in Phase 1).

---

## Phase 1 — B200: pull the new commit + ensure deps

Assuming `b200_exp_20260520_8x22b_repro/README.md` Phase 1 already
bootstrapped `$B200_ROOT/Mixture-Compressor-MoE` with a `uv venv`:

```bash
connect_ssh_b200.sh
B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
cd $B200_ROOT/Mixture-Compressor-MoE
git fetch mxmoe   # or `origin` if cloned that way
git checkout <SHA from Phase 0>
git checkout -- uv.lock || true
uv sync
exit
```

The c4 calibration shards (`data/c4-*.json`) should already be staged on B200
from `b200_exp_20260520_8x22b_repro/` Phase 2 — verify with
`ls $B200_ROOT/Mixture-Compressor-MoE/data/c4-*.json` and re-sftp if missing.

If the Mixtral-8x22B-v0.1 weights are not yet on B200, run
[`../b200_exp_20260520_8x22b_repro/b200_download.sh`](../b200_exp_20260520_8x22b_repro/b200_download.sh)
first (Phase 3 in that README).

---

## Phase 2 — gateway → B200 sftp of the bit_selection_correct pkls

The pkls were generated on gateway and need to ship to B200 because they
were not in `$SHA` (committed separately). Quick sftp:

```bash
# from gateway
cd /data_fast/home/deokjae/QUANT_works/Mixture-Compressor-MoE
connect_sftp_b200.sh <<EOF
cd /NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae/Mixture-Compressor-MoE
mkdir bit_selection_correct
cd bit_selection_correct
mkdir Mixtral-8x22B-v0.1
lcd bit_selection_correct/Mixtral-8x22B-v0.1
cd Mixtral-8x22B-v0.1
put experts_mixture_bitwidth_effavg_1.500.pkl
put experts_mixture_bitwidth_effavg_1.625.pkl
put experts_mixture_bitwidth_effavg_1.750.pkl
put experts_mixture_bitwidth_effavg_1.875.pkl
put experts_mixture_bitwidth_effavg_2.000.pkl
put experts_mixture_bitwidth_effavg_2.125.pkl
put experts_mixture_bitwidth_effavg_2.250.pkl
put experts_mixture_bitwidth_effavg_2.375.pkl
put experts_mixture_bitwidth_effavg_2.500.pkl
put experts_mixture_bitwidth_effavg_2.625.pkl
put experts_mixture_bitwidth_effavg_2.750.pkl
put experts_mixture_bitwidth_uniform.pkl
bye
EOF
```

(If you committed the pkls into git in Phase 0 instead — depends on repo
size policy — skip this; the `git checkout $SHA` will have placed them.)

---

## Phase 3 — B200 jobs

Each script is parameterised by a single positional arg and writes its own
log under `logs/`. `submit_b200.sh` runs the script as a one-shot wrapper.

### 3a. fp16 baseline (4× H200, ~282 GB sharded)

```bash
B200_ROOT=/NHNHOME/WORKSPACE/0226010285_A/mllab/deokjae
EXP=$B200_ROOT/Mixture-Compressor-MoE/exps/b200_exp_20260522_8x22b_correct

submit_b200.sh --user "$USER" --ngpus 4 --ncpus 32 \
    --command "bash $EXP/b200_fp16_eval.sh"
```

Driver: [`run_fp16_full.py`](run_fp16_full.py) loads with `device_map='auto'`,
chunked PPL on wiki2/c4, then 8 lm-eval tasks (HFLM wraps the sharded model).
Output:
- `logs/fp16_<ts>.log` — 2 `Perplexity:` lines (wiki2, c4) + lm-eval prints
- `results/fp16.json`  — merged lm-eval JSON

### 3b. Uniform W ∈ {1, 2, 3, 4} (1× H200 each)

```bash
for W in 1 2 3 4; do
    submit_b200.sh --user "$USER" --ngpus 1 --ncpus 8 \
        --command "bash $EXP/b200_uniform_eval.sh $W"
done
```

Each job: `main.py --mixed_type mixed --precisions <uniform.pkl> --wbits ${W}bit`
→ quantize + wiki2/c4 PPL + save under
`$B200_ROOT/mcmoe_checkpoint_uniform/Mixtral-8x22B-v0.1-atten_4-e_${W}.0/`,
then `run_lmeval_full.py` → 8 tasks JSON.

### 3c. MPQ correct T ∈ {1.500, …, 2.750} (1× H200 each)

```bash
for T in 1.500 1.625 1.750 1.875 2.000 2.125 2.250 2.375 2.500 2.625 2.750; do
    submit_b200.sh --user "$USER" --ngpus 1 --ncpus 8 \
        --command "bash $EXP/b200_mpq_eval.sh $T"
done
```

Each job: `main.py --mixed_type mixed --precisions effavg_<T>.pkl --wbits 2bit
--attn_bits 4bit` → quantize + PPL + save under
`$B200_ROOT/mcmoe_checkpoint_correct/Mixtral-8x22B-v0.1-atten_4-e_${T}/`,
then `run_lmeval_full.py` → 8 tasks JSON.

Watch:

```bash
get_b200_queue.sh
```

---

## Phase 4 — pull back & aggregate

```bash
# gateway
cd /data_fast/home/deokjae/QUANT_works/Mixture-Compressor-MoE
bash exps/b200_exp_20260522_8x22b_correct/gateway_pull.sh

.venv/bin/python exps/b200_exp_20260522_8x22b_correct/aggregate.py
# → exps/b200_exp_20260522_8x22b_correct/result.md
```

`aggregate.py` is the 8x22B sibling of
`exps/exp_20260522_mpq_correct_8x7b_base/aggregate.py`: same task list, same
metric conventions (`acc_norm` for arc/hellaswag/piqa, `acc` for boolq/winogrande,
`acc` for mmlu, `exact_match,strict-match` for gsm8k, mean of 6 zero-shot for
`avg_zeroshot`).

---

## File layout

```
exps/b200_exp_20260522_8x22b_correct/
├── README.md             # this file
├── run_lmeval_full.py    # B200: HFLM(qmodel) + simple_evaluate × 2
├── run_fp16_full.py      # B200: device_map='auto' fp16 PPL + lm-eval
├── b200_fp16_eval.sh     # wrapper (4× GPU)
├── b200_uniform_eval.sh  # wrapper (1× GPU, $1 = W ∈ {1..4})
├── b200_mpq_eval.sh      # wrapper (1× GPU, $1 = T ∈ {1.5..2.75})
├── gateway_pull.sh       # gateway: sftp logs/ + results/ back
├── aggregate.py          # gateway: render result.md
├── logs/                 # B200 → gateway landing
└── results/              # B200 → gateway landing
```

## Gotchas (recap from EXECUTION.md §B200)

- `submit_b200.sh --command` is not a real shell — every wrapper must be a
  single pre-staged `bash <abs path>` invocation; inline `&&` silently fails.
- `chmod 777 logs/ results/` so the `hosking` sftp user can write.
- Before `git checkout $SHA` on B200, `git checkout -- uv.lock` to discard the
  uv-sync-dirtied lock.
- lm-eval 0.4.5 is already pinned in `pyproject.toml` / `uv.lock`; no version
  upgrade needed.
