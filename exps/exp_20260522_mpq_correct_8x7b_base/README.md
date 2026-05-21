# exp_20260522_mpq_correct_8x7b_base

Mixtral-8x7B-v0.1 MPQ evaluation against the **effective-bitwidth-aware**
solutions in `bit_selection_correct/Mixtral-8x7B-v0.1/`.

## Why a fresh exp dir

The legacy MPQ sweep (`exp_20260521_eval_mixed_8x7b_base` +
`exp_20260521_lmeval_8x7b_base`) used the per-block ILP with *nominal* bit
coefficients (1, 2, 3) and a budget N ∈ {12..20}. The realized average
effective bit (1.125 / 2.25 / 3.25 with g=128 sym/asym overhead) was
**always higher than the target nominal avg**.

`precision_solver.py` was extended with a `--mode global_eff` global ILP that
optimizes under the true effective-bit constraint, producing 11 pkls in
`bit_selection_correct/Mixtral-8x7B-v0.1/` covering
T ∈ {1.500, 1.625, 1.750, 1.875, 2.000, 2.125, 2.250, 2.375, 2.500, 2.625, 2.750}
(0.125 step). The realized layer-wise effective avg matches T to within
~3e-3 (slack < min-transition cost of 9 weighted units).

## Single-pipeline-per-T design

Per target T one sbatch job does:

1. **Quantize + PPL + save** via `main.py`
   - `--mixed_type mixed --precisions bit_selection_correct/.../effavg_<T>.pkl`
   - reports calib / wikitext2 / c4 PPL to stdout
   - packs and saves to
     `/storage/deokjae/mcmoe_checkpoint_correct/Mixtral-8x7B-v0.1-atten_4-e_<T>/`
   - `main.py` was patched (effavg pkl filename regex) so `--save` derives
     `average_bits = T` directly from the pkl filename.
2. **lm-eval** via [`run_lmeval_full.py`](run_lmeval_full.py)
   - loads the packed model once, wraps in `HFLM`, calls `simple_evaluate`
     twice on the **same** in-memory wrapper:
       - 6 zero-shot tasks (num_fewshot=0):
         `arc_challenge, arc_easy, boolq, hellaswag, piqa, winogrande`
       - 2 5-shot tasks (num_fewshot=5):
         `mmlu, gsm8k`
   - merges both passes into a single `results/T_<T>.json`.

This keeps quantize/PPL/lm-eval under one exp dir as the user requested.

## Storage layout

```
/storage/deokjae/mcmoe_checkpoint_correct/
    Mixtral-8x7B-v0.1-atten_4-e_<T>/    # T = 1.5, 1.625, ..., 2.75
        qmodel.pt
        config.json
        tokenizer.*
```

This root is **separate** from the legacy `/storage/deokjae/mcmoe_checkpoint/`
(MPQ N=12..20 → e_1.5..e_2.5 collides path-wise with new T=1.5..2.5) and
from `/storage/deokjae/mcmoe_checkpoint_uniform/` (uniform W=1..4).

## Resources & expected wall-time per T

| step | resources | typical wall-time |
|---|---|---|
| quant + PPL + save | 1 GPU / 20 cpu / 150G / a100 or h100 | 60–90 min |
| lm-eval (6 zero-shot)     | (same node) | 20–30 min |
| lm-eval (mmlu + gsm8k 5-shot) | (same node) | 60–120 min |
| total per job  | 12 h walltime cap | ~2.5–4 h |

11 jobs → ~30–45 GPU-hours; runs in parallel as the queue allows.

## Run

```bash
# 1. (one-time) bit_selection_correct pkls — already produced by
#    precision_solver.py --mode global_eff and committed to
#    bit_selection_correct/Mixtral-8x7B-v0.1/. To regenerate:
.venv/bin/python precision_solver.py \
    --actnum_path     calib/Mixtral-8x7B-v0.1/experts_act_frequency.pkl \
    --weight_path     calib/Mixtral-8x7B-v0.1/experts_act_weight.pkl \
    --quant_loss_path calib/Mixtral-8x7B-v0.1/experts_quant_loss.pkl \
    --save_path       bit_selection_correct/Mixtral-8x7B-v0.1 \
    --mode global_eff

# 2. submit 11 sbatch jobs (one per T)
bash exps/exp_20260522_mpq_correct_8x7b_base/submit_all.sh

# 3. once jobs complete, build the markdown table
.venv/bin/python exps/exp_20260522_mpq_correct_8x7b_base/aggregate.py
# → exps/exp_20260522_mpq_correct_8x7b_base/result.md
```

## Output

- `results/T_<T>.json` — merged lm-eval JSON (6 zero-shot + mmlu + gsm8k)
- `logs/T_<T>.out`     — main.py stdout (PPL lines) + run_lmeval_full.py
- `result.md`          — final aggregated table (one row per T)
