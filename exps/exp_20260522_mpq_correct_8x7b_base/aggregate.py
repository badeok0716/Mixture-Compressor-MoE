"""Aggregate exp_20260522_mpq_correct_8x7b_base into result.md.

Per-target (T ∈ {1.500, 1.625, ..., 2.750}) source files:
    - logs/T_<T>.out      stdout from main.py + run_lmeval_full.py
                          (PPL: 3 "Perplexity:" lines in main.py order
                           calib / wikitext2 / c4)
    - results/T_<T>.json  merged lm-eval JSON
                          (6 zero-shot + mmlu/gsm8k 5-shot)

Columns reported:
    eff_bit_moe  = realized effective avg bit per expert across the layer
                   (recomputed from the bit_selection_correct pkl using
                    EFFECTIVE_BIT mapping below)
    wiki2_ppl    = wikitext2 (test) PPL  (main.py)
    c4_ppl       = c4 (validation 0) PPL (main.py)
    arc_challenge, arc_easy, hellaswag, piqa  : acc_norm
    boolq, winogrande                         : acc
    avg_zeroshot = arithmetic mean of the 6 zero-shot tasks above
    mmlu                                       : acc   (5-shot)
    gsm8k                                      : exact_match (strict_match) (5-shot)
    config       = path to the bit_selection_correct pkl

Output:
    exps/exp_20260522_mpq_correct_8x7b_base/result.md
    stdout listing of missing per-row entries.
"""

import json
import pickle
import re
from pathlib import Path

EXP = Path(__file__).resolve().parent
REPO = EXP.parent.parent
RESULTS = EXP / "results"
LOGS = EXP / "logs"

EFFECTIVE_BIT = {1: 1.125, 2: 2.25, 3: 3.25}

TARGETS = ["1.500", "1.625", "1.750", "1.875", "2.000", "2.125",
           "2.250", "2.375", "2.500", "2.625", "2.750"]

ZEROSHOT_TASKS = [
    ("arc_challenge", "acc_norm"),
    ("arc_easy",      "acc_norm"),
    ("boolq",         "acc"),
    ("hellaswag",     "acc_norm"),
    ("piqa",          "acc_norm"),
    ("winogrande",    "acc"),
]
# gsm8k 0.4.5 default metric in the strict_match variant. Fallback handled.
GSM8K_METRIC_CANDIDATES = ["exact_match,strict-match", "exact_match,flexible-extract",
                          "exact_match"]
MMLU_METRIC = "acc"


def _moe_eff_avg(pkl_path: Path) -> float | None:
    if not pkl_path.is_file():
        return None
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    total, n = 0.0, 0
    for blk in d.values():
        for v in blk.values():
            total += EFFECTIVE_BIT[v]
            n += 1
    return total / n if n else None


def _parse_ppl(log_path: Path) -> tuple[float | None, float | None]:
    """3 'Perplexity:' lines in main.py order: calib, wikitext2, c4."""
    if not log_path.exists():
        return None, None
    vals = []
    for line in log_path.read_text(errors="ignore").splitlines():
        m = re.match(r"^Perplexity:\s+([\d.eE+-]+)", line)
        if m:
            vals.append(float(m.group(1)))
    if len(vals) >= 3:
        return vals[1], vals[2]
    return None, None


def _pick_metric(metrics: dict, key: str) -> float | None:
    """Try '<key>,none' first then bare '<key>'."""
    for k in (f"{key},none", key):
        if k in metrics and isinstance(metrics[k], (int, float)):
            return metrics[k]
    return None


def _pick_gsm8k(metrics: dict) -> tuple[str, float] | tuple[None, None]:
    for k in GSM8K_METRIC_CANDIDATES:
        if k in metrics and isinstance(metrics[k], (int, float)):
            return k, metrics[k]
    # last resort: first numeric non-stderr
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not k.endswith("_stderr,none"):
            return k, v
    return None, None


def _pick_mmlu(results: dict) -> float | None:
    """lm-eval 0.4.5 emits a grouped 'mmlu' entry; fallback = mean of mmlu_*."""
    if "mmlu" in results:
        v = _pick_metric(results["mmlu"], MMLU_METRIC)
        if v is not None:
            return v
    subs = [_pick_metric(m, MMLU_METRIC)
            for k, m in results.items() if k.startswith("mmlu_")]
    subs = [s for s in subs if s is not None]
    if subs:
        return sum(subs) / len(subs)
    return None


def _fmt(v, prec=4):
    return f"{v:.{prec}f}" if isinstance(v, (int, float)) else "—"


def collect():
    rows = []
    for T in TARGETS:
        log = LOGS / f"T_{T}.out"
        res = RESULTS / f"T_{T}.json"
        cfg = (REPO / "bit_selection_correct" / "Mixtral-8x7B-v0.1"
               / f"experts_mixture_bitwidth_effavg_{T}.pkl")
        wiki, c4 = _parse_ppl(log)
        eff = _moe_eff_avg(cfg)

        row = {"name": f"MPQ correct T={T}",
               "T": T, "eff_bit_moe": eff,
               "wiki2": wiki, "c4": c4,
               "tasks": {}, "gsm8k_metric": None,
               "config": str(cfg.relative_to(REPO)) if cfg.exists() else None,
               "log_path": str(log.relative_to(REPO)) if log.exists() else None,
               "results_path": str(res.relative_to(REPO)) if res.exists() else None}

        if res.exists():
            with open(res) as f:
                obj = json.load(f)
            results = obj.get("results", {})
            for task, metric in ZEROSHOT_TASKS:
                v = _pick_metric(results.get(task, {}), metric)
                row["tasks"][task] = v
            row["tasks"]["mmlu"] = _pick_mmlu(results)
            gk, gv = _pick_gsm8k(results.get("gsm8k", {}))
            row["tasks"]["gsm8k"] = gv
            row["gsm8k_metric"] = gk
        rows.append(row)
    return rows


def render(rows):
    zs_names = [t for t, _ in ZEROSHOT_TASKS]
    header = ["model", "eff_bit_moe", "wiki2_ppl", "c4_ppl",
              *zs_names, "avg_zeroshot",
              "mmlu (5-shot, acc)", "gsm8k (5-shot, exact_match)", "config"]
    lines = []
    lines.append("# Mixtral-8x7B-v0.1 — bit_selection_correct (effective bitwidth MPQ)")
    lines.append("")
    lines.append("Built from `bit_selection_correct/Mixtral-8x7B-v0.1/` "
                 "(global ILP with effective bits 1→1.125, 2→2.25, 3→3.25, "
                 "targeting layer-wise avg T ∈ {1.500, 1.625, ..., 2.750}).")
    lines.append("")
    lines.append("Tasks:")
    lines.append("- Zero-shot (acc_norm, except acc for boolq & winogrande): "
                 "`arc_challenge`, `arc_easy`, `boolq`, `hellaswag`, `piqa`, "
                 "`winogrande`. `avg_zeroshot` = arithmetic mean.")
    lines.append("- 5-shot: `mmlu` (acc, mean across 57 subjects), "
                 "`gsm8k` (exact_match strict).")
    lines.append("")
    lines.append("PPL on wikitext2 (test) and c4 (validation 0); "
                 "GPTQ calib = wikitext2 train 128×2048.")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        cells = [r["name"], _fmt(r["eff_bit_moe"]), _fmt(r["wiki2"]), _fmt(r["c4"])]
        zs_vals = []
        for t, _ in ZEROSHOT_TASKS:
            v = r["tasks"].get(t)
            cells.append(_fmt(v))
            if isinstance(v, (int, float)):
                zs_vals.append(v)
        cells.append(_fmt(sum(zs_vals) / len(zs_vals)) if zs_vals else "—")
        cells.append(_fmt(r["tasks"].get("mmlu")))
        cells.append(_fmt(r["tasks"].get("gsm8k")))
        cells.append(f"`{r['config']}`" if r["config"] else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def missing(rows):
    miss = []
    zs_names = [t for t, _ in ZEROSHOT_TASKS]
    for r in rows:
        for k in ("wiki2", "c4"):
            if r[k] is None:
                miss.append((r["name"], f"{k}_ppl"))
        for t in zs_names + ["mmlu", "gsm8k"]:
            if r["tasks"].get(t) is None:
                miss.append((r["name"], f"lm_eval/{t}"))
    return miss


def main():
    rows = collect()
    md = render(rows)
    out = EXP / "result.md"
    out.write_text(md)
    print(f"Wrote {out}\n")
    miss = missing(rows)
    if not miss:
        print("All entries present.")
        return
    by_row = {}
    for name, w in miss:
        by_row.setdefault(name, []).append(w)
    print("Missing entries:")
    for name in by_row:
        print(f"  [{name}]")
        for w in by_row[name]:
            print(f"    - {w}")


if __name__ == "__main__":
    main()
