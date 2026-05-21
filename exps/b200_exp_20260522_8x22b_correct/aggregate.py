"""Aggregate Mixtral-8x22B-v0.1 B200 results into result.md.

Sources:
    - fp16 baseline:
        logs/fp16_*.log       2 "Perplexity:" lines (wiki2, c4)
        results/fp16.json     lm-eval merged JSON (8 tasks)
    - Uniform W ∈ {1, 2, 3, 4}:
        logs/uniform_W<W>_*.log     3 "Perplexity:" lines (calib, wiki2, c4)
        results/uniform_W<W>.json
    - MPQ T ∈ {1.500..2.750}:
        logs/mpq_T_<T>_*.log
        results/mpq_T_<T>.json

Columns: same shape as exp_20260522_mpq_correct_8x7b_base/aggregate.py
(eff_bit_moe / wiki2_ppl / c4_ppl / 6 zero-shot accs / avg_zeroshot /
mmlu acc 5-shot / gsm8k exact_match 5-shot / config).
"""

import glob
import json
import pickle
import re
from pathlib import Path

EXP = Path(__file__).resolve().parent
REPO = EXP.parent.parent
RESULTS = EXP / "results"
LOGS = EXP / "logs"

EFFECTIVE_BIT = {1: 1.125, 2: 2.25, 3: 3.25, 4: 4.25}

TARGETS = ["1.500", "1.625", "1.750", "1.875", "2.000", "2.125",
           "2.250", "2.375", "2.500", "2.625", "2.750"]
UNIFORM_WS = [1, 2, 3, 4]

ZEROSHOT_TASKS = [
    ("arc_challenge", "acc_norm"),
    ("arc_easy",      "acc_norm"),
    ("boolq",         "acc"),
    ("hellaswag",     "acc_norm"),
    ("piqa",          "acc_norm"),
    ("winogrande",    "acc"),
]
GSM8K_METRIC_CANDIDATES = ["exact_match,strict-match",
                          "exact_match,flexible-extract", "exact_match"]
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


def _latest_log(pattern: str) -> Path | None:
    cands = sorted(glob.glob(str(LOGS / pattern)))
    return Path(cands[-1]) if cands else None


def _parse_ppl_quant(log_path: Path | None) -> tuple[float | None, float | None]:
    """main.py: 3 "Perplexity:" lines (calib, wiki2, c4)."""
    if log_path is None or not log_path.exists():
        return None, None
    vals = []
    for line in log_path.read_text(errors="ignore").splitlines():
        m = re.match(r"^Perplexity:\s+([\d.eE+-]+)", line)
        if m:
            vals.append(float(m.group(1)))
    if len(vals) >= 3:
        return vals[1], vals[2]
    return None, None


def _parse_ppl_fp16(log_path: Path | None) -> tuple[float | None, float | None]:
    """run_fp16_full.py: 2 "Perplexity:" lines (wiki2, c4)."""
    if log_path is None or not log_path.exists():
        return None, None
    vals = []
    for line in log_path.read_text(errors="ignore").splitlines():
        m = re.match(r"^Perplexity:\s+([\d.eE+-]+)", line)
        if m:
            vals.append(float(m.group(1)))
    if len(vals) >= 2:
        return vals[0], vals[1]
    return None, None


def _pick_metric(metrics: dict, key: str) -> float | None:
    for k in (f"{key},none", key):
        if k in metrics and isinstance(metrics[k], (int, float)):
            return metrics[k]
    return None


def _pick_gsm8k(metrics: dict) -> float | None:
    for k in GSM8K_METRIC_CANDIDATES:
        if k in metrics and isinstance(metrics[k], (int, float)):
            return metrics[k]
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not k.endswith("_stderr,none"):
            return v
    return None


def _pick_mmlu(results: dict) -> float | None:
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


def _parse_lmeval(json_path: Path | None) -> dict:
    if json_path is None or not json_path.exists():
        return {}
    with open(json_path) as f:
        obj = json.load(f)
    results = obj.get("results", {})
    out = {}
    for task, metric in ZEROSHOT_TASKS:
        out[task] = _pick_metric(results.get(task, {}), metric)
    out["mmlu"] = _pick_mmlu(results)
    out["gsm8k"] = _pick_gsm8k(results.get("gsm8k", {}))
    return out


def _fmt(v, prec=4):
    return f"{v:.{prec}f}" if isinstance(v, (int, float)) else "—"


def collect():
    rows = []

    # fp16
    log = _latest_log("fp16_*.log")
    w, c = _parse_ppl_fp16(log)
    res = RESULTS / "fp16.json"
    rows.append({"name": "fp16 (no quant)",
                 "eff_bit_moe": 16.0,
                 "wiki2": w, "c4": c,
                 "tasks": _parse_lmeval(res),
                 "config": "—",
                 "log_path": str(log.relative_to(REPO)) if log else None})

    # Uniform
    for W in UNIFORM_WS:
        log = _latest_log(f"uniform_W{W}_*.log")
        w, c = _parse_ppl_quant(log)
        res = RESULTS / f"uniform_W{W}.json"
        cfg = (REPO / "bit_selection_correct" / "Mixtral-8x22B-v0.1"
               / "experts_mixture_bitwidth_uniform.pkl")
        rows.append({"name": f"uniform W={W}bit",
                     "eff_bit_moe": EFFECTIVE_BIT.get(W),
                     "wiki2": w, "c4": c,
                     "tasks": _parse_lmeval(res),
                     "config": str(cfg.relative_to(REPO)) if cfg.exists() else None,
                     "log_path": str(log.relative_to(REPO)) if log else None})

    # MPQ correct
    for T in TARGETS:
        log = _latest_log(f"mpq_T_{T}_*.log")
        w, c = _parse_ppl_quant(log)
        res = RESULTS / f"mpq_T_{T}.json"
        cfg = (REPO / "bit_selection_correct" / "Mixtral-8x22B-v0.1"
               / f"experts_mixture_bitwidth_effavg_{T}.pkl")
        rows.append({"name": f"MPQ correct T={T}",
                     "eff_bit_moe": _moe_eff_avg(cfg),
                     "wiki2": w, "c4": c,
                     "tasks": _parse_lmeval(res),
                     "config": str(cfg.relative_to(REPO)) if cfg.exists() else None,
                     "log_path": str(log.relative_to(REPO)) if log else None})

    return rows


def render(rows):
    zs_names = [t for t, _ in ZEROSHOT_TASKS]
    header = ["model", "eff_bit_moe", "wiki2_ppl", "c4_ppl",
              *zs_names, "avg_zeroshot",
              "mmlu (5-shot, acc)", "gsm8k (5-shot, exact_match)", "config"]
    lines = []
    lines.append("# Mixtral-8x22B-v0.1 — fp16 / uniform / bit_selection_correct MPQ (B200)")
    lines.append("")
    lines.append("Tasks:")
    lines.append("- Zero-shot (acc_norm, except acc for boolq & winogrande): "
                 "`arc_challenge`, `arc_easy`, `boolq`, `hellaswag`, `piqa`, "
                 "`winogrande`. `avg_zeroshot` = arithmetic mean.")
    lines.append("- 5-shot: `mmlu` (acc, mean across 57 subjects), "
                 "`gsm8k` (exact_match strict).")
    lines.append("")
    lines.append("`eff_bit_moe` = storage-faithful average effective bit per "
                 "expert (1→1.125, 2→2.25, 3→3.25, 4→4.25 for g=128 sym/asym; 16 for fp16).")
    lines.append("")
    lines.append("PPL on wikitext2 (test) and c4 (validation 0); "
                 "GPTQ calib = wikitext2 train 128×2048.")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        cells = [r["name"], _fmt(r["eff_bit_moe"]),
                 _fmt(r["wiki2"]), _fmt(r["c4"])]
        zs_vals = []
        for t, _ in ZEROSHOT_TASKS:
            v = r["tasks"].get(t)
            cells.append(_fmt(v))
            if isinstance(v, (int, float)):
                zs_vals.append(v)
        cells.append(_fmt(sum(zs_vals) / len(zs_vals)) if zs_vals else "—")
        cells.append(_fmt(r["tasks"].get("mmlu")))
        cells.append(_fmt(r["tasks"].get("gsm8k")))
        cells.append(f"`{r['config']}`" if r["config"] and r["config"] != "—" else "—")
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
