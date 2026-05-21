"""B200 lm-eval driver for packed quantized 8x22B checkpoints.

Mirrors exps/exp_20260522_mpq_correct_8x7b_base/run_lmeval_full.py with the
same 6 zero-shot + mmlu/gsm8k 5-shot task set. Loads the qmodel.pt-packed
checkpoint via inference.load_quantized_model (puts the whole model on
cuda:0), wraps in HFLM, calls simple_evaluate twice (one per fewshot setting),
merges into a single JSON.

Lives under exps/b200_exp_20260522_8x22b_correct/ so it ships to B200 via
git pull alongside the b200_*.sh wrappers.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoTokenizer

from inference import load_quantized_model
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


ZEROSHOT_TASKS = ["arc_challenge", "arc_easy", "boolq",
                  "hellaswag", "piqa", "winogrande"]
FEWSHOT5_TASKS = ["mmlu", "gsm8k"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--batch_size", default="8",
                    help="lm-eval batch_size; int or 'auto'. 8x22B is heavier "
                         "than 8x7B; default 8 to be safe on 1 H200.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not os.path.isfile(os.path.join(args.save_dir, "qmodel.pt")):
        raise FileNotFoundError(f"qmodel.pt not found in {args.save_dir}")

    print(f"[lmeval] Loading packed model from {args.save_dir}", flush=True)
    model = load_quantized_model(args.save_dir, {"attn_implementation": "eager"})
    tokenizer = AutoTokenizer.from_pretrained(args.save_dir)

    bs = args.batch_size
    if bs.isdigit():
        bs = int(bs)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=bs)

    merged = {}
    fewshot_used = {}

    print(f"[lmeval] zero-shot pass: {ZEROSHOT_TASKS}", flush=True)
    r0 = simple_evaluate(model=lm, tasks=ZEROSHOT_TASKS, num_fewshot=0,
                         limit=args.limit)
    for t, m in r0.get("results", {}).items():
        merged[t] = m
        fewshot_used[t] = 0

    print(f"[lmeval] 5-shot pass: {FEWSHOT5_TASKS}", flush=True)
    r5 = simple_evaluate(model=lm, tasks=FEWSHOT5_TASKS, num_fewshot=5,
                         limit=args.limit)
    for t, m in r5.get("results", {}).items():
        merged[t] = m
        fewshot_used[t] = 5

    out = {"results": merged, "fewshot": fewshot_used,
           "save_dir": args.save_dir,
           "zeroshot_tasks": ZEROSHOT_TASKS,
           "fewshot5_tasks": FEWSHOT5_TASKS}
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[lmeval] Wrote {args.output_path}", flush=True)

    for task, metrics in merged.items():
        nums = " ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                        if isinstance(v, (int, float)))
        print(f"  {task:>40}  (n_fewshot={fewshot_used.get(task, '?')}): {nums}")


if __name__ == "__main__":
    main()
