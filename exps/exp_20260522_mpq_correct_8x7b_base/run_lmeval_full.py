"""Custom lm-eval driver for the bit_selection_correct MPQ sweep.

Runs two sets of tasks against a single in-memory quantized model:

  - 6 zero-shot tasks: arc_challenge, arc_easy, boolq, hellaswag, piqa,
    winogrande  (num_fewshot=0)
  - 2 5-shot tasks:   mmlu, gsm8k                                       (num_fewshot=5)

Why a custom driver? HFs `from_pretrained` cannot read this repo's
qmodel.pt format (quant state lives inside QLinear modules), so we call
`inference.load_quantized_model`, wrap in `HFLM`, and drive `simple_evaluate`
twice — once per fewshot setting. Both passes share the same loaded model
and the same HFLM wrapper, so the model is loaded exactly once.

Outputs a single JSON merging both passes:
    {"results": {<task>: {...}}, "configs": {...}, "fewshot": {<task>: int}}
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True,
                        help="Directory holding qmodel.pt + config.json.")
    parser.add_argument("--output_path", required=True,
                        help="Where to write the merged results JSON.")
    parser.add_argument("--batch_size", default="32",
                        help="lm-eval batch_size; int or 'auto'.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap samples per task (smoke tests only).")
    parser.add_argument("--zeroshot_tasks", default=",".join(ZEROSHOT_TASKS))
    parser.add_argument("--fewshot5_tasks", default=",".join(FEWSHOT5_TASKS))
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.save_dir, "qmodel.pt")):
        raise FileNotFoundError(f"qmodel.pt not found in {args.save_dir}")

    print(f"[run_lmeval_full] Loading packed model from {args.save_dir}", flush=True)
    model = load_quantized_model(args.save_dir, {"attn_implementation": "eager"})
    tokenizer = AutoTokenizer.from_pretrained(args.save_dir)

    bs = args.batch_size
    if bs.isdigit():
        bs = int(bs)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=bs)

    zeroshot = [t.strip() for t in args.zeroshot_tasks.split(",") if t.strip()]
    fewshot5 = [t.strip() for t in args.fewshot5_tasks.split(",") if t.strip()]

    merged_results = {}
    fewshot_used = {}

    if zeroshot:
        print(f"[run_lmeval_full] Pass 1: zero-shot tasks = {zeroshot}", flush=True)
        res0 = simple_evaluate(model=lm, tasks=zeroshot, num_fewshot=0, limit=args.limit)
        for t, m in res0.get("results", {}).items():
            merged_results[t] = m
            fewshot_used[t] = 0

    if fewshot5:
        print(f"[run_lmeval_full] Pass 2: 5-shot tasks = {fewshot5}", flush=True)
        res5 = simple_evaluate(model=lm, tasks=fewshot5, num_fewshot=5, limit=args.limit)
        for t, m in res5.get("results", {}).items():
            merged_results[t] = m
            fewshot_used[t] = 5

    out = {
        "results": merged_results,
        "fewshot": fewshot_used,
        "save_dir": args.save_dir,
        "zeroshot_tasks": zeroshot,
        "fewshot5_tasks": fewshot5,
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[run_lmeval_full] Wrote {args.output_path}", flush=True)

    print("\n=== summary ===")
    for task, metrics in merged_results.items():
        nums = " ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                        if isinstance(v, (int, float)))
        print(f"  {task:>40}  (n_fewshot={fewshot_used.get(task, '?')}): {nums}")


if __name__ == "__main__":
    main()
