"""B200 fp16 baseline: wikitext2 + c4 PPL and 8-task lm-eval for the
unquantized Mixtral-8x22B-v0.1.

Loads the model with device_map='auto' across all visible GPUs (4× H200 is
the expected B200 partition), computes PPL via direct chunked forward, then
wraps in HFLM for lm-eval. main.py's per-layer GPU shuttling does not apply
to unquantized 8x22B (282 GB > a single GPU), hence a custom driver.

Outputs:
    - 2 "Perplexity:" lines (wikitext2, c4) on stdout for aggregate.py
    - {output_path}: lm-eval merged JSON (6 zero-shot + mmlu/gsm8k 5-shot)
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from datautils import get_loaders
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


ZEROSHOT_TASKS = ["arc_challenge", "arc_easy", "boolq",
                  "hellaswag", "piqa", "winogrande"]
FEWSHOT5_TASKS = ["mmlu", "gsm8k"]


@torch.no_grad()
def chunked_ppl(model, input_ids: torch.Tensor, seqlen: int, label: str) -> float:
    """Compute corpus PPL by chunking input_ids into `seqlen`-length windows.

    Mirrors the GPTQ-style PPL convention used by llama_eval (NLL over a
    flat concatenation of seqlen chunks, no overlap, no BOS handling).
    """
    nsamples = input_ids.numel() // seqlen
    # Place inputs on the embedding's device. accelerate's AlignDevicesHook
    # would re-dispatch a wrong-device input, but feeding it correctly first
    # avoids a transient copy onto a possibly full GPU.
    input_device = model.model.embed_tokens.weight.device
    nlls = []
    loss_fct = nn.CrossEntropyLoss()
    for i in range(nsamples):
        batch = input_ids[:, i * seqlen:(i + 1) * seqlen].to(input_device)
        out = model(batch)
        logits = out.logits.float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].to(shift_logits.device)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
        nll = loss.float() * seqlen
        nlls.append(nll.detach().cpu())
        if (i + 1) % 8 == 0 or i == nsamples - 1:
            print(f"  [{label}] {i + 1}/{nsamples}", flush=True)
    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / (nsamples * seqlen))
    return float(ppl.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mixtral-8x22B-v0.1")
    ap.add_argument("--output_path", required=True,
                    help="Where to write the lm-eval merged JSON.")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", default="4",
                    help="lm-eval batch_size; 8x22B is heavy → small default.")
    ap.add_argument("--skip_ppl", action="store_true",
                    help="Skip the PPL passes (lm-eval only).")
    ap.add_argument("--skip_lmeval", action="store_true",
                    help="Skip the lm-eval passes (PPL only).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Smoke-test cap for lm-eval samples per task.")
    args = ap.parse_args()

    print(f"[fp16] loading {args.model} with device_map='auto'", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model.eval()
    model.seqlen = args.seqlen
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if not args.skip_ppl:
        for ds in ["wikitext2", "c4"]:
            print(f"[fp16] loading {ds} testenc", flush=True)
            _, testenc = get_loaders(ds, seed=args.seed, seqlen=args.seqlen,
                                     model=args.model)
            ids = testenc.input_ids
            print(f"[fp16] {ds} PPL: chunks={ids.numel() // args.seqlen}", flush=True)
            ppl = chunked_ppl(model, ids, args.seqlen, ds)
            # Two "Perplexity:" lines so the existing log-parser pattern works.
            print(f"Perplexity: {ppl:.3f}", flush=True)

    if not args.skip_lmeval:
        bs = args.batch_size
        if bs.isdigit():
            bs = int(bs)
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=bs)

        merged = {}
        fewshot_used = {}
        print(f"[fp16] lm-eval zero-shot pass: {ZEROSHOT_TASKS}", flush=True)
        r0 = simple_evaluate(model=lm, tasks=ZEROSHOT_TASKS, num_fewshot=0,
                             limit=args.limit)
        for t, m in r0.get("results", {}).items():
            merged[t] = m
            fewshot_used[t] = 0

        print(f"[fp16] lm-eval 5-shot pass: {FEWSHOT5_TASKS}", flush=True)
        r5 = simple_evaluate(model=lm, tasks=FEWSHOT5_TASKS, num_fewshot=5,
                             limit=args.limit)
        for t, m in r5.get("results", {}).items():
            merged[t] = m
            fewshot_used[t] = 5

        out = {"results": merged, "fewshot": fewshot_used,
               "model": args.model,
               "zeroshot_tasks": ZEROSHOT_TASKS,
               "fewshot5_tasks": FEWSHOT5_TASKS}
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"[fp16] Wrote {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
