# CLAUDE.md (Mixture-Compressor-MoE)

MC-MoE-specific rules. Common rules (§1–§7) and Slurm / B200 / `exps/`
conventions auto-merged from [`../CLAUDE.md`](../CLAUDE.md) and
[`../EXECUTION.md`](../EXECUTION.md).

## Project

MC-MoE — expert-wise mixed-precision GPTQ for Mixtral. Three steps:

1. `awareness.py` (C4 calib) → factor pkls.
2. `precision_solver.py` (Gurobi ILP) → 9 precision pkls in
   `experts_mixture_bit_selection/`.
3. `main.py` → apply + GPTQ + PPL eval.

Steps 1+2 outputs are shipped, so step 3 alone reproduces the paper Table.
Hard-locked to `MixtralForCausalLM`; shipped pkls are 32×8 → **Mixtral-8x7B
only**. Run outputs in `exps/`.

## Environment

torch 2.2.1+cu121, transformers 4.36.2, huggingface-hub 0.36.2, Python 3.10
(uv-managed — compute nodes lack `python3.10-dev`).

One-time:

```bash
uv python install 3.10
uv venv --python 3.10 --python-preference only-managed
uv sync
```

Every shell:

```bash
export CUDA_HOME=/usr/local/cuda-12.1                 # match torch +cu121
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

For `--pack --save` only: rebuild `hqq_aten` once on a GPU node via
[quant/kernels/setup_cuda.py](quant/kernels/setup_cuda.py)
(`TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"`).

## Calibration / eval data

Defaults per `scripts/quant.sh` + `factors.sh`:

| Step | Calib | Tokens |
|---|---|---|
| `awareness.py` | C4 train shard 0 | 128 × 2048 |
| `main.py` GPTQ | wikitext2 train | 128 × 2048 |

Deterministic at `seed=0`. Loaders use hardcoded relative paths
([data/build.py:13](data/build.py#L13),
[datautils.py:64-65](datautils.py#L64)); cwd must be repo root. Required
files at [data/](data/):

- `c4-train.00000-of-01024.json` (~820 MB) — `awareness.py`, `--dataset c4`
- `c4-validation.00000-of-00008.json` (~100 MB) — `--eval_ppl` c4 branch

Source: `huggingface.co/datasets/allenai/c4/resolve/main/en/<name>.gz`.

`main.py` eval loop = `["wikitext2", "c4"]` (wikitext2 from HF cache).

## Fork-specific patches (applied)

- **Calibration cache** moved to `$HF_HOME/mcmoe_cache/...`
  ([datautils.py:93](datautils.py#L93); upstream `/mnt/afs/yliao/...`
  hardcoded).
- **Calibration-set PPL** reported alongside test PPL in
  [main.py](main.py).

### Triton GPTQ backend (default)

[kernel/gptq_triton.py](kernel/gptq_triton.py) — Triton inner-loop kernel
plus an MC-MoE-consistent `_find_group_params_inplace`: **fractional zero
point** (`-xmin/scale`) + **MSE search** over `p ∈ [0.9, 1.1]` (101
candidates, norm=2.4). `apply_inner_rank1` toggles the per-column rank-1
update (mimics upstream MC-MoE's `if wbits > 3` skip,
[gptq.py:169](gptq.py#L169) upstream). `binary` flag selects the XNOR-Net
1-bit path ([utils/quantizer_moe.py:10-20](utils/quantizer_moe.py#L10)).

```
--gptq_backend {pytorch,triton,triton_graph}    # default triton
--inner_rank1 {legacy,always,never}              # default legacy
```

Triton 2.2 API: `tl.math.rint`, `tl.minimum(tl.maximum(...))` in lieu of
newer `libdevice.rint` / `tl.clamp`.

**Quantization quality**: this default path matches MC-MoE's `--pack --save`
deployment-quality quantization (same scale / zero / q decisions as
upstream's `_quantize` pack=True branch). Upstream's `--pack`-off PPL path
runs a buggy double-dequant MSE metric in
[utils/quantizer_moe.py:191-192](utils/quantizer_moe.py#L191) when
`pack=False`, which makes sub-optimal scale picks for high-bit (wbits=4)
attention projections. Our path uses the standard metric:
- wbits=2 expert PPL matches upstream pytorch within ~0.05 PPL (noise
  level).
- wbits=4 attention sublayers quantize ~2× better (lower per-sublayer
  reconstruction loss).
- ~1.5–1.8× faster end-to-end than the upstream pytorch path on H100 PCIe.
