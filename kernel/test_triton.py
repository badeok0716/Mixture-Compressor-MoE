"""Numerics + reconstruction-loss sweep: reference vs Triton vs Triton+CUDA-graph
across Mixtral-8x7B sublayer shapes and all bit widths used by MC-MoE.

For each (shape, wbits, apply_inner_rank1) we report:
  - max-abs-diff of W_quant between ref/triton/triton+graph (numerical drift)
  - True GPTQ reconstruction loss L = tr((W_o - W_q)^T H (W_o - W_q))
    for each impl, and the relative gap of triton/triton+graph vs ref.

Lower L = better quantization. If L_tri ≈ L_ref, the implementations produce
equally good quantizations even when their outputs diverge by 1-ULP drift.

Run on a GPU (srun).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from kernel import gptq_triton as gt

# MC-MoE's gptq.py disables TF32 (L11-12). Mirror that here so our reference
# and triton outer rank-1 use the same fp32 reduction path.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

torch.manual_seed(0)
device = 'cuda'
GROUP = 128
BLOCK = 128

SHAPES = [
    (4096, 4096),     # attention proj
    (14336, 4096),    # FFN w1/w3
    (4096, 14336),    # FFN w2
]
BITS = [1, 2, 3, 4]


def make_inputs(rows, cols):
    W0 = (torch.randn(rows, cols, device=device, dtype=torch.float32) * 0.05).contiguous()
    H = torch.randn(cols, cols, device=device, dtype=torch.float32)
    H = H @ H.t() + float(cols) * torch.eye(cols, device=device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H))).t().contiguous()
    return W0, H, Hinv


def gptq_loss(W_orig, W_quant, H):
    """True GPTQ reconstruction objective: tr((W_o - W_q)^T H (W_o - W_q))."""
    delta = (W_orig - W_quant).float()
    return float(((delta @ H) * delta).sum().item())


def reference_fp16scale(
    W0, Hinv, maxq, group_size, blocksize=128,
    *, apply_inner_rank1=True, sym=False, binary=False,
):
    """PyTorch reference where scale/qzero are computed on fp16-cast W slice
    (mirroring MC-MoE's quantizer_moe.find_params which does `x = w.to(fp16)`
    before min/max). Internal arithmetic stays fp32, only the scale/qzero
    are subjected to fp16 representable precision.
    """
    W = W0.clone().contiguous()
    rows, cols = W.shape
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        block_w = i2 - i1

        # MC-MoE-style: cast W slice to fp16 before computing min/max/scale
        x = W[:, i1:i2].to(torch.float16)
        if binary:
            s_fp16 = 2.0 * x.abs().mean(dim=1)
            z_fp16 = torch.full_like(s_fp16, 0.5)
        else:
            xmax = x.amax(dim=1)
            xmin = x.amin(dim=1)
            zero_t = torch.zeros_like(xmin)
            xmin = torch.minimum(xmin, zero_t)
            xmax = torch.maximum(xmax, zero_t)
            if sym:
                xmax = torch.maximum(xmin.abs(), xmax)
                neg = xmin < 0
                xmin = torch.where(neg, -xmax, xmin)
            both_zero = (xmin == 0) & (xmax == 0)
            xmin = torch.where(both_zero, -torch.ones_like(xmin), xmin)
            xmax = torch.where(both_zero, torch.ones_like(xmax), xmax)
            s_fp16 = (xmax - xmin) / maxq
            if sym:
                z_fp16 = torch.full_like(s_fp16, (maxq + 1) / 2)
            else:
                z_fp16 = torch.round(-xmin / s_fp16)
        # Use as fp32 in subsequent ops (s_fp16/z_fp16 are fp16 tensors → auto-promoted)
        s = s_fp16.to(torch.float32)
        z = z_fp16.to(torch.float32)

        Err_block = torch.empty(rows, block_w, dtype=W.dtype, device=W.device)
        for i_local in range(block_w):
            i = i1 + i_local
            w = W[:, i]
            d = Hinv[i, i]
            if binary:
                q_code = (w >= 0).to(W.dtype)
            else:
                q_code = torch.clamp(torch.round(w / s) + z, 0.0, maxq)
            q_deq = s * (q_code - z)
            err = (w - q_deq) / d
            W[:, i] = q_deq
            Err_block[:, i_local] = err
            if apply_inner_rank1 and i + 1 < i2:
                W[:, i + 1 : i2] -= err.unsqueeze(1) * Hinv[i, i + 1 : i2].unsqueeze(0)
        if i2 < cols:
            W[:, i2:].addmm_(Err_block, Hinv[i1:i2, i2:], beta=1.0, alpha=-1.0)
    return W


def run(impl, W0, Hinv, *, wbits, sym, dynamic_groups, apply_inner_rank1):
    binary = (wbits == 1)
    maxq = float(2 ** wbits - 1)
    rows, cols = W0.shape
    n_groups = cols // GROUP

    W = W0.clone().contiguous()
    if dynamic_groups:
        scale = torch.zeros(rows, n_groups, dtype=W.dtype, device=device)
        qzero = torch.zeros_like(scale)
    else:
        if binary:
            scale, qzero = gt.make_scale_qzero_binary(W, GROUP)
        else:
            scale, qzero = gt.make_scale_qzero(W, GROUP, maxq, sym=sym)
    scale = scale.contiguous()
    qzero = qzero.contiguous()
    impl(
        W, Hinv, scale, qzero, maxq, GROUP, BLOCK,
        dynamic_groups=dynamic_groups, sym=sym,
        apply_inner_rank1=apply_inner_rank1, binary=binary,
    )
    return W


def test_combo(rows, cols, wbits, sym, dynamic_groups, apply_inner_rank1):
    W0, H, Hinv = make_inputs(rows, cols)
    Wref = run(gt.fasterquant_inner_loop_reference, W0, Hinv,
               wbits=wbits, sym=sym, dynamic_groups=dynamic_groups,
               apply_inner_rank1=apply_inner_rank1)
    Wtri = run(gt.fasterquant_inner_loop_triton, W0, Hinv,
               wbits=wbits, sym=sym, dynamic_groups=dynamic_groups,
               apply_inner_rank1=apply_inner_rank1)
    Wgr = run(gt.fasterquant_inner_loop_triton_graph, W0, Hinv,
              wbits=wbits, sym=sym, dynamic_groups=dynamic_groups,
              apply_inner_rank1=apply_inner_rank1)
    d_tri = (Wref - Wtri).abs().max().item()
    d_gr = (Wref - Wgr).abs().max().item()
    L_ref = gptq_loss(W0, Wref, H)
    L_tri = gptq_loss(W0, Wtri, H)
    L_gr = gptq_loss(W0, Wgr, H)
    return d_tri, d_gr, L_ref, L_tri, L_gr


HEADER = (
    f"{'Shape':>14} {'wbits':>5} {'apply':>5}"
    f"  {'maxd_tri':>9} {'maxd_gr':>9}"
    f"  {'L_ref':>11} {'L_tri':>11} {'L_gr':>11}"
    f"  {'tri/ref':>9} {'gr/ref':>9}"
)


def report_row(rows, cols, wbits, apply, out):
    d_tri, d_gr, L_ref, L_tri, L_gr = out
    print(
        f"{f'{rows}x{cols}':>14} {wbits:>5} {str(apply):>5}"
        f"  {d_tri:>9.2e} {d_gr:>9.2e}"
        f"  {L_ref:>11.4e} {L_tri:>11.4e} {L_gr:>11.4e}"
        f"  {(L_tri/L_ref):>9.5f} {(L_gr/L_ref):>9.5f}"
    )


# === Sweep 1: MC-MoE production path (legacy: skip inner rank-1 when wbits<=3) ===
print("=== DYNAMIC groups, LEGACY (skip rank-1 when wbits<=3) — MC-MoE production ===")
print(HEADER)
print("-" * len(HEADER))
for rows, cols in SHAPES:
    for wbits in BITS:
        apply = (wbits > 3)
        gt._GRAPH_CACHE.clear()
        torch.cuda.empty_cache()
        out = test_combo(rows, cols, wbits, sym=False, dynamic_groups=True,
                         apply_inner_rank1=apply)
        report_row(rows, cols, wbits, apply, out)

# === Sweep 2: standard GPTQ — always apply inner rank-1 ===
print("\n=== DYNAMIC groups, ALWAYS apply rank-1 — standard GPTQ ===")
print(HEADER)
print("-" * len(HEADER))
for rows, cols in SHAPES:
    for wbits in BITS:
        gt._GRAPH_CACHE.clear()
        torch.cuda.empty_cache()
        out = test_combo(rows, cols, wbits, sym=False, dynamic_groups=True,
                         apply_inner_rank1=True)
        report_row(rows, cols, wbits, True, out)

# === Sweep 2.5: fp16 vs fp32 scale — isolate scale precision effect ===
print("\n=== fp32 scale vs fp16 scale — pytorch reference, legacy rank-1 policy ===")
print(f"{'Shape':>14} {'wbits':>5}  {'L_fp32_ref':>13} {'L_fp16_ref':>13}  {'fp16/fp32':>10}")
print("-" * 72)
for rows, cols in SHAPES:
    for wbits in BITS:
        apply = (wbits > 3)
        W0, H, Hinv = make_inputs(rows, cols)
        # fp32-scale ref via existing kernel reference
        W_fp32 = run(gt.fasterquant_inner_loop_reference, W0, Hinv,
                     wbits=wbits, sym=False, dynamic_groups=True,
                     apply_inner_rank1=apply)
        # fp16-scale ref: same loop, scale/qzero computed on fp16-cast W slice
        W_fp16 = reference_fp16scale(
            W0, Hinv, float(2 ** wbits - 1), GROUP, BLOCK,
            apply_inner_rank1=apply, sym=False, binary=(wbits == 1),
        )
        L_fp32 = gptq_loss(W0, W_fp32, H)
        L_fp16 = gptq_loss(W0, W_fp16, H)
        print(
            f"{f'{rows}x{cols}':>14} {wbits:>5}"
            f"  {L_fp32:>13.4e} {L_fp16:>13.4e}"
            f"  {(L_fp16/L_fp32):>10.5f}"
        )

# === Sweep 3: B vs C comparison — for wbits<=3, legacy(skip) vs always — using TRITON only ===
print("\n=== B (legacy) vs C (always) — same impl, different rank-1 policy ===")
print(f"{'Shape':>14} {'wbits':>5}  {'L_triton_legacy':>16} {'L_triton_always':>16}  {'always/legacy':>14}")
print("-" * 78)
for rows, cols in SHAPES:
    for wbits in [1, 2, 3]:
        W0, H, Hinv = make_inputs(rows, cols)
        gt._GRAPH_CACHE.clear()
        torch.cuda.empty_cache()
        W_legacy = run(gt.fasterquant_inner_loop_triton, W0, Hinv,
                       wbits=wbits, sym=False, dynamic_groups=True,
                       apply_inner_rank1=False)
        gt._GRAPH_CACHE.clear()
        torch.cuda.empty_cache()
        W_always = run(gt.fasterquant_inner_loop_triton, W0, Hinv,
                       wbits=wbits, sym=False, dynamic_groups=True,
                       apply_inner_rank1=True)
        L_legacy = gptq_loss(W0, W_legacy, H)
        L_always = gptq_loss(W0, W_always, H)
        print(
            f"{f'{rows}x{cols}':>14} {wbits:>5}"
            f"  {L_legacy:>16.4e} {L_always:>16.4e}"
            f"  {(L_always/L_legacy):>14.5f}"
        )
