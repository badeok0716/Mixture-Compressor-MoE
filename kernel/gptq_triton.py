"""Triton kernels for GPTQ's inner column loop.

Ported from hbmpq/amq/kernel/gptq_triton.py with one MC-MoE-specific
extension: `apply_inner_rank1: bool` toggles the per-column rank-1
propagation kernel launch. MC-MoE's PyTorch path skips that propagation
when wbits <= 3 (see gptq.py::static_fasterquant L169 `if self.wbits > 3`).
Set apply_inner_rank1=False to match that semantics in the Triton path.

Targets the per-column quantize + rank-1 propagation inside GPTQ's
fasterquant. Follows the fused-kernel approach from IST-DASLab/MoE-Quant
(https://github.com/IST-DASLab/MoE-Quant/blob/main/src/gptq_loop.py),
adapted to fp32 / row-major / upper-triangular Hinv conventions.

Entry points:
  - `fasterquant_inner_loop_reference` — pure-PyTorch reference
  - `fasterquant_inner_loop_triton`    — Triton-kernel implementation
  - `fasterquant_inner_loop_triton_graph` — CUDA-Graph-wrapped Triton

All three modify W in place to its dequantized form.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------- Triton primitives ----------
@triton.jit
def _tl_round(x):
    # CUDA's rint = round-to-nearest-even (banker's), matches torch.round.
    # Triton 2.2 exposes it via tl.math.rint; newer versions use
    # `triton.language.extra.libdevice.rint`.
    return tl.math.rint(x)


@triton.jit
def _tl_quantize(x, scale, qzero, maxq):
    # fp64 round-trip to match torch.div semantics (avoids 1-ULP drift from
    # Triton's approximate reciprocal-multiply on fp32).
    ratio = (x.to(tl.float64) / scale.to(tl.float64)).to(tl.float32)
    # Triton 2.2 lacks tl.clamp; use minimum(maximum(...)) instead.
    q = _tl_round(ratio) + qzero
    return tl.minimum(tl.maximum(q, 0.0), maxq)


@triton.jit
def _tl_dequantize(q, scale, qzero):
    return scale * (q - qzero)


@triton.jit
def _quantize_column_kernel(
    W_ptr,            # (rows, cols), fp32, row-major
    err_ptr,          # (rows,), fp32 — output
    scale_col_ptr,    # (rows,), fp32 — per-row scale for THIS column's group
    qzero_col_ptr,    # (rows,), fp32 — per-row qzero for THIS column's group
    Hinv_diag_ptr,    # (cols,), fp32
    maxq,             # fp32 scalar
    col_idx,          # int32 scalar
    rows: tl.int64,
    cols_stride: tl.int64,
    BLOCK: tl.constexpr,
    BINARY: tl.constexpr,
    ADD_Z_BEFORE_ROUND: tl.constexpr,
    FP64_DIV: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < rows

    d = tl.load(Hinv_diag_ptr + col_idx)

    offs64 = offs.to(tl.int64)
    w_ptrs = W_ptr + offs64 * cols_stride + col_idx

    w = tl.load(w_ptrs, mask=mask, other=0.0)
    s = tl.load(scale_col_ptr + offs, mask=mask, other=1.0)
    z = tl.load(qzero_col_ptr + offs, mask=mask, other=0.0)
    if BINARY:
        q_code = tl.where(w >= 0.0, 1.0, 0.0)
    else:
        if FP64_DIV:
            ratio = (w.to(tl.float64) / s.to(tl.float64)).to(tl.float32)
        else:
            ratio = w / s
        if ADD_Z_BEFORE_ROUND:
            q_code = tl.minimum(tl.maximum(_tl_round(ratio + z), 0.0), maxq)
        else:
            q_code = tl.minimum(tl.maximum(_tl_round(ratio) + z, 0.0), maxq)
    q_deq = _tl_dequantize(q_code, s, z)
    if FP64_DIV:
        err = ((w - q_deq).to(tl.float64) / d.to(tl.float64)).to(tl.float32)
    else:
        err = (w - q_deq) / d

    tl.store(w_ptrs, q_deq, mask=mask)
    tl.store(err_ptr + offs, err, mask=mask)


@triton.jit
def _rank1_update_kernel(
    W_ptr,            # (rows, cols), fp32
    err_ptr,          # (rows,), fp32
    Hinv_ptr,         # (cols, cols), fp32 — full Hinv, indexed inside
    row_idx,          # int32 — Hinv row to broadcast (= the column-i being propagated)
    col_start,        # int32 — first col to update (= col_idx + 1)
    col_end,          # int32 — first col NOT to update (= i2)
    rows: tl.int64,
    W_row_stride: tl.int64,
    Hinv_row_stride: tl.int64,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_r = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    rows_offs = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    cols_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C) + col_start
    r_mask = rows_offs < rows
    c_mask = cols_offs < col_end
    mask = r_mask[:, None] & c_mask[None, :]

    err = tl.load(err_ptr + rows_offs, mask=r_mask, other=0.0)
    h = tl.load(Hinv_ptr + row_idx * Hinv_row_stride + cols_offs, mask=c_mask, other=0.0)
    update = err[:, None] * h[None, :]

    w_ptrs = W_ptr + rows_offs[:, None].to(tl.int64) * W_row_stride + cols_offs[None, :].to(tl.int64)
    w = tl.load(w_ptrs, mask=mask, other=0.0)
    w = w - update
    tl.store(w_ptrs, w, mask=mask)


# ---------- Pure-PyTorch reference ----------
def fasterquant_inner_loop_reference(
    W: torch.Tensor,
    Hinv: torch.Tensor,
    scale: torch.Tensor,
    qzero: torch.Tensor,
    maxq: float,
    group_size: int,
    blocksize: int = 128,
    *,
    dynamic_groups: bool = False,
    sym: bool = True,
    apply_inner_rank1: bool = True,
    binary: bool = False,
    add_z_before_round: bool = True,
) -> None:
    """Pure-PyTorch reference of GPTQ's inner column loop. Modifies W in place.

    apply_inner_rank1=False mirrors MC-MoE's `if self.wbits > 3` skip.
    binary=True activates the XNOR-Net-style 1-bit path:
      scale = 2*mean(|x|, dim=1) per group, qzero = 0.5,
      q_code = (sign(x)+1)/2.
    """
    assert W.dim() == 2 and Hinv.dim() == 2
    rows, cols = W.shape
    assert Hinv.shape == (cols, cols)
    n_groups = cols // group_size
    assert scale.shape == (rows, n_groups), f"scale {scale.shape} vs ({rows}, {n_groups})"
    assert qzero.shape == (rows, n_groups)
    if dynamic_groups:
        assert group_size == blocksize, (
            "dynamic_groups currently requires group_size == blocksize "
            f"(got {group_size} vs {blocksize})"
        )

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        g_block = i1 // group_size
        if dynamic_groups:
            if binary:
                _find_group_params_inplace_binary(
                    W[:, i1:i2], scale[:, g_block], qzero[:, g_block]
                )
            else:
                _find_group_params_inplace(
                    W[:, i1:i2], scale[:, g_block], qzero[:, g_block], maxq, sym
                )
        Err_block = torch.empty(rows, i2 - i1, dtype=W.dtype, device=W.device)
        for i_local in range(i2 - i1):
            i = i1 + i_local
            g = i // group_size
            w = W[:, i]
            d = Hinv[i, i]
            s = scale[:, g]
            z = qzero[:, g]
            if binary:
                q_code = (w >= 0).to(W.dtype)
            elif add_z_before_round:
                q_code = torch.clamp(torch.round(w / s + z), 0.0, maxq)
            else:
                q_code = torch.clamp(torch.round(w / s) + z, 0.0, maxq)
            q_deq = s * (q_code - z)
            err = (w - q_deq) / d
            W[:, i] = q_deq
            Err_block[:, i_local] = err
            if apply_inner_rank1 and i + 1 < i2:
                W[:, i + 1 : i2] -= err.unsqueeze(1) * Hinv[i, i + 1 : i2].unsqueeze(0)
        if i2 < cols:
            # Use addmm_ (same op as triton wrapper) so reduction order matches.
            W[:, i2:].addmm_(Err_block, Hinv[i1:i2, i2:], beta=1.0, alpha=-1.0)


# ---------- Triton-kernel implementation ----------
def _make_scratch(W: torch.Tensor, scale: torch.Tensor, qzero: torch.Tensor,
                  group_size: int, blocksize: int) -> dict:
    rows, cols = W.shape
    n_groups = cols // group_size
    device = W.device
    dtype = W.dtype
    return {
        'Hinv_diag': torch.empty(cols, dtype=dtype, device=device),
        'err_buf': torch.empty(rows, dtype=dtype, device=device),
        'Err_block': torch.empty(rows, blocksize, dtype=dtype, device=device),
        'scale_t': torch.empty(n_groups, rows, dtype=scale.dtype, device=device),
        'qzero_t': torch.empty(n_groups, rows, dtype=qzero.dtype, device=device),
    }


def fasterquant_inner_loop_triton(
    W: torch.Tensor,
    Hinv: torch.Tensor,
    scale: torch.Tensor,
    qzero: torch.Tensor,
    maxq: float,
    group_size: int,
    blocksize: int = 128,
    *,
    scratch: dict | None = None,
    dynamic_groups: bool = False,
    sym: bool = True,
    apply_inner_rank1: bool = True,
    binary: bool = False,
    add_z_before_round: bool = True,
    fp64_div: bool = True,
    outer_use_matmul: bool = False,
) -> None:
    """Triton implementation. Modifies W in place to dequantized values.

    apply_inner_rank1=False skips the per-column `_rank1_update_kernel` launch
    inside the blocksize block — matches MC-MoE's `if self.wbits > 3` skip.
    The block-end outer addmm_ is always applied.
    """
    assert W.is_cuda and Hinv.is_cuda
    assert W.is_contiguous(), "W must be row-major contiguous"
    assert Hinv.is_contiguous(), "Hinv must be row-major contiguous"
    assert scale.is_contiguous() and qzero.is_contiguous()
    assert W.dim() == 2 and Hinv.dim() == 2
    rows, cols = W.shape
    assert Hinv.shape == (cols, cols)
    n_groups = cols // group_size
    assert scale.shape == (rows, n_groups)
    assert qzero.shape == (rows, n_groups)
    if dynamic_groups:
        assert group_size == blocksize, (
            "dynamic_groups currently requires group_size == blocksize "
            f"(got {group_size} vs {blocksize})"
        )

    if scratch is None:
        scratch = _make_scratch(W, scale, qzero, group_size, blocksize)
    Hinv_diag = scratch['Hinv_diag']
    err_buf = scratch['err_buf']
    Err_block = scratch['Err_block']
    scale_t = scratch['scale_t']
    qzero_t = scratch['qzero_t']

    Hinv_diag.copy_(torch.diagonal(Hinv))
    scale_t.copy_(scale.t())
    qzero_t.copy_(qzero.t())

    maxq_f = float(maxq)
    BLOCK_QUANT = 256
    BLOCK_R, BLOCK_C = 64, 128

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        block_w = i2 - i1
        g_block = i1 // group_size
        if dynamic_groups:
            if binary:
                _find_group_params_inplace_binary(
                    W[:, i1:i2], scale_t[g_block], qzero_t[g_block]
                )
            else:
                _find_group_params_inplace(
                    W[:, i1:i2], scale_t[g_block], qzero_t[g_block], maxq_f, sym
                )
            scale[:, g_block].copy_(scale_t[g_block])
            qzero[:, g_block].copy_(qzero_t[g_block])
        for i_local in range(block_w):
            i = i1 + i_local
            g = i // group_size
            grid_q = (triton.cdiv(rows, BLOCK_QUANT),)
            _quantize_column_kernel[grid_q](
                W, err_buf, scale_t[g], qzero_t[g], Hinv_diag, maxq_f,
                i, rows, cols,
                BLOCK=BLOCK_QUANT, BINARY=binary,
                ADD_Z_BEFORE_ROUND=add_z_before_round,
                FP64_DIV=fp64_div,
            )
            Err_block[:, i_local].copy_(err_buf)
            if apply_inner_rank1 and i + 1 < i2:
                cwidth = i2 - i - 1
                grid_r1 = (triton.cdiv(rows, BLOCK_R), triton.cdiv(cwidth, BLOCK_C))
                _rank1_update_kernel[grid_r1](
                    W, err_buf, Hinv, i,
                    i + 1, i2, rows, cols, cols,
                    BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C,
                )
        if i2 < cols:
            if outer_use_matmul:
                # MC-MoE form: separate matmul + in-place subtract.
                W[:, i2:] -= Err_block[:, :block_w].matmul(Hinv[i1:i2, i2:])
            else:
                W[:, i2:].addmm_(Err_block[:, :block_w], Hinv[i1:i2, i2:],
                                 beta=1.0, alpha=-1.0)


# ---------- CUDA Graph wrapper ----------
_GRAPH_CACHE: dict = {}


def fasterquant_inner_loop_triton_graph(
    W: torch.Tensor,
    Hinv: torch.Tensor,
    scale: torch.Tensor,
    qzero: torch.Tensor,
    maxq: float,
    group_size: int,
    blocksize: int = 128,
    *,
    dynamic_groups: bool = False,
    sym: bool = True,
    apply_inner_rank1: bool = True,
    binary: bool = False,
    add_z_before_round: bool = True,
    fp64_div: bool = True,
    outer_use_matmul: bool = False,
) -> None:
    """CUDA-Graph-wrapped variant. All constexpr flags are baked into the
    captured graph (cache key includes them).
    """
    assert W.is_cuda
    rows, cols = W.shape
    device = W.device
    dtype = W.dtype
    key = (rows, cols, group_size, blocksize, float(maxq),
           bool(dynamic_groups), bool(sym), bool(apply_inner_rank1),
           bool(binary), bool(add_z_before_round), bool(fp64_div),
           bool(outer_use_matmul), str(dtype), str(device))

    if key not in _GRAPH_CACHE:
        W_buf = torch.empty_like(W)
        Hinv_buf = torch.empty_like(Hinv)
        scale_buf = torch.empty_like(scale)
        qzero_buf = torch.empty_like(qzero)
        scratch = _make_scratch(W, scale, qzero, group_size, blocksize)

        W_buf.copy_(W)
        Hinv_buf.copy_(Hinv)
        scale_buf.copy_(scale)
        qzero_buf.copy_(qzero)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                W_buf.copy_(W)
                fasterquant_inner_loop_triton(
                    W_buf, Hinv_buf, scale_buf, qzero_buf, maxq, group_size,
                    blocksize, scratch=scratch,
                    dynamic_groups=dynamic_groups, sym=sym,
                    apply_inner_rank1=apply_inner_rank1,
                    binary=binary,
                    add_z_before_round=add_z_before_round,
                    fp64_div=fp64_div,
                    outer_use_matmul=outer_use_matmul,
                )
        torch.cuda.current_stream().wait_stream(s)

        W_buf.copy_(W)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fasterquant_inner_loop_triton(
                W_buf, Hinv_buf, scale_buf, qzero_buf, maxq, group_size,
                blocksize, scratch=scratch,
                dynamic_groups=dynamic_groups, sym=sym,
                apply_inner_rank1=apply_inner_rank1,
                binary=binary,
            )
        _GRAPH_CACHE[key] = {
            'graph': graph,
            'W_buf': W_buf,
            'Hinv_buf': Hinv_buf,
            'scale_buf': scale_buf,
            'qzero_buf': qzero_buf,
            'scratch': scratch,
        }

    c = _GRAPH_CACHE[key]
    c['W_buf'].copy_(W)
    c['Hinv_buf'].copy_(Hinv)
    c['scale_buf'].copy_(scale)
    c['qzero_buf'].copy_(qzero)
    c['graph'].replay()
    W.copy_(c['W_buf'])
    if dynamic_groups:
        scale.copy_(c['scale_buf'])
        qzero.copy_(c['qzero_buf'])


# ---------- Helpers ----------
def make_scale_qzero(
    W: torch.Tensor,
    group_size: int,
    maxq: float,
    sym: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (scale, qzero) of shape (rows, n_groups) via per-group min-max."""
    rows, cols = W.shape
    assert cols % group_size == 0, f"cols={cols} not divisible by group_size={group_size}"
    n_groups = cols // group_size
    Wg = W.view(rows, n_groups, group_size)
    xmax = Wg.amax(dim=2)
    xmin = Wg.amin(dim=2)
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
    scale = (xmax - xmin) / maxq
    if sym:
        qzero = torch.full_like(scale, (maxq + 1) / 2)
    else:
        qzero = torch.round(-xmin / scale)
    return scale.contiguous(), qzero.contiguous()


def make_scale_qzero_binary(
    W: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary scale (MC-MoE XNOR-Net style): 2*mean(|x|) per group, qzero=0.5."""
    rows, cols = W.shape
    assert cols % group_size == 0, f"cols={cols} not divisible by group_size={group_size}"
    n_groups = cols // group_size
    Wg = W.view(rows, n_groups, group_size)
    scale = 2.0 * Wg.abs().mean(dim=2)
    qzero = torch.full_like(scale, 0.5)
    return scale.contiguous(), qzero.contiguous()


def _find_group_params_inplace_binary(
    W_slice: torch.Tensor,
    scale_out: torch.Tensor,
    qzero_out: torch.Tensor,
) -> None:
    """Per-row binary scale for ONE column slice. Mirrors
    `utils.quantizer_moe.binary_scale`: scale = 2*mean(|x|, dim=1), zero = 0.5.
    """
    scale_out.copy_(2.0 * W_slice.abs().mean(dim=1))
    qzero_out.fill_(0.5)


_MSE_TAU_RANGE = 0.1
_MSE_TAU_N = 50
_MSE_NORM = 2.4


def _find_group_params_inplace(
    W_slice: torch.Tensor,
    scale_out: torch.Tensor,
    qzero_out: torch.Tensor,
    maxq: float,
    sym: bool,
) -> None:
    """MC-MoE-consistent per-row (scale, zero) for ONE column slice.

    Matches `utils/quantizer_moe.py::Quantizer.find_params` semantics:
      - asymmetric: ``zero = -xmin/scale`` (FRACTIONAL, no round)
      - tau-range MSE search over ``p ∈ [1 - τ, 1 + τ]`` with `_MSE_TAU_N`
        candidates each side (default 50, plus the p=1 baseline), pick the
        ``(scale, zero)`` minimizing per-row L_NORM (NORM=2.4) quant error.

    Symmetric path uses ``zero = (maxq+1)/2`` (integer) and skips the MSE
    search (matches MC-MoE; sym is unused for our typical W2A4 setup but
    kept for completeness).

    Used together with the kernel's ADD_Z_BEFORE_ROUND=True path so that
    ``q_code = round(x/s + z)`` produces an integer code even when ``z`` is
    fractional.
    """
    xmax = W_slice.amax(dim=1)
    xmin = W_slice.amin(dim=1)
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
    scale = (xmax - xmin) / maxq
    if sym:
        zero = torch.full_like(scale, (maxq + 1) / 2)
    else:
        zero = -xmin / scale   # FRACTIONAL (MC-MoE)

    # MSE search: pick (scale, zero) per row that minimizes ‖q_deq − w‖_NORM.
    # Symmetric quantization keeps zero fixed (matches MC-MoE).
    if not sym:
        rows = W_slice.shape[0]
        best = torch.full((rows,), float('inf'), device=W_slice.device, dtype=W_slice.dtype)
        ps = torch.cat([
            torch.ones(1, device=W_slice.device, dtype=W_slice.dtype),
            torch.linspace(1.0, 1.0 + _MSE_TAU_RANGE, _MSE_TAU_N + 1,
                           device=W_slice.device, dtype=W_slice.dtype)[1:],
            torch.linspace(1.0, 1.0 - _MSE_TAU_RANGE, _MSE_TAU_N + 1,
                           device=W_slice.device, dtype=W_slice.dtype)[1:],
        ])
        for p in ps:
            xmin1 = p * xmin
            xmax1 = p * xmax
            scale1 = (xmax1 - xmin1) / maxq
            zero1 = -xmin1 / scale1
            s = scale1.unsqueeze(1)
            z = zero1.unsqueeze(1)
            q_code = torch.clamp(torch.round(W_slice / s + z), 0.0, maxq)
            q_deq = s * (q_code - z)
            err = (q_deq - W_slice).abs().pow(_MSE_NORM).sum(dim=1)
            update = err < best
            if update.any():
                best = torch.where(update, err, best)
                scale = torch.where(update, scale1, scale)
                zero = torch.where(update, zero1, zero)

    scale_out.copy_(scale)
    qzero_out.copy_(zero)
