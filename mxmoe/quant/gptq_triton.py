"""Triton + CUDA Graph implementation of GPTQ's inner column loop for MxMoE.

Kernels (`_quantize_column_kernel`, `_rank1_update_kernel`) are taken from
hbmpq/amq/kernel/gptq_triton.py with no algorithmic change — including the
libdevice.rint + fp64 round-trip trick that keeps results bit-identical to
PyTorch's torch.round / torch.div.

Wrapper added on top:
  - `fasterquant_triton(W, H, ...)` — public entry that takes RAW (W, H) the
    same way mxmoe.quant.gptq.GPTQ.fasterquant does, applies actorder
    permutation if requested, builds per-column scale/qzero in permuted
    column order, runs the kernel, and inv-permutes the output.
  - Supports groupsize=-1 (per-channel) by expanding the single (rows, 1)
    per-row scale across all columns (n_groups_kernel = 1, group_size_kernel
    = cols).
  - Supports static_groups + actorder: kernel sees per-column scale
    (group_size_kernel = 1, n_groups_kernel = cols) — each permuted column
    is looked up against its ORIGINAL position's static group.
  - Supports dynamic groups (static_groups=False, groupsize != -1) ONLY when
    actorder=False (matches hbmpq's existing relax — dynamic + actorder is
    semantically tricky and not needed for the primary MxMoE coverage).

Public API:
  - `fasterquant_triton(W, H, *, wbits, groupsize, actorder, static_groups,
                        sym, blocksize=128, percdamp=0.01)`
  - `fasterquant_triton_graph(W, H, *, ...)` — same signature, CUDA Graph
    captured. Caches per (rows, cols, group_size_kernel, dtype, ...).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# ============================================================================
# Triton kernels — verbatim from hbmpq/amq/kernel/gptq_triton.py
# ============================================================================

@triton.jit
def _tl_round(x):
    return libdevice.rint(x)


@triton.jit
def _tl_quantize(x, scale, qzero, maxq):
    # fp64 round-trip — see hbmpq comment for rationale (matches torch.div
    # bit-exactly; plain fp32 div drifts by 1 ULP and breaks ties).
    ratio = (x.to(tl.float64) / scale.to(tl.float64)).to(tl.float32)
    return tl.clamp(_tl_round(ratio) + qzero, 0.0, maxq)


@triton.jit
def _tl_dequantize(q, scale, qzero):
    return scale * (q - qzero)


@triton.jit
def _quantize_column_kernel(
    W_ptr, err_ptr, scale_col_ptr, qzero_col_ptr, Hinv_diag_ptr,
    maxq, col_idx,
    rows: tl.int64,
    cols_stride: tl.int64,
    BLOCK: tl.constexpr,
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
    q_code = _tl_quantize(w, s, z, maxq)
    q_deq = _tl_dequantize(q_code, s, z)
    err = ((w - q_deq).to(tl.float64) / d.to(tl.float64)).to(tl.float32)

    tl.store(w_ptrs, q_deq, mask=mask)
    tl.store(err_ptr + offs, err, mask=mask)


@triton.jit
def _rank1_update_kernel(
    W_ptr, err_ptr, Hinv_ptr,
    row_idx, col_start, col_end,
    rows: tl.int64,
    W_row_stride: tl.int64,
    Hinv_row_stride: tl.int64,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
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

    w_ptrs = (W_ptr + rows_offs[:, None].to(tl.int64) * W_row_stride
              + cols_offs[None, :].to(tl.int64))
    w = tl.load(w_ptrs, mask=mask, other=0.0)
    tl.store(w_ptrs, w - update, mask=mask)


# ============================================================================
# Quantizer math (mirrors mxmoe.quant.gptq.Quantizer)
# ============================================================================

def _find_params(x: torch.Tensor, bits: int, sym: bool
                 ) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Per-row find_params (perchannel=True, weight=True).

    Returns (scale, zero, maxq_float) where scale/zero are (rows, 1).
    """
    maxq = float(2 ** bits - 1)
    dev = x.device
    x = x.flatten(1)
    tmp = torch.zeros(x.shape[0], device=dev, dtype=x.dtype)
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)
    if sym:
        xmax = torch.maximum(xmin.abs(), xmax)
        neg = xmin < 0
        if torch.any(neg):
            xmin = torch.where(neg, -xmax, xmin)
    both_zero = (xmin == 0) & (xmax == 0)
    xmin = torch.where(both_zero, -torch.ones_like(xmin), xmin)
    xmax = torch.where(both_zero, torch.ones_like(xmax), xmax)
    scale = (xmax - xmin) / maxq
    if sym:
        zero = torch.full_like(scale, (maxq + 1) / 2)
    else:
        zero = torch.round(-xmin / scale)
    return scale.unsqueeze(1), zero.unsqueeze(1), maxq


def _find_params_inplace_scaled(
    W_slice: torch.Tensor, scale_out: torch.Tensor, qzero_out: torch.Tensor,
    maxq: float, sym: bool,
) -> None:
    """Per-row find_params writing into preallocated (rows,) buffers.

    Used for the dynamic-groups path (where we need to recompute at each
    block boundary, and reuse the same persistent buffers under CUDA Graph
    capture).
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
    scale_out.copy_((xmax - xmin) / maxq)
    if sym:
        qzero_out.fill_((maxq + 1) / 2)
    else:
        qzero_out.copy_(torch.round(-xmin / scale_out))


# ============================================================================
# Inner-loop driver (matches the column-flow of GPTQ.fasterquant)
# ============================================================================

def _make_scratch(W: torch.Tensor, scale_t: torch.Tensor, blocksize: int) -> dict:
    """Allocate persistent buffers (CUDA Graph-friendly)."""
    rows, cols = W.shape
    return {
        'Hinv_diag': torch.empty(cols, dtype=W.dtype, device=W.device),
        'err_buf': torch.empty(rows, dtype=W.dtype, device=W.device),
        'Err_block': torch.empty(rows, blocksize, dtype=W.dtype, device=W.device),
    }


def _run_inner_loop_triton(
    W: torch.Tensor,            # (rows, cols), updated in place to Q (dequant)
    Hinv: torch.Tensor,         # (cols, cols), upper-tri Cholesky of inv(H+damp)
    scale_t: torch.Tensor,      # (n_groups_kernel, rows) — per-group per-row scale
    qzero_t: torch.Tensor,      # (n_groups_kernel, rows) — per-group per-row qzero
    maxq: float,
    group_size_kernel: int,     # kernel's view of group span (1 / blocksize / cols)
    blocksize: int,
    *,
    scratch: dict | None = None,
    dynamic_groups: bool = False,
    sym: bool = True,
) -> None:
    """Run the triton inner loop. Writes dequantized values back into W."""
    assert W.is_cuda and Hinv.is_cuda
    assert W.is_contiguous() and Hinv.is_contiguous()
    assert scale_t.is_contiguous() and qzero_t.is_contiguous()
    rows, cols = W.shape
    assert Hinv.shape == (cols, cols)
    assert scale_t.shape[1] == rows and qzero_t.shape[1] == rows
    if dynamic_groups:
        assert group_size_kernel == blocksize, (
            "dynamic_groups requires group_size_kernel == blocksize "
            f"(got {group_size_kernel} vs {blocksize})"
        )

    if scratch is None:
        scratch = _make_scratch(W, scale_t, blocksize)
    Hinv_diag = scratch['Hinv_diag']
    err_buf = scratch['err_buf']
    Err_block = scratch['Err_block']

    # Pre-extract Hinv's diagonal once (constant for the run).
    Hinv_diag.copy_(torch.diagonal(Hinv))

    BLOCK_QUANT = 256
    BLOCK_R, BLOCK_C = 64, 128

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        block_w = i2 - i1
        g_block = i1 // group_size_kernel
        if dynamic_groups:
            # Recompute group params for this block from CURRENT W slice
            # (post all prior rank-1 updates).
            _find_params_inplace_scaled(
                W[:, i1:i2], scale_t[g_block], qzero_t[g_block], maxq, sym,
            )
        for i_local in range(block_w):
            i = i1 + i_local
            g = i // group_size_kernel
            grid_q = (triton.cdiv(rows, BLOCK_QUANT),)
            _quantize_column_kernel[grid_q](
                W, err_buf, scale_t[g], qzero_t[g], Hinv_diag, maxq,
                i, rows, cols, BLOCK=BLOCK_QUANT,
            )
            Err_block[:, i_local].copy_(err_buf)
            if i + 1 < i2:
                cwidth = i2 - i - 1
                grid_r1 = (triton.cdiv(rows, BLOCK_R), triton.cdiv(cwidth, BLOCK_C))
                _rank1_update_kernel[grid_r1](
                    W, err_buf, Hinv, i,
                    i + 1, i2, rows, cols, cols,
                    BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C,
                )
        if i2 < cols:
            W[:, i2:].addmm_(Err_block[:, :block_w], Hinv[i1:i2, i2:],
                             beta=1.0, alpha=-1.0)


# ============================================================================
# Public wrapper — same call surface as fasterquant_reference
# ============================================================================

def _prepare(
    W: torch.Tensor, H: torch.Tensor, *,
    wbits: int, groupsize: int, actorder: bool, static_groups: bool, sym: bool,
    blocksize: int, percdamp: float,
) -> dict:
    """Compute everything the kernel needs: permuted W, Hinv, per-column
    scale/qzero, invperm, kernel group_size.

    Returns a dict so the CUDA-Graph version can reuse the same setup.
    """
    assert W.dim() == 2 and H.dim() == 2
    rows, cols = W.shape
    assert H.shape == (cols, cols)
    dev = W.device

    W_proc = W.detach().clone().float().contiguous()
    H_proc = H.detach().clone().float()

    # Initial per-channel find_params (matches GPTQ.fasterquant).
    scale_init, zero_init, maxq = _find_params(W_proc, wbits, sym)

    dead = torch.diag(H_proc) == 0
    H_proc[dead, dead] = 1
    W_proc[:, dead] = 0

    static_group_scale = None
    static_group_zero = None
    if static_groups and groupsize != -1:
        # Pre-compute from ORIGINAL position. Each col c (original idx) maps
        # to group c // groupsize.
        s_list, z_list = [], []
        for i in range(0, cols, groupsize):
            s, z, _ = _find_params(W_proc[:, i:i + groupsize], wbits, sym)
            s_list.append(s)
            z_list.append(z)
        static_group_scale = torch.cat(s_list, dim=1)  # (rows, n_groups)
        static_group_zero = torch.cat(z_list, dim=1)

    perm = None
    invperm = None
    if actorder:
        perm = torch.argsort(torch.diag(H_proc), descending=True)
        W_proc = W_proc[:, perm].contiguous()
        H_proc = H_proc[perm][:, perm].contiguous()
        invperm = torch.argsort(perm)

    # Decide kernel group_size + build (n_groups_kernel, rows) scale_t/qzero_t.
    dynamic_groups = False
    if groupsize == -1:
        # Per-channel — single group spans all cols.
        scale_t = scale_init.squeeze(1).unsqueeze(0).contiguous()   # (1, rows)
        qzero_t = zero_init.squeeze(1).unsqueeze(0).contiguous()
        group_size_kernel = cols
    elif static_groups:
        # Permuted col c → original idx (perm[c] if actorder else c) →
        # static group (orig_idx // groupsize). Materialize per-col scale.
        if actorder:
            orig_idx = perm
        else:
            orig_idx = torch.arange(cols, device=dev)
        g_idx = orig_idx // groupsize                              # (cols,)
        scale_per_col = static_group_scale[:, g_idx]               # (rows, cols)
        zero_per_col = static_group_zero[:, g_idx]
        scale_t = scale_per_col.t().contiguous()                   # (cols, rows)
        qzero_t = zero_per_col.t().contiguous()
        group_size_kernel = 1
    else:
        # Dynamic groups: scale recomputed from current W slice each block.
        # Forbid actorder here — caller-side perm doesn't survive the
        # block-by-block find_params over CURRENT W (it would re-derive
        # scale from permuted columns, not from the original group).
        assert not actorder, "Triton: actorder + dynamic_groups not supported"
        assert groupsize == blocksize, (
            "Triton: dynamic_groups requires groupsize == blocksize "
            f"(got {groupsize} vs {blocksize})"
        )
        n_groups_kernel = cols // groupsize
        scale_t = torch.zeros(n_groups_kernel, rows,
                              dtype=W_proc.dtype, device=dev)
        qzero_t = torch.zeros_like(scale_t)
        group_size_kernel = groupsize
        dynamic_groups = True

    # Hinv
    damp = percdamp * torch.mean(torch.diag(H_proc))
    diag_idx = torch.arange(cols, device=dev)
    H_proc[diag_idx, diag_idx] += damp
    L = torch.linalg.cholesky(H_proc)
    Hinv_full = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv_full, upper=True).contiguous()

    return {
        'W': W_proc,
        'Hinv': Hinv,
        'scale_t': scale_t,
        'qzero_t': qzero_t,
        'maxq': maxq,
        'group_size_kernel': group_size_kernel,
        'blocksize': blocksize,
        'dynamic_groups': dynamic_groups,
        'sym': sym,
        'invperm': invperm,
        'cols': cols,
        'rows': rows,
    }


@torch.no_grad()
def fasterquant_triton(
    W: torch.Tensor,
    H: torch.Tensor,
    *,
    wbits: int,
    groupsize: int,
    actorder: bool,
    static_groups: bool,
    sym: bool,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> torch.Tensor:
    """Direct (non-graph) triton path. Returns Q (rows, cols), fp32."""
    p = _prepare(W, H, wbits=wbits, groupsize=groupsize, actorder=actorder,
                 static_groups=static_groups, sym=sym, blocksize=blocksize,
                 percdamp=percdamp)
    _run_inner_loop_triton(
        p['W'], p['Hinv'], p['scale_t'], p['qzero_t'], p['maxq'],
        p['group_size_kernel'], p['blocksize'],
        dynamic_groups=p['dynamic_groups'], sym=p['sym'],
    )
    Q = p['W']
    if p['invperm'] is not None:
        Q = Q[:, p['invperm']].contiguous()
    return Q


# ============================================================================
# CUDA Graph wrapper
# ============================================================================

_GRAPH_CACHE: dict = {}


def _graph_key(p: dict) -> tuple:
    return (
        p['rows'], p['cols'], p['group_size_kernel'], p['blocksize'],
        float(p['maxq']),
        bool(p['dynamic_groups']), bool(p['sym']),
        str(p['W'].dtype), str(p['W'].device),
        tuple(p['scale_t'].shape),
    )


@torch.no_grad()
def fasterquant_triton_graph(
    W: torch.Tensor,
    H: torch.Tensor,
    *,
    wbits: int,
    groupsize: int,
    actorder: bool,
    static_groups: bool,
    sym: bool,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> torch.Tensor:
    """CUDA-Graph captured inner loop. Setup (find_params, Cholesky, perm) is
    still eager — only the inner column loop is captured.
    """
    p = _prepare(W, H, wbits=wbits, groupsize=groupsize, actorder=actorder,
                 static_groups=static_groups, sym=sym, blocksize=blocksize,
                 percdamp=percdamp)
    key = _graph_key(p)

    if key not in _GRAPH_CACHE:
        rows, cols = p['rows'], p['cols']
        W_buf = torch.empty_like(p['W'])
        Hinv_buf = torch.empty_like(p['Hinv'])
        scale_buf = torch.empty_like(p['scale_t'])
        qzero_buf = torch.empty_like(p['qzero_t'])
        scratch = _make_scratch(W_buf, scale_buf, blocksize)

        # Warmup on a side stream — required before capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                W_buf.copy_(p['W'])
                Hinv_buf.copy_(p['Hinv'])
                scale_buf.copy_(p['scale_t'])
                qzero_buf.copy_(p['qzero_t'])
                _run_inner_loop_triton(
                    W_buf, Hinv_buf, scale_buf, qzero_buf, p['maxq'],
                    p['group_size_kernel'], p['blocksize'],
                    scratch=scratch,
                    dynamic_groups=p['dynamic_groups'], sym=p['sym'],
                )
        torch.cuda.current_stream().wait_stream(s)

        # Capture.
        W_buf.copy_(p['W'])
        Hinv_buf.copy_(p['Hinv'])
        scale_buf.copy_(p['scale_t'])
        qzero_buf.copy_(p['qzero_t'])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run_inner_loop_triton(
                W_buf, Hinv_buf, scale_buf, qzero_buf, p['maxq'],
                p['group_size_kernel'], p['blocksize'],
                scratch=scratch,
                dynamic_groups=p['dynamic_groups'], sym=p['sym'],
            )
        _GRAPH_CACHE[key] = {
            'graph': graph,
            'W_buf': W_buf, 'Hinv_buf': Hinv_buf,
            'scale_buf': scale_buf, 'qzero_buf': qzero_buf,
            'scratch': scratch,
        }

    c = _GRAPH_CACHE[key]
    c['W_buf'].copy_(p['W'])
    c['Hinv_buf'].copy_(p['Hinv'])
    c['scale_buf'].copy_(p['scale_t'])
    c['qzero_buf'].copy_(p['qzero_t'])
    c['graph'].replay()
    Q = c['W_buf'].clone()
    if p['invperm'] is not None:
        Q = Q[:, p['invperm']].contiguous()
    return Q
