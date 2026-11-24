"""Sparse-attention prefill on gfx90a: upstream's ragged kernel
(rocm_aiter_mla_sparse._sparse_attn_prefill_ragged_kernel), with two knobs to find what
limits it.

Upstream runs one program per query: 8 heads padded to a 16-row MFMA tile, ~640 gathered
1 KB bf16 rows (SWA 128 + top-512) in steps of BLOCK_K, an fp32 [16, 512] accumulator in
registers. At 2048 queries it takes 1.77 ms (2 warps, nonkdim 16, job 23983859) for
~1.34 GB of gathers, ~760 GB/s.

DV_SPLIT: P programs per query, each owning OUT_DV / P output columns. The scores still
need the whole 512-wide rows; each program re-reads its value slice (from cache). Smaller
accumulators, so more programs resident.
LOADS_ONLY: the same gathers, no attention math (the access pattern's own ceiling); the
output is garbage.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _prefill_kernel(q_ptr, kv_ptr, idx_ptr, indptr_ptr, sink_ptr, out_ptr,
                    q_st, q_sh, kv_sn, out_st, out_sh, num_heads, num_kv, scale,
                    HAS_SINK: tl.constexpr, OUT_DV: tl.constexpr, BLOCK_H: tl.constexpr,
                    BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr, DV_SPLIT: tl.constexpr,
                    LOADS_ONLY: tl.constexpr):
    query = tl.program_id(0)
    dvb = tl.program_id(1)
    BLOCK_DV: tl.constexpr = BLOCK_D // DV_SPLIT
    h = tl.arange(0, BLOCK_H)
    hmask = h < num_heads
    d = tl.arange(0, BLOCK_D)
    dv = dvb * BLOCK_DV + tl.arange(0, BLOCK_DV)
    q = tl.load(q_ptr + query * q_st + h[:, None] * q_sh + d[None, :], mask=hmask[:, None], other=0.0)
    neg = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_DV), dtype=tl.float32)
    start = tl.load(indptr_ptr + query)
    n = tl.load(indptr_ptr + query + 1) - start
    ko = tl.arange(0, BLOCK_K)
    for k0 in tl.range(0, n, BLOCK_K, num_stages=1):
        pos = k0 + ko
        slot = tl.load(idx_ptr + start + pos, mask=pos < n, other=-1)
        valid = (pos < n) & (slot >= 0) & (slot < num_kv)
        row = kv_ptr + tl.where(valid, slot, 0).to(tl.int64)[:, None] * kv_sn
        kv = tl.load(row + d[None, :], mask=valid[:, None], other=0.0)
        if LOADS_ONLY:  # keep the loads alive, nothing else
            acc += tl.sum(tl.sum(kv.to(tl.float32), 1), 0)
            if DV_SPLIT > 1:
                v = tl.load(row + dv[None, :], mask=valid[:, None], other=0.0)
                acc += tl.sum(tl.sum(v.to(tl.float32), 1), 0)
        else:
            s = tl.dot(q, tl.trans(kv)) * scale
            s = tl.where(hmask[:, None] & valid[None, :], s, neg)
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            p = tl.where(hmask[:, None] & valid[None, :], p, 0.0)
            l_i = l_i * alpha + tl.sum(p, 1)
            if DV_SPLIT == 1:
                v = kv
            else:
                v = tl.load(row + dv[None, :], mask=valid[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
    if HAS_SINK:
        sink = tl.load(sink_ptr + h, mask=hmask, other=neg).to(tl.float32)
        m_f = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_f)
        l_f = l_i * alpha + tl.exp(sink - m_f)
        o = tl.where(l_f[:, None] > 0.0, (acc * alpha[:, None]) / tl.maximum(l_f, 1e-30)[:, None], 0.0)
    else:
        o = tl.where(l_i[:, None] > 0.0, acc / tl.maximum(l_i, 1e-30)[:, None], 0.0)
    tl.store(out_ptr + query * out_st + h[:, None] * out_sh + dv[None, :], o,
             mask=hmask[:, None] & (dv[None, :] < OUT_DV))


def sparse_prefill(q, kv, indices, indptr, scale, attn_sink, out=None, out_dv=None,
                   dv_split=1, block_k=16, num_warps=2, loads_only=False, **launch):
    """q [T, H, 512] bf16, kv [N, 512] bf16 workspace, ragged indices/indptr (int32), as
    upstream's _rocm_sparse_attn_prefill_ragged_triton; out [T, H, out_dv]."""
    T, H, D = q.shape
    assert H <= 16 and D == 512 and kv.stride(-1) == 1 and q.stride(-1) == 1
    out_dv = out_dv or D
    if out is None:
        out = torch.empty(T, H, out_dv, dtype=q.dtype, device=q.device)
    if T == 0:
        return out
    kw = {"matrix_instr_nonkdim": 16, "waves_per_eu": 1, **launch}
    _prefill_kernel[(T, dv_split)](
        q, kv, indices, indptr, attn_sink if attn_sink is not None else q, out,
        q.stride(0), q.stride(1), kv.stride(0), out.stride(0), out.stride(1), H, kv.shape[0],
        float(scale), HAS_SINK=attn_sink is not None, OUT_DV=out_dv, BLOCK_H=16, BLOCK_D=D,
        BLOCK_K=block_k, DV_SPLIT=dv_split, LOADS_ONLY=loads_only, num_warps=num_warps, **kw,
    )
    return out
