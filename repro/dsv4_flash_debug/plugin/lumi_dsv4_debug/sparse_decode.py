"""Sparse MLA decode attention for gfx90a: split-K partial + reduce, with a lean inner loop.

Copied from the V4.1 port (branch exp/deepseek-v41-gfx90a, lumi_dsv41/sparse_decode.py at
fd3fd13); V4-Flash uses the same attention stack and fp8_ds_mla record (576 bytes per token,
then 8 scale bytes, per block). install() at the end hooks it into stock vLLM 0.26 for
DSV4_SPARSE_DECODE=1; it supersedes DSV4_SPARSE_SPLIT. The text below is V4.1's.

V4.1 decodes every layer's attention over a ragged list of KV slots: the sliding window
(main, 128 entries) plus, in 38 of 43 layers, the indexer's top-512 compressed entries
(extra). Off gfx942/950 upstream runs a one-pass kernel with one workgroup per query; its
own split-K path (gfx942/950) shares the same inner loop, and both run at ~50 GB/s at
32-64 queries (graph profile job 23701511: 19.3 ms of a 32-user step, the largest cost).
That loop walks 16 entries per iteration, gathers the UE8M0 scale byte once per element
(16 x 512 loads), and does the 448 FP8 dims (padded to 512) and the 64 bf16 RoPE dims as
two dot pairs.

Here (GPU time per top-k layer call vs the fallback, job 23707915: 409 -> 30 us at 1 query,
434 -> 108 at 32, 445 -> 188 at 64, 888 -> 336 at 128; the split-K path 62 / 278 / 584 /
1249), per program (query, split): BLOCK_K entries per iteration, scales loaded once per
token and broadcast over their 64 dims, NoPE and RoPE assembled into one [BLOCK_K, 512]
bf16 tile so that each step is one QK^T and one PV dot over the 8 (padded to 16) heads;
main and extra are walked as one concatenated list, split evenly across NUM_SPLITS
programs; partials (max, sum, acc) are combined with the attention sink by a second
kernel. Same arithmetic as the fallback: K dequantised to bf16 exactly (FP8 x 2^e), fp32
scores and accumulators, invalid slots and NaN K values skipped.
"""

import torch

from vllm.triton_utils import tl, triton

NOPE, ROPE, HEAD_DIM, TOKEN_BYTES, SCALE_BYTES = 448, 64, 512, 576, 8


@triton.jit
def _load_k(main_cache_ptr, extra_cache_ptr, main_idx_ptr, extra_idx_ptr, main_start,
            extra_start, main_len, total, pos, main_rows, extra_rows, main_bs, extra_bs,
            main_stride, extra_stride, IS_FNUZ: tl.constexpr, BLOCK_K: tl.constexpr,
            DEQ: tl.constexpr = 0):
    """K rows for list positions `pos` of the concatenated (main, extra) list:
    [BLOCK_K, 512] bf16 (448 dequantised FP8 NoPE dims, then 64 bf16 RoPE dims), and the
    validity mask."""
    is_main = pos < main_len
    in_list = pos < total
    slot_m = tl.load(main_idx_ptr + main_start + pos, mask=is_main, other=-1)
    slot_e = tl.load(extra_idx_ptr + extra_start + (pos - main_len),
                     mask=in_list & ~is_main, other=-1)
    slot = tl.where(is_main, slot_m, slot_e)
    rows = tl.where(is_main, main_rows, extra_rows)
    valid = in_list & (slot >= 0) & (slot < rows)
    slot = tl.where(valid, slot, 0)
    bs = tl.where(is_main, main_bs, extra_bs)
    blk = (slot // bs).to(tl.int64)
    off = slot % bs
    main_block = main_cache_ptr + blk * main_stride
    extra_block = extra_cache_ptr + blk * extra_stride
    block = tl.where(is_main, main_block, extra_block)
    data = block + off * 576
    scale_row = block + bs * 576 + off * 8

    d = tl.arange(0, 512)
    is_nope = d < 448
    x = tl.load(data[:, None] + d[None, :], mask=valid[:, None] & is_nope[None, :], other=0)
    if DEQ == 1:  # ablation only: skip the FP8 decode (wrong values)
        xf = x.to(tl.float32)
    elif IS_FNUZ:
        xf = x.to(tl.float8e4b8, bitcast=True).to(tl.float32)
    else:
        xf = x.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    e = tl.load(scale_row[:, None] + tl.arange(0, 8)[None, :], mask=valid[:, None], other=127)
    s = tl.exp2(e.to(tl.float32) - 127.0)  # UE8M0
    s = tl.reshape(tl.broadcast_to(s[:, :, None], (BLOCK_K, 8, 64)), (BLOCK_K, 512))
    rope = tl.load((data + 448).to(tl.pointer_type(tl.bfloat16))[:, None] + (d[None, :] - 448),
                   mask=valid[:, None] & ~is_nope[None, :], other=0.0)
    k = tl.where(is_nope[None, :], (xf * s).to(tl.bfloat16), rope)
    k = tl.where(valid[:, None] & (k == k), k, 0.0)
    return k, valid


@triton.jit
def _partial_kernel(q_ptr, main_cache_ptr, main_idx_ptr, main_indptr_ptr,
                    extra_cache_ptr, extra_idx_ptr, extra_indptr_ptr,
                    pm_ptr, pl_ptr, pacc_ptr,
                    q_s0, q_s1, main_rows, extra_rows, main_bs, extra_bs,
                    main_stride, extra_stride, scale, num_heads,
                    NUM_SPLITS: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
                    HAS_EXTRA: tl.constexpr, IS_FNUZ: tl.constexpr,
                    NUM_STAGES: tl.constexpr = 1, DEQ: tl.constexpr = 0):
    b = tl.program_id(0)
    sp = tl.program_id(1)
    h = tl.arange(0, BLOCK_H)
    hmask = h < num_heads
    d = tl.arange(0, 512)
    q = tl.load(q_ptr + b * q_s0 + h[:, None] * q_s1 + d[None, :], mask=hmask[:, None], other=0.0)

    main_start = tl.load(main_indptr_ptr + b)
    main_len = tl.load(main_indptr_ptr + b + 1) - main_start
    if HAS_EXTRA:
        extra_start = tl.load(extra_indptr_ptr + b)
        extra_len = tl.load(extra_indptr_ptr + b + 1) - extra_start
    else:
        extra_start = 0
        extra_len = 0
    total = main_len + extra_len
    chunk = tl.cdiv(tl.cdiv(total, NUM_SPLITS), BLOCK_K) * BLOCK_K
    lo = sp * chunk
    hi = tl.minimum(lo + chunk, total)

    neg = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg, tl.float32)
    l_i = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, 512), tl.float32)
    kk = tl.arange(0, BLOCK_K)
    for start in tl.range(lo, hi, BLOCK_K, num_stages=NUM_STAGES):
        pos = start + kk
        k, valid = _load_k(main_cache_ptr, extra_cache_ptr, main_idx_ptr, extra_idx_ptr,
                           main_start, extra_start, main_len, hi, pos, main_rows, extra_rows,
                           main_bs, extra_bs, main_stride, extra_stride, IS_FNUZ, BLOCK_K, DEQ)
        sc = tl.dot(q, tl.trans(k)) * scale
        sc = tl.where(hmask[:, None] & valid[None, :], sc, neg)
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new[:, None])
        p = tl.where(hmask[:, None] & valid[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), k)
        m_i = m_new

    row = b * NUM_SPLITS + sp
    tl.store(pm_ptr + row * BLOCK_H + h, m_i)
    tl.store(pl_ptr + row * BLOCK_H + h, l_i)
    tl.store(pacc_ptr + (row * BLOCK_H + h[:, None]) * 512 + d[None, :], acc)


@triton.jit
def _reduce_kernel(pm_ptr, pl_ptr, pacc_ptr, sink_ptr, out_ptr, out_s0, out_s1, num_heads,
                   NUM_SPLITS: tl.constexpr, SPLITS_PAD: tl.constexpr, BLOCK_H: tl.constexpr,
                   HAS_SINK: tl.constexpr):
    b = tl.program_id(0)
    hb = tl.program_id(1)  # one head per program: [splits, 512] fits easily
    s = tl.arange(0, SPLITS_PAD)
    smask = s < NUM_SPLITS
    row = (b * NUM_SPLITS + s) * BLOCK_H + hb
    neg = -3.4028234663852886e38
    m = tl.load(pm_ptr + row, mask=smask, other=neg)
    l = tl.load(pl_ptr + row, mask=smask, other=0.0)
    m_max = tl.max(m, 0)
    if HAS_SINK:
        sink = tl.load(sink_ptr + hb).to(tl.float32)
        m_max = tl.maximum(m_max, sink)
    w = tl.exp(m - m_max)
    w = tl.where(smask, w, 0.0)
    denom = tl.sum(l * w, 0)
    if HAS_SINK:
        denom += tl.exp(sink - m_max)
    d = tl.arange(0, 512)
    acc = tl.load(pacc_ptr + row[:, None] * 512 + d[None, :], mask=smask[:, None], other=0.0)
    o = tl.sum(acc * w[:, None], 0)
    o = tl.where(denom > 0.0, o / tl.maximum(denom, 1.0e-30), 0.0)
    tl.store(out_ptr + b * out_s0 + hb * out_s1 + d, o.to(out_ptr.dtype.element_ty))


def num_splits(num_queries: int, has_extra: bool) -> int:
    """KV splits per query: ~512 workgroups for top-k layers (128 + 512 entries), ~256 for
    SWA-only layers (128), as a power of two. From test_sparse_decode.py's sweep (job
    23707915, BLOCK_K 16, 4 warps); within 10 % of the best split count at 1-128 queries."""
    target, cap = (512, 16) if has_extra else (256, 8)
    s = max(1, min(cap, target // max(1, num_queries)))
    return 1 << (s.bit_length() - 1)


def sparse_decode(q, main_cache, main_indices, main_indptr, scale, attn_sink,
                  extra_cache=None, extra_indices=None, extra_indptr=None, out=None,
                  is_fnuz=False, splits=None, block_k=16, num_warps=4, num_stages=1, deq=0):
    """Decode attention like rocm_aiter_mla_sparse._rocm_sparse_attn_decode_ragged_triton
    for V4/V4.1's fp8_ds_mla record (448 FP8 NoPE + 64 bf16 RoPE, head dim 512) and up
    to 16 heads per rank. q [B, H, 512] bf16; indices/indptr int32 ragged lists."""
    B, H, D = q.shape
    assert D == HEAD_DIM and H <= 16 and q.stride(-1) == 1
    if out is None:
        out = torch.empty_like(q, dtype=torch.bfloat16)
    if B == 0:
        return out
    has_extra = extra_cache is not None and extra_indices is not None and extra_indptr is not None
    if not has_extra:
        extra_cache, extra_indices, extra_indptr = main_cache, main_indices, main_indptr
    S = splits or num_splits(B, has_extra)
    BH = 16
    pm = torch.empty(B * S * BH, device=q.device, dtype=torch.float32)
    pl = torch.empty_like(pm)
    pacc = torch.empty(B * S * BH, D, device=q.device, dtype=torch.float32)
    _partial_kernel[(B, S)](
        q, main_cache, main_indices, main_indptr, extra_cache, extra_indices, extra_indptr,
        pm, pl, pacc, q.stride(0), q.stride(1),
        main_cache.shape[0] * main_cache.shape[1], extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1], extra_cache.shape[1], main_cache.stride(0), extra_cache.stride(0),
        scale, H, NUM_SPLITS=S, BLOCK_H=BH, BLOCK_K=block_k, HAS_EXTRA=has_extra,
        IS_FNUZ=is_fnuz, NUM_STAGES=num_stages, DEQ=deq, num_warps=num_warps,
    )
    sink = attn_sink if attn_sink is not None else pm
    _reduce_kernel[(B, H)](
        pm, pl, pacc, sink, out, out.stride(0), out.stride(1), H,
        NUM_SPLITS=S, SPLITS_PAD=triton.next_power_of_2(S), BLOCK_H=BH,
        HAS_SINK=attn_sink is not None, num_warps=4,
    )
    return out


def install() -> None:
    """Replace stock 0.26's `_rocm_sparse_attn_decode_ragged_triton` (no `out` argument;
    called with keywords) off gfx942/950. FNUZ caches and other dims go to the stock path."""
    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as m

    if m._ON_GFX942 or m._ON_GFX950:
        return
    stock = m._rocm_sparse_attn_decode_ragged_triton
    names = ("q", "main_cache", "main_indices", "main_indptr", "scale", "attn_sink",
             "nope_head_dim", "rope_head_dim", "extra_cache", "extra_indices", "extra_indptr")

    def decode(*args, **kwargs):
        a = dict(zip(names, args))
        a.update(kwargs)
        q = a["q"]
        if (m.current_platform.is_fp8_fnuz() or a["nope_head_dim"] != NOPE
                or a["rope_head_dim"] != ROPE or q.shape[1] > 16 or q.stride(-1) != 1):
            return stock(*args, **kwargs)
        ei, ep = a.get("extra_indices"), a.get("extra_indptr")
        return sparse_decode(
            q, a["main_cache"], m._as_int32_contiguous_1d(a["main_indices"]),
            m._as_int32_contiguous_1d(a["main_indptr"]), a["scale"], a["attn_sink"],
            extra_cache=a.get("extra_cache"),
            extra_indices=None if ei is None else m._as_int32_contiguous_1d(ei),
            extra_indptr=None if ep is None else m._as_int32_contiguous_1d(ep),
        )

    decode.stock = stock
    m._rocm_sparse_attn_decode_ragged_triton = decode
