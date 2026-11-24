"""The sparse indexer's prefill logits in one Triton kernel (gfx90a, no aiter).

Without aiter, `rocm_fp8_mqa_logits` falls back to `fp8_mqa_logits_torch`: an einsum of
bf16-cast FP8 queries [M, 32, 128] and keys [N, 128] into a dense [32, M, N] score, then
scale, ReLU, the per-head weights and a sum over heads, then the causal mask. In a
16k-token prefill that was ~85 ms of every 2048-token chunk (prefill profile, job
23898197: FP8 casts, an fp32-output GEMM, elementwise passes and a reduction), plus the
row chunking `_chunked_mqa_logits` needs to stay within memory.

Here a program owns a [BLOCK_M, BLOCK_N] tile of logits: it loads the key block once and
loops over the 32 heads, each a bf16 matrix-core dot of the query rows with it, rounded to
bf16 as the einsum's output is, then scale, ReLU and the weighted sum in fp32. Tiles that
lie wholly outside every row's [ks, ke) are written as -inf without computing (in prefill
about half of them). Same arithmetic as the reference up to fp32 summation order.
"""

import torch

from vllm.triton_utils import tl, triton

# test_mqa_logits.py (job 23900362), 2048 query rows: torch 4.5 / 18.2 / 37.8 / 68.1 ms ->
# 0.33 / 1.90 / 4.03 / 7.50 ms at 2k / 8k / 16k / 30k keys; same mask, max rel err 2e-7,
# top-512 overlap >= 99.96 %.
BLOCK_M, BLOCK_N, NUM_WARPS = 32, 128, 4


@triton.jit
def _mqa_logits_kernel(q_ptr, k_ptr, s_ptr, w_ptr, ks_ptr, ke_ptr, o_ptr, M, N,
                       stride_qm, stride_qh, stride_kn, stride_wm, stride_om,
                       H: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr,
                       BLOCK_N: tl.constexpr, ROUND: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    mmask = rm < M
    nmask = rn < N
    ks = tl.load(ks_ptr + rm, mask=mmask, other=2147483647)
    ke = tl.load(ke_ptr + rm, mask=mmask, other=0)
    lo = tl.min(ks, 0)
    hi = tl.max(ke, 0)
    out = o_ptr + rm.to(tl.int64)[:, None] * stride_om + rn[None, :]
    omask = mmask[:, None] & nmask[None, :]
    neg_inf = float("-inf")
    if (pn * BLOCK_N >= hi) | (pn * BLOCK_N + BLOCK_N <= lo):
        tl.store(out, tl.full([BLOCK_M, BLOCK_N], neg_inf, tl.float32), mask=omask)
    else:
        d = tl.arange(0, D)
        k = tl.load(k_ptr + rn.to(tl.int64)[:, None] * stride_kn + d[None, :],
                    mask=nmask[:, None], other=0)
        k = k.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
        sc = tl.load(s_ptr + rn, mask=nmask, other=0.0)
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        q_rows = q_ptr + rm.to(tl.int64)[:, None] * stride_qm + d[None, :]
        for h in tl.range(0, H):
            q = tl.load(q_rows + h * stride_qh, mask=mmask[:, None], other=0)
            q = q.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
            s = tl.dot(q, tl.trans(k))
            if ROUND:  # the einsum on bf16 tensors returns bf16
                s = s.to(tl.bfloat16).to(tl.float32)
            s = tl.maximum(s * sc[None, :], 0.0)
            w = tl.load(w_ptr + rm * stride_wm + h, mask=mmask, other=0.0)
            acc += s * w[:, None]
        valid = (rn[None, :] >= ks[:, None]) & (rn[None, :] < ke[:, None])
        tl.store(out, tl.where(valid, acc, neg_inf), mask=omask)


# Extra launch options (AMD). test_mqa_logits.py (job 23983712), 2048 rows: 4.23 -> 3.77 ms
# at 16k keys, 7.91 -> 7.12 at 30k (-10 %); same mask, max rel diff <= 1e-3.
LAUNCH_KW: dict = {"matrix_instr_nonkdim": 32, "waves_per_eu": 2, "kpack": 2}


def fp8_mqa_logits_triton(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke,
                          block_m=None, block_n=None, num_warps=None, round_bf16=True,
                          launch_kw=None):
    """Same arguments and result as rocm_aiter_mla_sparse.fp8_mqa_logits_torch:
    q [M, H, D] float8_e4m3fn, kv = (k [N, D] float8_e4m3fn, scales [N] or [N, 1] fp32),
    weights [M, H] fp32, cu_seqlen_ks / cu_seqlen_ke [M] int32 -> logits [M, N] fp32."""
    k_fp8, scale = kv
    m, h, d = q.shape
    n = k_fp8.shape[0]
    out = torch.empty(m, n, dtype=torch.float32, device=q.device)
    if m == 0 or n == 0:
        return out
    qb = q.contiguous().view(torch.uint8)
    kb = k_fp8.contiguous().view(torch.uint8)
    sc = scale.reshape(-1).float().contiguous()
    w = weights.float().contiguous()
    bm, bn = block_m or BLOCK_M, block_n or BLOCK_N
    _mqa_logits_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        qb, kb, sc, w, cu_seqlen_ks.int().contiguous(), cu_seqlen_ke.int().contiguous(), out,
        m, n, qb.stride(0), qb.stride(1), kb.stride(0), w.stride(0), out.stride(0),
        H=h, D=d, BLOCK_M=bm, BLOCK_N=bn, ROUND=round_bf16, num_warps=num_warps or NUM_WARPS,
        **(LAUNCH_KW if launch_kw is None else launch_kw),
    )
    return out
