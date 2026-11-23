"""Compressor score GEMMs in Triton (DSV4_SCORE_GEMM=1).

Every compressor layer computes `torch.mm(hidden, fused_wkv_wgate.weight.T,
out_dtype=torch.float32)` (attention.py, attn_gemm_parallel_execute): bf16 [T, 4096] times a
replicated bf16 [N, 4096] weight, N = 2048 (ratio-4), 1024 (ratio-128) or 512 (ratio-4
indexer), 60 calls per decode step. For an fp32 output hipBLASLt on gfx90a picks
`Cijk_..._BSS_..._MT64x16x16_MI16x16x1`: fp32 MFMA on bf16 inputs, 66 us per call at 32
users and 111 us at 64 (profile job 23631328), 4-7 ms per step, where reading the weight
takes ~10 us. This kernel uses bf16 MFMA with fp32 accumulation (bf16 products are exact in
fp32, so the result matches to fp32 rounding) and splits K so that the narrow N still
fills the GCD. Tiles come from test_score_gemm.py (job 23706116, graph replay, weights
not in L2): 3.6-6.4x faster per call, half the error of torch.mm against fp64, and
per decode step 3.96 -> 1.11 ms at 32 tokens, 6.54 -> 1.45 ms at 64. Split-K partials are
summed by a second kernel rather than atomics (deterministic; atomics saved only 1-2 us).
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_round_tile(a, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, eps):
    """bf16_linear._fp8_round_kernel on a [BLOCK_M, BLOCK_K] tile, per 128-column group (the
    same fp32 operations, so bit-identical): scale by max(amax, eps) / 448, round to e4m3
    precision (RNE, subnormal grid 2^-9 below 2^-6), scale back, bf16."""
    G: tl.constexpr = BLOCK_K // 128
    x = tl.reshape(a.to(tl.float32), [BLOCK_M, G, 128])
    s = tl.maximum(tl.max(tl.abs(x), axis=2), eps) / 448.0
    v = tl.minimum(tl.maximum(x / s[:, :, None], -448.0), 448.0)
    bits = v.to(tl.int32, bitcast=True)
    bits = (bits + 0x7FFFF + ((bits >> 20) & 1)) & -1048576
    normal = bits.to(tl.float32, bitcast=True)
    sub = ((v * 512.0 + 12582912.0) - 12582912.0) / 512.0
    q = tl.where(tl.abs(v) < 0.015625, sub, normal)
    return tl.reshape((q * s[:, :, None]).to(tl.bfloat16), [BLOCK_M, BLOCK_K])


@triton.jit
def _score_mm_kernel(x_ptr, w_ptr, o_ptr, M, N, K, sxm, swn, som,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     SPLIT_K: tl.constexpr, ATOMIC: tl.constexpr, M_FAST: tl.constexpr = False,
                     FP8_ROUND: tl.constexpr = False, eps=1e-10):
    # M_FAST: the token blocks of one weight tile are adjacent programs, so they run together
    # and the weight tile comes from HBM once (n-major order re-reads it per token block).
    if M_FAST:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
    else:
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    k_len = K // SPLIT_K
    k0 = pid_k * k_len
    a_ptrs = x_ptr + rm[:, None] * sxm + (k0 + rk)[None, :]
    b_ptrs = w_ptr + rn[None, :] * swn + (k0 + rk)[:, None]  # [BLOCK_K, BLOCK_N] of w.T
    mask_m = rm[:, None] < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, k_len, BLOCK_K):
        a = tl.load(a_ptrs, mask=mask_m, other=0.0)
        if FP8_ROUND:  # BF16_LINEAR's activation rounding, fused (BLOCK_K a multiple of 128)
            a = _fp8_round_tile(a, BLOCK_M, BLOCK_K, eps)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    o_ptrs = o_ptr + rm[:, None] * som + rn[None, :]
    if ATOMIC:
        tl.atomic_add(o_ptrs, acc, mask=mask_m, sem="relaxed")
    else:  # split k writes partial pid_k of [SPLIT_K, M, N]; SPLIT_K 1 may store bf16
        tl.store(o_ptrs + pid_k * M * som, acc.to(o_ptr.dtype.element_ty), mask=mask_m)


@triton.jit
def _splitk_sum_kernel(part_ptr, out_ptr, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    """out[i] = sum over k of part[k, i], in order k = 0.. (deterministic), cast to out's
    dtype. One pass, instead of torch's .sum(0) (~11 us at these sizes) plus a cast."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + k * MN + i, mask=mask, other=0.0)
    tl.store(out_ptr + i, acc.to(out_ptr.dtype.element_ty), mask=mask)


# (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps[, num_stages, m_fast]) per N and largest M;
# from the sweep.
CONFIGS: dict[int, list[tuple[int, tuple[int, int, int, int, int]]]] = {
    2048: [(16, (16, 64, 128, 4, 4)), (48, (16, 64, 128, 2, 4)), (64, (64, 64, 128, 2, 4)),
           (96, (16, 64, 128, 1, 4)), (128, (64, 64, 64, 2, 4))],
    1024: [(32, (16, 64, 128, 8, 4)), (64, (16, 64, 128, 4, 4)), (96, (16, 64, 128, 2, 4)),
           (128, (64, 64, 128, 4, 4))],
    512: [(8, (16, 16, 256, 8, 4)), (16, (16, 16, 128, 8, 4)), (24, (16, 64, 128, 8, 4)),
          (32, (32, 16, 128, 8, 4)), (64, (16, 64, 128, 8, 4)), (128, (16, 64, 128, 4, 4))],
    # MoE router (moe_small.install_gate), job 23811434
    256: [(16, (16, 16, 256, 8, 4)), (32, (32, 16, 256, 8, 4)), (64, (64, 16, 128, 8, 4)),
          (128, (16, 64, 128, 8, 4))],
}


def pick(M: int, N: int):
    for m_max, cfg in CONFIGS.get(N, ()):
        if M <= m_max:
            return cfg
    return None


SUM_BLOCK = 1024
# DSV4_SPLITK_SUM=torch keeps torch's .sum(0) for the split-K partials (as validated up to
# Nov 19); the default sums them with _splitk_sum_kernel.
TORCH_SUM = os.environ.get("DSV4_SPLITK_SUM", "triton") == "torch"


def score_mm(x: torch.Tensor, w: torch.Tensor, cfg=None, atomic: bool = False,
             out_dtype: torch.dtype = torch.float32, fp8_round: bool = False) -> torch.Tensor:
    """x [M, K] bf16 @ w[N, K].T -> [M, N] in out_dtype (fp32 accumulation either way).
    fp8_round: round x through FP8 per 128 columns first, in the kernel (cfg's BLOCK_K must
    be a multiple of 128)."""
    M, K = x.shape
    N = w.shape[0]
    cfg = cfg or pick(M, N)
    if cfg is None or x.stride(1) != 1 or w.stride(1) != 1:
        return torch.mm(x, w.T, out_dtype=out_dtype)
    bm, bn, bk, sk, warps = cfg[:5]
    stages, m_fast = (cfg[5], cfg[6]) if len(cfg) > 5 else (2, False)
    assert N % bn == 0 and K % (bk * sk) == 0, (M, N, K, cfg)
    grid = ((triton.cdiv(M, bm), N // bn, sk) if m_fast else (N // bn, triton.cdiv(M, bm), sk))
    assert not fp8_round or bk % 128 == 0, cfg
    kw = dict(M_FAST=m_fast, FP8_ROUND=fp8_round, num_warps=warps, num_stages=stages)
    if sk == 1:
        out = torch.empty(M, N, dtype=out_dtype, device=x.device)
        _score_mm_kernel[grid](x, w, out, M, N, K, x.stride(0), w.stride(0), out.stride(0),
                               bm, bn, bk, sk, False, **kw)
        return out
    if atomic:
        out = torch.zeros(M, N, dtype=torch.float32, device=x.device)
        _score_mm_kernel[grid](x, w, out, M, N, K, x.stride(0), w.stride(0), out.stride(0),
                               bm, bn, bk, sk, True, **kw)
        return out.to(out_dtype)
    part = torch.empty(sk, M, N, dtype=torch.float32, device=x.device)
    _score_mm_kernel[grid](x, w, part, M, N, K, x.stride(0), w.stride(0), N,
                           bm, bn, bk, sk, False, **kw)
    if TORCH_SUM:
        return part.sum(0).to(out_dtype)
    out = torch.empty(M, N, dtype=out_dtype, device=x.device)
    _splitk_sum_kernel[(triton.cdiv(M * N, SUM_BLOCK),)](part, out, M * N, sk, SUM_BLOCK,
                                                         num_warps=4)
    return out


def install() -> None:
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention

    def attn_gemm_parallel_execute(self, hidden_states):
        # ROCm passes no aux streams, so vLLM runs these four serially too.
        assert self.aux_stream_list is None
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        kv_score = indexer_kv_score = indexer_weights = None
        if self.compressor is not None:
            kv_score = score_mm(hidden_states, self.compressor.fused_wkv_wgate.weight)
        if self.indexer is not None:
            indexer_weights, _ = self.indexer.weights_proj(hidden_states)
            indexer_kv_score = score_mm(hidden_states, self.indexer.compressor.fused_wkv_wgate.weight)
        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    DeepseekV4Attention.attn_gemm_parallel_execute = attn_gemm_parallel_execute
