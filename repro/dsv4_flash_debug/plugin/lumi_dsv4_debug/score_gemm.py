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

import torch
import triton
import triton.language as tl


@triton.jit
def _score_mm_kernel(x_ptr, w_ptr, o_ptr, M, N, K, sxm, swn, som,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     SPLIT_K: tl.constexpr, ATOMIC: tl.constexpr):
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
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    o_ptrs = o_ptr + rm[:, None] * som + rn[None, :]
    if ATOMIC:
        tl.atomic_add(o_ptrs, acc, mask=mask_m, sem="relaxed")
    else:  # split k writes partial pid_k of [SPLIT_K, M, N]
        tl.store(o_ptrs + pid_k * M * som, acc, mask=mask_m)


# (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps) per N and largest M; from the sweep.
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


def score_mm(x: torch.Tensor, w: torch.Tensor, cfg=None, atomic: bool = False) -> torch.Tensor:
    """x [M, K] bf16 @ w[N, K].T -> [M, N] fp32."""
    M, K = x.shape
    N = w.shape[0]
    cfg = cfg or pick(M, N)
    if cfg is None or x.stride(1) != 1 or w.stride(1) != 1:
        return torch.mm(x, w.T, out_dtype=torch.float32)
    bm, bn, bk, sk, warps = cfg
    assert N % bn == 0 and K % (bk * sk) == 0, (M, N, K, cfg)
    grid = (N // bn, triton.cdiv(M, bm), sk)
    if sk == 1 or atomic:
        out = (torch.zeros if sk > 1 else torch.empty)(M, N, dtype=torch.float32, device=x.device)
        _score_mm_kernel[grid](x, w, out, M, N, K, x.stride(0), w.stride(0), out.stride(0),
                               bm, bn, bk, sk, sk > 1, num_warps=warps)
        return out
    part = torch.empty(sk, M, N, dtype=torch.float32, device=x.device)
    _score_mm_kernel[grid](x, w, part, M, N, K, x.stride(0), w.stride(0), N,
                           bm, bn, bk, sk, False, num_warps=warps)
    return part.sum(0)


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
