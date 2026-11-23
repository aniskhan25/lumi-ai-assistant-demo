"""Dense bf16 GEMMs at decode sizes in Triton (DSV4_DENSE_GEMM=1).

vLLM's ROCm unquantized GEMM (`rocm_unquantized_gemm`, used by every bf16 linear and by
DSV4_BF16_LINEAR's dequantised FP8 linears) runs wvSplitK at <= 5 tokens and otherwise
F.linear, where hipBLASLt takes ~33 us for a [T, 4096] x [1024, 4096].T GEMM at any T up to
128 (test_wo_a.py). At 32 users hipBLASLt GEMMs were ~10 ms of a 54 ms decode step (graph
profile, job 23861348). score_gemm's Triton kernel with a bf16 output, with a tile per
(N, K) and token count from test_dense_gemm.py, takes the shapes and sizes where it won by
more than 5 %; everything else (other shapes, prefill, bias, non-bf16) is the stock call.
Installed before DSV4_BF16_LINEAR, which binds the GEMM function once.
"""

import torch

# (N, K): [(max tokens, (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, warps) or None), ...] for every
# swept token count, from test_dense_gemm.py; None = the stock path won (or was within 5 %).
BF16_CONFIGS: dict[tuple[int, int], list[tuple[int, tuple[int, int, int, int, int] | None]]] = {
    # job 23880540; us per call at 2 / 8 / 32 / 64 tokens, vLLM -> chosen path
    # fused wq_a + wkv (replicated): 13 / 35 / 37 / 37 -> 13 / 15 / 17 / 25
    (1536, 4096): [(2, None), (4, (16, 64, 128, 8, 4)), (6, (16, 64, 128, 8, 4)),
                   (8, (16, 64, 128, 8, 4)), (12, (16, 64, 128, 8, 4)), (16, (16, 64, 128, 8, 4)),
                   (24, (16, 64, 128, 4, 4)), (32, (16, 64, 128, 4, 4)), (48, (16, 64, 128, 4, 4)),
                   (64, (16, 64, 128, 2, 4)), (96, (64, 64, 128, 4, 4)), (128, (64, 64, 256, 2, 4))],
    # wq_b, wo_b: 11 / 16 / 17 / 21 -> 11 / 11 / 12 / 17
    (4096, 1024): [(2, (16, 16, 256, 1, 4)), (4, (16, 16, 256, 1, 4)), (6, (16, 16, 256, 1, 4)),
                   (8, (16, 16, 256, 1, 4)), (12, (16, 16, 256, 1, 4)), (16, (16, 16, 256, 1, 4)),
                   (24, (16, 64, 128, 1, 4)), (32, (16, 64, 128, 1, 4)), (48, (16, 64, 128, 1, 4)),
                   (64, (16, 64, 128, 1, 4)), (96, None), (128, (64, 64, 64, 1, 4))],
    # shared expert w1 + w3: 7 / 33 / 32 / 32 -> 7 / 9 / 10 / 12
    (512, 4096): [(2, None), (4, (16, 16, 256, 8, 4)), (6, (16, 16, 256, 8, 4)),
                  (8, (16, 16, 256, 8, 4)), (12, (16, 16, 256, 8, 4)), (16, (16, 64, 128, 8, 4)),
                  (24, (16, 64, 128, 8, 4)), (32, (32, 16, 128, 8, 4)), (48, (16, 64, 128, 8, 4)),
                  (64, (16, 64, 128, 8, 4)), (96, (16, 64, 128, 4, 4)), (128, (16, 64, 128, 8, 4))],
    # shared expert w2: 8 / 9 / 9 / 8 -> 5 / 5 / 5 / 6
    (4096, 256): [(2, (16, 16, 64, 1, 4)), (4, (16, 16, 64, 1, 4)), (6, (16, 16, 64, 1, 4)),
                  (8, (16, 16, 64, 1, 4)), (12, (16, 16, 64, 1, 4)), (16, (16, 16, 64, 1, 4)),
                  (24, (16, 64, 128, 1, 4)), (32, (32, 16, 128, 1, 4)), (48, (16, 64, 128, 1, 4)),
                  (64, (16, 64, 64, 1, 4)), (96, (16, 64, 64, 1, 4)), (128, (16, 64, 64, 1, 4))],
    # indexer wq_b: 5 / 12 / 11 / 12 -> 5 / 6 / 7 / 8
    (1024, 1024): [(2, None), (4, (16, 16, 128, 8, 4)), (6, (16, 16, 128, 8, 4)),
                   (8, (16, 16, 128, 8, 4)), (12, (16, 16, 128, 8, 4)), (16, (16, 64, 128, 8, 4)),
                   (24, (16, 64, 128, 8, 4)), (32, (16, 64, 128, 8, 4)), (48, (16, 64, 128, 4, 4)),
                   (64, (16, 64, 128, 4, 4)), (96, (16, 64, 128, 2, 4)), (128, (16, 64, 128, 2, 4))],
}


def pick(M: int, N: int, K: int):
    """The decision of the smallest swept token count >= M."""
    for m_max, cfg in BF16_CONFIGS.get((N, K), ()):
        if M <= m_max:
            return cfg
    return None


def gemm_fp8_round(x2d: torch.Tensor, weight: torch.Tensor, group: int):
    """BF16_LINEAR's FP8 activation rounding and the GEMM in one kernel, where this module's
    table picks a Triton tile whose BLOCK_K holds whole 128-column groups; else None (the
    caller rounds separately). Bit-identical to rounding first (test_fp8_fuse.py)."""
    from .score_gemm import score_mm

    M, K = x2d.shape
    if (group != 128 or M > 128 or x2d.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or not x2d.is_contiguous() or weight.stride(1) != 1):
        return None
    cfg = pick(M, weight.shape[0], K)
    if cfg is None or cfg[2] % 128:
        return None
    return score_mm(x2d, weight, cfg, out_dtype=torch.bfloat16, fp8_round=True)


def install() -> None:
    import vllm.model_executor.layers.utils as u

    from .score_gemm import score_mm

    stock = u.rocm_unquantized_gemm
    if getattr(stock, "_lumi", False):
        return

    def rocm_unquantized_gemm(layer, x, weight, bias=None):
        if bias is None and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16:
            K = x.shape[-1]
            M = x.numel() // K
            cfg = pick(M, weight.shape[0], K) if M <= 128 else None
            if cfg is not None and weight.stride(1) == 1 and x.is_contiguous():
                y = score_mm(x.reshape(M, K), weight, cfg, out_dtype=torch.bfloat16)
                return y.view(*x.shape[:-1], weight.shape[0])
        return stock(layer, x, weight, bias)

    rocm_unquantized_gemm._lumi = True
    u.rocm_unquantized_gemm = rocm_unquantized_gemm
