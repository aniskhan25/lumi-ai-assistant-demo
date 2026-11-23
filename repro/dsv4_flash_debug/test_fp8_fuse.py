"""BF16_LINEAR's FP8 activation rounding fused into the dense Triton GEMM (DSV4_FP8_FUSE): is
it bit-identical to rounding first (bf16_linear.fp8_round_triton, then score_mm with the same
tile), and how much does dropping the separate launch save? (one GCD, graph replay, weights
cycled from HBM)

    python test_fp8_fuse.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plugin"))
sys.path.insert(0, os.path.dirname(__file__))
import test_score_gemm as tsg  # noqa: E402
from lumi_dsv4_debug import dense_gemm  # noqa: E402
from lumi_dsv4_debug.bf16_linear import fp8_round_triton  # noqa: E402
from lumi_dsv4_debug.score_gemm import score_mm  # noqa: E402

dev = "cuda"
torch.manual_seed(0)
print(torch.cuda.get_device_name())
bad, saved = [], {}
for (N, K), rows in dense_gemm.BF16_CONFIGS.items():
    copies = max(4, -(-64 * 2**20 // (N * K * 2)))
    ws = [torch.randn(N, K, device=dev).bfloat16() * 0.02 for _ in range(copies)]
    for M, cfg in rows:
        x = torch.randn(M, K, device=dev).bfloat16()
        x[:, ::97] *= 40  # outliers: groups with very different scales
        if dense_gemm.gemm_fp8_round(x, ws[0], 128) is None:
            print(f"N={N:5d} K={K:5d} M={M:4d}  not fused (cfg {cfg})")
            continue
        want = score_mm(fp8_round_triton(x, 128), ws[0], cfg, out_dtype=torch.bfloat16)
        got = dense_gemm.gemm_fp8_round(x, ws[0], 128)
        diff = (got.float() - want.float()).abs().max().item()
        if diff != 0:
            bad.append((N, K, M, diff))
        t_sep = tsg.timed(lambda w: score_mm(fp8_round_triton(x, 128), w, cfg, out_dtype=torch.bfloat16), ws)
        t_fus = tsg.timed(lambda w: dense_gemm.gemm_fp8_round(x, w, 128), ws)
        saved.setdefault(M, []).append(t_sep - t_fus)
        print(f"N={N:5d} K={K:5d} M={M:4d}  separate {t_sep:6.1f} us  fused {t_fus:6.1f} us  "
              f"max |diff| {diff:.1e}  cfg {cfg}", flush=True)
print("\nmean us saved per call:", {m: round(sum(v) / len(v), 2) for m, v in saved.items()})
print("NOT BIT-IDENTICAL:", bad or "none")
