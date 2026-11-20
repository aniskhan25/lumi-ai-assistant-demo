"""V4-Flash dense bf16 GEMMs (BF16_LINEAR's dequantised linears and the shared expert) at
decode sizes on one GCD: vLLM's ROCm dispatch (wvSplitK at <= 5 tokens, else hipBLASLt via
F.linear: ~33 us flat for skinny shapes, test_wo_a.py) against score_gemm's Triton kernel with
a bf16 output, tile-swept per shape and token count. Weights cycled from HBM (>= 64 MB).

Per-rank shapes at TP=8, (N, K): fused wq_a+wkv (replicated) 1536 x 4096; wq_b and wo_b
4096 x 1024; shared expert w1+w3 512 x 4096, w2 4096 x 256; indexer wq_b 1024 x 1024;
wo_a 1024 x 4096. Prints a BF16_CONFIGS table with the entries where Triton wins by > 5 %.

    sbatch repro/dsv4_flash_debug/run_dense_gemm_test.sh
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plugin"))
sys.path.insert(0, os.path.dirname(__file__))
import test_score_gemm as tsg  # noqa: E402
from lumi_dsv4_debug.score_gemm import score_mm  # noqa: E402
from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl  # noqa: E402

dev = "cuda"
SHAPES = {"wqa_wkv": (1536, 4096), "wq_b/wo_b": (4096, 1024), "shared w13": (512, 4096),
          "shared w2": (4096, 256), "indexer wq_b": (1024, 1024), "wo_a": (1024, 4096)}
MS = (2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
torch.manual_seed(0)
print(torch.cuda.get_device_name())
table = {}
total = {}
for name, (N, K) in SHAPES.items():
    tsg.K = K
    copies = max(4, -(-64 * 2**20 // (N * K * 2)))
    ws = [torch.randn(N, K, device=dev).bfloat16() * 0.02 for _ in range(copies)]
    rows = []
    for M in MS:
        x = torch.randn(M, K, device=dev).bfloat16()
        ref = x.double() @ ws[0].double().T
        t_v = tsg.timed(lambda w: rocm_unquantized_gemm_impl(x, w), ws)
        sweep = []
        for cfg in tsg.configs(M):
            if N % cfg[1]:
                continue
            try:
                sweep.append((tsg.timed(lambda w, c=cfg: score_mm(x, w, c, out_dtype=torch.bfloat16), ws), cfg))
            except Exception:  # out of resources
                torch.cuda.synchronize()
        t_t, cfg = min(sweep)
        err = ((score_mm(x, ws[0], cfg, out_dtype=torch.bfloat16).double() - ref).abs().max()
               / ref.abs().max()).item()
        win = t_t < 0.95 * t_v
        if win:
            rows.append((M, cfg))
        total.setdefault(M, [0.0, 0.0])
        total[M][0] += t_v
        total[M][1] += min(t_v, t_t)
        print(f"{name:<13} N={N:5d} K={K:5d} M={M:4d}  vllm {t_v:6.1f} us  triton {t_t:6.1f} us "
              f"{cfg} err {err:.0e}{'  <-' if win else ''}", flush=True)
    table[(N, K)] = rows
print("\nper token count, sum over the 6 shapes (one call each): vllm -> best, us")
for M, (a, b) in total.items():
    print(f"  M={M:4d}: {a:7.1f} -> {b:7.1f}")
print("\nBF16_CONFIGS = {")
for (N, K), rows in table.items():
    print(f"    ({N}, {K}): {rows},")
print("}")
