"""V4.1's MoE router GEMM on one GCD: bf16 [M, 5120] x [384, 5120].T -> fp32 logits.

GateLinear has no ROCm tier in 0.26: it runs bf16 F.linear (hipBLASLt) and casts, so the
logits are rounded to bf16 (V4: rel err 2e-3, 40 us per call; V4's GATE_GEMM=1 routes it
through score_gemm.score_mm, bf16 MFMA with fp32 accumulation). V4's score_mm has tiles for
N = 256 / 512 / 1024 / 2048 at K = 4096 only, so for V4.1 it falls back to torch.mm. This
reuses ../dsv4_flash_debug/test_score_gemm.py's sweep (weights cycled through >= 64 MB so
they come from HBM, graph replay, error vs fp64) for V4.1's shape, after timing what
GateLinear does now.

    sbatch repro/deepseek_v41_gfx90a/run_router_gemm_test.sh
"""

import os
import sys

import torch

DEBUG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dsv4_flash_debug")
sys.path.insert(0, DEBUG)
import test_score_gemm as tsg  # noqa: E402

K, N = 5120, 384
tsg.K, tsg.NS = K, (N,)
tsg.MS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)

dev = "cuda"
torch.manual_seed(0)
ws = [torch.randn(N, K, device=dev).bfloat16() * 0.02 for _ in range(17)]  # ~67 MB
print(f"router GEMM [M, {K}] x [{N}, {K}].T -> fp32; GateLinear today = bf16 F.linear + cast")
for M in tsg.MS:
    x = torch.randn(M, K, device=dev).bfloat16()
    ref = x.double() @ ws[0].double().T
    scale = ref.abs().max().item()
    lin = torch.nn.functional.linear(x, ws[0]).float()
    t = tsg.timed(lambda w: torch.nn.functional.linear(x, w).float(), ws)
    print(f"  M={M:4d}: F.linear + cast {t:6.1f} us, err {(lin.double() - ref).abs().max().item() / scale:.1e}")
del ws
torch.cuda.empty_cache()
tsg.main()
