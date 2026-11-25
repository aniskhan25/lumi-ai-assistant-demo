"""score_mm (SCORE_GEMM / GATE_GEMM) at prefill sizes: the compressor score GEMMs and the
router GEMM (bf16 [M, 4096] x [N, 4096].T, fp32 out) for 256-2048 tokens, where score_gemm
had no tile and fell back to torch.mm (hipBLASLt's fp32-MFMA MT64x16x16 kernel: 1.4 ms a call,
13 % of a prefill, profile job 24005447). Tile sweep against torch.mm, weights cycled from HBM,
error against fp64; prints CONFIGS rows to append.

    python test_score_prefill.py
"""

import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plugin"))
sys.path.insert(0, os.path.dirname(__file__))
import test_score_gemm as tsg  # noqa: E402
from lumi_dsv4_debug.score_gemm import score_mm  # noqa: E402

dev = "cuda"
K = 4096
MS = [int(m) for m in os.environ.get("SCORE_MS", "256 512 1024 2048").split()]
torch.manual_seed(0)
print(torch.cuda.get_device_name())
rows = {}
for N in (2048, 1024, 512, 256):
    copies = max(4, -(-64 * 2**20 // (N * K * 2)))
    ws = [torch.randn(N, K, device=dev).bfloat16() * 0.02 for _ in range(copies)]
    for M in MS:
        x = torch.randn(M, K, device=dev).bfloat16()
        ref = x.double() @ ws[0].double().T
        t0 = tsg.timed(lambda w: torch.mm(x, w.T, out_dtype=torch.float32), ws)
        e0 = ((torch.mm(x, ws[0].T, out_dtype=torch.float32).double() - ref).abs().max()
              / ref.abs().max()).item()
        best = (float("inf"), None, None)
        for bm, bn, bk, sk, w, st in itertools.product((32, 64, 128), (32, 64, 128),
                                                      (32, 64, 128), (1, 2, 4), (4, 8), (1, 2)):
            if N % bn or K % (bk * sk):
                continue
            cfg = (bm, bn, bk, sk, w, st, False)
            try:
                t = tsg.timed(lambda ww, c=cfg: score_mm(x, ww, c), ws)
            except Exception:  # noqa: BLE001  out of resources
                torch.cuda.synchronize()
                continue
            if t < best[0]:
                err = ((score_mm(x, ws[0], cfg).double() - ref).abs().max() / ref.abs().max()).item()
                best = (t, cfg, err)
        rows.setdefault(N, []).append((M, best[1]))
        print(f"N={N:5d} M={M:5d}  torch.mm {t0:7.1f} us (err {e0:.0e})  score_mm {best[0]:7.1f} us "
              f"{best[1]} (err {best[2]:.0e})  {t0 / best[0]:.2f}x", flush=True)
print("\nCONFIGS rows to append:")
for N, r in rows.items():
    print(f"  {N}: {r},")
