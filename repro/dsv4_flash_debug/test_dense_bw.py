"""How close are the Triton dense GEMMs (score_mm) to the memory bound at decode sizes? (one GCD)

score_mm runs every bf16 dense GEMM in the recipe at 4-128 tokens (DENSE_GEMM, WO_A,
SCORE_GEMM, GATE_GEMM): 370 calls and 7.1 ms of a 54-56 ms step at 32 users (graph profiles
23901341, 23908712). From the dense_gemm sweep (job 23880540) they read their weights at
0.3-0.5 TB/s; the GCD streams 1.25 TB/s (test_moe_roofline.py). Two things the earlier
sweeps held fixed:
  - grid order: programs ran weight-tile-major, so with BLOCK_M < M each token block of a
    tile re-read it, far apart in time (M_FAST puts a tile's token blocks side by side);
  - num_warps / num_stages (4 / 2).
For each per-rank shape and token count: the served tile, then a sweep of block_m/n/k,
split_k and grid order, then warps x stages for the best few. Weights cycled from HBM
(>= 64 MB of copies); us per call in graph replay; bandwidth on the weight bytes.
The last table weights the gains by calls per decode step.

    python test_dense_bw.py
"""

import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plugin"))
sys.path.insert(0, os.path.dirname(__file__))
import test_score_gemm as tsg  # noqa: E402
from lumi_dsv4_debug import dense_gemm, score_gemm  # noqa: E402
from lumi_dsv4_debug.score_gemm import score_mm  # noqa: E402

dev = "cuda"
# name: (N, K, output dtype, calls per decode step with MTP: 43 layers + 1 MTP layer)
SHAPES = {
    "wqa_wkv": (1536, 4096, torch.bfloat16, 44), "wq_b/wo_b": (4096, 1024, torch.bfloat16, 88),
    "shared w13": (512, 4096, torch.bfloat16, 44), "shared w2": (4096, 256, torch.bfloat16, 44),
    "indexer wq_b": (1024, 1024, torch.bfloat16, 21), "wo_a": (1024, 4096, torch.bfloat16, 44),
    "compressor r4": (2048, 4096, torch.float32, 20), "compressor r128": (1024, 4096, torch.float32, 20),
    "indexer compr": (512, 4096, torch.float32, 20), "router": (256, 4096, torch.float32, 41),
}
MS = [int(m) for m in os.environ.get("DENSE_MS", "8 32 64 128").split()]
torch.manual_seed(0)


def served(name, M, N, K):
    if SHAPES[name][2] == torch.bfloat16 and name != "wo_a":
        return dense_gemm.pick(M, N, K)
    return score_gemm.pick(M, N)


def stage1(M, K):
    bms = sorted({b for b in (16, 32, 64, 128) if b <= max(16, M)})
    for bm, bn, bk, sk, mf in itertools.product(bms, (32, 64, 128), (64, 128, 256),
                                               (1, 2, 4, 8, 16), (False, True)):
        if K % (bk * sk) == 0:
            yield (bm, bn, bk, sk, 4, 2, mf)


print(torch.cuda.get_device_name())
gain = {M: [0.0, 0.0] for M in MS}
for name, (N, K, odt, calls) in SHAPES.items():
    tsg.K = K
    copies = max(4, -(-64 * 2**20 // (N * K * 2)))
    ws = [torch.randn(N, K, device=dev).bfloat16() * 0.02 for _ in range(copies)]
    nbytes = N * K * 2
    for M in MS:
        x = torch.randn(M, K, device=dev).bfloat16()
        ref = x.double() @ ws[0].double().T

        def run(cfg):
            try:
                out = score_mm(x, ws[0], cfg, out_dtype=odt)
                err = ((out.double() - ref).abs().max() / ref.abs().max()).item()
                return tsg.timed(lambda w: score_mm(x, w, cfg, out_dtype=odt), ws), err
            except Exception:  # noqa: BLE001  out of resources
                torch.cuda.synchronize()
                return float("inf"), 9.9

        cur_cfg = served(name, M, N, K)
        t_cur = run(cur_cfg)[0] if cur_cfg else tsg.timed(
            lambda w: torch.nn.functional.linear(x, w).to(odt), ws)
        rows = sorted(((run(c) + (c,)) for c in stage1(M, K) if N % c[1] == 0),
                      key=lambda r: r[0])
        rows2 = sorted((run(c[:4] + (w, s, c[6])) + (c[:4] + (w, s, c[6]),)
                       for c in [r[2] for r in rows[:4]]
                       for w, s in itertools.product((2, 4, 8), (1, 2, 3))), key=lambda r: r[0])
        best = min(rows[0], rows2[0], key=lambda r: r[0])
        gain[M][0] += t_cur * calls
        gain[M][1] += min(t_cur, best[0]) * calls
        mf_best = min((r for r in rows if r[2][6]), default=(float("inf"),), key=lambda r: r[0])
        nf_best = min((r for r in rows if not r[2][6]), default=(float("inf"),), key=lambda r: r[0])
        print(f"{name:<15} N={N:5d} K={K:5d} M={M:4d}  served {t_cur:6.1f} us "
              f"{nbytes / t_cur / 1e6:4.2f} TB/s {cur_cfg}  | best {best[0]:6.1f} us "
              f"{nbytes / best[0] / 1e6:4.2f} TB/s {best[2]} err {best[1]:.0e}  | "
              f"n-major {nf_best[0]:.1f} m-fast {mf_best[0]:.1f}", flush=True)
print("\nper decode step (calls x us), served -> best, ms")
for M, (a, b) in gain.items():
    print(f"  M={M:4d}: {a / 1e3:6.2f} -> {b / 1e3:6.2f}")
