"""Compressor score GEMMs: hipBLASLt (torch.mm, fp32 out) vs the Triton kernel, on one GCD.

V4-Flash shapes: bf16 [M, 4096] @ [N, 4096].T -> fp32, N = 2048 / 1024 / 512, 20 calls of
each per decode step. Each timed call reads a different weight copy (>= 64 MB in rotation),
since in serving the weights are not in the 8 MB L2. Times are GPU us per call from graph
replays. Errors are against an fp64 product. Prints the best tile per (N, M) for
score_gemm.CONFIGS and the per-step total at each batch.
"""

import itertools
import os
import sys

import torch
from triton.runtime.errors import OutOfResources

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plugin"))
from lumi_dsv4_debug import score_gemm  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
K = 4096
NS = (2048, 1024, 512)
MS = (1, 4, 8, 16, 24, 32, 48, 64, 96, 128)
ITERS = 24


def timed(fn, ws):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for w in ws[:2]:
            fn(w)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(ITERS):
            fn(ws[i % len(ws)])
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(5):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / (5 * ITERS)


def configs(M):
    bms = [b for b in (16, 32, 64, 128) if b <= max(16, M)]
    for bm, bn, bk, sk in itertools.product(bms, (16, 32, 64, 128), (64, 128, 256), (1, 2, 4, 8, 16)):
        if K % (bk * sk) == 0:
            yield (bm, bn, bk, sk, 4)


def main():
    print(torch.cuda.get_device_name(), torch.__version__)
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention  # what install() patches

    assert hasattr(DeepseekV4Attention, "attn_gemm_parallel_execute")
    best = {}
    stock_us = {}
    for N in NS:
        copies = max(4, -(-64 * 2**20 // (N * K * 2)))
        ws = [torch.randn(N, K, device=dev).bfloat16() * 0.02 for _ in range(copies)]
        for M in MS:
            x = torch.randn(M, K, device=dev).bfloat16()
            ref = (x.double() @ ws[0].double().T)
            scale = ref.abs().max().item()
            stock = torch.mm(x, ws[0].T, out_dtype=torch.float32)
            err_stock = (stock.double() - ref).abs().max().item() / scale
            t_stock = timed(lambda w: torch.mm(x, w.T, out_dtype=torch.float32), ws)
            stock_us[N, M] = t_stock
            rows = []
            for cfg in configs(M):
                for atomic in ((False, True) if cfg[3] > 1 else (False,)):
                    try:  # compiles here, outside capture; big tiles overflow LDS
                        out = score_gemm.score_mm(x, ws[0], cfg, atomic)
                    except OutOfResources:
                        continue
                    err = (out.double() - ref).abs().max().item() / scale
                    t = timed(lambda w: score_gemm.score_mm(x, w, cfg, atomic), ws)
                    rows.append((t, cfg, atomic, err))
            rows.sort(key=lambda r: r[0])
            det = next(r for r in rows if not r[2])
            ato = next((r for r in rows if r[2]), None)
            best[N, M] = det
            worst_err = max(r[3] for r in rows)
            print(f"N={N:5d} M={M:4d}  torch.mm {t_stock:7.1f} us (err {err_stock:.1e})  "
                  f"triton {det[0]:6.1f} us {det[1]} (err {det[3]:.1e})  "
                  f"atomic {ato[0] if ato else float('nan'):6.1f} us {ato[1] if ato else ''}  "
                  f"worst err over tiles {worst_err:.1e}", flush=True)
            if worst_err > 1e-4:
                print("  FAIL: a tile is off by more than fp32 accumulation error")
        del ws
        torch.cuda.empty_cache()
    print("\nper decode step (20 calls of each N), us: torch.mm -> triton")
    for M in MS:
        a = sum(20 * stock_us[N, M] for N in NS)
        b = sum(20 * best[N, M][0] for N in NS)
        print(f"  M={M:4d}: {a:8.0f} -> {b:6.0f}  ({a / b:.1f}x)")
    print("\nCONFIGS = {")
    for N in NS:
        print(f"    {N}: [" + ", ".join(f"({M}, {best[N, M][1]})" for M in MS) + "],")
    print("}")


if __name__ == "__main__":
    main()
