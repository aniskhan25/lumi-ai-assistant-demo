"""lumi_dsv41/mxfp4_gemm.py on one GCD: what an MXFP4 expert GEMM costs on gfx90a, by mode
(0 tl.dot_scaled as matmul_ogs, 1 explicit integer upcast + tl.dot, 2 bf16 weights, no
upcast, 3 the FP4 bytes loaded only). 48 experts of V4.1's gate/up shape ([5120 -> 4608]
per rank under EP), M tokens per expert, weights cold (566 MB of FP4 > L2); best tile
per mode, outputs checked against an fp32 dequantize-and-matmul reference.

    MOE_TEST=test_mxfp4_gemm.py sbatch repro/deepseek_v41_gfx90a/run_moe_matmul_test.sh
"""

import itertools
import os

import torch

from lumi_dsv41.mxfp4_gemm import dequant, grouped_gemm

torch.manual_seed(0)
dev = "cuda"
E, K, N = 48, 5120, 4608
w = torch.randint(0, 256, (E, K // 2, N), dtype=torch.uint8, device=dev)
s = torch.randint(118, 125, (E, N, K // 32), dtype=torch.uint8, device=dev)
wf = dequant(w, s)  # [E, K, N] fp32
w16 = wf.bfloat16()  # exact: e2m1 x 2^k is representable in bf16
print(torch.cuda.get_device_name(), f"{E} experts [{K} -> {N}], FP4 weights "
      f"{w.numel() / 1e6:.0f} MB + scales {s.numel() / 1e6:.0f} MB")


def graph_us(fn, reps=5):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(4):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / (4 * reps)


bad = []
NAMES = {0: "dot_scaled", 1: "int upcast", 2: "bf16 weights", 3: "FP4 loads only"}
for M in (int(m) for m in os.environ.get("MX_M", "16 32 64").split()):
    x = torch.randn(E, M, K, device=dev).bfloat16()
    ref = torch.bmm(x.float(), wf)
    best = {}
    for mode in (0, 1, 2, 3):
        times = {}
        for bm, bn, bk, nw, nk in itertools.product(
                (16, 32, 64), (64, 128, 256), (128, 256), (4, 8), (16, 32)):
            if bm > max(16, M):
                continue
            kw = dict(block_m=bm, block_n=bn, block_k=bk, num_warps=nw, matrix_instr_nonkdim=nk)
            try:
                y = grouped_gemm(x, w, s, mode, w16=w16, **kw)
            except Exception as e:  # out of resources / unsupported
                continue
            if mode != 3:
                err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
                if not err < 1e-2:
                    bad.append((M, mode, kw, err))
                    continue
            times[(bm, bn, bk, nw, nk)] = graph_us(lambda: grouped_gemm(x, w, s, mode, w16=w16, **kw))
        if not times:
            print(f"  M={M} mode {mode} ({NAMES[mode]}): no valid config", flush=True)
            continue
        c = min(times, key=times.get)
        best[mode] = times[c]
        gb = (w.numel() + s.numel()) * (4 if mode == 2 else 1) / 1e9
        print(f"  M={M:3d} {NAMES[mode]:<15} best {times[c]:8.1f} us (bm, bn, bk, warps, nonkdim)={c}"
              f"  weight bytes {gb / times[c] * 1e6:6.0f} GB/s", flush=True)
    if 0 in best and 1 in best:
        print(f"  M={M:3d} int upcast vs dot_scaled: {best[0] / best[1]:.2f}x; dot_scaled vs FP4 "
              f"floor: {best[0] / best[3]:.2f}x" if 3 in best else "", flush=True)
print("FAILED:", bad[:5] if bad else "none")
