"""Fused FP8 activation rounding (bf16_linear.fp8_round_triton) vs vLLM's quantiser + dequant.

DSV4_BF16_LINEAR rounds every dense linear's input to FP8 per 128-group before the bf16 GEMM,
which the model needs (job 23535685). As served, that is vLLM's per_token_group_quant_fp8
kernel plus a separate dequantise elementwise: ~6 ms per decode step (job 23547966). On one
GCD, for V4-Flash's linear input widths and decode/prefill row counts:

  - bit-exactness of the fused kernel against the served path (random, outlier, tiny and
    all-zero groups; contiguous and strided rows)
  - time per call of both, in CUDA-graph replay

    python test_fp8_round.py
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent / "plugin"))
from lumi_dsv4_debug.bf16_linear import fp8_round_torch, fp8_round_triton  # noqa: E402

dev = torch.device("cuda")
torch.manual_seed(0)
GROUP = 128


def graph_us(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(5):
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1000 / reps)
    return best


def inputs(m, k, kind):
    x = torch.randn(m, k, device=dev)
    if kind == "outliers":
        x[:, torch.randint(0, k, (max(1, k // 256),), device=dev)] *= 500
    elif kind == "tiny":
        x *= 1e-6
    elif kind == "zeros":
        x[:, :GROUP] = 0
    elif kind == "wide":
        x = torch.randn(m, k + 512, device=dev)[:, 256:256 + k]  # strided rows
    return x.to(torch.bfloat16)


print(f"device {torch.cuda.get_device_properties(0).gcnArchName}")
print("bit-exactness vs vLLM quantiser + dequant (FAIL if any element differs)")
bad = 0
for k in (4096, 1024, 256):
    for m in (1, 4, 32, 300):
        for kind in ("normal", "outliers", "tiny", "zeros", "wide"):
            x = inputs(m, k, kind)
            want = fp8_round_torch(x.contiguous(), GROUP)
            got = fp8_round_triton(x, GROUP)
            diff = (got.float() - want.float()).abs()
            n = int((got != want).sum()) - int((torch.isnan(got) & torch.isnan(want)).sum())
            if n:
                bad += 1
                print(f"  FAIL K={k} M={m} {kind}: {n} of {got.numel()} differ, max abs diff "
                      f"{diff.nan_to_num(0).max().item():.3g}")
print(f"  {'all identical' if not bad else f'{bad} cases differ'}")

print("\ntime per call (us, graph replay): served path (2 kernels) vs fused")
total = {"torch": 0.0, "triton": 0.0}
for k in (4096, 1024):
    for m in (1, 4, 32, 64):
        x = inputs(m, k, "normal")
        t_torch = graph_us(lambda: fp8_round_torch(x, GROUP))
        t_triton = graph_us(lambda: fp8_round_triton(x, GROUP))
        print(f"  K={k:<5} M={m:<3} served {t_torch:7.1f}   fused {t_triton:7.1f}   "
              f"{t_torch / t_triton:4.1f}x")
