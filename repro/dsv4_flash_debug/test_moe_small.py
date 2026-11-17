"""MoE routing at decode sizes, on one GCD: plugin moe_small.py.

1. pack_bitmatrix: vLLM's launch (BLOCK_SIZE_M = 512) vs row blocks of 16-512, on V4's 256
   experts and top-6 ids, n_rows 1-8192 with -1 (non-local) slots: bit-identical output,
   GPU us per call from graph replays.
2. Router GEMM shapes for score_gemm (N = 256 experts, K = 4096): the tile sweep and the
   error against fp64, next to bf16 F.linear + cast (what GateLinear does on ROCm).
"""

import os
import sys

import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plugin"))
from lumi_dsv4_debug import moe_small  # noqa: E402

from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (  # noqa: E402
    pack_bitmatrix)

dev = "cuda"
E, TOPK = 256, 6
torch.manual_seed(0)
fail = []


def graph_us(fn, iters=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(5):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / (5 * iters)


print("1. pack_bitmatrix, us per call (graph replay)")
BMS = (16, 32, 64, 128, 256, 512)
print(f"{'n_rows':>7} {'vLLM':>7} " + " ".join(f"{'bm' + str(b):>7}" for b in BMS) + "   plugin pick")
for n_rows in (1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 8192):
    ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(n_rows)]).to(torch.int16)
    ids[torch.rand(n_rows, TOPK, device=dev) < 0.05] = -1
    bm_cols = triton.cdiv(E, 32)
    ref = torch.zeros(n_rows, bm_cols, dtype=torch.uint32, device=dev)
    pack_bitmatrix[(triton.cdiv(n_rows, 512),)](ref, ids, n_rows, bm_cols, TOPK,
                                                BLOCK_SIZE_M=512, BLOCK_SIZE_K=32)
    times = {}
    for bm in (512,) + BMS:
        out = torch.zeros_like(ref)

        def run(bm=bm, out=out):
            pack_bitmatrix[(triton.cdiv(n_rows, bm),)](out, ids, n_rows, bm_cols, TOPK,
                                                       BLOCK_SIZE_M=bm, BLOCK_SIZE_K=32)
        run()
        if not torch.equal(out.view(torch.int32), ref.view(torch.int32)):
            fail.append(f"pack bm={bm} n_rows={n_rows}")
        times[bm] = graph_us(run)
    out = torch.zeros_like(ref)
    moe_small._PackBitmatrix(pack_bitmatrix)[None](out, ids, n_rows, bm_cols, TOPK,
                                                    BLOCK_SIZE_M=512, BLOCK_SIZE_K=32)
    if not torch.equal(out.view(torch.int32), ref.view(torch.int32)):
        fail.append(f"pack plugin n_rows={n_rows}")
    print(f"{n_rows:7d} {times[512]:7.1f} " + " ".join(f"{times[b]:7.1f}" for b in BMS)
          + f"   bm{moe_small.pack_block_m(n_rows)}")

print("\n2. router GEMM [M, 4096] x [256, 4096].T -> fp32")
import test_score_gemm as tsg  # noqa: E402

tsg.NS = (256,)
tsg.MS = (1, 2, 4, 8, 16, 32, 64, 128)
w = torch.randn(256, 4096, device=dev).bfloat16() * 0.02
for M in (1, 2, 16, 64):
    x = torch.randn(M, 4096, device=dev).bfloat16()
    ref = x.double() @ w.double().T
    scale = ref.abs().max().item()
    lin = torch.nn.functional.linear(x, w).float()
    print(f"  M={M:3d}: bf16 F.linear + cast err {(lin.double() - ref).abs().max().item() / scale:.1e}, "
          f"{graph_us(lambda: torch.nn.functional.linear(x, w).float()):.1f} us")
tsg.main()

print("\nFAILED:", fail or "none")
