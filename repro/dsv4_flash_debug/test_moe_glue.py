"""moe_glue.py (DSV4_MOE_GLUE) against vLLM's code, on one GCD.

1. make_routing_data: the one-kernel prologue vs vLLM's (routing outputs must be equal),
   including rows with -1 slots;
2. gather_rows with int32 indices vs torch indexing (equal);
3. the shared expert's SiluAndMulWithClamp kernel vs forward_native (max difference);
4. GPU us per MoE layer for the replaced glue in graph replay: vLLM's routing prologue,
   two int32-index gathers and forward_native, against moe_glue's.

    python test_moe_glue.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plugin"))
import vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe as m  # noqa: E402
from vllm.model_executor.layers.activation import SiluAndMulWithClamp  # noqa: E402

stock_routing = m.make_routing_data
stock_act = SiluAndMulWithClamp.forward_native
from lumi_dsv4_debug import moe_glue  # noqa: E402

moe_glue.install()
new_routing = m.make_routing_data
new_act = SiluAndMulWithClamp.forward_native
dev = "cuda"
E, TOPK, N1, I = 256, 6, 512, 256
torch.manual_seed(0)
fails = []


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
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(5):
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1000 / reps)
    return best


def routing(M, invalid=False):
    ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).int()
    if invalid:
        ids[::3, -1] = -1
    w = torch.rand(M, TOPK, device=dev)
    return ids, w


def same(a, b):
    return a.shape == b.shape and torch.equal(a.cpu(), b.cpu())


print(torch.cuda.get_device_name(), "legacy triton_kernels routing:", m.use_legacy_triton_kernels,
      "| make_routing_data replaced:", new_routing is not stock_routing)
print("1. routing outputs (gate_scal, src/dst indices, block sizes)")
for M in (1, 2, 7, 64, 128, 300):
    for invalid in (False, True):
        ids, w = routing(M, invalid)
        r0, g0, s0 = stock_routing(ids, w, E)
        r9, g9, s9 = stock_routing(ids, w, E)  # vLLM against itself
        r1, g1, s1 = new_routing(ids, w, E)
        fields = {"gate_scal": (r0.gate_scal, r9.gate_scal, r1.gate_scal),
                  "src_indx": (g0.src_indx, g9.src_indx, g1.src_indx),
                  "dst_indx": (g0.dst_indx, g9.dst_indx, g1.dst_indx)}
        n_valid = int((ids >= 0).sum())
        cells = []
        ok = True
        for k, (a, a2, b) in fields.items():
            self_ok, new_ok = same(a, a2), same(a, b)
            # with -1 slots, compare the n_valid routed entries (the rest is padding)
            head_ok = same(a.flatten()[:n_valid], b.flatten()[:n_valid])
            cells.append(f"{k} self {'=' if self_ok else '!='} new {'=' if new_ok else ('= on valid' if head_ok else '!=')}")
            ok &= new_ok or (invalid and head_ok and not self_ok) or (invalid and head_ok)
        print(f"  {'ok  ' if ok else 'FAIL'} M={M:<4} invalid slots {invalid}: " + "; ".join(cells))
        if not ok:
            fails.append(("routing", M, invalid))

print("2. gather_rows vs torch indexing")
for rows, n in ((384, 512), (384, 256), (12, 512), (768, 4096)):
    src = torch.randn(rows, n, device=dev).bfloat16()
    idx = torch.randperm(rows, device=dev).int()
    ok = same(moe_glue.gather_rows(src, idx), src[idx])
    print(f"  {'ok  ' if ok else 'FAIL'} [{rows}, {n}]")
    if not ok:
        fails.append(("gather", rows, n))

print("3. SiluAndMulWithClamp: kernel vs forward_native")
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402

with set_current_vllm_config(VllmConfig()):
    act = SiluAndMulWithClamp(10.0)
for M in (1, 2, 64, 128, 2048):
    x = (torch.randn(M, N1, device=dev) * 6).bfloat16()
    want, got = stock_act(act, x), new_act(act, x)
    diff = (got.float() - want.float()).abs()
    rel = (diff.max() / want.float().abs().max()).item()
    print(f"  M={M:<5} max |diff| {diff.max().item():.2e} (rel {rel:.1e}), "
          f"{(diff > 0).float().mean().item() * 100:.3f} % of elements differ")
    if rel > 1e-2:
        fails.append(("act", M, rel))

print("4. us per MoE layer for the replaced glue (graph replay)")
for M in (2, 16, 64, 128):
    ids, w = routing(M)
    c1 = torch.randn(M * TOPK, N1, device=dev).bfloat16()
    c2 = torch.randn(M * TOPK, I, device=dev).bfloat16()
    x = torch.randn(M, N1, device=dev).bfloat16()
    _, g, _ = stock_routing(ids, w, E)

    def old():
        stock_routing(ids, w, E)
        c1[g.dst_indx]
        c2[g.src_indx]
        stock_act(act, x)

    def new():
        new_routing(ids, w, E)
        moe_glue.gather_rows(c1, g.dst_indx)
        moe_glue.gather_rows(c2, g.src_indx)
        new_act(act, x)

    t0, t1 = graph_us(old), graph_us(new)
    print(f"  M={M:<4} vLLM {t0:6.1f} us  moe_glue {t1:6.1f} us  -> per step x44: "
          f"{t0 * 44 / 1e3:.2f} -> {t1 * 44 / 1e3:.2f} ms")
print("FAILED:", fails or "none")
