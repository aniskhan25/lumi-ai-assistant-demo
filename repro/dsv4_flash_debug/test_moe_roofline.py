"""How far are V4-Flash's MXFP4 expert GEMMs from the memory bound at decode sizes? (one GCD)

At 32-64 users (64-128 tokens with MTP) the two matmul_ogs kernels are ~30 % of a decode
step (graph profiles 23901341/2). Per rank at TP=8: 256 experts, top-6, gate/up 4096 -> 512
and down 256 -> 4096, MXFP4 (4 bits + one ue8m0 scale per 32). gfx90a has no FP4 math, so
the weights are converted to bf16 in software before the bf16 matrix cores.

For each token count and both GEMMs:
  1. the served flags (moe_small's tables) and the library default: us per call and the
     bandwidth on the weights of the experts actually routed to (distinct experts x bytes);
  2. the same GEMM with bf16 weights (4x the bytes, no conversion): if it moves bytes much
     faster, the FP4 path is limited by the conversion, not by memory;
  3. a sweep of block_m / block_n / block_k / split_k, then num_warps / num_stages for the
     best few (overridden on the returned flags; the library fixes them), with the max error
     against the served output.
A plain read of a 512 MiB buffer gives the GCD's practical HBM bandwidth.

    python test_moe_roofline.py
"""

import dataclasses
import importlib
import itertools
import os
import pathlib
import sys

import torch

import vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe as g
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import _swizzle_mxfp4

sys.path.insert(0, str(pathlib.Path(__file__).parent / "plugin"))
from lumi_dsv4_debug import moe_small  # noqa: E402

mo = importlib.import_module(g.matmul_ogs.__module__)
of = importlib.import_module(mo.__name__.rsplit(".", 1)[0] + ".matmul_ogs_details.opt_flags")
FlexCtx, PrecisionConfig, matmul_ogs = mo.FlexCtx, mo.PrecisionConfig, mo.matmul_ogs

dev = "cuda"
E, TOPK, H, I = 256, 6, 4096, 256
N1 = 2 * I
SETS = 8
TOKENS = [int(t) for t in os.environ.get("MOE_TOKENS", "2 8 32 64 128").split()]
torch.manual_seed(0)

OVERRIDE = {}
_amd = of.make_default_opt_flags_amd


def _amd_flags(*a, **k):
    ret = _amd(*a, **k)
    return dataclasses.replace(ret, **OVERRIDE) if OVERRIDE else ret


of.make_default_opt_flags_amd = _amd_flags  # looked up at call time by make_opt_flags
LAST = {}
_make = of.make_opt_flags


def _record(*a, **k):
    ret = _make(*a, **k)
    LAST["flags"] = ret
    return ret


of.make_opt_flags = _record
for name, mod in list(sys.modules.items()):  # matmul_ogs imported the name directly
    if "triton_kernels" in name and mod is not of and mod.__dict__.get("make_opt_flags") is _make:
        mod.make_opt_flags = _record


def mxfp4(n, k):
    w = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev)
    s = torch.randint(118, 124, (E, n, k // 32), dtype=torch.uint8, device=dev)
    w, flex, scale = _swizzle_mxfp4(w, s)
    return w, PrecisionConfig(weight_scale=scale, flex_ctx=FlexCtx(rhs_data=flex))


W = {"gate_up": mxfp4(N1, H), "down": mxfp4(H, I)}
WB = {"gate_up": (torch.randn(E, H, N1, device=dev).bfloat16() * 0.02, PrecisionConfig()),
      "down": (torch.randn(E, I, H, device=dev).bfloat16() * 0.02, PrecisionConfig())}
FP4_BYTES = {"gate_up": N1 * H // 2 + N1 * H // 32, "down": H * I // 2 + H * I // 32}
BF16_BYTES = {"gate_up": N1 * H * 2, "down": H * I * 2}


def graph_us(fns, reps=6):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(reps):
            for f in fns:
                f()
    gr.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(5):
        a.record()
        gr.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1000 / (reps * len(fns)))
    return best


def with_flags(cons, over, fn):
    of.reset_opt_flags_constraints()
    OVERRIDE.clear()
    if cons:
        of.update_opt_flags_constraints(cons)
    OVERRIDE.update(over or {})
    try:
        return fn()
    finally:
        of.reset_opt_flags_constraints()
        OVERRIDE.clear()


def calls(M, which, weights):
    """One closure per routing set (different experts each), and the mean distinct experts."""
    w, pc = weights
    out, distinct = [], 0
    for _ in range(SETS):
        ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).int()
        distinct += ids.unique().numel() / SETS
        wts = torch.rand(M, TOPK, device=dev)
        rd, gi, si = g.make_routing_data(ids, wts, E)
        if which == "gate_up":
            x = torch.randn(M, H, device=dev).bfloat16()
            y = torch.empty(1, M * TOPK, N1, device=dev, dtype=torch.bfloat16)
            out.append((lambda x=x, y=y, rd=rd, gi=gi:
                        matmul_ogs(x, w, None, rd, gather_indx=gi, precision_config=pc, y=y), y))
        else:
            rd.n_expts_act = 1
            x = torch.randn(M * TOPK, I, device=dev).bfloat16()[gi.src_indx]
            y = torch.empty(1, M * TOPK, H, device=dev, dtype=torch.bfloat16)
            out.append((lambda x=x, y=y, rd=rd, si=si:
                        matmul_ogs(x, w, None, rd, scatter_indx=si, precision_config=pc,
                                   gammas=rd.gate_scal, y=y), y))
    return out, distinct


def served(M, which):
    table = moe_small.GATE_UP if which == "gate_up" else moe_small.DOWN
    return moe_small._pick(table, M)  # {"block_n", "split_k"} or None


buf = torch.empty(512 << 20, dtype=torch.uint8, device=dev).view(torch.float32)
hbm = (512 << 20) / graph_us([lambda: buf.sum()]) / 1e6
print(f"{torch.cuda.get_device_name()}  practical HBM read: {hbm:.2f} TB/s (sum over 512 MiB)")

CONS = [dict(block_m=bm, block_n=bn, block_k=bk, split_k=sk) for bm, bn, bk, sk in
        itertools.product([int(b) for b in os.environ.get("MOE_BM", "16 32").split()],
                          (64, 128, 256, 512), (128, 256), (1, 2, 4))]
OVERS = [dict(num_warps=w, num_stages=s) for w, s in itertools.product((2, 4, 8), (1, 2, 3))]

for which in ("gate_up", "down"):
    print(f"\n{which}: us per call, TB/s on the routed experts' weights")
    for M in TOKENS:
        cs, distinct = calls(M, which, W[which])
        fns = [c[0] for c in cs]
        scons = served(M, which)
        with_flags(scons, None, lambda: [f() for f in fns])
        sflags = LAST.get("flags") or of.OptFlags(*([0] * 15))
        ref = [c[1].clone() for c in cs]
        bw = lambda us: distinct * FP4_BYTES[which] / us / 1e6  # noqa: E731
        t_served = with_flags(scons, None, lambda: graph_us(fns))
        t_default = with_flags(None, None, lambda: graph_us(fns))
        bcs, _ = calls(M, which, WB[which])
        t_bf16 = with_flags(None, None, lambda: graph_us([c[0] for c in bcs]))
        bf16_bw = distinct * BF16_BYTES[which] / t_bf16 / 1e6

        def run(cons, over):
            try:
                with_flags(cons, over, lambda: [f() for f in fns])
                err = max(((c[1].float() - r.float()).abs().max() / r.float().abs().max()).item()
                          for c, r in zip(cs, ref))
                if err > 2e-2:
                    return float("inf"), f"err {err:.1e}"
                return with_flags(cons, over, lambda: graph_us(fns)), f"{err:.0e}"
            except Exception as e:  # noqa: BLE001  unsupported combination
                torch.cuda.synchronize()
                return float("inf"), repr(e)[:50]

        rows = sorted(((*run(c, None), c, None) for c in CONS), key=lambda r: r[0])
        rows2 = sorted(((*run(c, o), c, o) for c in [r[2] for r in rows[:3]] for o in OVERS),
                       key=lambda r: r[0])
        best = min(rows[0], rows2[0], key=lambda r: r[0])
        print(f"  M={M:<4} experts {distinct:5.1f}  served {t_served:6.1f} us {bw(t_served):.2f} TB/s"
              f" (bm {sflags.block_m} bn {sflags.block_n} bk {sflags.block_k} sk {sflags.split_k}"
              f" w {sflags.num_warps} st {sflags.num_stages})  default {t_default:6.1f}"
              f"  | bf16 weights {t_bf16:6.1f} us {bf16_bw:.2f} TB/s"
              f"  | best {best[0]:6.1f} us {bw(best[0]):.2f} TB/s {best[2]} {best[3] or ''} err {best[1]}",
              flush=True)
        print("         next: " + "  ".join(
            f"{r[0]:.0f} {tuple(r[2].values())}{tuple(r[3].values()) if r[3] else ''}"
            for r in sorted(rows[1:4] + rows2[1:4], key=lambda r: r[0])[:5]))
