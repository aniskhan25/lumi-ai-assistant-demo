"""V4-Flash MXFP4 expert GEMMs (triton_kernels matmul_ogs) at decode sizes, on one GCD.

Per-rank shapes at TP=8 without EP: 256 experts, top-6; gate/up [4096 -> 512] (2 x 2048 / 8),
down [256 -> 4096]. Called as vLLM's UnfusedOAITritonExperts does (gather into gate/up,
scatter out of down with n_expts_act = 1). For each batch, the default opt flags vs forced
block_n / split_k (opt_flags constraints), GPU us per call from graph replays that cycle
through 8 routings with different experts (the weights do not stay in the 8 MB L2), and
the max error against the default flags' output.
"""

import importlib
import itertools

import torch

import vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe as g
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import _swizzle_mxfp4

# The triton_kernels package vLLM imported (its bundled copy, aliased): the opt-flag
# constraints are a module global, so they must be set on that same module.
mo = importlib.import_module(g.matmul_ogs.__module__)
of = importlib.import_module(mo.__name__.rsplit(".", 1)[0] + ".matmul_ogs_details.opt_flags")
FlexCtx, PrecisionConfig, matmul_ogs = mo.FlexCtx, mo.PrecisionConfig, mo.matmul_ogs

dev = "cuda"
E, TOPK, H, I = 256, 6, 4096, 256
N1 = 2 * I
SETS = 8
torch.manual_seed(0)


def weights(n, k):
    w = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev)
    s = torch.randint(118, 124, (E, n, k // 32), dtype=torch.uint8, device=dev)
    w, flex, scale = _swizzle_mxfp4(w, s)
    return w, PrecisionConfig(weight_scale=scale, flex_ctx=FlexCtx(rhs_data=flex))


w13, pc13 = weights(N1, H)
w2, pc2 = weights(H, I)


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
    a.record()
    for _ in range(5):
        gr.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / (5 * reps * len(fns))


def with_flags(cons, fn):
    of.reset_opt_flags_constraints()
    if cons:
        of.update_opt_flags_constraints(cons)
    try:
        return fn()
    finally:
        of.reset_opt_flags_constraints()


def calls(M, which):
    """One closure per routing set, each running the gate/up or the down GEMM."""
    out = []
    for _ in range(SETS):
        ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).int()
        wts = torch.rand(M, TOPK, device=dev)
        rd, gi, si = g.make_routing_data(ids, wts, E)
        if which == "gate_up":
            x = torch.randn(M, H, device=dev).bfloat16()
            y = torch.empty(1, M * TOPK, N1, device=dev, dtype=torch.bfloat16)
            out.append((lambda x=x, y=y, rd=rd, gi=gi:
                        matmul_ogs(x, w13, None, rd, gather_indx=gi, precision_config=pc13, y=y), y))
        else:
            rd.n_expts_act = 1
            x = torch.randn(M * TOPK, I, device=dev).bfloat16()[gi.src_indx]
            y = torch.empty(1, M * TOPK, H, device=dev, dtype=torch.bfloat16)
            out.append((lambda x=x, y=y, rd=rd, si=si:
                        matmul_ogs(x, w2, None, rd, scatter_indx=si, precision_config=pc2,
                                   gammas=rd.gate_scal, y=y), y))
    return out


print(torch.cuda.get_device_name())
for which in ("gate_up", "down"):
    print(f"\n{which}: us per call; best forced flags vs default (block_n, split_k)")
    for M in (1, 2, 4, 8, 16, 32, 64):
        cs = calls(M, which)
        fns = [c[0] for c in cs]
        with_flags(None, lambda: [f() for f in fns])
        ref = [c[1].clone() for c in cs]
        t0 = with_flags(None, lambda: graph_us(fns))
        rows = []
        for bn, sk in itertools.product((32, 64, 128, 256), (1, 2, 4, 8, 16)):
            cons = {"block_n": bn, "split_k": sk}
            try:
                with_flags(cons, lambda: [f() for f in fns])
                err = max(((c[1].float() - r.float()).abs().max() / r.float().abs().max()).item()
                          for c, r in zip(cs, ref))
                rows.append((with_flags(cons, lambda: graph_us(fns)), cons, f"{err:.1e}"))
            except Exception as e:  # unsupported combination
                torch.cuda.synchronize()
                rows.append((float("inf"), cons, repr(e)[:60]))
        rows.sort(key=lambda r: r[0])
        best = rows[0]
        print(f"  M={M:3d}  default {t0:6.1f}  best {best[0]:6.1f} {best[1]} err {best[2]}   "
              + "  ".join(f"{r[1]['block_n']}/{r[1]['split_k']}:{r[0]:.0f}" for r in rows[1:5]),
              flush=True)
