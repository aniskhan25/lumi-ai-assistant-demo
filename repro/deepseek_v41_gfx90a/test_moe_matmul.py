"""V4.1-Flash MXFP4 expert GEMMs (triton_kernels matmul_ogs) at decode sizes, on one GCD.

V4.1 serves with expert parallelism (TP=8 + EP): each rank holds 48 of the 384 experts
whole, so gate/up is [5120 -> 4608] (2 x 2304) and down [2304 -> 5120], against V4's
TP-sharded [4096 -> 512] / [256 -> 4096] (../dsv4_flash_debug/test_moe_matmul.py, whose
tiles V4's MOE_SMALL uses). Tokens pick 6 of 384 experts, so a rank sees ~6/8 of a routed
row per token: the routing below is the global top-6 mapped through the rank's expert map
(-1 elsewhere), as vLLM's UnfusedOAITritonExperts gets it under EP.

For each batch, the default opt flags vs forced block_n / split_k, GPU us per call from
graph replays cycling 8 routings (the 48 experts' weights, 1.1 GB, never stay in L2), and
the max error against the default flags' output.

    sbatch repro/deepseek_v41_gfx90a/run_moe_matmul_test.sh
"""

import importlib
import itertools

import torch

import vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe as g
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import _swizzle_mxfp4

mo = importlib.import_module(g.matmul_ogs.__module__)
of = importlib.import_module(mo.__name__.rsplit(".", 1)[0] + ".matmul_ogs_details.opt_flags")
FlexCtx, PrecisionConfig, matmul_ogs = mo.FlexCtx, mo.PrecisionConfig, mo.matmul_ogs

dev = "cuda"
E_GLOBAL, EP, TOPK, H, I = 384, 8, 6, 5120, 2304
E = E_GLOBAL // EP  # local experts on this rank
N1 = 2 * I
SETS = 8
RANK = 3  # which rank's experts; any
torch.manual_seed(0)


def weights(n, k):
    w = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev)
    s = torch.randint(118, 124, (E, n, k // 32), dtype=torch.uint8, device=dev)
    w, flex, scale = _swizzle_mxfp4(w, s)
    return w, PrecisionConfig(weight_scale=scale, flex_ctx=FlexCtx(rhs_data=flex))


w13, pc13 = weights(N1, H)
w2, pc2 = weights(H, I)
expert_map = torch.full((E_GLOBAL,), -1, dtype=torch.int32, device=dev)
expert_map[RANK * E:(RANK + 1) * E] = torch.arange(E, dtype=torch.int32, device=dev)


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


def routing(M):
    """Global top-6 of 384 per token."""
    ids = torch.stack([torch.randperm(E_GLOBAL, device=dev)[:TOPK] for _ in range(M)]).int()
    wts = torch.rand(M, TOPK, device=dev)
    return ROUTE(ids, wts)


def calls(M, which):
    """One closure per routing set, each running the gate/up or the down GEMM."""
    out = []
    for _ in range(SETS):
        rd, gi, si, rows = routing(M)
        if which == "gate_up":
            x = torch.randn(M, H, device=dev).bfloat16()
            y = torch.empty(1, rows, N1, device=dev, dtype=torch.bfloat16)
            y.zero_()  # under EP only the routed rows are written; vLLM reads y[dst_indx]
            out.append((lambda x=x, y=y, rd=rd, gi=gi:
                        matmul_ogs(x, w13, None, rd, gather_indx=gi, precision_config=pc13, y=y),
                        y, gi.dst_indx))
        else:
            rd.n_expts_act = 1
            x = torch.randn(rows, I, device=dev).bfloat16()[gi.src_indx]
            y = torch.empty(1, rows, H, device=dev, dtype=torch.bfloat16)
            y.zero_()
            out.append((lambda x=x, y=y, rd=rd, si=si:
                        matmul_ogs(x, w2, None, rd, scatter_indx=si, precision_config=pc2,
                                   gammas=rd.gate_scal, y=y), y, None))
    return out


def ROUTE(ids, wts):
    """As UnfusedOAITritonExperts.apply under EP: global ids -> local (-1 elsewhere), routing
    over the local experts; its buffers have M * topk rows."""
    local = g.remap_topk_to_local(ids, expert_map)
    rd, gi, si = g.make_routing_data(local, wts, E)
    return rd, gi, si, ids.shape[0] * TOPK


if __name__ == "__main__":
    print(torch.cuda.get_device_name(), f"local experts {E}, gate/up [{H} -> {N1}], down [{I} -> {H}]")
    for which in ("gate_up", "down"):
        print(f"\n{which}: us per call; best forced flags vs default (block_n, split_k)")
        for M in (1, 2, 4, 8, 16, 32, 64, 128):
            cs = calls(M, which)
            fns = [c[0] for c in cs]
            with_flags(None, lambda: [f() for f in fns])
            def read_rows(c, t):  # the rows vLLM reads: routed slots for gate/up, all for down
                t = t.view(-1, t.shape[-1]).float()
                return t if c[2] is None else t[c[2].long()]

            ref = [read_rows(c, c[1]).clone() for c in cs]
            assert all(torch.isfinite(r).all() for r in ref), "default output not finite"
            t0 = with_flags(None, lambda: graph_us(fns))
            rows = []
            for bn, sk in itertools.product((32, 64, 128, 256), (1, 2, 4, 8, 16)):
                cons = {"block_n": bn, "split_k": sk}
                try:
                    with_flags(cons, lambda: [f() for f in fns])
                    err = max(((read_rows(c, c[1]) - r).abs().max()
                               / r.abs().max().clamp_min(1e-6)).item()
                              for c, r in zip(cs, ref))
                    rows.append((with_flags(cons, lambda: graph_us(fns)), cons, f"{err:.1e}"))
                except Exception as e:  # unsupported combination
                    torch.cuda.synchronize()
                    rows.append((float("inf"), cons, repr(e)[:60]))
            rows.sort(key=lambda r: r[0])
            best = rows[0]
            print(f"  M={M:3d}  default {t0:7.1f}  best {best[0]:7.1f} {best[1]} err {best[2]}   "
                  + "  ".join(f"{r[1]['block_n']}/{r[1]['split_k']}:{r[0]:.0f}" for r in rows[1:5]),
                  flush=True)
