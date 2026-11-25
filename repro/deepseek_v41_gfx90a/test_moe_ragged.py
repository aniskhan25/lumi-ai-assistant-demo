"""Is matmul_ogs slower than a plain grouped dot_scaled GEMM on real routing?

test_mxfp4_gemm.py: a plain grouped dot_scaled GEMM takes 1148 us at 32 tokens per expert
(uniform); matmul_ogs gate/up took ~1545 us at a similar average in test_moe_matmul.py.
Here both run the same real routing: router counts (EXPERT_COUNTS=1, job 23995740) of a
few layers scaled to a 2048-token chunk, rank 0's 48 experts under the linear and the
balanced placement. matmul_ogs as served (gathers rows of x, served tiles);
lumi_dsv41.mxfp4_gemm.ragged_gemm on x already sorted by expert (the gather it skips reads
~1536 rows x 10 KB, small next to 566 MB of weights). Gate/up only.

    MOE_TEST=test_moe_ragged.py COUNTS=... sbatch repro/deepseek_v41_gfx90a/run_moe_matmul_test.sh
"""

import itertools
import json
import os
import random
import re

import torch

import expert_placement as ep
import test_moe_matmul as t  # weights, matmul_ogs, graph timing, served tiles (main guarded)
from lumi_dsv41.mxfp4_gemm import dequant, ragged_gemm

M, TOPK, E, PER = 2048, t.TOPK, t.E_GLOBAL, t.E
counts = json.load(open(os.environ["COUNTS"]))
LAYERS = [f"model.layers.{i}.ffn.experts" for i in (0, 10, 20, 39)]


def rank_routing(local_counts):
    """[M, TOPK] local expert ids (-1 = another rank's) with these per-expert counts."""
    ids = torch.full((M, TOPK), -1, dtype=torch.int32)
    free = [list(range(TOPK)) for _ in range(M)]
    rng = random.Random(0)
    for e, n in sorted(enumerate(local_counts), key=lambda x: -x[1]):
        for tok in rng.sample(range(M), min(n, M)):
            if free[tok]:
                ids[tok, free[tok].pop(rng.randrange(len(free[tok])))] = e
    return ids.to(t.dev)


# Our kernel's weights: the same shapes and bytes as t.w13 (gate/up), in its own layout.
w_r = torch.randint(0, 256, (PER, t.H // 2, t.N1), dtype=torch.uint8, device=t.dev)
s_r = torch.randint(118, 125, (PER, t.N1, t.H // 32), dtype=torch.uint8, device=t.dev)
print(torch.cuda.get_device_name(), f"rank 0, {PER} experts [{t.H} -> {t.N1}], {M}-token chunk")

checked = False
for name in LAYERS:
    c = counts[name]
    scale = M * TOPK / sum(c)
    c = [round(v * scale) for v in c]
    for pname, place in (("linear", [e // PER for e in range(E)]), ("balanced", ep.balanced(c))):
        mine = [e for e in range(E) if place[e] == 0]
        ids = rank_routing([c[e] for e in mine])
        cnt = torch.bincount(ids[ids >= 0].long(), minlength=PER)
        rows = int(cnt.sum())
        # matmul_ogs, as served
        wts = torch.rand(M, TOPK, device=t.dev)
        rd, gi, si = t.g.make_routing_data(ids, wts, PER)
        x = torch.randn(M, t.H, device=t.dev).bfloat16()
        y1 = torch.zeros(1, M * TOPK, t.N1, device=t.dev, dtype=torch.bfloat16)
        ogs = lambda: t.matmul_ogs(x, t.w13, None, rd, gather_indx=gi, precision_config=t.pc13, y=y1)  # noqa: E731
        t_ogs = t.with_flags(t.served_cons("gate_up", M), lambda: t.graph_us([ogs]))
        # ragged dot_scaled on expert-sorted rows
        xs = torch.randn(rows, t.H, device=t.dev).bfloat16()
        offs = torch.zeros(PER + 1, dtype=torch.int32, device=t.dev)
        offs[1:] = torch.cumsum(cnt, 0)
        mx = int(cnt.max())
        if not checked:  # correctness once, against an fp32 reference
            y = ragged_gemm(xs, w_r, s_r, offs, mx).float()
            wf = dequant(w_r, s_r)
            ref = torch.cat([xs[offs[e]:offs[e + 1]].float() @ wf[e] for e in range(PER)])
            err = ((y - ref).abs().max() / ref.abs().max()).item()
            print(f"ragged_gemm vs fp32 reference: max rel err {err:.1e}", flush=True)
            del wf
            checked = True
        times = {}
        for bm, bn, nw, nk in itertools.product((16, 32, 64), (128, 256), (4, 8), (16, 32)):
            kw = dict(block_m=bm, block_n=bn, num_warps=nw, matrix_instr_nonkdim=nk)
            try:
                ragged_gemm(xs, w_r, s_r, offs, mx, **kw)
            except Exception:
                continue
            times[(bm, bn, nw, nk)] = t.graph_us([lambda: ragged_gemm(xs, w_r, s_r, offs, mx, **kw)])
        best = min(times, key=times.get)
        print(f"  {name:<28} {pname:8} rows {rows:5d} (max expert {mx:4d}): matmul_ogs {t_ogs:7.0f} us, "
              f"ragged dot_scaled {times[best]:7.0f} us {best} ({t_ogs / times[best]:.2f}x)", flush=True)
