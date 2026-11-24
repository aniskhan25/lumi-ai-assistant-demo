"""Would a balanced expert placement shorten the MoE critical path? Timed, not estimated.

Under EP each MoE layer ends in an all-reduce that waits for the slowest rank, so a
2048-token prefill chunk's MoE critical path is the sum over layers of the slowest rank's
expert GEMMs. Expert GEMMs at this size are largely weight-bound, so token counts alone
overstate the imbalance. Here, for every target MoE layer, the measured router counts
(EXPERT_COUNTS=1) are scaled to one 2048-token chunk, each rank's share under the linear
placement and under expert_placement.py's balanced one is rebuilt as routing over its 48
local experts, and gate/up + down are timed on one GCD with the served tiles
(test_moe_matmul.py's weights and graph timing).

    COUNTS=expert_counts.json sbatch repro/deepseek_v41_gfx90a/run_moe_matmul_test.sh
    (with MOE_TEST=test_moe_placement.py)
"""

import json
import os
import random
import re

import torch

import test_moe_matmul as t  # weights, routing helpers, graph timing (its main is guarded)

M, TOPK, EP, PER, E = 2048, t.TOPK, t.EP, t.E, t.E_GLOBAL
counts = json.load(open(os.environ["COUNTS"]))
layers = sorted((k for k in counts if int(re.findall(r"\d+", k)[0]) < int(os.environ.get("TARGET_LAYERS", 40))),
                key=lambda k: int(re.findall(r"\d+", k)[0]))
import expert_placement as ep  # noqa: E402  (its module code reads sys.argv: guard below)


def rank_routing(local_counts):
    """[M, TOPK] local expert ids (-1 = another rank's expert) with these per-expert
    counts, no expert twice in a token."""
    ids = torch.full((M, TOPK), -1, dtype=torch.int32)
    free = [list(range(TOPK)) for _ in range(M)]
    rng = random.Random(0)
    for e, n in sorted(enumerate(local_counts), key=lambda x: -x[1]):
        tokens = rng.sample(range(M), min(n, M))
        for tok in tokens:
            if free[tok]:
                ids[tok, free[tok].pop(rng.randrange(len(free[tok])))] = e
    return ids.to(t.dev)


def time_rank(local_ids):
    wts = torch.rand(M, TOPK, device=t.dev)
    rd, gi, si = t.g.make_routing_data(local_ids, wts, PER)
    rows = M * TOPK
    x = torch.randn(M, t.H, device=t.dev).bfloat16()
    y1 = torch.zeros(1, rows, t.N1, device=t.dev, dtype=torch.bfloat16)
    rd2, gi2, si2 = t.g.make_routing_data(local_ids, wts, PER)
    rd2.n_expts_act = 1
    x2 = torch.randn(rows, t.I, device=t.dev).bfloat16()[gi2.src_indx]
    y2 = torch.zeros(1, rows, t.H, device=t.dev, dtype=torch.bfloat16)
    up = lambda: t.matmul_ogs(x, t.w13, None, rd, gather_indx=gi, precision_config=t.pc13, y=y1)  # noqa: E731
    down = lambda: t.matmul_ogs(x2, t.w2, None, rd2, scatter_indx=si2, precision_config=t.pc2,  # noqa: E731
                                gammas=rd2.gate_scal, y=y2)
    return (t.with_flags(t.served_cons("gate_up", M), lambda: t.graph_us([up]))
            + t.with_flags(t.served_cons("down", M), lambda: t.graph_us([down])))


print(torch.cuda.get_device_name(), f"{len(layers)} target MoE layers, {M}-token chunk")
tot = {"linear": [0.0, 0.0], "balanced": [0.0, 0.0]}
for name in layers:
    c = counts[name]
    scale = M * TOPK / sum(c)
    c = [round(v * scale) for v in c]
    places = {"linear": [e // PER for e in range(E)], "balanced": ep.balanced(c)}
    line = []
    for pname, place in places.items():
        ts = []
        for r in range(EP):
            mine = [e for e in range(E) if place[e] == r]
            ts.append(time_rank(rank_routing([c[e] for e in mine])))
        tot[pname][0] += max(ts)
        tot[pname][1] += sum(ts) / EP
        line.append(f"{pname} max {max(ts):6.0f} mean {sum(ts) / EP:6.0f} us")
    print(f"  {name:<28} " + ";  ".join(line), flush=True)
for pname, (mx, mean) in tot.items():
    print(f"{pname:9}: MoE critical path {mx / 1e3:6.1f} ms per chunk (slowest rank per layer), "
          f"mean rank {mean / 1e3:6.1f} ms, waiting {(mx - mean) / 1e3:5.1f} ms")
print(f"balanced saves {(tot['linear'][0] - tot['balanced'][0]) / 1e3:.1f} ms of MoE critical path "
      "per 2048-token chunk")
