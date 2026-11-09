"""V4 MoE router (ops.topk_hash_softplus_sqrt, the ROCm path) on non-finite rows.

Graph-padded batches carry pad rows that hold whatever the graph pool left there. The
router is the first op that is not row-local in effect: an expert id outside [0, E) in a
pad row would corrupt the routing of real tokens. This checks, on one GCD, what the
kernel returns for NaN / Inf / huge rows, and that finite rows match the torch reference.
"""

import torch

from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    _topk_softplus_sqrt_torch,
    fused_topk_bias,
)

torch.manual_seed(0)
dev = "cuda"
E, K, H = 256, 6, 4096
bias = (8.0 + 0.1 * torch.randn(E, device=dev)).float()
table = torch.randint(0, E, (129280, K), device=dev, dtype=torch.int32)


def rows(M):
    g = torch.randn(M, E, device=dev) * 3
    kinds = ["finite"] * M
    special = {
        "all_nan": float("nan"),
        "all_posinf": float("inf"),
        "all_neginf": float("-inf"),
        "all_huge": 3e38,
    }
    for i, (name, v) in enumerate(special.items()):
        if i < M:
            g[i] = v
            kinds[i] = name
    if M > 5:
        g[4, ::7] = float("nan")
        kinds[4] = "some_nan"
        g[5, ::5] = float("inf")
        kinds[5] = "some_inf"
    return g, kinds


def check(M, use_hash, dtype):
    g, kinds = rows(M)
    g = g.to(dtype)
    hs = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
    toks = torch.randint(0, 129280, (M,), device=dev, dtype=torch.int32)
    w, ids = fused_topk_bias(
        hs, g, "sqrtsoftplus", None if use_hash else bias, K, True,
        input_tokens=toks if use_hash else None,
        hash_indices_table=table if use_hash else None,
        routed_scaling_factor=1.5,
    )
    torch.cuda.synchronize()
    rw = torch.empty(M, K, device=dev)
    ri = torch.empty(M, K, device=dev, dtype=torch.int32)
    _topk_softplus_sqrt_torch(rw, ri, torch.empty_like(ri), g, True,
                              None if use_hash else bias, toks if use_hash else None,
                              table if use_hash else None, 1.5)
    bad = []
    for r in range(M):
        i = ids[r].tolist()
        oor = [x for x in i if x < 0 or x >= E]
        dup = len(set(i)) != K
        nanw = bool(torch.isnan(w[r]).any())
        if kinds[r] == "finite":
            same = sorted(i) == sorted(ri[r].tolist())
            werr = (w[r].sort().values - rw[r].sort().values).abs().max().item()
            if not same or werr > 1e-4 or oor or dup or nanw:
                bad.append((r, kinds[r], i, ri[r].tolist(), werr))
        elif oor or dup or nanw:
            bad.append((r, kinds[r], i, w[r].tolist()))
    special = [(r, kinds[r], ids[r].tolist(), [round(x, 3) for x in w[r].tolist()])
               for r in range(min(M, 6)) if kinds[r] != "finite"]
    print(f"M={M:4d} hash={use_hash} {str(dtype):15s} finite-row mismatches / "
          f"bad special rows: {len(bad)}")
    for b in bad[:6]:
        print("   BAD", b)
    if M == 16:
        for s in special:
            print("   ", s)


for dtype in (torch.float32, torch.bfloat16):
    for use_hash in (False, True):
        for M in (1, 4, 13, 16, 64, 128, 512):
            check(M, use_hash, dtype)
