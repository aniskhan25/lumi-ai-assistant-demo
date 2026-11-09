"""Do graph pad rows with duplicate expert ids corrupt the real rows of the MoE?

ROCm's V4 router (ops.topk_hash_softplus_sqrt) returns ids [0]*6 for an all-NaN row
(test_topk_nan.py). The triton_unfused experts build dispatch/combine maps from a
bitmatrix, where a duplicated id sets one bit for six top-k slots. This runs the same
path as UnfusedOAITritonExperts.apply (make_routing_data, matmul_ogs gather / scatter,
masked_moe_sum) with bf16 weights on one GCD, 12 real rows plus 4 pad rows, and compares
the real rows with a dense reference.
"""

import torch
import torch.nn.functional as F

import vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe as moe
from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
    make_routing_data,
    masked_moe_sum,
)

matmul_ogs = moe.matmul_ogs
torch.manual_seed(0)
dev = "cuda"
E, K, I, TOPK = 256, 512, 128, 6
N = 2 * I
w1 = (torch.randn(E, K, N, device=dev) / K**0.5).bfloat16()
w2 = (torch.randn(E, I, K, device=dev) / I**0.5).bfloat16()


def experts(x, ids, wts):
    M = x.shape[0]
    rd, g, s = make_routing_data(ids, wts, E)
    ic1 = torch.empty(1, M * TOPK, N, device=dev, dtype=torch.bfloat16)
    matmul_ogs(x, w1, None, rd, gather_indx=g, y=ic1)
    a = ic1.view(-1, N)[g.dst_indx]
    ic2 = F.silu(a[:, :I]) * a[:, I:]
    rd.n_expts_act = 1
    ic3 = torch.zeros(1, M * TOPK, K, device=dev, dtype=torch.bfloat16)
    matmul_ogs(ic2[g.src_indx], w2, None, rd, scatter_indx=s, gammas=rd.gate_scal, y=ic3)
    out = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
    masked_moe_sum(ic3.view(-1, TOPK, K), ids, out)
    return out, rd


def reference(x, ids, wts):
    out = torch.zeros(x.shape[0], K, device=dev)
    for t in range(x.shape[0]):
        for j in range(TOPK):
            e = int(ids[t, j])
            h = x[t].float() @ w1[e].float()
            h = F.silu(h[:I]) * h[I:]
            out[t] += wts[t, j].float() * (h.bfloat16().float() @ w2[e].float())
    return out


def distinct_ids(M):
    return torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).int()


R, P = 12, 4
x_real = torch.randn(R, K, device=dev).bfloat16()
ids_real = distinct_ids(R)
w_real = torch.softmax(torch.randn(R, TOPK, device=dev), -1)
ref = reference(x_real, ids_real, w_real)
alone, _ = experts(x_real, ids_real, w_real)
tol = 3e-2 * ref.abs().max().item()
print(f"real rows alone: max err {(alone.float() - ref).abs().max().item():.3e} (tol {tol:.2e})")

nan = float("nan")
cases = {
    "pad: distinct ids, finite": (distinct_ids(P), torch.randn(P, K), 0.2),
    "pad: distinct ids, NaN hidden + NaN weights": (distinct_ids(P), torch.full((P, K), nan), nan),
    "pad: ids [0..5] (Inf router rows), NaN weights": (
        torch.arange(TOPK, device=dev).int().repeat(P, 1), torch.randn(P, K), nan),
    "pad: ids [0]*6 (NaN router rows), finite hidden": (
        torch.zeros(P, TOPK, device=dev, dtype=torch.int32), torch.randn(P, K), 0.2),
    "pad: ids [0]*6, NaN hidden, router weights as returned": (
        torch.zeros(P, TOPK, device=dev, dtype=torch.int32), torch.full((P, K), nan),
        torch.tensor([nan] + [-15011.86] * 5)),
    "one pad row [0]*6 at the front": None,
}
for name, case in cases.items():
    if case is None:
        ids = torch.cat([torch.zeros(1, TOPK, device=dev, dtype=torch.int32), ids_real])
        x = torch.cat([torch.randn(1, K, device=dev).bfloat16(), x_real])
        w = torch.cat([torch.full((1, TOPK), 0.2, device=dev), w_real])
        out, rd = experts(x, ids, w)
        real = out[1:]
    else:
        pid, px, pw = case
        pw = (pw.to(dev).expand(P, TOPK) if torch.is_tensor(pw)
              else torch.full((P, TOPK), pw, device=dev))
        ids = torch.cat([ids_real, pid])
        x = torch.cat([x_real, px.to(dev).bfloat16()])
        w = torch.cat([w_real, pw.float()])
        out, rd = experts(x, ids, w)
        real = out[:R]
    err = (real.float() - ref)
    bad_rows = [r for r in range(R) if not torch.isfinite(err[r]).all()
                or err[r].abs().max().item() > tol]
    print(f"{name:55s} real rows corrupted: {len(bad_rows)}/{R} {bad_rows}  "
          f"max err {err.abs().nan_to_num(float('inf')).max().item():.3e}")


# Sweep: R real rows padded to vLLM's capture sizes, pad rows as an all-NaN router
# row comes back ([0]*6), i.e. exactly the serving layout.
sizes = [1, 2, 4] + list(range(8, 129, 8))
print("\nsweep: real rows R padded to the next capture size with [0]*6 pad rows")
hits = []
for R in range(1, 65):
    Mp = next(s for s in sizes if s >= R)
    P = Mp - R
    xr = torch.randn(R, K, device=dev).bfloat16()
    ir = distinct_ids(R)
    wr = torch.softmax(torch.randn(R, TOPK, device=dev), -1)
    alone, _ = experts(xr, ir, wr)
    if P == 0:
        continue
    x = torch.cat([xr, torch.randn(P, K, device=dev).bfloat16()])
    ids = torch.cat([ir, torch.zeros(P, TOPK, device=dev, dtype=torch.int32)])
    w = torch.cat([wr, torch.full((P, TOPK), 0.2, device=dev)])
    out, _ = experts(x, ids, w)
    d = (out[:R].float() - alone.float()).abs().amax(1)
    n = int((d > 1e-3 * alone.float().abs().max()).sum())
    if n:
        hits.append((R, Mp, n))
print("corrupted (R, padded, rows):", hits if hits else "none")
