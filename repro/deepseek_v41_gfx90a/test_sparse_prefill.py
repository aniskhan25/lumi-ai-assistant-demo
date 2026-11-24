"""Upstream's ROCm sparse-attention prefill kernel (_sparse_attn_prefill_ragged_kernel, the
path off gfx950 without aiter) on one GCD: its launch settings against a sweep.

V4.1's prefill at TP=8: 8 heads per rank, head dim 512, each query attends to a ragged
list of rows in a gathered bf16 workspace (SWA window 128 + top-512 compressed). The
wrapper launches one program per query with BLOCK_H 16, BLOCK_K 16, 4 warps, no software
pipelining: 4.65 ms per call for a 2048-query chunk at 16k context (prefill profile, job
23898197), ~29 % of prefill. This times the stock launch and BLOCK_K x warps x stages on
the same inputs, each checked against the stock output.

    sbatch repro/deepseek_v41_gfx90a/run_sparse_prefill_test.sh
"""

import itertools

import torch

import vllm.v1.attention.ops.rocm_aiter_mla_sparse as m  # noqa: E402 (vendored)

torch.manual_seed(0)
dev = "cuda"
H, D, NOPE, ROPE = 8, 512, 448, 64
SCALE = D**-0.5
kernel = m._sparse_attn_prefill_ragged_kernel
print(f"device {torch.cuda.get_device_properties(0).gcnArchName}")


def inputs(T, n_kv, swa=128, topk=512):
    """T queries at the end of an n_kv-row workspace: a causal SWA window plus top-k rows
    drawn from the earlier part (some -1), as the V4.1 prefill gather builds them."""
    q = torch.randn(T, H, D, device=dev).bfloat16()
    kv = torch.randn(n_kv, D, device=dev).bfloat16()
    lists = []
    for t in range(T):
        pos = n_kv - T + t
        win = torch.arange(max(0, pos - swa + 1), pos + 1)
        k = min(topk, max(0, pos - swa))
        top = torch.randperm(max(1, pos - swa + 1))[:k] if k else torch.empty(0, dtype=torch.long)
        lst = torch.cat([win, top])
        lst[torch.rand(len(lst)) < 0.02] = -1
        lists.append(lst.int())
    indptr = torch.tensor([0] + list(itertools.accumulate(len(x) for x in lists)),
                          dtype=torch.int32, device=dev)
    return q, kv, torch.cat(lists).to(dev), indptr


def launch(q, kv, idx, ptr, sink, out, block_k, warps, stages, block_h=16, **kw):
    kernel[(q.shape[0], triton_cdiv(H, block_h))](
        q, kv, idx, ptr, sink, out,
        q.stride(0), q.stride(1), q.stride(2), kv.stride(0), kv.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        H, D, kv.shape[0], float(SCALE),
        HAS_ATTN_SINK=True, OUT_DV=out.shape[-1], BLOCK_H=block_h, BLOCK_D=D,
        BLOCK_K=block_k, num_warps=warps, num_stages=stages, **kw,
    )
    return out


def triton_cdiv(a, b):
    return -(-a // b)


def timeit(fn, iters=10):
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


bad = []
for T, n_kv in ((2048, 4096), (2048, 16384)):
    q, kv, idx, ptr = inputs(T, n_kv)
    sink = torch.randn(H, device=dev)
    ref = m._rocm_sparse_attn_prefill_ragged_triton(q, kv, idx, ptr, SCALE, sink, NOPE, ROPE)
    t_stock = timeit(lambda: m._rocm_sparse_attn_prefill_ragged_triton(
        q, kv, idx, ptr, SCALE, sink, NOPE, ROPE))
    times = {}
    for bk, nw, st in itertools.product((16, 32, 64), (4, 8), (1, 2, 3)):
        out = torch.empty_like(q)
        try:
            launch(q, kv, idx, ptr, sink, out, bk, nw, st)
        except Exception:  # out of resources
            continue
        err = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
        if err > 2e-2:
            bad.append((T, n_kv, bk, nw, st, err))
            continue
        times[(bk, nw, st)] = timeit(lambda: launch(q, kv, idx, ptr, sink, out, bk, nw, st))
    best = min(times, key=times.get)
    top = sorted(times, key=times.get)[:5]
    print(f"T={T} kv={n_kv}: stock {t_stock:6.2f} ms  best {times[best]:6.2f} ms "
          f"(BLOCK_K, warps, stages)={best}  " + " ".join(f"{c}:{times[c]:.2f}" for c in top[1:]),
          flush=True)
print("AMD launch options, num_stages 1 (ms per call; served: BLOCK_K 16, 4 warps)")
for T, n_kv in ((2048, 16384), (2048, 30000)):
    q, kv, idx, ptr = inputs(T, n_kv)
    sink = torch.randn(H, device=dev)
    ref = m._rocm_sparse_attn_prefill_ragged_triton(q, kv, idx, ptr, SCALE, sink, NOPE, ROPE)
    times = {}
    for bh, bk, nw, nk, we, kp in itertools.product(
            (8, 16), (16, 32, 64), (2, 4, 8), (16, 32), (0, 1, 2, 4), (1, 2)):
        kw = dict(matrix_instr_nonkdim=nk, waves_per_eu=we, kpack=kp)
        out = torch.empty_like(q)
        try:
            launch(q, kv, idx, ptr, sink, out, bk, nw, 1, block_h=bh, **kw)
        except Exception:  # unsupported or out of resources
            continue
        err = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
        if err > 2e-2:
            bad.append((T, n_kv, bh, bk, nw, kw, err))
            continue
        times[(bh, bk, nw, nk, we, kp)] = timeit(
            lambda: launch(q, kv, idx, ptr, sink, out, bk, nw, 1, block_h=bh, **kw))
    served = timeit(lambda: launch(q, kv, idx, ptr, sink, torch.empty_like(q), 16, 4, 1))
    top = sorted(times, key=times.get)[:6]
    print(f"kv={n_kv}: served {served:.2f} ms; best (BLOCK_H, BLOCK_K, warps, nonkdim, "
          f"waves_per_eu, kpack): " + "  ".join(f"{c}:{times[c]:.2f}" for c in top), flush=True)
print("Diagnosis: lumi_dsv41/sparse_prefill.py variants (ms per call; output columns split "
      "over DV_SPLIT programs per query; loads only = the gathers without the math)")
import sys  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lumi_dsv41.sparse_prefill import sparse_prefill  # noqa: E402

for T, n_kv in ((2048, 16384), (2048, 30000)):
    q, kv, idx, ptr = inputs(T, n_kv)
    sink = torch.randn(H, device=dev)
    ref = m._rocm_sparse_attn_prefill_ragged_triton(q, kv, idx, ptr, SCALE, sink, NOPE, ROPE)
    gb = (idx >= 0).sum().item() * D * 2 / 1e9
    served = timeit(lambda: launch(q, kv, idx, ptr, sink, torch.empty_like(q), 16, 2, 1,
                                   matrix_instr_nonkdim=16, waves_per_eu=1))
    row = [f"served {served:.2f}"]
    for split, bk, nw in itertools.product((1, 2, 4), (16, 32), (2, 4)):
        try:
            got = sparse_prefill(q, kv, idx, ptr, SCALE, sink, dv_split=split, block_k=bk, num_warps=nw)
        except Exception as e:
            row.append(f"s{split}/k{bk}/w{nw}: {type(e).__name__}")
            continue
        err = ((got.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
        if err > 2e-2:
            bad.append(("variant", n_kv, split, bk, nw, err))
            continue
        t = timeit(lambda: sparse_prefill(q, kv, idx, ptr, SCALE, sink, dv_split=split,
                                          block_k=bk, num_warps=nw))
        row.append(f"s{split}/k{bk}/w{nw} {t:.2f}")
    lo = {}
    for split, nw in ((1, 2), (1, 4), (2, 2), (4, 2)):
        lo[(split, nw)] = timeit(lambda: sparse_prefill(q, kv, idx, ptr, SCALE, sink, dv_split=split,
                                                        num_warps=nw, loads_only=True))
    print(f"kv={n_kv}: {gb:.2f} GB gathered; " + "  ".join(row), flush=True)
    print(f"   loads only: " + "  ".join(f"s{s_}/w{w}: {t:.2f} ms ({gb / t:.0f} GB/s)"
                                         for (s_, w), t in lo.items()), flush=True)
print("FAILED:", bad if bad else "none")
