"""lumi_dsv41/mqa_logits.py on one GCD against the torch fallback it replaces
(rocm_aiter_mla_sparse.fp8_mqa_logits_torch, as served: row-chunked by _chunked_mqa_logits).

V4.1's indexer: 32 heads, head dim 128, FP8 queries and keys with fp32 per-key scales.
A 2048-token prefill chunk at the end of an N-key context (causal ends), N = 2k-30k:
same -inf mask, values within bf16 rounding, the same top-512 per row; then time per call
for a few tiles.

    sbatch repro/deepseek_v41_gfx90a/run_mqa_logits_test.sh
"""

import itertools

import torch

import vllm.v1.attention.ops.rocm_aiter_mla_sparse as m  # noqa: E402 (vendored)
from lumi_dsv41.mqa_logits import fp8_mqa_logits_triton  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
H, D, TOPK, M = 32, 128, 512, 2048
FP8 = torch.float8_e4m3fn
ref_fn = m.fp8_mqa_logits_torch
print(f"device {torch.cuda.get_device_properties(0).gcnArchName}")
bad = []


def timeit(fn, iters=5):
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


for N in (2048, 8192, 16384, 30000):
    q = (torch.randn(M, H, D, device=dev) * 0.5).to(FP8)
    k = (torch.randn(N, D, device=dev) * 0.5).to(FP8)
    scale = (torch.rand(N, 1, device=dev) * 0.05 + 0.01)
    w = torch.randn(M, H, device=dev) * 0.1
    ke = (torch.arange(M, device=dev, dtype=torch.int32) + (N - M) + 1).clamp(1, N)
    ks = torch.zeros(M, device=dev, dtype=torch.int32)
    ks[M // 2:] = 64  # some rows with a nonzero start
    ref = ref_fn(q, (k, scale), w, ks, ke)
    got = fp8_mqa_logits_triton(q, (k, scale), w, ks, ke)
    same_mask = torch.equal(torch.isinf(ref), torch.isinf(got))
    fin = torch.isfinite(ref)
    err = ((got[fin] - ref[fin]).abs().max() / ref[fin].abs().max()).item()
    kk = min(TOPK, N)
    ti = ref.topk(kk, dim=1).indices.sort(dim=1).values
    gi = got.topk(kk, dim=1).indices.sort(dim=1).values
    overlap = (ti == gi).float().mean().item()
    ok = same_mask and err < 2e-2 and overlap > 0.98
    if not ok:
        bad.append((N, same_mask, err, overlap))
    t_ref = timeit(lambda: ref_fn(q, (k, scale), w, ks, ke))
    times = {}
    for bm, bn, nw in itertools.product((32, 64, 128), (32, 64, 128), (4, 8)):
        try:
            fp8_mqa_logits_triton(q, (k, scale), w, ks, ke, block_m=bm, block_n=bn, num_warps=nw)
        except Exception:  # out of resources
            continue
        times[(bm, bn, nw)] = timeit(lambda: fp8_mqa_logits_triton(
            q, (k, scale), w, ks, ke, block_m=bm, block_n=bn, num_warps=nw))
    best = min(times, key=times.get)
    top = sorted(times, key=times.get)[:4]
    print(f"N={N:6d}: mask {same_mask} max rel err {err:.1e} top-{kk} overlap {overlap:.4f}  "
          f"torch {t_ref:7.2f} ms  triton best {times[best]:6.2f} ms {best}  "
          + " ".join(f"{c}:{times[c]:.2f}" for c in top[1:]), flush=True)
print("FAILED:", bad if bad else "none")
