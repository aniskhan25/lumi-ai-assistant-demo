"""Check plugin/lumi_dsv4_debug/paged_logits.py on one GCD against vLLM 0.26's own fallback.

For V4-Flash's indexer shape (64 heads, head dim 128, FP8 cache with float32 scales):
  1. the Triton kernel vs `fp8_paged_mqa_logits_torch`: same -inf mask, finite values within
     fp32 rounding, same top-512 indices;
  2. CUDA-graph capture with one set of sequence lengths, replay with another;
  3. time per call, torch loop vs Triton, at 1-32 sequences;
  4. MTP (next_n = 2): per-token [B, 2] causal lengths, as vLLM's indexer passes them, and
     per-sequence [B] lengths, vs the fallback's next_n == 1 path run per query token. (Its
     next_n > 1 path reads the cache as interleaved [D values, 4 scale bytes] per position,
     not the per-block layout V4 writes, so it is wrong for V4.)

    python test_paged_logits.py
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "plugin"))
from lumi_dsv4_debug.paged_logits import fp8_paged_mqa_logits_triton  # noqa: E402

from vllm.platforms import current_platform  # noqa: E402
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import fp8_paged_mqa_logits_torch  # noqa: E402

H, D, TOPK, MAX_LEN = 64, 128, 512, 32768
FP8 = current_platform.fp8_dtype()
dev = torch.device("cuda")
print(f"device {torch.cuda.get_device_properties(0).gcnArchName}  fp8 {FP8}")
failures = []


def make_cache(num_blocks: int, block_size: int) -> torch.Tensor:
    kv = torch.empty(num_blocks, block_size, 1, D + 4, dtype=torch.uint8, device=dev)
    flat = kv.view(num_blocks, -1)
    vals = (torch.randn(num_blocks, block_size * D, device=dev) * 2).to(FP8)
    scales = torch.exp2(torch.randint(-8, 0, (num_blocks, block_size), device=dev).float())
    flat[:, : block_size * D] = vals.view(torch.uint8)
    flat[:, block_size * D :] = scales.view(torch.uint8)
    return kv


def make_batch(lens: list[int], block_size: int, num_blocks: int):
    b = len(lens)
    q = torch.randn(b, 1, H, D, device=dev).to(FP8)
    w = torch.randn(b, H, device=dev)
    max_blocks = MAX_LEN // block_size
    bt = torch.stack([torch.randperm(num_blocks, device=dev)[:max_blocks] for _ in lens]).int()
    return q, w, torch.tensor(lens, dtype=torch.int32, device=dev), bt


def compare(name: str, got: torch.Tensor, want: torch.Tensor, lens: list[int]) -> None:
    mask_ok = torch.equal(torch.isinf(got), torch.isinf(want))
    fin = torch.isfinite(want)
    rel = ((got[fin] - want[fin]).norm() / want[fin].norm().clamp_min(1e-12)).item()
    overlap = []
    for i, n in enumerate(lens):
        k = min(TOPK, n)
        a = set(torch.topk(got[i, :n], k).indices.tolist())
        b = set(torch.topk(want[i, :n], k).indices.tolist())
        overlap.append(len(a & b) / k)
    worst = min(overlap)
    bad = not mask_ok or rel > 1e-4 or worst < 0.99
    if bad:
        failures.append(name)
    print(f"  {'FAIL' if bad else 'ok  '} {name:<34} -inf mask {'same' if mask_ok else 'DIFF'}"
          f"  rel err {rel:.1e}  top-{TOPK} overlap min {worst:.4f}")


torch.manual_seed(0)
print("1. correctness vs fp8_paged_mqa_logits_torch")
for block_size in (64, 256):
    num_blocks = 2 * MAX_LEN // block_size
    kv = make_cache(num_blocks, block_size)
    for lens in ([1], [block_size], [block_size + 1], [5000], [1, 63, 64, 65, 300, 2047, 8192, MAX_LEN],
                 [377] * 32):
        q, w, cl, bt = make_batch(lens, block_size, num_blocks)
        want = fp8_paged_mqa_logits_torch(q, kv, w, cl, bt, MAX_LEN)
        got = fp8_paged_mqa_logits_triton(q, kv, w, cl, bt, MAX_LEN)
        label = f"bs={block_size} lens={lens if len(lens) < 9 else f'{len(lens)}x{lens[0]}'}"
        compare(label[:34], got, want, lens)

print("2. CUDA graph: capture with lengths A, replay with lengths B")
block_size = 64
num_blocks = 2 * MAX_LEN // block_size
kv = make_cache(num_blocks, block_size)
lens_a, lens_b = [100, 2000, 7, 4096], [3000, 1, 650, 12000]
q, w, cl, bt = make_batch(lens_a, block_size, num_blocks)
for _ in range(2):
    fp8_paged_mqa_logits_triton(q, kv, w, cl, bt, MAX_LEN)  # compile outside capture
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out = fp8_paged_mqa_logits_triton(q, kv, w, cl, bt, MAX_LEN)
cl.copy_(torch.tensor(lens_b, dtype=torch.int32, device=dev))
q.copy_(torch.randn_like(q, dtype=torch.float32).to(FP8))
graph.replay()
torch.cuda.synchronize()
compare("graph replay, new lengths", out.clone(), fp8_paged_mqa_logits_torch(q, kv, w, cl, bt, MAX_LEN), lens_b)

print("3. time per call (ms), torch loop vs Triton")


def timeit(fn, iters: int = 20) -> float:
    for _ in range(3):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


for length in (384, 4096):
    for batch in (1, 4, 16, 32):
        q, w, cl, bt = make_batch([length] * batch, block_size, num_blocks)
        t_torch = timeit(lambda: fp8_paged_mqa_logits_torch(q, kv, w, cl, bt, MAX_LEN))
        t_triton = timeit(lambda: fp8_paged_mqa_logits_triton(q, kv, w, cl, bt, MAX_LEN))
        print(f"  len {length:>5} batch {batch:>2}: torch {t_torch:7.3f}  triton {t_triton:6.3f}"
              f"  ({t_torch / t_triton:5.1f}x)")

print("4. MTP, next_n = 2, vs fp8_paged_mqa_logits_torch per query token")
for block_size in (64, 256):
    num_blocks = 2 * MAX_LEN // block_size
    kv = make_cache(num_blocks, block_size)
    for lens in ([2], [block_size + 1], [5000], [2, 64, 65, 300, 2047, 8192, MAX_LEN], [377] * 32):
        b, n = len(lens), 2
        _, _, cl, bt = make_batch(lens, block_size, num_blocks)
        q = torch.randn(b, n, H, D, device=dev).to(FP8)
        w = torch.randn(b * n, H, device=dev)
        rows = [L - n + j + 1 for L in lens for j in range(n)]
        per_token = torch.tensor(rows, dtype=torch.int32, device=dev).view(b, n)
        want = torch.cat([
            fp8_paged_mqa_logits_torch(q[r // n : r // n + 1, r % n : r % n + 1], kv, w[r : r + 1],
                                       per_token.view(-1)[r : r + 1], bt[r // n : r // n + 1], MAX_LEN)
            for r in range(b * n)])
        for name, c in (("[B,2] lens", per_token), ("[B] lens", cl)):
            got = fp8_paged_mqa_logits_triton(q, kv, w, c, bt, MAX_LEN)
            label = f"bs={block_size} {name} {lens if len(lens) < 8 else f'{len(lens)}x{lens[0]}'}"
            compare(label[:34], got, want, rows)

print("FAILED:", failures or "none")
