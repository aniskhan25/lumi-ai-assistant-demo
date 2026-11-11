"""lumi_dsv41/paged_logits.py on one GCD against vLLM 0.26's own torch fallback.

V4.1's indexer shape (32 heads, head dim 128, FP8 values with float32 scales). The same
random keys are stored twice: row-major for vLLM's `fp8_paged_mqa_logits_torch`, and in the
16x16-tiled (SHUFFLE) layout V4.1's ROCm writer uses for the Triton kernel, so the test
checks the kernel and the tiling together:
  1. same -inf mask, values within fp32 rounding, same top-512;
  2. CUDA-graph capture with one set of lengths, replay with another;
  3. time per call, torch loop vs Triton, at 1-32 sequences.

    sbatch repro/deepseek_v41_gfx90a/run_paged_logits_test.sh
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lumi_dsv41.paged_logits import TILE, fp8_paged_mqa_logits_triton  # noqa: E402

from vllm.platforms import current_platform  # noqa: E402
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import fp8_paged_mqa_logits_torch  # noqa: E402

H, D, TOPK, MAX_LEN = 32, 128, 512, 16384
FP8 = current_platform.fp8_dtype()
dev = torch.device("cuda")
print(f"device {torch.cuda.get_device_properties(0).gcnArchName}  fp8 {FP8}")
failures = []


def make_caches(num_blocks: int, block_size: int):
    """(row-major cache, tiled cache) holding the same keys and scales."""
    flat = torch.empty(num_blocks, block_size * (D + 4), dtype=torch.uint8, device=dev)
    vals = (torch.randn(num_blocks, block_size, D, device=dev) * 2).to(FP8).view(torch.uint8)
    scales = torch.exp2(torch.randint(-8, 0, (num_blocks, block_size), device=dev).float())
    flat[:, block_size * D:] = scales.view(torch.uint8)
    rows, tiled = flat.clone(), flat.clone()
    rows[:, : block_size * D] = vals.reshape(num_blocks, -1)
    tiled[:, : block_size * D] = (vals.view(num_blocks, block_size // TILE, TILE, D // TILE, TILE)
                                  .permute(0, 1, 3, 2, 4).reshape(num_blocks, -1))
    shape = (num_blocks, block_size, 1, D + 4)
    return rows.view(shape), tiled.view(shape)


def make_batch(lens: list[int], block_size: int, num_blocks: int):
    q = torch.randn(len(lens), 1, H, D, device=dev).to(FP8)
    w = torch.randn(len(lens), H, device=dev)
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
    print(f"  {'FAIL' if bad else 'ok  '} {name:<36} -inf mask {'same' if mask_ok else 'DIFF'}"
          f"  rel err {rel:.1e}  top-{TOPK} overlap min {worst:.4f}")


torch.manual_seed(0)
print("1. tiled Triton vs fp8_paged_mqa_logits_torch on the row-major copy")
for block_size in (64, 128):
    num_blocks = 2 * MAX_LEN // block_size
    rows, tiled = make_caches(num_blocks, block_size)
    for lens in ([1], [block_size], [block_size + 1], [5000],
                 [1, 15, 16, 17, 300, 2047, 8192, MAX_LEN], [377] * 32):
        q, w, cl, bt = make_batch(lens, block_size, num_blocks)
        want = fp8_paged_mqa_logits_torch(q, rows, w, cl, bt, MAX_LEN)
        got = fp8_paged_mqa_logits_triton(q, tiled, w, cl, bt, MAX_LEN, shuffle=True)
        label = f"bs={block_size} lens={lens if len(lens) < 9 else f'{len(lens)}x{lens[0]}'}"
        compare(label[:36], got, want, lens)
    q, w, cl, bt = make_batch([700, 3000], block_size, num_blocks)
    compare(f"bs={block_size} untiled (shuffle=False)",
            fp8_paged_mqa_logits_triton(q, rows, w, cl, bt, MAX_LEN, shuffle=False),
            fp8_paged_mqa_logits_torch(q, rows, w, cl, bt, MAX_LEN), [700, 3000])

print("2. CUDA graph: capture with lengths A, replay with lengths B")
block_size = 64
num_blocks = 2 * MAX_LEN // block_size
rows, tiled = make_caches(num_blocks, block_size)
lens_a, lens_b = [100, 2000, 7, 4096], [3000, 1, 650, 12000]
q, w, cl, bt = make_batch(lens_a, block_size, num_blocks)
for _ in range(2):
    fp8_paged_mqa_logits_triton(q, tiled, w, cl, bt, MAX_LEN, shuffle=True)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out = fp8_paged_mqa_logits_triton(q, tiled, w, cl, bt, MAX_LEN, shuffle=True)
cl.copy_(torch.tensor(lens_b, dtype=torch.int32, device=dev))
q.copy_(torch.randn_like(q, dtype=torch.float32).to(FP8))
graph.replay()
torch.cuda.synchronize()
compare("graph replay, new lengths", out.clone(),
        fp8_paged_mqa_logits_torch(q, rows, w, cl, bt, MAX_LEN), lens_b)

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


for length in (512, 8192):
    for batch in (1, 4, 16, 32):
        q, w, cl, bt = make_batch([length] * batch, block_size, num_blocks)
        t_torch = timeit(lambda: fp8_paged_mqa_logits_torch(q, rows, w, cl, bt, MAX_LEN))
        t_triton = timeit(lambda: fp8_paged_mqa_logits_triton(q, tiled, w, cl, bt, MAX_LEN, True))
        print(f"  len {length:>5} batch {batch:>2}: torch {t_torch:7.3f}  triton {t_triton:6.3f}"
              f"  ({t_torch / t_triton:5.1f}x)")

print("FAILED:", failures or "none")
