"""DSV41_SPARSE_SPLIT (hooks.py:_sparse_split) on one GCD: V4.1's vendored split-K sparse
decode, taken as gfx942, against the gfx90a one-pass fallback.

V4.1's per-rank shapes at TP=8 are V4's (8 of 64 heads, head_dim 512 = 448 FP8 nope + 64
bf16 rope in the 584-byte fp8_ds_mla record, SWA window 128, top-k 512), so this is
../dsv4_flash_debug/test_sparse_split.py on the vendored module: outputs compared, the
hook checked to split up to 16 queries and fall back above, GPU time per call from graph
replays.

    sbatch repro/deepseek_v41_gfx90a/run_sparse_split_test.sh
"""

import os

os.environ["DSV41_SPARSE_SPLIT"] = "1"  # before the vendored module (and its hooks) loads

import torch  # noqa: E402

import vllm.v1.attention.ops.rocm_aiter_mla_sparse as m  # noqa: E402
from lumi_dsv41.hooks import SPARSE_SPLIT_MAX_QUERIES  # noqa: E402

shim = m._rocm_sparse_attn_decode_ragged_triton
assert shim.__name__ == "split_decode", "DSV41_SPARSE_SPLIT hook not installed"
stock = next(c.cell_contents for c in shim.__closure__
             if getattr(c.cell_contents, "__name__", "") == "_rocm_sparse_attn_decode_ragged_triton")
print(f"vendored module {m.__file__}  CUs={m._decode_cu_count()}")

torch.manual_seed(0)
dev = "cuda"
H, NOPE, ROPE, BS = 8, 448, 64, 64
HD = NOPE + ROPE
SCALE = HD**-0.5


def make_cache(nblocks):
    """fp8_ds_mla blocks: per token 448 FP8 bytes + 64 bf16, then 8 exponent bytes."""
    rows = nblocks * BS
    nope = torch.randn(rows, NOPE).to(torch.float8_e4m3fn).view(torch.uint8)
    rope = torch.randn(rows, ROPE).bfloat16().view(torch.uint8)
    scales = torch.randint(122, 128, (rows, 8), dtype=torch.uint8)
    data = torch.cat([nope, rope], 1).view(nblocks, BS * 576)
    return torch.cat([data, scales.view(nblocks, BS * 8)], 1).view(nblocks, BS, 584).to(dev)


def ragged(B, length, rows, invalid=0.05):
    idx = torch.stack([torch.randperm(rows)[:length] for _ in range(B)]).int()
    idx[torch.rand(B, length) < invalid] = -1  # masked slots occur in served batches
    indptr = torch.arange(B + 1, dtype=torch.int32) * length
    return idx.flatten().to(dev), indptr.to(dev)


def run(split, *args):
    m._ON_GFX942 = split
    try:
        return stock(*args)
    finally:
        m._ON_GFX942 = False


def timed(fn, iters=50):
    """GPU us per call from replays of a graph holding `iters` calls."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(5):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (5 * iters) * 1e3


swa = make_cache(256)
comp = make_cache(512)
sink = torch.randn(H, device=dev)
print(f"{'layer':<16} {'B':>4} {'rel err':>8} {'fallback us':>12} {'split us':>9} {'splits':>7}")
ok = True
for layer, extra_len in (("SWA only", 0), ("ratio 4, top-k", 512), ("ratio 128", 16)):
    for B in (1, 4, 8, 16, 32, 64):
        q = torch.randn(B, H, HD, device=dev).bfloat16()
        mi, mp = ragged(B, 128, swa.shape[0] * BS)
        args = [q, swa, mi, mp, SCALE, sink, NOPE, ROPE]
        if extra_len:
            ei, ep = ragged(B, extra_len, comp.shape[0] * BS)
            args += [comp, ei, ep]
        ref = run(False, *args).float()
        out = run(True, *args).float()
        via_shim = shim(*args).float()
        err = ((out - ref).abs().max() / ref.abs().max()).item()
        want = out if B <= SPARSE_SPLIT_MAX_QUERIES else ref
        good = (err < 2e-2 and torch.isfinite(out).all().item() and torch.equal(want, via_shim)
                and not m._ON_GFX942)
        ok &= good
        splits = m._decode_num_splits(B, 1, 128, extra_len, 32)
        t_ref = timed(lambda: run(False, *args))
        t_split = timed(lambda: run(True, *args))
        print(f"{layer:<16} {B:>4} {err:>8.1e} {t_ref:>12.1f} {t_split:>9.1f} {splits:>7}"
              + ("" if good else "  MISMATCH"))
print("PASS" if ok else "FAIL")
