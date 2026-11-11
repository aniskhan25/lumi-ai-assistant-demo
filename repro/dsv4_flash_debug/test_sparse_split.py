"""Split-K sparse attention decode vs the gfx90a fallback, on one GCD.

vLLM 0.26 gates its split-K decode kernels (partial + reduce) to gfx942/gfx950; gfx90a gets
a one-pass kernel with one workgroup per query and head block (sparse_split.py). This runs
both on V4-Flash's per-rank shapes at TP=8 (8 heads, head_dim 512 = 448 FP8 nope + 64 bf16
rope, SWA window 128, top-k 512) with fp8_ds_mla caches, compares the outputs and times them.
Also checks that the plugin shim gives the split result up to MAX_QUERIES and the
fallback result above it.
"""

import os
import sys

import torch

import vllm.v1.attention.ops.rocm_aiter_mla_sparse as m

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plugin"))
from lumi_dsv4_debug import sparse_split  # noqa: E402

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


def run(gate, *args):
    saved = m._ON_GFX950
    m._ON_GFX950 = gate
    try:
        return m._rocm_sparse_attn_decode_ragged_triton(*args)
    finally:
        m._ON_GFX950 = saved


def timed(fn, iters=50):
    """GPU time per call in us, from replays of a graph holding `iters` calls (serving
    replays graphs, so eager host overhead would bias toward the one-launch fallback)."""
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


def forced(splits, *args):
    """Split path with a fixed split count instead of the gfx950-tuned heuristic."""
    saved = m._decode_num_splits
    m._decode_num_splits = lambda *a, **k: splits
    try:
        return run(True, *args)
    finally:
        m._decode_num_splits = saved


swa = make_cache(256)
comp = make_cache(512)
sink = torch.randn(H, device=dev)
orig = m._rocm_sparse_attn_decode_ragged_triton
sparse_split.install()
shim = m._rocm_sparse_attn_decode_ragged_triton
m._rocm_sparse_attn_decode_ragged_triton = orig

SPLITS = (1, 2, 4, 8, 16)
print(f"GPU us per call (graph replay); sN = split path forced to N splits; CUs={m._decode_cu_count()}")
print(f"{'layer':<16} {'B':>4} {'rel err':>8} {'fallback':>9} {'auto':>11}"
      + "".join(f"{'s' + str(n):>8}" for n in SPLITS))
ok = True
for layer, extra_len in (("SWA only", 0), ("ratio 4, top-k", 512), ("ratio 128", 16)):
    for B in (1, 4, 16, 32, 64, 128):
        q = torch.randn(B, H, HD, device=dev).bfloat16()
        mi, mp = ragged(B, 128, swa.shape[0] * BS)
        args = [q, swa, mi, mp, SCALE, sink, NOPE, ROPE]
        if extra_len:
            ei, ep = ragged(B, extra_len, comp.shape[0] * BS)
            args += [comp, ei, ep]
        ref = run(False, *args).float()
        out = run(True, *args).float()
        m._rocm_sparse_attn_decode_ragged_triton = shim
        via_shim = m._rocm_sparse_attn_decode_ragged_triton(*args).float()
        m._rocm_sparse_attn_decode_ragged_triton = orig
        err = ((out - ref).abs().max() / ref.abs().max()).item()
        want = out if B <= sparse_split.MAX_QUERIES else ref  # shim splits small batches only
        good = err < 2e-2 and torch.isfinite(out).all().item() and torch.equal(want, via_shim)
        for n in SPLITS:
            e = ((forced(n, *args).float() - ref).abs().max() / ref.abs().max()).item()
            good &= e < 2e-2
        ok &= good
        auto = m._decode_num_splits(B, 1, 128, extra_len, 32)
        t_ref = timed(lambda: run(False, *args))
        t_auto = timed(lambda: run(True, *args))
        ts = [timed(lambda: forced(n, *args)) for n in SPLITS]
        print(f"{layer:<16} {B:>4} {err:>8.1e} {t_ref:>9.1f} {t_auto:>7.1f}/s{auto:<2}"
              + "".join(f"{t:>8.1f}" for t in ts) + ("" if good else "  MISMATCH"))
print("PASS" if ok else "FAIL")
