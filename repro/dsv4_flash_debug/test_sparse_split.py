"""Split-K sparse attention decode vs the gfx90a fallback, on one GCD.

vLLM 0.26 gates its split-K decode kernels (partial + reduce) to gfx942/gfx950; gfx90a gets
a one-pass kernel with one workgroup per query and head block (sparse_split.py). This runs
both on V4-Flash's per-rank shapes at TP=8 (8 heads, head_dim 512 = 448 FP8 nope + 64 bf16
rope, SWA window 128, top-k 512) with fp8_ds_mla caches, compares the outputs and times them.
Also checks that the plugin shim gives the split result.
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
    for _ in range(3):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3


swa = make_cache(256)
comp = make_cache(512)
sink = torch.randn(H, device=dev)
orig = m._rocm_sparse_attn_decode_ragged_triton
sparse_split.install()
shim = m._rocm_sparse_attn_decode_ragged_triton
m._rocm_sparse_attn_decode_ragged_triton = orig

print(f"{'layer':<18} {'B':>4} {'max rel err':>12} {'fallback us':>12} {'split us':>9} {'speedup':>8}")
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
        good = err < 2e-2 and torch.isfinite(out).all().item() and torch.equal(out, via_shim)
        ok &= good
        t_ref = timed(lambda: run(False, *args))
        t_out = timed(lambda: run(True, *args))
        print(f"{layer:<18} {B:>4} {err:>12.2e} {t_ref:>12.1f} {t_out:>9.1f} "
              f"{t_ref / t_out:>7.1f}x{'' if good else '  MISMATCH'}")
print("PASS" if ok else "FAIL")
