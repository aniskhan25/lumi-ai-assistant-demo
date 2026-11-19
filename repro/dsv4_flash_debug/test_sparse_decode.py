"""plugin sparse_decode.py (the V4.1 port's kernel) on one GCD against stock vLLM 0.26's
gfx90a fallback, at V4-Flash's per-rank decode shapes (TP=8: 8 heads, head dim 512 = 448 FP8
+ 64 bf16 in the 584-byte fp8_ds_mla record; SWA window 128 as main; top-512 (ratio 4) or up
to ~256 (ratio 128 at 32k) compressed entries as extra).

  1. output vs the fallback: fixed and ragged lengths, invalid (-1) slots, NaN in unused cache
     bytes, with and without the sink; through the installed hook with keywords, as vLLM calls;
  2. GPU us per call from graph replays: fallback, the split-K path SPARSE_SPLIT uses, and the
     kernel, per layer type and batch (MTP doubles the queries: 1 user = 2).

    sbatch repro/dsv4_flash_debug/run_sparse_decode_test.sh
"""

import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plugin"))
import vllm.v1.attention.ops.rocm_aiter_mla_sparse as m  # noqa: E402
from lumi_dsv4_debug import sparse_decode as sd  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
H, NOPE, ROPE, BS = 8, 448, 64, 64
HD = NOPE + ROPE
SCALE = HD**-0.5
stock = m._rocm_sparse_attn_decode_ragged_triton
sd.install()
hook = m._rocm_sparse_attn_decode_ragged_triton
assert hook is not stock and hook.stock is stock
print(f"device {torch.cuda.get_device_properties(0).gcnArchName}")


def make_cache(nblocks, nan_pad=False):
    rows = nblocks * BS
    nope = torch.randn(rows, NOPE).to(torch.float8_e4m3fn).view(torch.uint8)
    rope = torch.randn(rows, ROPE).bfloat16().view(torch.uint8)
    scales = torch.randint(122, 128, (rows, 8), dtype=torch.uint8)
    data = torch.cat([nope, rope], 1).view(nblocks, BS * 576)
    cache = torch.cat([data, scales.view(nblocks, BS * 8)], 1).view(nblocks, BS, 584).to(dev)
    if nan_pad:  # a NaN bf16 in rows nobody points at: must not leak through masked loads
        cache.view(-1)[-584:-584 + 576].view(torch.bfloat16)[:] = float("nan")
    return cache


def ragged(B, length, rows, invalid=0.05, vary=False):
    lens = (torch.randint(1, length + 1, (B,)) if vary else torch.full((B,), length)).tolist()
    idx = [torch.randperm(rows - BS)[:n].int() for n in lens]  # last block left unused
    for t in idx:
        t[torch.rand(len(t)) < invalid] = -1
    indptr = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int32)
    return torch.cat(idx).to(dev), indptr.to(dev)


def split_k(**kw):
    m._ON_GFX950 = True
    try:
        return stock(**kw)
    finally:
        m._ON_GFX950 = False


def timed(fn, iters=30):
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
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    for _ in range(3):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / (3 * iters) * 1e3


def call_kw(q, mi, mp, sink, extra):
    kw = dict(q=q, main_cache=swa, main_indices=mi, main_indptr=mp, scale=SCALE,
              attn_sink=sink, nope_head_dim=NOPE, rope_head_dim=ROPE)
    if extra:
        ei, ep = ragged(q.shape[0], extra, comp.shape[0] * BS)
        kw.update(extra_cache=comp, extra_indices=ei, extra_indptr=ep)
    return kw


swa, comp = make_cache(256, nan_pad=True), make_cache(512, nan_pad=True)
sink = torch.randn(H, device=dev)
bad = []
print("1. accuracy vs the stock fallback (max |diff| / max |ref|)")
for B, extra, vary, use_sink in itertools.product((1, 2, 7, 64, 128), (0, 64, 512),
                                                  (False, True), (True, False)):
    q = torch.randn(B, H, HD, device=dev).bfloat16()
    mi, mp = ragged(B, 128, swa.shape[0] * BS, vary=vary)
    kw = call_kw(q, mi, mp, sink if use_sink else None, 0)
    if extra:
        ei, ep = ragged(B, extra, comp.shape[0] * BS, vary=vary)
        kw.update(extra_cache=comp, extra_indices=ei, extra_indptr=ep)
    r = stock(**kw).float()
    o = hook(**kw).float()
    err = ((o - r).abs().max() / r.abs().max().clamp_min(1e-6)).item()
    ok = err < 2e-2 and torch.isfinite(o).all().item()
    if not ok:
        bad.append((B, extra, vary, use_sink, err))
    print(f"  {'ok  ' if ok else 'FAIL'} B={B:3d} extra={extra:3d} ragged={vary:d} sink={use_sink:d}: err {err:.1e}")

print("2. GPU us per call (graph replay)")
print(f"{'layer':<12}{'queries':>8}{'fallback':>10}{'split-K':>10}{'kernel':>10}")
for layer, extra in (("SWA", 0), ("ratio 128", 64), ("ratio 4", 512)):
    for B in (2, 4, 8, 16, 32, 64, 128):
        q = torch.randn(B, H, HD, device=dev).bfloat16()
        mi, mp = ragged(B, 128, swa.shape[0] * BS)
        kw = call_kw(q, mi, mp, sink, extra)
        t = [timed(lambda: stock(**kw)), timed(lambda: split_k(**kw)), timed(lambda: hook(**kw))]
        print(f"{layer:<12}{B:>8}" + "".join(f"{x:10.1f}" for x in t))
print("FAILED:", bad if bad else "none")
