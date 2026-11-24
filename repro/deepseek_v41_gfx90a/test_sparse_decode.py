"""lumi_dsv41/sparse_decode.py on one GCD against the gfx90a fallback in V4.1's vendored ops.

V4.1's per-rank decode shapes at TP=8: 8 heads, head dim 512 (448 FP8 + 64 bf16 in the
584-byte fp8_ds_mla record), SWA window 128 (main), top-512 compressed entries (extra) in
38 of 43 layers. Checks, then GPU time per call from graph replays:
  1. output vs the stock one-pass kernel: fixed and ragged lengths, invalid (-1) slots,
     NaN in unused cache bytes, with and without the attention sink;
  2. sweep of splits x BLOCK_K x num_warps per batch size, against the fallback and the
     split-K path (as gfx942).

    sbatch repro/deepseek_v41_gfx90a/run_sparse_decode_test.sh
"""

import itertools
import os

os.environ["DSV41_SPARSE_DECODE"] = "1"  # before the vendored module (and its hooks) loads

import torch  # noqa: E402

import vllm.v1.attention.ops.rocm_aiter_mla_sparse as m  # noqa: E402 (vendored, via sitecustomize)
from lumi_dsv41.sparse_decode import num_splits, sparse_decode  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
H, NOPE, ROPE, BS = 8, 448, 64, 64
HD = NOPE + ROPE
SCALE = HD**-0.5
shim = m._rocm_sparse_attn_decode_ragged_triton
assert shim.__name__ == "decode", "DSV41_SPARSE_DECODE hook not installed"
stock = next(c.cell_contents for c in shim.__closure__
             if getattr(c.cell_contents, "__name__", "") == "_rocm_sparse_attn_decode_ragged_triton")
print(f"device {torch.cuda.get_device_properties(0).gcnArchName}, CUs {m._decode_cu_count()}")


def make_cache(nblocks, nan_pad=False, nan_codes=False):
    rows = nblocks * BS
    nope = torch.randn(rows, NOPE).to(torch.float8_e4m3fn).view(torch.uint8)
    # every e4m3 code somewhere (subnormals; NaN 0x7f / 0xff only with nan_codes, which
    # the cache writer never produces and the stock kernel decodes as finite values)
    hit = torch.rand(rows, NOPE) < 0.02
    codes = torch.randint(0, 256, (int(hit.sum()),), dtype=torch.uint8)
    if not nan_codes:
        codes[(codes & 0x7F) == 0x7F] = 0x7E
    nope[hit] = codes
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


def ref(*args):
    m._ON_GFX942 = False
    return stock(*args)


def split_k(*args):
    m._ON_GFX942 = True
    try:
        return stock(*args)
    finally:
        m._ON_GFX942 = False


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


swa, comp = make_cache(256, nan_pad=True), make_cache(512, nan_pad=True)
sink = torch.randn(H, device=dev)
bad = []
print("1. accuracy vs the stock fallback (max |diff| / max |ref|)")
for B, extra, vary, use_sink in itertools.product((1, 7, 32, 128), (0, 512), (False, True), (True, False)):
    q = torch.randn(B, H, HD, device=dev).bfloat16()
    mi, mp = ragged(B, 128, swa.shape[0] * BS, vary=vary)
    args = [q, swa, mi, mp, SCALE, sink if use_sink else None, NOPE, ROPE]
    kw = {}
    if extra:
        ei, ep = ragged(B, extra, comp.shape[0] * BS, vary=vary)
        args += [comp, ei, ep]
        kw = dict(extra_cache=comp, extra_indices=ei, extra_indptr=ep)
    r = ref(*args).float()
    names = ["q", "main_cache", "main_indices", "main_indptr", "scale", "attn_sink",
             "nope_head_dim", "rope_head_dim", "extra_cache", "extra_indices", "extra_indptr"]
    via_hook = shim(**dict(zip(names, args))).float()  # keywords, as vLLM's caller passes
    direct = sparse_decode(q, swa, mi, mp, SCALE, sink if use_sink else None, **kw).float()
    if not torch.equal(via_hook, direct):
        bad.append((B, extra, vary, use_sink, "hook != direct"))
    for S in (1, 3, 8):
        o = sparse_decode(q, swa, mi, mp, SCALE, sink if use_sink else None, splits=S, **kw).float()
        err = ((o - r).abs().max() / r.abs().max().clamp_min(1e-6)).item()
        if not (err < 2e-2 and torch.isfinite(o).all().item()):
            bad.append((B, extra, vary, use_sink, S, err))
    print(f"  B={B:3d} extra={extra:3d} ragged={vary:d} sink={use_sink:d}: last err {err:.1e}")

# SD_SWEEP_B: DSpark verifies 6 tokens per sequence, so 16-128 users give 96-768 queries.
SWEEP_B = [int(x) for x in os.environ.get("SD_SWEEP_B", "1 4 8 16 24 32 48 64 96 128").split()]
print("2. GPU us per call (graph replay); config = (splits, BLOCK_K, warps, stages)")
CONFIGS = list(itertools.product((1, 2, 4, 8, 16), (16,), (4,), (1,)))
results = {}
for layer, extra in (("SWA", 0), ("top-k", 512)):
    for B in SWEEP_B:
        q = torch.randn(B, H, HD, device=dev).bfloat16()
        mi, mp = ragged(B, 128, swa.shape[0] * BS)
        args = [q, swa, mi, mp, SCALE, sink, NOPE, ROPE]
        kw = {}
        if extra:
            ei, ep = ragged(B, extra, comp.shape[0] * BS)
            args += [comp, ei, ep]
            kw = dict(extra_cache=comp, extra_indices=ei, extra_indptr=ep)
        t_ref = timed(lambda: ref(*args))
        t_split = timed(lambda: split_k(*args))
        t_auto = timed(lambda: sparse_decode(q, swa, mi, mp, SCALE, sink, **kw))
        times = {}
        for S, bk, nw, st in CONFIGS:
            try:
                times[(S, bk, nw, st)] = timed(lambda: sparse_decode(
                    q, swa, mi, mp, SCALE, sink, splits=S, block_k=bk, num_warps=nw,
                    num_stages=st, **kw))
            except Exception as e:  # out of resources for big tiles
                times[(S, bk, nw, st)] = float("inf")
        results[(layer, B)] = times
        cfg = min(times, key=times.get)
        top = sorted(times, key=times.get)[:4]
        print(f"{layer:<6} B={B:>4} fallback {t_ref:6.1f} split-K {t_split:6.1f} auto {t_auto:6.1f}"
              f" (S{num_splits(B, bool(extra))}) best {times[cfg]:6.1f} "
              + " ".join(f"{c}:{times[c]:.0f}" for c in top))
        if layer == "top-k" and B in (32, 64):
            c = cfg
            t_nodeq = timed(lambda: sparse_decode(q, swa, mi, mp, SCALE, sink, splits=c[0],
                            block_k=c[1], num_warps=c[2], num_stages=c[3], deq=1, **kw))
            print(f"         ablation at {c}: without the FP8 decode {t_nodeq:.1f} us")
print("per-split-count best (top-k): for each B, the best time with S fixed")
for B in SWEEP_B:
    t = results[("top-k", B)]
    row = []
    for S in (1, 2, 4, 8, 16):
        c = min((k for k in t if k[0] == S), key=t.get)
        row.append(f"S{S}:{t[c]:.0f}{c[1:]}")
    print(f"  B={B:>4} " + "  ".join(row))
print("3. bit-level FP8 decode (deq=2) vs the generic one (deq=0): equal outputs, us per call")
swa, comp = make_cache(256, nan_pad=True, nan_codes=True), make_cache(512, nan_pad=True, nan_codes=True)
for B, extra in itertools.product((1, 16, 32, 64, 128), (0, 512)):
    q = torch.randn(B, H, HD, device=dev).bfloat16()
    mi, mp = ragged(B, 128, swa.shape[0] * BS, vary=True)
    kw = {}
    if extra:
        ei, ep = ragged(B, extra, comp.shape[0] * BS, vary=True)
        kw = dict(extra_cache=comp, extra_indices=ei, extra_indptr=ep)
    a = sparse_decode(q, swa, mi, mp, SCALE, sink, deq=0, **kw)
    b = sparse_decode(q, swa, mi, mp, SCALE, sink, deq=2, **kw)
    same = torch.equal(a, b)
    if not same:
        bad.append((B, extra, "deq2 != deq0", ((a.float() - b.float()).abs().max()).item()))
    t0 = timed(lambda: sparse_decode(q, swa, mi, mp, SCALE, sink, deq=0, **kw))
    t2 = timed(lambda: sparse_decode(q, swa, mi, mp, SCALE, sink, deq=2, **kw))
    print(f"  {'top-k' if extra else 'SWA':<6} B={B:4d}: equal {same}  deq0 {t0:6.1f}  deq2 {t2:6.1f}")
print("4. AMD launch options on the auto split count (us per call; DSpark verify batches "
      "are 6 queries per user)")
for layer, extra in (("SWA", 0), ("top-k", 512)):
    for B in (6, 96, 192, 384):
        q = torch.randn(B, H, HD, device=dev).bfloat16()
        mi, mp = ragged(B, 128, swa.shape[0] * BS)
        kw = {}
        if extra:
            ei, ep = ragged(B, extra, comp.shape[0] * BS)
            kw = dict(extra_cache=comp, extra_indices=ei, extra_indptr=ep)
        base = sparse_decode(q, swa, mi, mp, SCALE, sink, **kw).float()
        t0 = timed(lambda: sparse_decode(q, swa, mi, mp, SCALE, sink, **kw))
        times = {}
        for bk, nw, nk, we, kp in itertools.product((16, 32), (2, 4, 8), (16, 32), (0, 1, 2, 4), (1, 2)):
            lk = dict(matrix_instr_nonkdim=nk, waves_per_eu=we, kpack=kp)
            try:
                got = sparse_decode(q, swa, mi, mp, SCALE, sink, block_k=bk, num_warps=nw,
                                    launch_kw=lk, **kw).float()
            except Exception:  # unsupported or out of resources
                continue
            err = ((got - base).abs().max() / base.abs().max()).item()
            if not err <= 2e-2:
                bad.append(("launch", layer, B, bk, nw, lk, err))
                continue
            times[(bk, nw, nk, we, kp)] = timed(lambda: sparse_decode(
                q, swa, mi, mp, SCALE, sink, block_k=bk, num_warps=nw, launch_kw=lk, **kw))
        top = sorted(times, key=times.get)[:5]
        print(f"  {layer:<6} B={B:>4}: served {t0:7.1f}  best {times[top[0]]:7.1f} "
              f"(BLOCK_K, warps, nonkdim, waves_per_eu, kpack)={top[0]} ({t0 / times[top[0]]:.2f}x)  "
              + " ".join(f"{c}:{times[c]:.0f}" for c in top[1:]), flush=True)
print("FAILED:", bad if bad else "none")
