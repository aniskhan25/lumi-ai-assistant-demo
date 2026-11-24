"""lumi_dsv41/mhc_triton.py on one GCD against upstream's torch references
(vendor/.../kernels/mhc/torch.py: mhc_pre_delayed_torch, mhc_post_torch), V4.1's shapes
(hc 4, hidden 5120, 20 Sinkhorn iterations): the delayed pre with and without a carried
pre-mix, the layer-0 form (embedding x with the folded weights), the post; then time per
call.

    sbatch repro/deepseek_v41_gfx90a/run_mhc_triton_test.sh
"""

import importlib.util
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lumi_dsv41.mhc_triton import mhc_post_triton, mhc_pre_delayed_triton  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "mhc_ref", HERE / "vendor/vllm/model_executor/kernels/mhc/torch.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

HC, H, SK = 4, 5120, 20
HC3 = HC * (2 + HC)
EPS = (1e-6, 1e-6, 1e-6, 2.0)  # rms_eps, pre_eps, sinkhorn_eps, post_mult
dev = torch.device("cuda")
print(f"device {torch.cuda.get_device_properties(0).gcnArchName}")
failures = []


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def report(name, pairs, tol):
    errs = {k: rel(a, b) for k, (a, b) in pairs.items()}
    finite = all(torch.isfinite(a).all().item() for a, _ in pairs.values())
    bad = not finite or any(e > tol[k] for k, e in errs.items())
    if bad:
        failures.append(name)
    print(f"  {'FAIL' if bad else 'ok  '} {name:<34} "
          + "  ".join(f"{k} {e:.1e}" for k, e in errs.items()))


TOL = {"post": 1e-5, "comb": 1e-4, "input": 1e-2, "next_pre": 1e-5}
torch.manual_seed(0)
print("1. delayed pre vs mhc_pre_delayed_torch")
for t in (1, 5, 32, 300):
    residual = torch.randn(t, HC, H, device=dev).bfloat16()
    scale = torch.rand(3, device=dev) + 0.5
    base = torch.randn(HC3, device=dev) * 0.5
    for case in ("carried", "no carried", "layer0 x"):
        k = H if case == "layer0 x" else HC * H
        fn = torch.randn(HC3, k, device=dev) * 0.02
        pre_mix = torch.rand(t, HC, device=dev) + 0.1 if case == "carried" else None
        x = torch.randn(t, H, device=dev).bfloat16() if case == "layer0 x" else None
        args = (residual, fn, scale, base, *EPS[:3], EPS[3], SK)
        got = mhc_pre_delayed_triton(*args, pre_mix=pre_mix, x=x)
        want = ref.mhc_pre_delayed_torch(*args, pre_mix=pre_mix, x=x)
        assert [tuple(g.shape) for g in got] == [tuple(w.shape) for w in want], \
            ([g.shape for g in got], [w.shape for w in want])
        report(f"T={t} {case}", dict(zip(TOL, zip(got, want))), TOL)

print("2. post vs mhc_post_torch")
for t in (1, 5, 32, 300):
    x = torch.randn(t, H, device=dev).bfloat16()
    residual = torch.randn(t, HC, H, device=dev).bfloat16()
    post = torch.rand(t, HC, 1, device=dev) * 2
    comb = torch.rand(t, HC, HC, device=dev)
    report(f"T={t}", {"out": (mhc_post_triton(x, residual, post, comb),
                              ref.mhc_post_torch(x, residual, post, comb))}, {"out": 1e-2})

print("3. time per call (ms): torch reference vs Triton")


def timeit(fn, iters=20):
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


import itertools  # noqa: E402

import lumi_dsv41.mhc_triton as mt  # noqa: E402

print("4. prenorm GEMM paths inside the delayed pre (whole call, ms): per-token split-K, "
      "torch GEMM, tiled fp32-MFMA (best config)")
TILED = list(itertools.product((16, 32, 64), (32, 64, 128), (512, 1024, 2048, 4096), (4, 8)))
for t in [int(x) for x in __import__('os').environ.get('MHC_SWEEP_T', '16 32 96 192 384 768').split()]:
    residual = torch.randn(t, HC, H, device=dev).bfloat16()
    fn = torch.randn(HC3, HC * H, device=dev) * 0.02
    scale, base = torch.rand(3, device=dev) + 0.5, torch.randn(HC3, device=dev) * 0.5
    pre_mix = torch.rand(t, HC, device=dev) + 0.1
    args = (residual, fn, scale, base, *EPS[:3], EPS[3], SK)
    want = ref.mhc_pre_delayed_torch(*args, pre_mix=pre_mix)
    saved = (mt.GEMM_MAX_TOKENS, mt.TILED_MIN_TOKENS, mt.TILED_CFG)
    mt.TILED_MIN_TOKENS = None
    mt.GEMM_MAX_TOKENS = 1 << 20
    t_sk = timeit(lambda: mhc_pre_delayed_triton(*args, pre_mix=pre_mix))
    mt.GEMM_MAX_TOKENS = 0
    t_mm = timeit(lambda: mhc_pre_delayed_triton(*args, pre_mix=pre_mix))
    mt.TILED_MIN_TOKENS = 1
    times = {}
    for cfg in TILED:
        mt.TILED_CFG = cfg
        try:
            got = mhc_pre_delayed_triton(*args, pre_mix=pre_mix)
        except Exception as e:  # out of resources
            continue
        err = max(rel(x, y) for x, y in zip(got, want))
        if err > 1e-4:
            failures.append(f"tiled {cfg} T={t}: {err:.1e}")
            continue
        times[cfg] = timeit(lambda: mhc_pre_delayed_triton(*args, pre_mix=pre_mix))
    # layer-0 form (x = embedding, K = hidden), with the best config
    best = min(times, key=times.get)
    mt.TILED_CFG = best
    x0 = torch.randn(t, H, device=dev).bfloat16()
    fn0 = torch.randn(HC3, H, device=dev) * 0.02
    a0 = (residual, fn0, scale, base, *EPS[:3], EPS[3], SK)
    e0 = max(rel(x, y) for x, y in zip(mhc_pre_delayed_triton(*a0, x=x0),
                                       ref.mhc_pre_delayed_torch(*a0, x=x0)))
    if e0 > 1e-4:
        failures.append(f"tiled layer0 T={t}: {e0:.1e}")
    mt.GEMM_MAX_TOKENS, mt.TILED_MIN_TOKENS, mt.TILED_CFG = saved
    top = sorted(times, key=times.get)[:3]
    print(f"  T={t:4d}: split-K {t_sk:6.3f}  torch {t_mm:6.3f}  tiled {times[best]:6.3f} {best}  "
          + " ".join(f"{c}:{times[c]:.3f}" for c in top[1:]) + f"  (layer0 err {e0:.1e})")

print("5. gates: fused in the collapse's block-0 programs vs their own token-blocked kernel "
      "(whole delayed-pre call, ms; carried pre-mix)")
for t in (1, 16, 96, 384, 2048, 4096):
    residual = torch.randn(t, HC, H, device=dev).bfloat16()
    fn = torch.randn(HC3, HC * H, device=dev) * 0.02
    scale, base = torch.rand(3, device=dev) + 0.5, torch.randn(HC3, device=dev) * 0.5
    pre_mix = torch.rand(t, HC, device=dev) + 0.1
    args = (residual, fn, scale, base, *EPS[:3], EPS[3], SK)
    saved = (mt.GATES_MIN_TOKENS, mt.GATES_BLOCK_T)
    mt.GATES_MIN_TOKENS = None
    fused = mhc_pre_delayed_triton(*args, pre_mix=pre_mix)
    t_fused = timeit(lambda: mhc_pre_delayed_triton(*args, pre_mix=pre_mix))
    mt.GATES_MIN_TOKENS = 1
    times = {}
    for bt in (16, 32, 64, 128):
        mt.GATES_BLOCK_T = bt
        got = mhc_pre_delayed_triton(*args, pre_mix=pre_mix)
        diff = max((a.float() - b.float()).abs().max().item() for a, b in zip(got, fused))
        if diff > 1e-6:
            failures.append(f"split gates BLOCK_T={bt} T={t}: max |diff| {diff:.1e}")
        times[bt] = timeit(lambda: mhc_pre_delayed_triton(*args, pre_mix=pre_mix))
    mt.GATES_MIN_TOKENS, mt.GATES_BLOCK_T = saved
    best = min(times, key=times.get)
    print(f"  T={t:5d}: fused {t_fused:7.3f}  split " + "  ".join(
        f"{bt}:{times[bt]:.3f}" for bt in times) + f"  (best BLOCK_T {best}, "
        f"{t_fused / times[best]:.1f}x)", flush=True)

print("6. kernels inside one delayed-pre call at prefill sizes (torch profiler, us per call)")
from torch.profiler import ProfilerActivity, profile  # noqa: E402

for t in (2048, 4096):
    residual = torch.randn(t, HC, H, device=dev).bfloat16()
    fn = torch.randn(HC3, HC * H, device=dev) * 0.02
    scale, base = torch.rand(3, device=dev) + 0.5, torch.randn(HC3, device=dev) * 0.5
    pre_mix = torch.rand(t, HC, device=dev) + 0.1
    args = (residual, fn, scale, base, *EPS[:3], EPS[3], SK)
    for _ in range(3):
        mhc_pre_delayed_triton(*args, pre_mix=pre_mix)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            mhc_pre_delayed_triton(*args, pre_mix=pre_mix)
        torch.cuda.synchronize()
    rows = sorted(((e.key, e.device_time_total / 10) for e in prof.key_averages()
                   if e.device_time_total > 0), key=lambda r: -r[1])
    gb = t * HC * H * 2 / 1e9
    print(f"  T={t}: residual {gb * 1e3:.0f} MB; " + "; ".join(f"{k[:40]} {v:.0f}" for k, v in rows[:6]),
          flush=True)

print("7. collapse form (row loads vs [hc, BLOCK] tile), hidden block, warps (whole call, ms)")
for t in (16, 384, 2048):
    residual = torch.randn(t, HC, H, device=dev).bfloat16()
    fn = torch.randn(HC3, HC * H, device=dev) * 0.02
    scale, base = torch.rand(3, device=dev) + 0.5, torch.randn(HC3, device=dev) * 0.5
    pre_mix = torch.rand(t, HC, device=dev) + 0.1
    args = (residual, fn, scale, base, *EPS[:3], EPS[3], SK)
    want = ref.mhc_pre_delayed_torch(*args, pre_mix=pre_mix)
    saved = (mt.PRE_ROWS, mt.PRE_BLOCK, mt.PRE_WARPS)
    times = {}
    for rows, block, warps in itertools.product((False, True), (256, 512, 1024), (1, 2, 4, 8)):
        if H % block:
            continue
        mt.PRE_ROWS, mt.PRE_BLOCK, mt.PRE_WARPS = rows, block, warps
        got = mhc_pre_delayed_triton(*args, pre_mix=pre_mix)
        err = max(rel(x, y) for x, y in zip(got, want))
        if err > 1e-2:
            failures.append(f"collapse rows={rows} block={block} warps={warps} T={t}: {err:.1e}")
            continue
        times[(rows, block, warps)] = timeit(lambda: mhc_pre_delayed_triton(*args, pre_mix=pre_mix))
    mt.PRE_ROWS, mt.PRE_BLOCK, mt.PRE_WARPS = saved
    best = min(times, key=times.get)
    tile = times.get((False, 512, 4))
    print(f"  T={t:5d}: tile/512/4 (old) {tile:.3f}  best {times[best]:.3f} (rows, block, warps)="
          f"{best}  " + " ".join(f"{c}:{times[c]:.3f}" for c in sorted(times, key=times.get)[1:5]),
          flush=True)

print("8. post: row form vs [hc, hc, BLOCK] tile, hidden block, warps (ms per call)")
for t in (16, 384, 2048):
    x = torch.randn(t, H, device=dev).bfloat16()
    residual = torch.randn(t, HC, H, device=dev).bfloat16()
    post = torch.rand(t, HC, 1, device=dev) * 2
    comb = torch.rand(t, HC, HC, device=dev)
    want = ref.mhc_post_torch(x, residual, post, comb)
    saved = (mt.POST_ROWS, mt.POST_BLOCK, mt.POST_WARPS)
    times = {}
    for rows, block, warps in itertools.product((False, True), (256, 512, 1024), (1, 2, 4)):
        if H % block:
            continue
        mt.POST_ROWS, mt.POST_BLOCK, mt.POST_WARPS = rows, block, warps
        got = mhc_post_triton(x, residual, post, comb)
        err = rel(got, want)
        if err > 1e-2:
            failures.append(f"post rows={rows} block={block} warps={warps} T={t}: {err:.1e}")
            continue
        times[(rows, block, warps)] = timeit(lambda: mhc_post_triton(x, residual, post, comb))
    mt.POST_ROWS, mt.POST_BLOCK, mt.POST_WARPS = saved
    best = min(times, key=times.get)
    print(f"  T={t:5d}: tile/512/4 (old) {times.get((False, 512, 4)):.3f}  default "
          f"{times.get(saved):.3f}  best {times[best]:.3f} {best}", flush=True)

print("9. tiled prenorm GEMM: AMD launch options on (BLOCK_M, BLOCK_K, CHUNK, warps) "
      "(whole delayed-pre call, ms)")
for t in (96, 384, 2048):
    residual = torch.randn(t, HC, H, device=dev).bfloat16()
    fn = torch.randn(HC3, HC * H, device=dev) * 0.02
    scale, base = torch.rand(3, device=dev) + 0.5, torch.randn(HC3, device=dev) * 0.5
    pre_mix = torch.rand(t, HC, device=dev) + 0.1
    args = (residual, fn, scale, base, *EPS[:3], EPS[3], SK)
    want = ref.mhc_pre_delayed_torch(*args, pre_mix=pre_mix)
    saved = (mt.TILED_CFG, mt.TILED_KW, mt.TILED_MIN_TOKENS, mt.TILED_LARGE_MIN)
    mt.TILED_MIN_TOKENS, mt.TILED_LARGE_MIN = 1, None
    t0 = timeit(lambda: mhc_pre_delayed_triton(*args, pre_mix=pre_mix))
    times = {}
    for cfg, nk, we, kp in itertools.product(
            ((64, 64, 512, 4), (64, 32, 2048, 4), (32, 64, 1024, 4), (64, 64, 1024, 8),
             (128, 64, 2048, 8), (128, 32, 1024, 4)), (16, 32), (0, 1, 2), (1, 2)):
        mt.TILED_CFG, mt.TILED_KW = cfg, dict(matrix_instr_nonkdim=nk, waves_per_eu=we, kpack=kp)
        try:
            got = mhc_pre_delayed_triton(*args, pre_mix=pre_mix)
        except Exception:
            continue
        err = max(rel(x, y) for x, y in zip(got, want))
        if err > 1e-4:
            failures.append(f"tiled launch {cfg} nk{nk} we{we} kp{kp} T={t}: {err:.1e}")
            continue
        times[(cfg, nk, we, kp)] = timeit(lambda: mhc_pre_delayed_triton(*args, pre_mix=pre_mix))
    mt.TILED_CFG, mt.TILED_KW, mt.TILED_MIN_TOKENS, mt.TILED_LARGE_MIN = saved
    top = sorted(times, key=times.get)[:4]
    print(f"  T={t:5d}: served {t0:.3f}  best {times[top[0]]:.3f} {top[0]} ({t0 / times[top[0]]:.2f}x)  "
          + " ".join(f"{c}:{times[c]:.3f}" for c in top[1:]), flush=True)

print("FAILED:", failures or "none")
