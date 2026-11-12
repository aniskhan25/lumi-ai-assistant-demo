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


for t in (1, 32):
    residual = torch.randn(t, HC, H, device=dev).bfloat16()
    fn = torch.randn(HC3, HC * H, device=dev) * 0.02
    scale, base = torch.rand(3, device=dev) + 0.5, torch.randn(HC3, device=dev) * 0.5
    pre_mix = torch.rand(t, HC, device=dev) + 0.1
    args = (residual, fn, scale, base, *EPS[:3], EPS[3], SK)
    a = timeit(lambda: ref.mhc_pre_delayed_torch(*args, pre_mix=pre_mix))
    b = timeit(lambda: mhc_pre_delayed_triton(*args, pre_mix=pre_mix))
    x = torch.randn(t, H, device=dev).bfloat16()
    post, comb = torch.rand(t, HC, 1, device=dev), torch.rand(t, HC, HC, device=dev)
    c = timeit(lambda: ref.mhc_post_torch(x, residual, post, comb))
    d = timeit(lambda: mhc_post_triton(x, residual, post, comb))
    print(f"  T={t:>2}: pre torch {a:6.3f}  triton {b:6.3f}   post torch {c:6.3f}  triton {d:6.3f}")

print("FAILED:", failures or "none")
