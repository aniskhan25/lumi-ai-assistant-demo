"""Do the mHC pre fixes (plugin mhc_fix.py) make vLLM's fused TileLang path correct and fast?

On one GCD, with DeepSeek-V4-Flash's real layer-5 parameters, for each big-fuse kernel
  stock   vLLM's TileLang kernel (wrong layer_input on gfx90a, job 23509871)
  wave64  the same kernel at 128 threads split at 64
  triton  mhc_pre_big_fuse_triton
run vLLM's mhc_pre_tilelang and mhc_fused_post_pre_tilelang wrappers and compare every
output with the torch fallback that served V4 correctly. Then time one decoder layer's
hyper-connections, as served, in CUDA-graph replay:
  torch   2 x (mhc_pre_torch + mhc_post_torch)        (DSV4_TORCH_MHC=1, unfused layer)
  fused   2 x mhc_fused_post_pre_tilelang              (TileLang path, fused layer)

    python test_mhc_fix.py <DeepSeek-V4-Flash snapshot dir>
"""

import json
import pathlib
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, str(pathlib.Path(__file__).parent / "plugin"))
import vllm.model_executor.kernels.mhc.tilelang  # noqa: F401,E402  registers *_tilelang ops
import vllm.model_executor.kernels.mhc.tilelang_kernels as tk  # noqa: E402
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch  # noqa: E402

from lumi_dsv4_debug import mhc_fix  # noqa: E402

snapshot = pathlib.Path(sys.argv[1])
config = json.loads((snapshot / "config.json").read_text())
HC, H = config["hc_mult"], config["hidden_size"]
ARGS = (config["rms_norm_eps"], config["hc_eps"], config["hc_eps"], 2.0, config["hc_sinkhorn_iters"])
LAYERS = config["num_hidden_layers"]
dev = torch.device("cuda")
index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]


def param(name: str) -> torch.Tensor:
    with safe_open(snapshot / index[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name).float().to(dev)


attn = [param(f"layers.5.hc_attn_{p}") for p in ("fn", "scale", "base")]
ffn = [param(f"layers.5.hc_ffn_{p}") for p in ("fn", "scale", "base")]
print(f"device {torch.cuda.get_device_properties(0).gcnArchName}  hc={HC} hidden={H}  "
      f"sinkhorn iters {ARGS[-1]}  layers {LAYERS}")

VARIANTS = {"stock": tk.mhc_pre_big_fuse_tilelang,
            "wave64": mhc_fix.wave64_tilelang_module().mhc_pre_big_fuse_tilelang,
            "triton": mhc_fix.mhc_pre_big_fuse_triton}
failures = set()


def rel(got, want) -> float:
    got, want = got.float(), want.float()
    if got.shape != want.shape or not torch.isfinite(got).all():
        return float("inf")
    return ((got - want).norm() / want.norm().clamp_min(1e-12)).item()


def graph_us(fn, reps: int = 10) -> tuple[float, str]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                fn()
        run, how = g.replay, "graph"
    except Exception as e:  # noqa: BLE001
        torch.cuda.synchronize()
        print(f"    (graph capture failed: {type(e).__name__}: {str(e)[:100]}; timing eager)")
        run, how = (lambda: [fn() for _ in range(reps)]), "eager"
    run()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(5):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1000 / reps)
    return best, how


torch.manual_seed(0)
print("\ncorrectness vs torch fallback (rel err; FAIL > 2e-2)")
for tokens in (1, 4, 7, 16, 32, 64, 300, 2048):
    residual = torch.randn(tokens, HC, H, dtype=torch.bfloat16, device=dev)
    x = torch.randn(tokens, H, dtype=torch.bfloat16, device=dev)
    want_pre = mhc_pre_torch(residual, *attn, *ARGS)
    post_mix, res_mix, _ = want_pre
    new_res = mhc_post_torch(x, residual, post_mix, res_mix)
    want_fused = (new_res, *mhc_pre_torch(new_res, *ffn, *ARGS))
    for name, kernel in VARIANTS.items():
        tk.mhc_pre_big_fuse_tilelang = kernel
        got_pre = torch.ops.vllm.mhc_pre_tilelang(residual, *attn, *ARGS, 1, None, 1e-6)
        got_fused = torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x, residual, post_mix, res_mix, *ffn, *ARGS, 1, 1, None, 1e-6)
        errs = [rel(g, w) for g, w in zip(got_pre, want_pre)]
        errs += [rel(g, w) for g, w in zip(got_fused, want_fused)]
        bad = max(errs) > 2e-2
        if bad:
            failures.add(name)
        print(f"  {'FAIL' if bad else 'ok  '} T={tokens:<5} {name:<7} pre post/res/input "
              + " ".join(f"{e:.1e}" for e in errs[:3]) + "   fused residual/post/res/input "
              + " ".join(f"{e:.1e}" for e in errs[3:]))

print(f"\none decoder layer's hyper-connections (us); per decode step = x{LAYERS} layers")
print(f"  {'T':>4} {'torch':>14} " + " ".join(f"{'fused ' + n:>16}" for n in VARIANTS))
for tokens in (1, 4, 16, 32, 64):
    residual = torch.randn(tokens, HC, H, dtype=torch.bfloat16, device=dev)
    x = torch.randn(tokens, H, dtype=torch.bfloat16, device=dev)
    post_mix, res_mix, _ = mhc_pre_torch(residual, *attn, *ARGS)

    def torch_layer():
        for p in (attn, ffn):
            post, comb, _ = mhc_pre_torch(residual, *p, *ARGS)
            mhc_post_torch(x, residual, post, comb)

    def fused_layer():
        for p in (attn, ffn):
            torch.ops.vllm.mhc_fused_post_pre_tilelang(
                x, residual, post_mix, res_mix, *p, *ARGS, 1, 1, None, 1e-6)

    times = {"torch": graph_us(torch_layer)}
    for name, kernel in VARIANTS.items():
        tk.mhc_pre_big_fuse_tilelang = kernel
        times[name] = graph_us(fused_layer)
    print(f"  {tokens:>4} " + " ".join(f"{t:9.1f} {how:>5}" for t, how in times.values())
          + "   per step ms: " + " ".join(f"{t * LAYERS / 1e3:.1f}" for t, _ in times.values()))

print("\nWRONG:", sorted(failures) or "none")
