"""Is mhc_fast.py (DSV4_MHC_FAST) right, and faster than the served mHC path, at decode sizes?

On one GCD with V4-Flash's real layer-5 hyper-connection parameters:
  1. accuracy of fused post+pre and pre against the torch references (mhc_post_torch, then
     mhc_pre_torch), next to the served path's own error (MHC_FIX=triton: TileLang post /
     prenorm GEMM / fused post+GEMM, Triton big fuse);
  2. one fused post+pre call in CUDA-graph replay, served path vs each prenorm variant
     (rows / tiled / fused) and big-fuse warps, per token count;
  3. the parts at the best variant: post, prenorm GEMM, big fuse, big fuse with one
     Sinkhorn iteration (how much of it is Sinkhorn latency).

    python test_mhc_fast.py <DeepSeek-V4-Flash snapshot dir>
"""

import json
import os
import pathlib
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, str(pathlib.Path(__file__).parent / "plugin"))
import vllm.model_executor.kernels.mhc.tilelang  # noqa: F401,E402  registers *_tilelang ops
import vllm.model_executor.kernels.mhc.tilelang_kernels as tk  # noqa: E402
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch  # noqa: E402

from lumi_dsv4_debug import mhc_fast, mhc_fix  # noqa: E402

snapshot = pathlib.Path(sys.argv[1])
config = json.loads((snapshot / "config.json").read_text())
HC, H = config["hc_mult"], config["hidden_size"]
ARGS = (config["rms_norm_eps"], config["hc_eps"], config["hc_eps"], 2.0, config["hc_sinkhorn_iters"])
dev = torch.device("cuda")
index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
tk.mhc_pre_big_fuse_tilelang = mhc_fix.mhc_pre_big_fuse_triton  # as served (MHC_FIX=triton)


def param(name: str) -> torch.Tensor:
    with safe_open(snapshot / index[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name).float().to(dev)


ffn = [param(f"layers.5.hc_ffn_{p}") for p in ("fn", "scale", "base")]
attn = [param(f"layers.5.hc_attn_{p}") for p in ("fn", "scale", "base")]
print(f"device {torch.cuda.get_device_properties(0).gcnArchName}  hc={HC} hidden={H}  "
      f"sinkhorn iters {ARGS[-1]}")

PRENORM = [("rows", 512, 256, 4), ("rows", 256, 256, 4), ("rows", 1024, 256, 4)]
PRENORM += [("tiled", bm, bk, ch, w) for bm, bk, ch, w in
            [(16, 64, 256, 4), (16, 64, 512, 4), (16, 128, 512, 4), (16, 128, 1024, 4),
             (32, 64, 512, 4), (32, 64, 256, 4), (64, 64, 512, 4)]]
PRENORM += [("fused", bm, ch, w) for bm, ch, w in
            [(16, 64, 4), (16, 128, 4), (16, 256, 4), (16, 128, 8), (32, 64, 4), (32, 128, 4),
             (32, 128, 8), (64, 64, 4), (64, 128, 8)]]
TOKENS = [int(t) for t in os.environ.get("MHC_TOKENS", "2 4 8 16 24 32 48 64 96 128").split()]
failures = []


def rel(got, want) -> float:
    got, want = got.float(), want.float()
    if got.shape != want.shape or not torch.isfinite(got).all():
        return float("inf")
    return ((got - want).norm() / want.norm().clamp_min(1e-12)).item()


def graph_us(fn, reps: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(7):
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1000 / reps)
    return best


def inputs(t: int):
    torch.manual_seed(t)
    residual = torch.randn(t, HC, H, dtype=torch.bfloat16, device=dev)
    x = torch.randn(t, H, dtype=torch.bfloat16, device=dev)
    post_mix, comb_mix, _ = mhc_pre_torch(residual, *attn, *ARGS)
    return x, residual, post_mix, comb_mix


def served(x, residual, post_mix, comb_mix):
    return torch.ops.vllm.mhc_fused_post_pre_tilelang(
        x, residual, post_mix, comb_mix, *ffn, *ARGS, 1, 1, None, 1e-6)


print("\n1. rel err vs torch references (residual_cur / post / comb / layer_input; FAIL > 2e-2)")
for t in (1, 2, 5, 16, 17, 64, 100, 128):
    x, residual, post_mix, comb_mix = inputs(t)
    cur = mhc_post_torch(x, residual, post_mix, comb_mix)
    want = (cur, *mhc_pre_torch(cur, *ffn, *ARGS))
    want_pre = mhc_pre_torch(residual, *ffn, *ARGS)
    print(f"  T={t:<4} served  " + " ".join(f"{rel(g, w):.1e}"
                                            for g, w in zip(served(x, residual, post_mix, comb_mix), want)))
    worst = {}
    for cfg in PRENORM:
        for bf in (1, 4):
            got = mhc_fast.fused_post_pre(x, residual, post_mix, comb_mix, *ffn, *ARGS,
                                          choice=(cfg, bf))
            errs = [rel(g, w) for g, w in zip(got, want)]
            errs += [rel(g, w) for g, w in zip(
                mhc_fast.pre(residual, *ffn, *ARGS, choice=(cfg, bf)), want_pre)]
            if max(errs) > 2e-2:
                failures.append((t, cfg, bf, errs))
            worst[cfg[0]] = [max(a, b) for a, b in zip(worst.get(cfg[0], [0] * 7), errs)]
    for kind, errs in worst.items():
        print(f"         {kind:<7} " + " ".join(f"{e:.1e}" for e in errs[:4])
              + "   pre " + " ".join(f"{e:.1e}" for e in errs[4:]) + "  (worst config)")

print("\n2. one fused post+pre call, graph replay, us (big fuse at 4 warps; then warps for the best)")
best_rows = []
for t in TOKENS:
    x, residual, post_mix, comb_mix = inputs(t)
    stock = graph_us(lambda: served(x, residual, post_mix, comb_mix))
    times = {}
    for cfg in PRENORM:
        if cfg[0] == "fused" and t < 8 and cfg[1] > 16:
            continue
        times[cfg] = graph_us(lambda: mhc_fast.fused_post_pre(
            x, residual, post_mix, comb_mix, *ffn, *ARGS, choice=(cfg, 4)))
    best = {k: min((v, c) for c, v in times.items() if c[0] == k) for k in ("rows", "tiled", "fused")}
    cfg = min(times, key=times.get)
    bf = {w: graph_us(lambda: mhc_fast.fused_post_pre(
        x, residual, post_mix, comb_mix, *ffn, *ARGS, choice=(cfg, w))) for w in (1, 2, 4, 8)}
    w = min(bf, key=bf.get)
    best_rows.append((t, cfg, w))
    print(f"  T={t:<4} served {stock:6.1f}  " + "  ".join(
        f"{k} {v:6.1f} {c[1:]}" for k, (v, c) in best.items())
        + f"  | big-fuse warps " + " ".join(f"{k}:{v:.1f}" for k, v in bf.items())
        + f"  -> {bf[w]:.1f} ({stock / bf[w]:.2f}x)")

print("\n3. parts at the best config, us")
for t, cfg, w in best_rows:
    x, residual, post_mix, comb_mix = inputs(t)
    cur = mhc_post_torch(x, residual, post_mix, comb_mix)
    flat = cur.view(t, HC * H)
    post_in = post_mix.float().reshape(t, HC).contiguous()
    comb_in = comb_mix.float().reshape(t, HC * HC).contiguous()
    out = torch.empty_like(cur)

    def post_only():
        mhc_fast._post_kernel[(t, H // mhc_fast.BLOCK_H)](
            x, residual, post_in, comb_in, out, x.stride(0), residual.stride(0),
            residual.stride(1), H=H, HC=HC, BLOCK=mhc_fast.BLOCK_H, num_warps=4)

    gcfg = cfg if cfg[0] != "fused" else ("tiled", cfg[1], 64, 512, 4)
    mix, sqr = mhc_fast._prenorm(flat, ffn[0], gcfg)
    p_post = graph_us(post_only)
    p_gemm = graph_us(lambda: mhc_fast._prenorm(flat, ffn[0], gcfg))
    p_tl = graph_us(lambda: torch.ops.vllm.mhc_pre_tilelang(cur, *ffn, *ARGS, 1, None, 1e-6))
    p_bf = graph_us(lambda: mhc_fast._big_fuse(mix, sqr, *ffn[1:], cur, H, *ARGS, w))
    a1 = (*ARGS[:-1], 1)
    p_bf1 = graph_us(lambda: mhc_fast._big_fuse(mix, sqr, *ffn[1:], cur, H, *a1, w))
    print(f"  T={t:<4} post {p_post:5.1f}  prenorm {gcfg[1:]} {p_gemm:5.1f}  "
          f"big fuse {p_bf:5.1f} (1 Sinkhorn iter {p_bf1:5.1f})  | served pre (TileLang GEMM + "
          f"big fuse) {p_tl:5.1f}")

print("\nCONFIGS = [")
for t, cfg, w in best_rows:
    print(f"    ({t}, {cfg!r}, {w}),")
print("]")
print("\nFAILED:", failures[:10] or "none")
