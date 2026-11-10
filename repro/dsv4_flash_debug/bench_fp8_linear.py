"""DeepSeek-V4-Flash's dense FP8 linears on one MI250X GCD: stock vs tuned vs bf16.

The decode profile (job 23530527) put `_w8a8_triton_block_scaled_mm` at ~89 ms of a ~182 ms
step. vLLM ships no MI250X config, so it runs BLOCK_M=64, BLOCK_N=128, BLOCK_K=128: 4-64
programs for these shapes on 110 CUs. For the five shapes vLLM warned about, per batch size M:

  stock  per-token-group activation quant + w8a8_triton_block_scaled_mm, default config
         (as served, including the per-call E8M0 -> fp32 weight-scale upcast)
  tuned  the same, with the best of a config sweep
  bf16   weight dequantised once (exact: e4m3 x power-of-two scale), vLLM's ROCm bf16 GEMM
         dispatch (rocm_unquantized_gemm_impl: wvSplitK for M <= 5, else hipBLASLt)

Times are CUDA-graph replays (the linears run inside graphs when served). Accuracy is the
relative error against an fp32 product of the unquantised activations and dequantised
weights. Tuned configs are written as vLLM config files to OUT_DIR.

    python bench_fp8_linear.py OUT_DIR
"""

import itertools
import json
import pathlib
import sys

import torch

import vllm.model_executor.layers.quantization.utils.fp8_utils as fu
from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
from vllm.platforms import current_platform

SHAPES = [(1536, 4096), (4096, 1024), (4096, 256), (512, 4096), (8192, 1024)]  # (N, K)
MS = [1, 2, 4, 8, 16, 32, 64]
BLOCK = [128, 128]
FP8 = current_platform.fp8_dtype()
out_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "fp8_configs")
out_dir.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda")
torch.manual_seed(0)
stock_lookup = fu.get_w8a8_block_fp8_configs


def graph_time_us(fn, reps: int = 20, rounds: int = 5) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1000 / reps)
    return best


def w8a8(x, w, ws, config):
    """The served path: quantise activations per 128-group, then the block-scaled kernel."""
    fu.get_w8a8_block_fp8_configs = (lambda *a: {1: config}) if config else (lambda *a: None)
    xq, xs = fu.per_token_group_quant_fp8(x, BLOCK[1])
    return fu.w8a8_triton_block_scaled_mm(xq, w, xs, ws, BLOCK, torch.bfloat16)


def rel_err(out, ref) -> float:
    return ((out.float() - ref).norm() / ref.norm()).item()


CANDIDATES = [
    dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=1, num_warps=nw, num_stages=ns)
    for bm, bn, bk, nw, ns in itertools.product((16, 32, 64), (16, 32, 64, 128), (64, 128), (2, 4, 8), (1, 2))
]
print(f"device {torch.cuda.get_device_properties(0).gcnArchName}  fp8 {FP8}  "
      f"{len(CANDIDATES)} configs per (shape, M)")
totals = {"stock": 0.0, "tuned": 0.0, "bf16": 0.0}
for n, k in SHAPES:
    w = (torch.randn(n, k, device=dev) * 0.5).to(FP8)
    exps = torch.randint(117, 124, (n // BLOCK[0], k // BLOCK[1]), device=dev, dtype=torch.uint8)
    ws = exps.view(torch.float8_e8m0fnu)  # 2^-10 .. 2^-4, as V4's ue8m0 weight scales
    ws32 = torch.exp2(exps.float() - 127)
    w_bf16 = (w.float() * ws32.repeat_interleave(BLOCK[0], 0).repeat_interleave(BLOCK[1], 1)).to(torch.bfloat16)
    tuned_file = {}
    print(f"\nN={n} K={k}")
    print(f"  {'M':>3} {'stock us':>9} {'tuned us':>9} {'bf16 us':>8}  {'err stock':>9} {'err tuned':>9} "
          f"{'err bf16':>8}  best config")
    for m in MS:
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        ref = x.float() @ w_bf16.float().T
        t_stock = graph_time_us(lambda: w8a8(x, w, ws, None))
        e_stock = rel_err(w8a8(x, w, ws, None), ref)
        best, best_cfg = float("inf"), None
        for cfg in CANDIDATES:
            try:
                t = graph_time_us(lambda: w8a8(x, w, ws, cfg), reps=10, rounds=2)
            except Exception:  # resources exceeded for this config
                torch.cuda.synchronize()
                continue
            if t < best:
                best, best_cfg = t, cfg
        best = graph_time_us(lambda: w8a8(x, w, ws, best_cfg))
        e_tuned = rel_err(w8a8(x, w, ws, best_cfg), ref)
        t_bf16 = graph_time_us(lambda: rocm_unquantized_gemm_impl(x, w_bf16))
        e_bf16 = rel_err(rocm_unquantized_gemm_impl(x, w_bf16), ref)
        tuned_file[str(m)] = best_cfg
        if m == 1:
            totals["stock"] += t_stock
            totals["tuned"] += best
            totals["bf16"] += t_bf16
        cfg_s = "{BLOCK_SIZE_M}x{BLOCK_SIZE_N}x{BLOCK_SIZE_K} w{num_warps} s{num_stages}".format(**best_cfg)
        print(f"  {m:>3} {t_stock:9.1f} {best:9.1f} {t_bf16:8.1f}  {e_stock:9.2e} {e_tuned:9.2e} "
              f"{e_bf16:8.2e}  {cfg_s}", flush=True)
    name = (f"N={n},K={k},device_name={fu.get_device_name_as_file_name()},"
            f"dtype=fp8_w8a8,block_shape=[{BLOCK[0]},{BLOCK[1]}].json")
    (out_dir / name).write_text(json.dumps(tuned_file, indent=2))
fu.get_w8a8_block_fp8_configs = stock_lookup
print(f"\nM=1, one call per shape: stock {totals['stock']:.0f} us, tuned {totals['tuned']:.0f} us, "
      f"bf16 {totals['bf16']:.0f} us")
print(f"config files written to {out_dir} for device_name={fu.get_device_name_as_file_name()}")
