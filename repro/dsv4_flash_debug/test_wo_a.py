"""V4-Flash attention output projection wo_a on one GCD: the GEMM behind
rocm_inv_rope_einsum (torch.einsum("tgd,grd->tgr"), hipBLASLt; 43 us per call at 2 tokens in
the graph profile, job 23861347, 1.9 ms per step). At TP=8 a rank holds 1 of the 8 o_groups,
so it is [T, 4096] x [1024, 4096].T -> [T, 1024] bf16, with an 8 MB bf16 weight.

Candidates per token count, GPU us per call from graph replays over rotating weight copies
(>= 64 MB, so the weight comes from HBM as in serving), and max error vs fp64:
einsum (as served), F.linear, vLLM's ROCm unquantized dispatch (wvSplitK for <= 5 tokens,
else F.linear), and score_gemm's Triton kernel + cast to bf16 (stock tile, best tile).

Then: the DSV4_WO_A hook against the stock rocm_inv_rope_einsum (same inverse RoPE and
cached weight, T = 1-200), and score_mm's split-K sum (Triton kernel vs torch .sum(0)) at the
compressor and router shapes.

    sbatch repro/dsv4_flash_debug/run_wo_a_test.sh
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plugin"))
sys.path.insert(0, os.path.dirname(__file__))
import test_score_gemm as tsg  # noqa: E402
from lumi_dsv4_debug.score_gemm import pick, score_mm  # noqa: E402
from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl  # noqa: E402

dev = "cuda"
K, N = 4096, 1024
tsg.K = K
torch.manual_seed(0)
ws = [torch.randn(N, K, device=dev).bfloat16() * 0.02 for _ in range(8)]
print(torch.cuda.get_device_name())
best_table = []
SWEEP_MS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
for M in SWEEP_MS if os.environ.get("WO_A_SWEEP", "1") == "1" else ():  # 0: skip (job 23866833)
    x = torch.randn(M, K, device=dev).bfloat16()
    ref = x.double() @ ws[0].double().T
    scale = ref.abs().max().item()

    def err(fn):
        return (fn(ws[0]).double() - ref).abs().max().item() / scale

    cands = {
        "einsum": lambda w: torch.einsum("tgd,grd->tgr", x.view(M, 1, K), w.view(1, N, K)),
        "F.linear": lambda w: torch.nn.functional.linear(x, w),
        "vllm": lambda w: rocm_unquantized_gemm_impl(x, w),
        "triton": lambda w: score_mm(x, w).to(torch.bfloat16),
    }
    row = {k: (tsg.timed(f, ws), err(f)) for k, f in cands.items()}
    sweep = []
    for cfg in tsg.configs(M):
        try:
            sweep.append((tsg.timed(lambda w, c=cfg: score_mm(x, w, c).to(torch.bfloat16), ws), cfg))
        except Exception:  # out of resources
            torch.cuda.synchronize()
    sweep.sort()
    t_best, cfg_best = sweep[0]
    e_best = err(lambda w: score_mm(x, w, cfg_best).to(torch.bfloat16))
    best_table.append((M, cfg_best))
    print(f"M={M:4d} " + "  ".join(f"{k} {t:6.1f} us ({e:.0e})" for k, (t, e) in row.items())
          + f"  triton-best {t_best:6.1f} us {cfg_best} ({e_best:.0e})  [stock tile {pick(M, N)}]",
          flush=True)
print("best tiles:", best_table)

print("\nDSV4_WO_A hook vs the stock rocm_inv_rope_einsum (max |diff| / max |ref|)")
import types  # noqa: E402

import vllm.models.deepseek_v4.amd.rocm as r  # noqa: E402
from lumi_dsv4_debug import score_gemm, wo_a  # noqa: E402

stock_fn = r.rocm_inv_rope_einsum
wo_a.install()
hook = r.rocm_inv_rope_einsum
assert hook.stock is stock_fn
H, D, R = 8, 512, 64
rot = types.SimpleNamespace(cos_sin_cache=torch.randn(8192, R, device=dev))
mod = types.SimpleNamespace(weight=ws[0].clone())
fail = []
for T in (1, 2, 3, 4, 7, 16, 64, 128, 200):
    o = torch.randn(T, H, D, device=dev).bfloat16()
    pos = torch.randint(0, 8192, (T,), device=dev)
    ref = stock_fn(rot, o, pos, R, 1, N, mod).float()
    out = hook(rot, o, pos, R, 1, N, mod).float()
    e = ((out - ref).abs().max() / ref.abs().max()).item()
    ok = out.shape == ref.shape and e < 1e-2
    fail += [] if ok else [(T, e, tuple(out.shape))]
    print(f"  {'ok  ' if ok else 'FAIL'} T={T:3d} shape {tuple(out.shape)} err {e:.1e}")

print("\nsplit-K sum: Triton kernel vs torch .sum(0), us per score_mm call (and max |diff|)")
for N2 in (2048, 1024, 512, 256):
    w2 = [torch.randn(N2, K, device=dev).bfloat16() * 0.02 for _ in range(max(4, 64 * 2**20 // (N2 * K * 2)))]
    for M in (2, 8, 32, 64):
        x = torch.randn(M, K, device=dev).bfloat16()
        score_gemm.TORCH_SUM = True
        a = score_mm(x, w2[0])
        t_torch = tsg.timed(lambda w: score_mm(x, w), w2)
        score_gemm.TORCH_SUM = False
        b = score_mm(x, w2[0])
        t_tri = tsg.timed(lambda w: score_mm(x, w), w2)
        d = (a - b).abs().max().item()
        print(f"  N={N2:5d} M={M:3d} cfg {pick(M, N2)}  torch sum {t_torch:6.1f}  triton sum {t_tri:6.1f}  diff {d:.1e}")
print("FAILED:", fail or "none")
