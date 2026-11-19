"""V4-Flash attention output projection wo_a on one GCD: the GEMM behind
rocm_inv_rope_einsum (torch.einsum("tgd,grd->tgr"), hipBLASLt; 43 us per call at 2 tokens in
the graph profile, job 23861347, 1.9 ms per step). At TP=8 a rank holds 1 of the 8 o_groups,
so it is [T, 4096] x [1024, 4096].T -> [T, 1024] bf16, with an 8 MB bf16 weight.

Candidates per token count, GPU us per call from graph replays over rotating weight copies
(>= 64 MB, so the weight comes from HBM as in serving), and max error vs fp64:
einsum (as served), F.linear, vLLM's ROCm unquantized dispatch (wvSplitK for <= 5 tokens,
else F.linear), and score_gemm's Triton kernel + cast to bf16 (stock tile, best tile).

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
for M in (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128):
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
