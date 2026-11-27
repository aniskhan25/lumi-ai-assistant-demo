"""MoE at decode batch sizes (DSV4_MOE_SMALL=1) and the router GEMM (DSV4_GATE_GEMM=1).

Eager profile at one user (job 23782732, rank 0, per decode step over 43 MoE layers):
`pack_bitmatrix` 96 us per call (4.1 ms), the two expert GEMMs 93 + 69 us (7 ms), the
router GEMM 40 us (1.7 ms).

pack_bitmatrix  vLLM's `make_routing_data` (gpt_oss_triton_kernels_moe.py) launches it with
                BLOCK_SIZE_M = 512 whatever the token count, so one or two tokens run in a
                single workgroup holding [512, 32] tiles through bm_cols = 8 column passes.
                Row blocks of 16 give the same bitmatrix and are the fastest at every row
                count (test_moe_small.py, job 23811434, graph replay: 43.7 -> 3.3 us at 1-512
                rows, 43.6 -> 6.7 us at 8192).
expert GEMMs    triton_kernels' AMD opt flags pick block_n 256 for the gate/up GEMM (N 512 per
                rank) and split_k from a grid that counts every expert, so few workgroups do
                the work. Forced block_n / split_k per token count (test_moe_matmul.py, job
                23811580, 8 routings cycled so the weights are cold): gate/up 1.3-1.6x at 1-64
                tokens (37.6 -> 23.6 us at 1, 525 -> 390 us at 64), down 1.2-1.6x at 1-2.
                Then block_m / block_k and num_warps / num_stages too (test_moe_roofline.py,
                job 23908569): block_m 16 (a token or two per expert fill no more), 4 warps
                and one stage beat that table by 16-17 % on gate/up and 22-24 % on down at
                32-128 tokens. The FP4 -> bf16 conversion, not memory, bounds both: bf16
                weights move 4x the bytes at 1.0-1.3 TB/s against 0.5-0.7 TB/s for MXFP4.
                DSV4_MOE_TABLE=v1 keeps the earlier block_n / split_k table.
router GEMM     `GateLinear` (4096 -> 256 experts, out_dtype float32) has no ROCm tier and
                falls through to bf16 F.linear (hipBLASLt MT32x32 with fp32 MFMA) followed by
                a cast, so the logits are rounded to bf16 first. score_gemm.score_mm computes
                them with bf16 MFMA and fp32 accumulation, as the CUDA tiers do with cuBLAS.
"""

import dataclasses
import os

import torch
import triton


def pack_block_m(n_rows: int) -> int:
    return 16


# (max tokens, cfg); larger batches (prefill) keep the library's choice. cfg is
# (block_n, split_k), or (block_m, block_n, block_k, split_k, num_warps, num_stages).
GATE_UP_V1 = [(1, (32, 4)), (2, (128, 4)), (4, (128, 2)), (8, (128, 4)), (128, (128, 2))]
DOWN_V1 = [(2, (64, 1))]  # 4+ tokens: the default was already best
# job 23908569, us per call (8 cold routings, uniform experts): gate/up 37.5 -> 30.1 at 2
# tokens, 98.7 -> 84.9 at 8, 260 -> 220 at 32, 390 -> 326 at 64, 469 -> 391 at 128; down
# 27.8 -> 21.1, 64.3 -> 49.9, 163 -> 126, 235 -> 178, 293 -> 225.
GATE_UP_V2 = [(2, (16, 64, 128, 4, 4, 1)), (8, (16, 128, 256, 4, 4, 1)),
              (64, (16, 128, 256, 2, 4, 1)), (128, (16, 128, 256, 1, 8, 2))]
DOWN_V2 = [(2, (16, 64, 128, 1, 2, 1)), (32, (16, 128, 128, 1, 4, 1)),
           (128, (16, 256, 128, 1, 4, 1))]
# Prefill chunks, 256-2048 tokens (test_moe_roofline.py, job 24069742, all 256 experts hit),
# us per call default -> table: gate/up 712 -> 516 at 256 tokens, 718 -> 533 at 512, 766 ->
# 558 at 1024, 929 -> 828 at 2048; down 336 -> 297, 385 -> 334, 460 -> 402, 744 -> 632.
# Bit-identical. DSV4_MOE_PREFILL=0 leaves prefill on the library's tile.
if os.environ.get("DSV4_MOE_PREFILL", "1") == "1":
    GATE_UP_V2 += [(1024, (32, 128, 256, 1, 8, 2)), (2048, (64, 128, 128, 1, 2, 2))]
    DOWN_V2 += [(256, (32, 256, 128, 1, 4, 1)), (1024, (32, 128, 128, 1, 4, 1)),
                (2048, (64, 128, 128, 1, 4, 1))]
V1 = os.environ.get("DSV4_MOE_TABLE", "v2") == "v1"
GATE_UP = GATE_UP_V1 if V1 else GATE_UP_V2
DOWN = DOWN_V1 if V1 else DOWN_V2
TOPK = 6  # V4-Flash num_experts_per_tok: the down GEMM's rows are tokens * TOPK


class _PackBitmatrix:
    """Stands in for the `pack_bitmatrix` JIT function: `[grid](...)` ignores vLLM's grid
    and BLOCK_SIZE_M and launches with pack_block_m rows per program."""

    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        def launch(bitmatrix, topk_ids, n_rows, bm_cols, n_expts_act, BLOCK_SIZE_M, BLOCK_SIZE_K):
            bm = pack_block_m(n_rows)
            return self.kernel[(triton.cdiv(n_rows, bm),)](
                bitmatrix, topk_ids, n_rows, bm_cols, n_expts_act,
                BLOCK_SIZE_M=bm, BLOCK_SIZE_K=BLOCK_SIZE_K)
        return launch


def install_pack() -> None:
    import vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe as m

    if not isinstance(m.pack_bitmatrix, _PackBitmatrix):
        m.pack_bitmatrix = _PackBitmatrix(m.pack_bitmatrix)


def _pick_full(table, tokens):
    """(opt-flag constraints, overrides of the returned flags) or None."""
    for m_max, cfg in table:
        if tokens <= m_max:
            if len(cfg) == 2:
                return {"block_n": cfg[0], "split_k": cfg[1]}, {}
            bm, bn, bk, sk, warps, stages = cfg
            return ({"block_m": bm, "block_n": bn, "block_k": bk, "split_k": sk},
                    {"num_warps": warps, "num_stages": stages})
    return None


def _pick(table, tokens):
    got = _pick_full(table, tokens)
    return got[0] if got else None


def install_matmul() -> None:
    import importlib

    import vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe as m

    stock = m.matmul_ogs
    if getattr(stock, "_lumi", False):
        return
    pkg = stock.__module__.rsplit(".", 1)[0]  # vLLM's (aliased) triton_kernels
    of = importlib.import_module(pkg + ".matmul_ogs_details.opt_flags")
    # num_warps / num_stages are not constraints: override them on the flags the AMD
    # heuristic returns (make_opt_flags looks the function up at call time).
    over = {}
    amd = of.make_default_opt_flags_amd

    def make_default_opt_flags_amd(*a, **k):
        ret = amd(*a, **k)
        return dataclasses.replace(ret, **over) if over else ret

    of.make_default_opt_flags_amd = make_default_opt_flags_amd

    def matmul_ogs(x, w, bias, routing_data=None, gather_indx=None, scatter_indx=None, **kw):
        got = None
        if routing_data is not None and gather_indx is not None:  # gate/up: x is [tokens, K]
            got = _pick_full(GATE_UP, x.shape[0])
        elif routing_data is not None and scatter_indx is not None:  # down: x is [tokens*topk, I]
            got = _pick_full(DOWN, x.shape[0] // TOPK)
        if got is None:
            return stock(x, w, bias, routing_data, gather_indx, scatter_indx, **kw)
        saved = of._opt_flags_constraints
        of._opt_flags_constraints = {**saved, **got[0]}
        over.update(got[1])
        try:
            return stock(x, w, bias, routing_data, gather_indx, scatter_indx, **kw)
        finally:
            of._opt_flags_constraints = saved
            over.clear()

    matmul_ogs._lumi = True
    m.matmul_ogs = matmul_ogs


def install_gate() -> None:
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    from .score_gemm import pick, score_mm

    stock = GateLinear.forward

    def forward(self, x):
        w = self.weight
        if (self.out_dtype == torch.float32 and x.dtype == torch.bfloat16 and x.dim() == 2
                and w.dtype == torch.bfloat16 and pick(x.shape[0], w.shape[0]) is not None):
            return score_mm(x, w), None
        return stock(self, x)

    GateLinear.forward = forward
