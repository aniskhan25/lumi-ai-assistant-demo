"""MoE routing at decode batch sizes (DSV4_MOE_SMALL=1) and the router GEMM (DSV4_GATE_GEMM=1).

Eager profile at one user (job 23782732, rank 0, per decode step over 43 MoE layers):
`pack_bitmatrix` 96 us per call (4.1 ms), the router GEMM 40 us per call (1.7 ms).

pack_bitmatrix  vLLM's `make_routing_data` (gpt_oss_triton_kernels_moe.py) launches it with
                BLOCK_SIZE_M = 512 whatever the token count, so one or two tokens run in a
                single workgroup holding [512, 32] tiles through bm_cols = 8 column passes.
                The launcher here picks the row block from the row count; the output is the
                same bitmatrix (test_moe_small.py).
router GEMM     `GateLinear` (4096 -> 256 experts, out_dtype float32) has no ROCm tier and
                falls through to bf16 F.linear (hipBLASLt MT32x32 with fp32 MFMA) followed by
                a cast, so the logits are rounded to bf16 first. score_gemm.score_mm computes
                them with bf16 MFMA and fp32 accumulation, as the CUDA tiers do with cuBLAS.
"""

import torch
import triton


def pack_block_m(n_rows: int) -> int:
    return max(16, min(512, triton.next_power_of_2(n_rows)))  # set from test_moe_small.py


class _PackBitmatrix:
    """Stands in for the `pack_bitmatrix` JIT function: `[grid](...)` ignores vLLM's grid
    and BLOCK_SIZE_M and launches with a row block sized to n_rows."""

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
