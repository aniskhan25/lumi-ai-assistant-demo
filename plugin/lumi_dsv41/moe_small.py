"""MoE expert routing and GEMM tiles for V4.1 on gfx90a (triton_unfused experts under EP).

pack_bitmatrix  vLLM's `make_routing_data` launches it with BLOCK_SIZE_M = 512 whatever
                the token count, so a few tokens run in one workgroup over [512, 32]
                tiles. Row blocks of 16 give the same bitmatrix and are the fastest at
                every row count (43.7 -> 3.3 us at 1-512 rows).
expert GEMMs    triton_kernels' AMD opt flags pick tiles from a grid that counts every
                expert. V4.1 holds 48 whole experts a rank (gate/up 5120 -> 4608, down
                2304 -> 5120); forced block_n / split_k per token count: gate/up 143 -> 38 us
                at 1 token, 1068 -> 933 at 192, 1780 -> 1548 at 2048. Same output up to the
                split-K summation order (<= 2e-3).
"""

import triton

# (max tokens, (block_n, split_k)); DSpark verifies 6 tokens per sequence, so decode
# batches reach 96-768 rows; prefill chunks are 2048.
GATE_UP = [(1, (32, 2)), (2, (32, 4)), (8, (128, 2)), (8192, (128, 1))]
DOWN = [(2, (32, 2)), (4, (128, 2)), (8192, (128, 1))]
TOPK = 6  # num_experts_per_tok: the down GEMM's rows are tokens * TOPK
PACK_BLOCK_M = 16


class _PackBitmatrix:
    """Stands in for the `pack_bitmatrix` JIT function: `[grid](...)` ignores vLLM's grid
    and BLOCK_SIZE_M and launches with PACK_BLOCK_M rows per program."""

    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        def launch(bitmatrix, topk_ids, n_rows, bm_cols, n_expts_act, BLOCK_SIZE_M, BLOCK_SIZE_K):
            return self.kernel[(triton.cdiv(n_rows, PACK_BLOCK_M),)](
                bitmatrix, topk_ids, n_rows, bm_cols, n_expts_act,
                BLOCK_SIZE_M=PACK_BLOCK_M, BLOCK_SIZE_K=BLOCK_SIZE_K)
        return launch


def _pick(table, tokens):
    for m_max, cfg in table:
        if tokens <= m_max:
            return {"block_n": cfg[0], "split_k": cfg[1]}
    return None


def install() -> None:
    import importlib

    import vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe as m

    if not isinstance(m.pack_bitmatrix, _PackBitmatrix):
        m.pack_bitmatrix = _PackBitmatrix(m.pack_bitmatrix)
    stock = m.matmul_ogs
    if getattr(stock, "_lumi", False):
        return
    pkg = stock.__module__.rsplit(".", 1)[0]  # vLLM's (aliased) triton_kernels
    of = importlib.import_module(pkg + ".matmul_ogs_details.opt_flags")

    def matmul_ogs(x, w, bias, routing_data=None, gather_indx=None, scatter_indx=None, **kw):
        cons = None
        if routing_data is not None and gather_indx is not None:  # gate/up: x is [tokens, K]
            cons = _pick(GATE_UP, x.shape[0])
        elif routing_data is not None and scatter_indx is not None:  # down: x is [tokens*topk, I]
            cons = _pick(DOWN, x.shape[0] // TOPK)
        if cons is None:
            return stock(x, w, bias, routing_data, gather_indx, scatter_indx, **kw)
        saved = of._opt_flags_constraints
        of._opt_flags_constraints = {**saved, **cons}
        try:
            return stock(x, w, bias, routing_data, gather_indx, scatter_indx, **kw)
        finally:
            of._opt_flags_constraints = saved

    matmul_ogs._lumi = True
    m.matmul_ogs = matmul_ogs
