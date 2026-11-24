"""Fewer small kernels around the MoE (DSV4_MOE_GLUE=1).

An eager profile at 32 users with stacks (job 23974135, small_kernels.py) traced the small
torch kernels that the graph profiles count by the hundred. Per MoE layer (44 a step):

make_routing_data   topk_ids.to(int16), topk_weights.to(bf16), torch.zeros(bitmatrix),
                    pack_bitmatrix, topk_ids == -1, torch.where(...): six launches, most on
                    torch's non-vectorised path for tiny tensors (5-13 us each). Here one Triton
                    kernel writes the int16 ids, the bf16 weights (-1 where the id is -1, as
                    matmul_ogs expects) and the bitmatrix (pack_bitmatrix's bits).
UnfusedOAITritonExperts.apply
                    intermediate_cache1[dst_indx] and intermediate_cache2[src_indx] index with
                    int32 tensors: torch casts each index to int64 (a strided copy, ~13 us)
                    before its gather (so does make_routing_data's gate_scal gather). Here a
                    Triton row gather takes the int32 index directly.
                    test_moe_glue.py (job 23983883): routing outputs equal to vLLM's on every
                    routed entry, gathers and the activation bit-identical; the replaced glue
                    101-116 -> 70-82 us per MoE layer (1.2-1.6 ms a step at 2-128 tokens).
SiluAndMulWithClamp (the shared expert; vLLM sends ROCm to forward_native)
                    clamp, clamp, mul, sigmoid, mul, add, mul: ~6 launches a layer. Here one
                    kernel with the same bf16 rounding after each op.

Everything else (the routing metadata kernels, matmul_ogs, the masked top-k sum) is vLLM's.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _or(a, b):
    return a | b


@triton.jit
def _route_prep_kernel(ids_ptr, w_ptr, ids16_ptr, w16_ptr, bm_ptr, n_rows,
                       TOPK: tl.constexpr, TOPK_P2: tl.constexpr, BM_COLS: tl.constexpr,
                       BM_P2: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    j = tl.arange(0, TOPK_P2)
    m = (rows < n_rows)[:, None] & (j < TOPK)[None, :]
    off = rows[:, None] * TOPK + j[None, :]
    ids = tl.load(ids_ptr + off, mask=m, other=-1).to(tl.int32)
    w = tl.load(w_ptr + off, mask=m, other=0.0).to(tl.bfloat16)
    valid = ids >= 0
    tl.store(ids16_ptr + off, ids.to(tl.int16), mask=m)
    tl.store(w16_ptr + off, tl.where(valid, w, -1.0).to(tl.bfloat16), mask=m)
    c = tl.arange(0, BM_P2)
    bit = tl.where(valid, tl.cast(1, tl.uint32) << (ids & 31).to(tl.uint32), 0)
    x = tl.where(valid[:, :, None] & ((ids >> 5)[:, :, None] == c[None, None, :]),
                 bit[:, :, None], 0)
    words = tl.reduce(x, 1, _or)  # [BLOCK, BM_P2]
    tl.store(bm_ptr + rows[:, None] * BM_COLS + c[None, :], words,
             mask=(rows < n_rows)[:, None] & (c < BM_COLS)[None, :])


@triton.jit
def _gather_rows_kernel(src_ptr, idx_ptr, out_ptr, N, stride_s, stride_o, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    i = tl.load(idx_ptr + r).to(tl.int64)
    v = tl.load(src_ptr + i * stride_s + cols, mask=cols < N)
    tl.store(out_ptr + r * stride_o + cols, v, mask=cols < N)


def gather_rows(src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """src[idx] for a 2-D src and a 1-D integer index of any width."""
    n = src.shape[1]
    out = torch.empty(idx.shape[0], n, dtype=src.dtype, device=src.device)
    if idx.numel():
        block = min(1024, triton.next_power_of_2(n))
        _gather_rows_kernel[(idx.shape[0], triton.cdiv(n, block))](
            src, idx, out, n, src.stride(0), out.stride(0), BLOCK=block, num_warps=4)
    return out


@triton.jit
def _silu_mul_clamp_kernel(x_ptr, out_ptr, D, stride_x, stride_o, limit, alpha, beta,
                           BLOCK: tl.constexpr):
    """SiluAndMulWithClamp.forward_native with its bf16 rounding after each torch op:
    gate = clamp(x[:d], max=limit); up = clamp(x[d:], -limit, limit);
    out = gate * sigmoid(alpha * gate) * (up + beta)."""
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < D
    g = tl.minimum(tl.load(x_ptr + r * stride_x + c, mask=m, other=0.0).to(tl.float32), limit)
    u = tl.load(x_ptr + r * stride_x + D + c, mask=m, other=0.0).to(tl.float32)
    u = tl.minimum(tl.maximum(u, -limit), limit)
    ag = (alpha * g).to(tl.bfloat16).to(tl.float32)
    s = (1.0 / (1.0 + tl.exp(-ag))).to(tl.bfloat16).to(tl.float32)
    t = (g * s).to(tl.bfloat16).to(tl.float32)
    ub = (u + beta).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + r * stride_o + c, (t * ub).to(tl.bfloat16), mask=m)


def install() -> None:
    import vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe as m
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    if getattr(m, "_lumi_glue", False):
        return
    m._lumi_glue = True

    if not m.use_legacy_triton_kernels:
        def make_routing_data(topk_ids, topk_weights, num_local_experts):
            n_rows, topk = topk_ids.shape
            if topk_ids.stride(-1) != 1 or topk_weights.stride(-1) != 1 or n_rows == 0:
                return stock_routing(topk_ids, topk_weights, num_local_experts)
            dev = topk_ids.device
            bm_cols = triton.cdiv(num_local_experts, 32)
            ids16 = torch.empty(n_rows, topk, dtype=torch.int16, device=dev)
            w16 = torch.empty(n_rows, topk, dtype=torch.bfloat16, device=dev)
            bitmatrix = torch.empty(n_rows, bm_cols, dtype=torch.uint32, device=dev)
            _route_prep_kernel[(triton.cdiv(n_rows, 16),)](
                topk_ids, topk_weights, ids16, w16, bitmatrix, n_rows, TOPK=topk,
                TOPK_P2=triton.next_power_of_2(topk), BM_COLS=bm_cols,
                BM_P2=triton.next_power_of_2(bm_cols), BLOCK=16, num_warps=4)
            bitmatrix = m.Bitmatrix(bitmatrix, dtype=m.BIT, shape=[n_rows, bm_cols * 32],
                                    shape_max=[n_rows, None])
            sparse_logits = m.SparseMatrix(indx=ids16, vals=w16, mask=bitmatrix)
            dispatch_indx = sparse_logits.mask_metadata.row_sorted_indx
            combine_indx = sparse_logits.mask_metadata.col_sorted_indx
            ragged = m.make_ragged_tensor_metadata(sparse_logits.mask_metadata.col_sum,
                                                   dispatch_indx.shape[0])
            gate_scal = gather_rows(sparse_logits.vals.reshape(-1, 1), combine_indx).view(-1)
            routing_data = m.RoutingData(gate_scal, ragged.block_sizes, num_local_experts, topk,
                                         ragged)
            return (routing_data, m.GatherIndx(combine_indx, dispatch_indx),
                    m.ScatterIndx(dispatch_indx, combine_indx))

        stock_routing = m.make_routing_data
        m.make_routing_data = make_routing_data  # _make_routing_data looks it up at call time

    cls = m.UnfusedOAITritonExperts
    stock_apply = cls.apply

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
              global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2,
              expert_tokens_meta, apply_router_weight_on_input):
        if expert_map is not None or getattr(self, "_lora_context", None) is not None:
            return stock_apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
                               activation, global_num_experts, expert_map, a1q_scale,
                               a2_scale, workspace13, workspace2, expert_tokens_meta,
                               apply_router_weight_on_input)
        qc = self.quant_config or m.FUSED_MOE_UNQUANTIZED_CONFIG
        routing_data, gather_indx, scatter_indx = self._make_routing_data(
            topk_ids, topk_weights, w1.shape[0])
        topk = topk_ids.size(1)
        M, K = hidden_states.shape
        N = w1.shape[2]
        cache1 = m._resize_cache(workspace2, (1, M * topk, N))
        cache3 = m._resize_cache(workspace2, (1, M * topk, K))
        cache2 = m._resize_cache(workspace13, (M * topk, self.adjust_N_for_activation(N, activation)))
        gammas = routing_data.gate_scal if routing_data else None
        m.matmul_ogs(hidden_states, w1, qc.w1_bias, routing_data, gather_indx=gather_indx,
                     precision_config=qc.w1_precision,
                     gammas=gammas if apply_router_weight_on_input else None,
                     fused_activation=None, y=cache1)
        self.activation(activation, cache2, gather_rows(cache1.view(-1, N), gather_indx.dst_indx))
        routing_data.n_expts_act = 1
        m.matmul_ogs(gather_rows(cache2, gather_indx.src_indx), w2, qc.w2_bias, routing_data,
                     scatter_indx=scatter_indx, precision_config=qc.w2_precision,
                     gammas=None if apply_router_weight_on_input else gammas, y=cache3)
        m.masked_moe_sum(cache3.view(-1, topk, K), topk_ids, output)

    cls.apply = apply

    def forward_native(self, x):
        d = x.shape[-1] // 2
        if x.dtype != torch.bfloat16 or x.stride(-1) != 1 or x.numel() == 0:
            return stock_act(self, x)
        x2 = x.reshape(-1, 2 * d)
        out = torch.empty(x2.shape[0], d, dtype=x.dtype, device=x.device)
        block = min(1024, triton.next_power_of_2(d))
        _silu_mul_clamp_kernel[(x2.shape[0], triton.cdiv(d, block))](
            x2, out, d, x2.stride(0), out.stride(0), self.swiglu_limit, self.alpha, self.beta,
            BLOCK=block, num_warps=4)
        return out.view(*x.shape[:-1], d)

    stock_act = SiluAndMulWithClamp.forward_native
    SiluAndMulWithClamp.forward_native = forward_native  # bound as _forward_method at init
