"""Attention output projection wo_a at decode sizes (DSV4_WO_A=1).

On ROCm, DeepseekV4ROCMAiterMLAAttention._o_proj calls rocm_inv_rope_einsum: the fused
inverse RoPE, then torch.einsum("tgd,grd->tgr") against the cached bf16 wo_a weight. At
TP=8 a rank holds 1 of V4-Flash's 8 o_groups, so the einsum is a plain [T, 4096] x
[1024, 4096].T GEMM, and hipBLASLt runs it in ~33 us at any T from 1 to 128 (43 us in the
1-user graph profile, job 23861347: 1.9 ms per step). test_wo_a.py (job 23866833, weights
cycled from HBM): vLLM's wvSplitK 9-12 us at 1-3 tokens, score_gemm's Triton kernel with a
bf16 output 15.6 us at 4-16, 18.6 at 32, 24 at 64, 34 at 128.

Same inverse RoPE and cached weight as the stock function; more than one local group, or
more than 128 tokens (prefill), go to the stock function.
"""

import torch

WVSPLITK_MAX = 3


def install() -> None:
    import vllm.models.deepseek_v4.amd.rocm as r
    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as m

    from .score_gemm import pick, score_mm

    stock = r.rocm_inv_rope_einsum
    if getattr(stock, "_lumi", False):
        return

    def rocm_inv_rope_einsum(rotary_emb, o, positions, rope_head_dim, n_local_groups,
                             o_lora_rank, wo_a):
        T = o.shape[0]
        if n_local_groups != 1 or T == 0 or (T > WVSPLITK_MAX and pick(T, o_lora_rank) is None):
            return stock(rotary_emb, o, positions, rope_head_dim, n_local_groups, o_lora_rank,
                         wo_a)
        o_ref = m._fused_inverse_rope_gptj(o, positions, rotary_emb.cos_sin_cache, rope_head_dim)
        x = o_ref.reshape(T, -1)
        w = m._get_cached_wo_a_bf16(wo_a, 1, o_lora_rank, x.shape[-1]).view(o_lora_rank, -1)
        if T <= WVSPLITK_MAX:
            y = torch.ops.vllm.rocm_unquantized_gemm(x, w, None)
        else:
            y = score_mm(x, w, out_dtype=torch.bfloat16)
        return y.view(T, 1, o_lora_rank)

    rocm_inv_rope_einsum._lumi = True
    rocm_inv_rope_einsum.stock = stock
    r.rocm_inv_rope_einsum = rocm_inv_rope_einsum
