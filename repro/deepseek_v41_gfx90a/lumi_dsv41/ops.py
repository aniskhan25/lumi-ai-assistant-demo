"""Upstream compiled (_C) ops whose signature changed after vLLM 0.26.

fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert: upstream added apply_q_norm, kv_mxfp8,
apply_q_rope and is_q_interleaved. 0.26's kernel always applies V4's per-head query
RMSNorm, which V4.1 dropped (DeepSeek's inference/model.py: q = wq_b(qr), then RoPE).
Its KV side (GPT-J RoPE, UE8M0 FP8 quantisation, paged insert of the 584 B record) is
V4.1's on gfx90a, so the KV side still runs there and the query is computed here.
"""

import torch


def rope_gptj_(x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor) -> None:
    """In place, GPT-J (interleaved pairs) RoPE on x[..., -rot:]; x is [T, H, D].
    cos_sin_cache is vLLM's [max_pos, rot]: cos for the first rot/2, then sin."""
    rot = cos_sin_cache.shape[-1]
    cos, sin = cos_sin_cache.index_select(0, positions).float().chunk(2, dim=-1)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    part = x[..., -rot:].float()
    even, odd = part[..., 0::2], part[..., 1::2]
    out = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
    x[..., -rot:] = out.to(x.dtype)


def qnorm_rope_kv_rope_quant_insert(
    q, kv, k_cache, slot_mapping, positions, cos_sin_cache, q_head_padded, eps,
    cache_block_size, apply_q_norm=True, kv_mxfp8=False, apply_q_rope=True,
    is_q_interleaved=False,
):
    if kv_mxfp8 or is_q_interleaved:
        raise NotImplementedError("V4.1 MXFP8 KV records / interleaved Q are SM100 paths")
    num_heads = q.shape[1]
    # KV side (and V4's query, discarded): 0.26's kernel. q is only read.
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q, kv, k_cache, slot_mapping, positions, cos_sin_cache,
        max(q_head_padded, num_heads), eps, cache_block_size,
    )
    width = max(q_head_padded, num_heads)
    out = q.new_zeros((q.shape[0], width, q.shape[2]))
    qq = q.float()
    if apply_q_norm:
        qq = qq * torch.rsqrt(qq.square().mean(-1, keepdim=True) + eps)
    out[:, :num_heads] = qq.to(q.dtype)
    if apply_q_rope:
        rope_gptj_(out[:, :num_heads], positions, cos_sin_cache)
    return out
