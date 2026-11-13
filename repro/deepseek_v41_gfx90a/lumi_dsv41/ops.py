"""Upstream compiled (_C) ops whose signature changed after vLLM 0.26.

fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert: upstream added apply_q_norm, kv_mxfp8,
apply_q_rope and is_q_interleaved. 0.26's kernel always applies V4's per-head query
RMSNorm, which V4.1 dropped (DeepSeek's inference/model.py: q = wq_b(qr), then RoPE).
Its KV side (GPT-J RoPE, UE8M0 FP8 quantisation, paged insert of the 584 B record) is
V4.1's on gfx90a, so the KV side still runs there and the query is computed here: one Triton
launch (head padding, optional norm, GPT-J RoPE on the tail) instead of ~10 strided torch
kernels per layer (graph profile, job 23701511: V4.1 spent 3 ms more per 32-user step than
V4 in generic elementwise kernels). _query_torch is the reference (test_ops.py).
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _query_kernel(q_ptr, out_ptr, pos_ptr, cs_ptr, stride_qt, stride_qh, stride_cs, eps,
                  H: tl.constexpr, D: tl.constexpr, ROT: tl.constexpr, NORM: tl.constexpr,
                  ROPE: tl.constexpr):
    """Program (token t, output head h): out[t, h] = rope(norm(q[t, h])) for h < H, else 0.
    Same roundings as _query_torch: fp32, the normed value cast to the output dtype before
    RoPE. GPT-J pairs (2i, 2i+1) of the last ROT dims; the partner comes from a second load."""
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    W = tl.num_programs(1)
    d = tl.arange(0, D)
    out = out_ptr + (t * W + h) * D + d
    if h >= H:
        tl.store(out, tl.zeros([D], dtype=out_ptr.dtype.element_ty))
    else:
        row = q_ptr + t * stride_qt + h * stride_qh
        x = tl.load(row + d).to(tl.float32)
        partner = tl.load(row + (d ^ 1)).to(tl.float32)
        if NORM:
            inv = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
            x = (x * inv).to(out_ptr.dtype.element_ty).to(tl.float32)
            partner = (partner * inv).to(out_ptr.dtype.element_ty).to(tl.float32)
        if ROPE:
            r = d - (D - ROT)
            in_rot = r >= 0
            j = tl.where(in_rot, r // 2, 0)
            pos = tl.load(pos_ptr + t)
            cos = tl.load(cs_ptr + pos * stride_cs + j).to(tl.float32)
            sin = tl.load(cs_ptr + pos * stride_cs + ROT // 2 + j).to(tl.float32)
            sign = tl.where(d % 2 == 0, -1.0, 1.0)
            x = tl.where(in_rot, x * cos + sign * partner * sin, x)
        tl.store(out, x.to(out_ptr.dtype.element_ty))


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
    t, _, dim = q.shape
    rot = cos_sin_cache.shape[-1]
    out = q.new_empty((t, width, dim))
    if t == 0:
        return out
    if q.stride(-1) != 1 or cos_sin_cache.stride(-1) != 1 or dim & (dim - 1) or dim < rot:
        return _query_torch(q, positions, cos_sin_cache, width, eps, apply_q_norm, apply_q_rope)
    _query_kernel[(t, width)](
        q, out, positions, cos_sin_cache, q.stride(0), q.stride(1), cos_sin_cache.stride(0), eps,
        H=num_heads, D=dim, ROT=rot, NORM=apply_q_norm, ROPE=apply_q_rope, num_warps=4,
    )
    return out


def _query_torch(q, positions, cos_sin_cache, width, eps, apply_q_norm, apply_q_rope):
    """Reference for _query_kernel: the query side in plain torch."""
    num_heads = q.shape[1]
    out = q.new_zeros((q.shape[0], width, q.shape[2]))
    qq = q.float()
    if apply_q_norm:
        qq = qq * torch.rsqrt(qq.square().mean(-1, keepdim=True) + eps)
    out[:, :num_heads] = qq.to(q.dtype)
    if apply_q_rope:
        rope_gptj_(out[:, :num_heads], positions, cos_sin_cache)
    return out
