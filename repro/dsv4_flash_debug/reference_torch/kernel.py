"""Plain-PyTorch drop-in for DeepSeek-V4's reference inference/kernel.py.

The upstream kernels are TileLang, written for NVIDIA (TMA, warp specialisation), and do
not run on MI250X. This module reimplements the same six functions with the same
signatures and the same arithmetic. Activations are quantised to FP8 with 128-wide blocks,
FP8/FP4 weights are dequantised with their e8m0 block scales, and accumulation is in FP32,
so DeepSeek's own model.py can run unmodified on gfx90a as a numerical ground truth.
Speed is irrelevant here; only the maths has to match.
"""

from typing import Optional, Tuple

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
# e2m1 code -> value, for the low three bits; bit 3 is the sign.
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2**ceil(log2(x)) for positive fp32 x, exactly as the kernel's bit trick does."""
    bits = x.float().contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & ((1 << 23) - 1)
    e = exponent - 127 + (mantissa != 0).to(torch.int32)
    return ((e + 127) << 23).view(torch.float32)


def _blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    return x.float().reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32, inplace: bool = False,
):
    """Block-wise FP8 quantisation; inplace=True does quant+dequant back into x."""
    assert x.size(-1) % block_size == 0
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _pow2_ceil(amax / FP8_MAX) if scale_fmt is not None else amax / FP8_MAX
    y = (xb / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((y.float() * s).reshape(x.shape).to(x.dtype))
        return x
    return y.reshape(x.shape), s.squeeze(-1).to(scale_dtype)


def _to_e2m1_codes(v: torch.Tensor) -> torch.Tensor:
    """Round |v| <= 6 to the nearest e2m1 value, ties to even; return 4-bit codes."""
    sign = (v < 0).to(torch.uint8) << 3
    a = v.abs()
    grid = torch.tensor(_E2M1, device=v.device)
    mids = (grid[1:] + grid[:-1]) / 2
    # bucketize(right=False) puts a value equal to a midpoint in the lower bucket.
    code = torch.bucketize(a, mids)
    # Exact ties go to the even code (code & 1 == 0), as IEEE round-to-nearest-even.
    at_mid = a == mids[code.clamp(max=len(mids) - 1)]
    tie = at_mid & (code < len(mids)) & (code % 2 == 1)
    code = code + tie.to(code.dtype)
    return code.to(torch.uint8) | sign


def _from_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    grid = torch.tensor(_E2M1, device=codes.device)
    value = grid[(codes & 0x7).long()]
    return torch.where((codes & 0x8) != 0, -value, value)


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False):
    """Block-wise FP4 (e2m1) quantisation with power-of-two e8m0 scales."""
    assert x.size(-1) % block_size == 0
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(6 * 2.0**-126)
    s = _pow2_ceil(amax / FP4_MAX)
    codes = _to_e2m1_codes((xb / s).clamp(-FP4_MAX, FP4_MAX))
    if inplace:
        x.copy_((_from_e2m1_codes(codes) * s).reshape(x.shape).to(x.dtype))
        return x
    flat = codes.reshape(*x.shape[:-1], x.shape[-1])
    packed = flat[..., 0::2] | (flat[..., 1::2] << 4)  # first element in the low nibble
    return packed.view(torch.float4_e2m1fn_x2), s.squeeze(-1).to(torch.float8_e8m0fnu)


def _expand(scale: torch.Tensor, rows: int, cols: int, row_block: int, col_block: int):
    s = scale.float()
    s = s.repeat_interleave(row_block, 0)[:rows] if row_block > 1 else s
    return s.repeat_interleave(col_block, 1)[:, :cols]


def fp8_gemm(a, a_s, b, b_s, scale_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """C[M,N] = A[M,K] @ B[N,K]^T; A per-128 scales on K, B per 128x128 block scales."""
    K = a.size(-1)
    N = b.size(0)
    a2 = a.reshape(-1, K).float() * a_s.reshape(-1, K // 128).float().repeat_interleave(128, -1)
    b2 = b.float() * _expand(b_s, N, K, 128, 128)
    c = a2 @ b2.t()
    return c.reshape(*a.shape[:-1], N).to(torch.get_default_dtype())


def _unpack_fp4(b: torch.Tensor) -> torch.Tensor:
    raw = b.view(torch.uint8)
    codes = torch.stack((raw & 0xF, raw >> 4), dim=-1).reshape(raw.shape[0], raw.shape[1] * 2)
    return _from_e2m1_codes(codes)


def fp4_gemm(a, a_s, b, b_s, scale_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T; A per-128 scales, B per-32 e8m0 scales on K.
    The kernel casts FP4 to FP8 before the GEMM; every e2m1 value is exact in e4m3."""
    K = a.size(-1)
    N = b.size(0)
    a2 = a.reshape(-1, K).float() * a_s.reshape(-1, K // 128).float().repeat_interleave(128, -1)
    b2 = _unpack_fp4(b) * b_s.float().repeat_interleave(32, -1)[:, :K]
    c = a2 @ b2.t()
    return c.reshape(*a.shape[:-1], N).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale) -> torch.Tensor:
    """Attention over gathered KV rows (-1 = none), with a per-head sink logit.
    kv serves as both K and V (MQA); q [b,s,h,d], kv [b,n,d], topk_idxs [b,s,k]."""
    b, s, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    rows = kv[torch.arange(b, device=kv.device)[:, None, None], idx.clamp_min(0)]  # b,s,k,d
    scores = torch.einsum("bshd,bskd->bshk", q.float(), rows.float()) * softmax_scale
    scores = scores.masked_fill(~valid[:, :, None, :], float("-inf"))
    m = torch.maximum(scores.amax(-1), attn_sink.float()[None, None, :])  # include the sink
    p = torch.exp(scores - m[..., None])
    denom = p.sum(-1) + torch.exp(attn_sink.float()[None, None, :] - m)
    # The kernel rounds probabilities to bf16 before the PV GEMM.
    o = torch.einsum("bshk,bskd->bshd", p.to(torch.bfloat16).float(), rows.float())
    return (o / denom[..., None]).to(torch.bfloat16)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    hc = hc_mult
    m = mixes.float()
    sc = hc_scale.float()
    base = hc_base.float()
    pre = torch.sigmoid(m[..., :hc] * sc[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * sc[1] + base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * sc[2] + base[2 * hc:]).reshape(*m.shape[:-1], hc, hc)
    comb = torch.softmax(comb, -1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
