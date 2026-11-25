"""Batched, graph-capturable FP8 paged MQA logits for the V4.1 indexer decode on gfx90a.

Without aiter, upstream computes the decode logits with `fp8_paged_mqa_logits_torch`: a
Python loop over the batch with a host sync per sequence (slow, grows with users, cannot be
captured in a CUDA graph). Its own Triton kernel (`rocm_fp8_paged_mqa_logits_triton`) is
gated to gfx942/950, uses fp8 MFMA, and reads only the block-flat layout.

This is the DeepSeek-V4 kernel (validated on gfx90a under piecewise graphs) with V4.1's ROCm cache layout: values in 16x16
tiles (indexer_k_store.py, SHUFFLE), so byte d of position p within a block sits at
(p // 16) * 16 * D + (d // 16) * 256 + (p % 16) * 16 + d % 16. Scales (float32, one per
position) follow the values untiled. For next_n == 1, batch row b and position t < L_b:

    logits[b, t] = scale[t] * sum_h weights[b, h] * relu(q[b, h, :] . k[t, :])

and -inf past L_b, as the torch fallback. FP8 (e4m3) is exact in fp16, so the dot runs in
fp16 with fp32 accumulation.
"""

import torch

from vllm.triton_utils import tl, triton

TILE = 16


@triton.jit
def _paged_mqa_logits_kernel(
    q_ptr,  # [B, 1, H, D] fp8
    kv_ptr,  # uint8, block stride KV_BLOCK_STRIDE bytes
    w_ptr,  # [B, H] float32
    lens_ptr,  # [B] int
    bt_ptr,  # [B, max_blocks] int32
    out_ptr,  # [B, max_model_len] float32
    q_stride_b,
    q_stride_h,
    w_stride_b,
    bt_stride_b,
    out_stride_b,
    max_model_len,
    H,
    KV_BLOCK_STRIDE,
    BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SHUFFLE: tl.constexpr,
    TILE: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    pid_t = tl.program_id(1)
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    seq_len = tl.load(lens_ptr + b)
    out_row = out_ptr + b * out_stride_b
    if pid_t * BLOCK_T >= seq_len:
        tl.store(out_row + t, float("-inf"), mask=t < max_model_len)
        return

    valid = t < seq_len
    page = tl.load(bt_ptr + b * bt_stride_b + t // BLOCK_SIZE, mask=valid, other=0).to(tl.int64)
    pos = t % BLOCK_SIZE
    block_base = kv_ptr + page * KV_BLOCK_STRIDE
    d = tl.arange(0, D)
    if SHUFFLE:
        off = ((pos // TILE) * (TILE * D) + (pos % TILE) * TILE)[:, None] \
            + ((d // TILE) * (TILE * TILE) + d % TILE)[None, :]
    else:
        off = pos[:, None] * D + d[None, :]
    k = tl.load(block_base[:, None] + off, mask=valid[:, None], other=0)
    k = k.to(q_ptr.dtype.element_ty, bitcast=True).to(tl.float32).to(tl.float16)
    scale = tl.load((block_base + BLOCK_SIZE * D + pos * 4).to(tl.pointer_type(tl.float32)),
                    mask=valid, other=0.0)

    h = tl.arange(0, BLOCK_H)
    q = tl.load(q_ptr + b * q_stride_b + h[:, None] * q_stride_h + d[None, :],
                mask=(h < H)[:, None], other=0.0)
    q = q.to(tl.float32).to(tl.float16)
    w = tl.load(w_ptr + b * w_stride_b + h, mask=h < H, other=0.0).to(tl.float32)

    s = tl.dot(k, tl.trans(q))  # [BLOCK_T, BLOCK_H], fp32
    s = tl.maximum(s, 0.0) * w[None, :]
    logits = tl.sum(s, axis=1) * scale
    tl.store(out_row + t, tl.where(valid, logits, float("-inf")), mask=t < max_model_len)


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,  # [B, 1, H, D] fp8
    kv_cache: torch.Tensor,  # [num_blocks, block_size, 1, D + 4] uint8
    weights: torch.Tensor,  # [B, H] float32
    context_lens: torch.Tensor,  # [B] or [B, 1]
    block_tables: torch.Tensor,  # [B, max_blocks]
    max_model_len: int,
    shuffle: bool,
    block_t: int = 64,
) -> torch.Tensor:
    batch, next_n, heads, dim = q.shape
    assert next_n == 1
    block_size = kv_cache.shape[1]
    assert kv_cache.dtype == torch.uint8 and kv_cache.shape[-1] == dim + 4
    assert block_size % block_t == 0 or block_t % block_size == 0
    assert not shuffle or (block_size % TILE == 0 and dim % TILE == 0)
    assert q.stride(-1) == 1 and weights.stride(-1) == 1
    lens = context_lens.reshape(-1)
    out = torch.empty(batch, max_model_len, dtype=torch.float32, device=q.device)
    grid = (batch, triton.cdiv(max_model_len, block_t))
    _paged_mqa_logits_kernel[grid](
        q, kv_cache, weights, lens, block_tables, out,
        q.stride(0), q.stride(2), weights.stride(0), block_tables.stride(0), out.stride(0),
        max_model_len, heads, kv_cache.stride(0),
        BLOCK_SIZE=block_size, D=dim, BLOCK_H=triton.next_power_of_2(heads), BLOCK_T=block_t,
        SHUFFLE=shuffle, TILE=TILE, num_warps=4,
    )
    return out
