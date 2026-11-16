"""Batched, graph-capturable FP8 paged MQA logits for the sparse indexer's decode, without aiter.

vLLM 0.26 computes the indexer's decode logits on ROCm with aiter's
`deepgemm_fp8_paged_mqa_logits`, or else with `fp8_paged_mqa_logits_torch`, which loops over
the batch in Python with one `.item()` host sync per sequence. That fallback is slow, grows
with the number of users, and cannot be captured in a CUDA graph (job 23512857).

This kernel computes the same thing in one launch. The grid is fixed by the batch size and
`max_model_len`; each program reads its sequence length on the device and fills positions
past it with -inf. For output row r = b * next_n + j (query token j of sequence b; next_n is
2 with one MTP draft token) and cached position t < L_r:

    logits[r, t] = scale[t] * sum_h weights[r, h] * relu(q[b, j, h, :] . k[t, :])

L_r is the token's causal length: vLLM's indexer passes it per token as context_lens [B,
next_n] for MTP (indexer.py, native path); a [B] context_lens means L_b - next_n + j + 1, as
in the torch fallback.

The cache is the layout the V4 compressor writes (`_fused_kv_compress_norm_rope_insert_
indexer_attn`) and the torch fallback reads: per block, `block_size * D` FP8 values, row-major
by position, then `block_size` float32 scales. FP8 (e4m3) values are exact in fp16, so the
dot products run in fp16 with fp32 accumulation.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _paged_mqa_logits_kernel(
    q_ptr,  # [B, NEXT_N, H, D] fp8
    kv_ptr,  # uint8, block stride KV_BLOCK_STRIDE bytes
    w_ptr,  # [B * NEXT_N, H] float32
    lens_ptr,  # [B * NEXT_N] int, causal length per query token
    bt_ptr,  # [B, max_blocks] int32
    out_ptr,  # [B * NEXT_N, max_model_len] float32
    q_stride_b,
    q_stride_n,
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
    NEXT_N: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    b = r // NEXT_N
    j = r % NEXT_N
    pid_t = tl.program_id(1)
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    seq_len = tl.load(lens_ptr + r)
    out_row = out_ptr + r * out_stride_b
    if pid_t * BLOCK_T >= seq_len:
        tl.store(out_row + t, float("-inf"), mask=t < max_model_len)
        return

    valid = t < seq_len
    page = tl.load(bt_ptr + b * bt_stride_b + t // BLOCK_SIZE, mask=valid, other=0).to(tl.int64)
    pos = t % BLOCK_SIZE
    block_base = kv_ptr + page * KV_BLOCK_STRIDE
    d = tl.arange(0, D)
    k = tl.load(block_base[:, None] + pos[:, None] * D + d[None, :], mask=valid[:, None], other=0)
    k = k.to(q_ptr.dtype.element_ty, bitcast=True).to(tl.float32).to(tl.float16)
    scale = tl.load((block_base + BLOCK_SIZE * D + pos * 4).to(tl.pointer_type(tl.float32)),
                    mask=valid, other=0.0)

    h = tl.arange(0, BLOCK_H)
    q = tl.load(q_ptr + b * q_stride_b + j * q_stride_n + h[:, None] * q_stride_h + d[None, :],
                mask=(h < H)[:, None], other=0.0)
    q = q.to(tl.float32).to(tl.float16)
    w = tl.load(w_ptr + r * w_stride_b + h, mask=h < H, other=0.0).to(tl.float32)

    s = tl.dot(k, tl.trans(q))  # [BLOCK_T, BLOCK_H], fp32
    s = tl.maximum(s, 0.0) * w[None, :]
    logits = tl.sum(s, axis=1) * scale
    tl.store(out_row + t, tl.where(valid, logits, float("-inf")), mask=t < max_model_len)


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,  # [B, next_n, H, D] fp8
    kv_cache: torch.Tensor,  # [num_blocks, block_size, 1, D + 4] uint8
    weights: torch.Tensor,  # [B * next_n, H] float32
    context_lens: torch.Tensor,  # [B, next_n] per token, or [B]
    block_tables: torch.Tensor,  # [B, max_blocks]
    max_model_len: int,
    block_t: int = 64,
) -> torch.Tensor:
    batch, next_n, heads, dim = q.shape
    block_size = kv_cache.shape[1]
    assert kv_cache.dtype == torch.uint8 and kv_cache.shape[-1] == dim + 4
    assert block_size % block_t == 0 or block_t % block_size == 0
    assert q.stride(-1) == 1 and weights.stride(-1) == 1
    lens = context_lens.reshape(-1)
    if next_n > 1 and lens.numel() == batch:  # per sequence: token j sees L - next_n + j + 1
        offs = torch.arange(1 - next_n, 1, device=lens.device, dtype=lens.dtype)
        lens = (lens[:, None] + offs).reshape(-1)
    assert lens.numel() == batch * next_n and weights.shape[0] == batch * next_n
    out = torch.empty(batch * next_n, max_model_len, dtype=torch.float32, device=q.device)
    grid = (batch * next_n, triton.cdiv(max_model_len, block_t))
    _paged_mqa_logits_kernel[grid](
        q, kv_cache, weights, lens, block_tables, out,
        q.stride(0), q.stride(1), q.stride(2), weights.stride(0), block_tables.stride(0),
        out.stride(0), max_model_len, heads, kv_cache.stride(0),
        BLOCK_SIZE=block_size, D=dim, BLOCK_H=triton.next_power_of_2(heads), BLOCK_T=block_t,
        NEXT_N=next_n, num_warps=4,
    )
    return out


def install() -> None:
    """Route rocm_fp8_paged_mqa_logits to the kernel above when aiter is off."""
    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as ops

    import os

    from vllm.logger import init_logger

    log = init_logger("vllm.lumi_dsv4_debug")
    stock = ops.rocm_fp8_paged_mqa_logits
    if getattr(stock, "_lumi_triton", False):
        return
    # DSV4_PAGED_LOGITS_CHECK=N: compare the first N calls with the torch fallback (syncs).
    checks = {"left": int(os.environ.get("DSV4_PAGED_LOGITS_CHECK", "0")), "described": False}

    def describe(q, kv, w, lens, bt, max_model_len):
        t = lambda x: f"{tuple(x.shape)} {x.dtype} stride {x.stride()}"  # noqa: E731
        log.info("paged logits call: q %s | kv %s | w %s | lens %s | bt %s | max_model_len %d",
                 t(q), t(kv), t(w), t(lens), t(bt), max_model_len)

    def rocm_fp8_paged_mqa_logits(q_fp8, kv_cache_fp8, weights, context_lens, block_tables,
                                  schedule_metadata, max_model_len):
        from vllm._aiter_ops import rocm_aiter_ops

        if rocm_aiter_ops.is_enabled():
            return stock(q_fp8, kv_cache_fp8, weights, context_lens, block_tables,
                         schedule_metadata, max_model_len)
        if not checks["described"]:
            checks["described"] = True
            describe(q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len)
        out = fp8_paged_mqa_logits_triton(q_fp8, kv_cache_fp8, weights, context_lens,
                                          block_tables, max_model_len)
        if checks["left"] > 0:
            checks["left"] -= 1
            want = ops.fp8_paged_mqa_logits_torch(q_fp8, kv_cache_fp8, weights, context_lens,
                                                  block_tables, max_model_len)
            same_mask = torch.equal(torch.isinf(out), torch.isinf(want))
            fin = torch.isfinite(want)
            err = ((out[fin] - want[fin]).abs().max().item() if fin.any() else 0.0)
            log.info("paged logits check: lens %s, -inf mask %s, max abs err %.3g",
                     context_lens.reshape(-1).tolist()[:8], "same" if same_mask else "DIFF", err)
        return out

    rocm_fp8_paged_mqa_logits._lumi_triton = True
    ops.rocm_fp8_paged_mqa_logits = rocm_fp8_paged_mqa_logits
