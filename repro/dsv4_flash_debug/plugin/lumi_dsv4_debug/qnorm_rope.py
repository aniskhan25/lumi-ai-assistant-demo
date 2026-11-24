"""V4 attention's q-norm / RoPE / KV insert in one Triton kernel (DSV4_QNORM_ROPE=1).

`DeepseekV4Attention._fused_qnorm_rope_kv_insert` (fp8_ds_mla cache) calls vLLM's HIP op
`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`: per-head RMSNorm (no weight) and
GPT-J RoPE on q into a head-padded q tensor, GPT-J RoPE on the new KV row, UE8M0 FP8
quantisation and the paged cache insert. On gfx90a it takes 32-34 us a call at 1, 32 and
64 users alike (graph profiles 23861347, 23908712, 23900446), 44 calls and ~1.4 ms a step:
latency, not data (two tokens move ~10 KB).

Here one program per (token, q head) normalises and rotates a head (or zero-fills a
padding head); one more per token rotates the KV row's RoPE part and writes the
fp8_ds_mla record exactly as vLLM's Triton `quantize_and_insert_k_kernel` does (scale =
2^ceil(log2(max(amax, 1e-4) / 448)) per 64 values, OCP e4m3, the RoPE part as bf16).
The XPU backend's `xpu_qnorm_rope_kv_fp8_insert` does the same work in two kernels.
"""

import torch

from vllm.triton_utils import tl, triton

HEAD_DIM, ROPE_DIM = 512, 64
NOPE_DIM = HEAD_DIM - ROPE_DIM


@triton.jit
def _qnorm_rope_kv_insert_kernel(
    q_ptr, kv_ptr, qo_ptr, cache_ptr, slot_ptr, pos_ptr, cs_ptr,
    num_q, num_kv, stride_qt, stride_qh, stride_kv, block_stride, eps,
    N_HEADS: tl.constexpr, PADDED: tl.constexpr, CACHE_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr, ROPE_DIM: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    NOPE: tl.constexpr = HEAD_DIM - ROPE_DIM
    HALF: tl.constexpr = ROPE_DIM // 2
    i = tl.arange(0, HALF)
    if h < PADDED:
        if t >= num_q:
            return
        offs = tl.arange(0, HEAD_DIM)
        out = qo_ptr + t.to(tl.int64) * (PADDED * HEAD_DIM) + h * HEAD_DIM
        if h >= N_HEADS:
            tl.store(out + offs, tl.zeros([HEAD_DIM], dtype=tl.bfloat16))
            return
        pos = tl.load(pos_ptr + t).to(tl.int64)
        cos = tl.load(cs_ptr + pos * ROPE_DIM + i).to(tl.float32)
        sin = tl.load(cs_ptr + pos * ROPE_DIM + HALF + i).to(tl.float32)
        base = q_ptr + t.to(tl.int64) * stride_qt + h * stride_qh
        xq = tl.load(base + offs).to(tl.float32)
        rms = tl.rsqrt(tl.sum(xq * xq, 0) / HEAD_DIM + eps)
        tl.store(out + offs, (xq * rms).to(tl.bfloat16), mask=offs < NOPE)
        qe = tl.load(base + NOPE + 2 * i).to(tl.float32) * rms
        qo = tl.load(base + NOPE + 2 * i + 1).to(tl.float32) * rms
        tl.store(out + NOPE + 2 * i, (qe * cos - qo * sin).to(tl.bfloat16))
        tl.store(out + NOPE + 2 * i + 1, (qe * sin + qo * cos).to(tl.bfloat16))
    else:  # the KV row of token t
        if t >= num_kv:
            return
        slot = tl.load(slot_ptr + t)
        if slot == -1:
            return
        pos = tl.load(pos_ptr + t).to(tl.int64)
        cos = tl.load(cs_ptr + pos * ROPE_DIM + i).to(tl.float32)
        sin = tl.load(cs_ptr + pos * ROPE_DIM + HALF + i).to(tl.float32)
        row = kv_ptr + t.to(tl.int64) * stride_kv
        blk = cache_ptr + (slot // CACHE_BLOCK).to(tl.int64) * block_stride
        p = slot % CACHE_BLOCK
        data = blk + p * (NOPE + 2 * ROPE_DIM)
        scales = blk + CACHE_BLOCK * (NOPE + 2 * ROPE_DIM) + p * 8
        for qb in tl.static_range(NOPE // 64):
            o = qb * 64 + tl.arange(0, 64)
            xk = tl.load(row + o)
            amax = tl.maximum(tl.max(tl.abs(xk), axis=0), 1e-4)
            e = tl.ceil(tl.log2(amax / 448.0))
            xs = tl.clamp(xk / tl.exp2(e), -448.0, 448.0)
            tl.store(data + o, xs.to(tl.float8e4nv).to(tl.uint8, bitcast=True))
            tl.store(scales + qb, tl.maximum(tl.minimum(e + 127.0, 255.0), 0.0).to(tl.uint8))
        tl.store(scales + 7, tl.zeros((), dtype=tl.uint8))
        ke = tl.load(row + NOPE + 2 * i).to(tl.float32)
        ko = tl.load(row + NOPE + 2 * i + 1).to(tl.float32)
        rp = (data + NOPE).to(tl.pointer_type(tl.bfloat16))
        tl.store(rp + 2 * i, (ke * cos - ko * sin).to(tl.bfloat16))
        tl.store(rp + 2 * i + 1, (ke * sin + ko * cos).to(tl.bfloat16))


def qnorm_rope_kv_insert(q, kv, cache_2d, slot_mapping, positions, cos_sin_cache,
                         padded_heads, eps, block_size):
    """Same arguments and result as torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert:
    returns q normalised and rotated, [T, padded_heads, 512] bf16 (padding heads zero)."""
    T, H, D = q.shape
    out = torch.empty(T, padded_heads, D, dtype=torch.bfloat16, device=q.device)
    n_kv = slot_mapping.shape[0]
    if T or n_kv:
        _qnorm_rope_kv_insert_kernel[(max(T, n_kv), padded_heads + 1)](
            q, kv, out, cache_2d, slot_mapping, positions, cos_sin_cache, T, n_kv,
            q.stride(0), q.stride(1), kv.stride(0), cache_2d.stride(0), eps,
            N_HEADS=H, PADDED=padded_heads, CACHE_BLOCK=block_size, HEAD_DIM=D,
            ROPE_DIM=cos_sin_cache.shape[-1], num_warps=4)
    return out


def install() -> None:
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention

    stock = DeepseekV4Attention._fused_qnorm_rope_kv_insert

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        if isinstance(attn_metadata, dict):
            swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
            cache = self.swa_cache_layer.kv_cache
            cs = self.rotary_emb.cos_sin_cache
            if (swa_metadata is not None and cache.dtype == torch.uint8
                    and q.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16
                    and q.shape[-1] == HEAD_DIM and cs.shape[-1] == ROPE_DIM
                    and q.stride(-1) == 1 and kv.stride(-1) == 1
                    and positions.dtype == torch.int64):
                return qnorm_rope_kv_insert(
                    q, kv, cache.view(cache.shape[0], -1), swa_metadata.slot_mapping,
                    positions, cs, self.padded_heads, self.eps, swa_metadata.block_size)
        return stock(self, q, kv, positions, attn_metadata)

    DeepseekV4Attention._fused_qnorm_rope_kv_insert = _fused_qnorm_rope_kv_insert
