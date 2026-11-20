"""V4.1's mHC (delayed pre and post) on gfx90a in two Triton kernels instead of torch.

Without aiter or TileLang (wrong on gfx90a, repro/dsv4_flash_debug), vLLM runs V4.1's
hyper-connections with the torch references in kernels/mhc/torch.py: about 40 small kernels
per call (gates, softmax, 20 Sinkhorn iterations, the collapse; the post's einsum), two
calls per layer, ~47 ms per decode step in the eager profile (job 23660212).

Here, per delayed pre: the fp32 prenorm GEMM and squared sum in one split-K kernel for
decode-sized batches (torch/hipBLASLt for larger ones), then one kernel computes the
pre/post/comb gates,
Sinkhorn and the collapse with the carried pre-mix; it is V4's big-fuse kernel
(repro/dsv4_flash_debug/plugin/lumi_dsv4_debug/mhc_fix.py, served on V4) with the
delayed collapse. The post is one kernel. Same arithmetic as mhc_pre_delayed_torch and
mhc_post_torch, in fp32.
"""

import torch

from vllm.triton_utils import tl, triton

BLOCK_H = 512
# Split-K prenorm GEMM for decode-sized batches; larger batches use torch (hipBLASLt).
# Split-K beats the torch GEMM (fp32 copy + hipBLASLt fp32 GEMM + square-sum) up to ~256
# tokens (job 23871480: 0.099 vs 0.195 ms at 96, 0.155 vs 0.197 at 192, equal at 384);
# DSpark's verify batches (6 tokens per sequence) land in that range.
GEMM_MAX_TOKENS, GEMM_CHUNK, GEMM_BK = 256, 512, 256


@triton.jit
def _prenorm_gemm_kernel(
    x_ptr, fn_ptr, mix_ptr, sqr_ptr, num_tokens, stride_xt, K,
    HC3: tl.constexpr, R: tl.constexpr, CHUNK: tl.constexpr, BK: tl.constexpr,
):
    """Program (token t, split s): partial x[t, chunk] @ fn[:, chunk].T and sum x^2 over the
    chunk, in fp32, into mix[s, t, :] and sqr[s, t]."""
    t = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    r = tl.arange(0, R)
    acc = tl.zeros([R], dtype=tl.float32)
    sq = tl.zeros([BK], dtype=tl.float32)
    for k0 in tl.static_range(0, CHUNK, BK):
        k = s * CHUNK + k0 + tl.arange(0, BK)
        xk = tl.load(x_ptr + t * stride_xt + k).to(tl.float32)
        f = tl.load(fn_ptr + r[:, None] * K + k[None, :], mask=(r < HC3)[:, None], other=0.0)
        acc += tl.sum(f * xk[None, :], 1)
        sq += xk * xk
    tl.store(mix_ptr + (s * num_tokens + t) * HC3 + r, acc, mask=r < HC3)
    tl.store(sqr_ptr + s * num_tokens + t, tl.sum(sq, 0))


@triton.jit
def _prenorm_gemm_tiled_kernel(
    x_ptr, fn_ptr, mix_ptr, sqr_ptr, num_tokens, stride_xt, K,
    HC3: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Program (token block m, split s): x[m-block, chunk] @ fn[:, chunk].T with an fp32
    matrix-core dot (IEEE, no bf16/tf32 rounding of fn), plus sum x^2 over the chunk, into
    mix[s, rows, :] and sqr[s, rows]. The per-token kernel above is a vector sum per token,
    so its cost grows with T; this one reads fn once per token block."""
    pm = tl.program_id(0)
    s = tl.program_id(1)
    rows = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < num_tokens
    n = tl.arange(0, BLOCK_N)
    nmask = n < HC3
    kk = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k0 in tl.range(0, CHUNK, BLOCK_K):
        k = s * CHUNK + k0 + kk
        xk = tl.load(x_ptr + rows.to(tl.int64)[:, None] * stride_xt + k[None, :],
                     mask=rmask[:, None], other=0.0).to(tl.float32)
        f = tl.load(fn_ptr + n[None, :] * K + k[:, None], mask=nmask[None, :], other=0.0)
        acc = tl.dot(xk, f, acc, input_precision="ieee")
        sq += tl.sum(xk * xk, 1)
    tl.store(mix_ptr + (s * num_tokens + rows)[:, None] * HC3 + n[None, :], acc,
             mask=rmask[:, None] & nmask[None, :])
    tl.store(sqr_ptr + s * num_tokens + rows, sq, mask=rmask)


# Tiled prenorm GEMM from this many tokens on (None: off); (BLOCK_M, BLOCK_K, CHUNK, warps).
TILED_MIN_TOKENS = None
TILED_CFG = (32, 64, 1024, 4)


@triton.jit
def _pre_delayed_kernel(
    mix_ptr, sqr_ptr, scale_ptr, base_ptr, res_ptr, carried_ptr,
    post_ptr, comb_ptr, next_pre_ptr, out_ptr,
    num_tokens, stride_rt, stride_rm, inv_k,
    rms_eps, pre_eps, sk_eps, post_mult,
    H: tl.constexpr, HC: tl.constexpr, SK_REPEAT: tl.constexpr, HAS_CARRIED: tl.constexpr,
    N_SPLITS: tl.constexpr, BLOCK: tl.constexpr,
):
    """One program per (token, hidden block). Block 0 also writes the token's gates.
    mix [N_SPLITS, T, hc3] and sqr [N_SPLITS, T] are summed over the splits."""
    t = tl.program_id(0).to(tl.int64)
    hb = tl.program_id(1)
    HC3: tl.constexpr = HC * (2 + HC)
    j = tl.arange(0, HC)
    jk = j[:, None] * HC + j[None, :]

    if HAS_CARRIED:
        w = tl.load(carried_ptr + t * HC + j)
    else:  # no carried pre-mix: the layer input is stream 0
        w = tl.where(j == 0, 1.0, 0.0)

    if hb == 0:
        sq = tl.load(sqr_ptr + t)
        m_pre = tl.load(mix_ptr + t * HC3 + j)
        m_post = tl.load(mix_ptr + t * HC3 + HC + j)
        m_comb = tl.load(mix_ptr + t * HC3 + 2 * HC + jk)
        for s in tl.static_range(1, N_SPLITS):
            row = (s * num_tokens + t) * HC3
            sq += tl.load(sqr_ptr + s * num_tokens + t)
            m_pre += tl.load(mix_ptr + row + j)
            m_post += tl.load(mix_ptr + row + HC + j)
            m_comb += tl.load(mix_ptr + row + 2 * HC + jk)
        rms = tl.rsqrt(sq * inv_k + rms_eps)
        pre = tl.sigmoid(m_pre * rms * tl.load(scale_ptr) + tl.load(base_ptr + j)) + pre_eps
        tl.store(next_pre_ptr + t * HC + j, pre)
        post = tl.sigmoid(m_post * rms * tl.load(scale_ptr + 1) + tl.load(base_ptr + HC + j))
        tl.store(post_ptr + t * HC + j, post * post_mult)
        cm = m_comb * rms * tl.load(scale_ptr + 2) + tl.load(base_ptr + 2 * HC + jk)
        cm = tl.exp(cm - tl.max(cm, 1)[:, None])
        cm = cm / tl.sum(cm, 1)[:, None] + sk_eps
        cm = cm / (tl.sum(cm, 0)[None, :] + sk_eps)
        for _ in tl.static_range(SK_REPEAT - 1):
            cm = cm / (tl.sum(cm, 1)[:, None] + sk_eps)
            cm = cm / (tl.sum(cm, 0)[None, :] + sk_eps)
        tl.store(comb_ptr + t * HC * HC + jk, cm)

    offs = hb * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(res_ptr + t * stride_rt + j[:, None] * stride_rm + offs[None, :]).to(tl.float32)
    tl.store(out_ptr + t * H + offs,
             tl.sum(w[:, None] * x, 0).to(out_ptr.dtype.element_ty))


@triton.jit
def _post_kernel(
    x_ptr, res_ptr, post_ptr, comb_ptr, out_ptr,
    stride_xt, stride_rt, stride_rm,
    H: tl.constexpr, HC: tl.constexpr, BLOCK: tl.constexpr,
):
    """out[t, j, h] = post[t, j] * x[t, h] + sum_i comb[t, i, j] * res[t, i, h]."""
    t = tl.program_id(0).to(tl.int64)
    hb = tl.program_id(1)
    j = tl.arange(0, HC)
    offs = hb * BLOCK + tl.arange(0, BLOCK)
    res = tl.load(res_ptr + t * stride_rt + j[:, None] * stride_rm + offs[None, :]).to(tl.float32)
    comb = tl.load(comb_ptr + t * HC * HC + j[:, None] * HC + j[None, :])  # [i, j]
    post = tl.load(post_ptr + t * HC + j)
    x = tl.load(x_ptr + t * stride_xt + offs).to(tl.float32)
    mixed = tl.sum(comb[:, :, None] * res[:, None, :], 0)  # [j, h]
    out = mixed + post[:, None] * x[None, :]
    tl.store(out_ptr + t * HC * H + j[:, None] * H + offs[None, :],
             out.to(out_ptr.dtype.element_ty))


def _rows(t: torch.Tensor) -> torch.Tensor:
    return t if t.stride(-1) == 1 else t.contiguous()


def mhc_pre_delayed_triton(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                           hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                           pre_mix=None, x=None):
    """Same arguments and returns as kernels/mhc/torch.py:mhc_pre_delayed_torch:
    post_mix [T, hc, 1], comb_mix [T, hc, hc], layer_input [T, H], next_pre_mix [T, hc]."""
    t, hc, hidden = residual.shape
    dev = residual.device
    post = torch.empty(t, hc, 1, dtype=torch.float32, device=dev)
    comb = torch.empty(t, hc, hc, dtype=torch.float32, device=dev)
    next_pre = torch.empty(t, hc, dtype=torch.float32, device=dev)
    layer_input = torch.empty(t, hidden, dtype=residual.dtype, device=dev)
    if t == 0:
        return post, comb, layer_input, next_pre
    hc3 = hc * (2 + hc)
    assert hidden % BLOCK_H == 0 and fn.shape[0] == hc3
    src = residual.reshape(t, hc * hidden) if x is None else x  # a view when contiguous
    k = src.shape[-1]
    bm, bk, chunk, warps = TILED_CFG
    if (TILED_MIN_TOKENS is not None and t >= TILED_MIN_TOKENS and k % chunk == 0
            and fn.is_contiguous() and fn.dtype == torch.float32):
        src = _rows(src)
        splits = k // chunk
        mixes = torch.empty(splits, t, hc3, dtype=torch.float32, device=dev)
        sqrsum = torch.empty(splits, t, dtype=torch.float32, device=dev)
        _prenorm_gemm_tiled_kernel[(triton.cdiv(t, bm), splits)](
            src, fn, mixes, sqrsum, t, src.stride(0), k,
            HC3=hc3, BLOCK_M=bm, BLOCK_N=triton.next_power_of_2(hc3), BLOCK_K=bk,
            CHUNK=chunk, num_warps=warps,
        )
    elif t <= GEMM_MAX_TOKENS and k % GEMM_CHUNK == 0 and fn.is_contiguous():
        src = _rows(src)
        splits = k // GEMM_CHUNK
        mixes = torch.empty(splits, t, hc3, dtype=torch.float32, device=dev)
        sqrsum = torch.empty(splits, t, dtype=torch.float32, device=dev)
        _prenorm_gemm_kernel[(t, splits)](
            src, fn, mixes, sqrsum, t, src.stride(0), k,
            HC3=hc3, R=triton.next_power_of_2(hc3), CHUNK=GEMM_CHUNK, BK=GEMM_BK, num_warps=4,
        )
    else:
        src = src.float()
        mixes = (src @ fn.t()).unsqueeze(0)  # [1, T, hc3] fp32
        sqrsum = src.square().sum(-1).unsqueeze(0)  # [1, T]
    residual = _rows(residual)
    carried = pre_mix.float().contiguous() if pre_mix is not None else next_pre
    _pre_delayed_kernel[(t, hidden // BLOCK_H)](
        mixes, sqrsum, hc_scale, hc_base, residual, carried,
        post, comb, next_pre, layer_input,
        t, residual.stride(0), residual.stride(1), 1.0 / k,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        H=hidden, HC=hc, SK_REPEAT=sinkhorn_repeat, HAS_CARRIED=pre_mix is not None,
        N_SPLITS=mixes.shape[0], BLOCK=BLOCK_H, num_warps=4,
    )
    return post, comb, layer_input, next_pre


def mhc_post_triton(x, residual, post_layer_mix, comb_res_mix):
    """Same as kernels/mhc/torch.py:mhc_post_torch for x [T, H], residual [T, hc, H]."""
    t, hc, hidden = residual.shape
    out = torch.empty(t, hc, hidden, dtype=residual.dtype, device=residual.device)
    if t == 0:
        return out
    assert hidden % BLOCK_H == 0
    x, residual = _rows(x), _rows(residual)
    post = post_layer_mix.float().reshape(t, hc).contiguous()
    comb = comb_res_mix.float().reshape(t, hc, hc).contiguous()
    _post_kernel[(t, hidden // BLOCK_H)](
        x, residual, post, comb, out,
        x.stride(0), residual.stride(0), residual.stride(1),
        H=hidden, HC=hc, BLOCK=BLOCK_H, num_warps=4,
    )
    return out
