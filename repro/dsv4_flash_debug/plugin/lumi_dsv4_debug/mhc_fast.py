"""DSV4_MHC_FAST=1: V4's hyper-connections (fused post+pre, and the first pre) in Triton.

With MHC_FIX=triton the served path is vLLM's TileLang wrapper with our big-fuse kernel:
for more than 16 tokens TileLang's post kernel, then its prenorm GEMM with one split (no
split-K without DeepGEMM), then the big fuse; up to 16 tokens TileLang's fused post+GEMM.
At 32 users with MTP (64 tokens) that is 18.7 + 43.6 + 18.0 us per call, 88 calls, ~7 ms
of a 54.5 ms step (graph profile job 23861348); the GEMM alone reads 1.5 MB of weights and
2 MB of activations per call, ~3 us of HBM time.

From the V4.1 port (repro/deepseek_v41_gfx90a/lumi_dsv41/mhc_triton.py): the post as a
Triton kernel and the prenorm GEMM split over K with an fp32 matrix-core dot (IEEE), the
partial sums left for the big-fuse kernel, which already adds up any number of splits.
Here the post and the GEMM can also be one kernel: each program mixes a hidden slice of
all four streams, stores it, and multiplies the bf16-rounded result by the same slice of
fn, as the reference does (mhc_post_torch, then mhc_pre_torch on its bf16 output).
"""

import torch

from vllm.triton_utils import tl, triton

from .mhc_fix import BLOCK_H, _pre_big_fuse_kernel

HC = 4  # V4's hc_mult; the kernels below are written for it


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


@triton.jit
def _prenorm_rows_kernel(
    x_ptr, fn_ptr, mix_ptr, sqr_ptr, num_tokens, stride_xt, K,
    HC3: tl.constexpr, R: tl.constexpr, CHUNK: tl.constexpr, BK: tl.constexpr,
):
    """Program (token t, split s): x[t, chunk] @ fn[:, chunk].T and sum x^2 over the chunk,
    fp32, into mix[s, t, :] and sqr[s, t]. A vector sum per token (V4.1's split-K kernel)."""
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
def _prenorm_tiled_kernel(
    x_ptr, fn_ptr, mix_ptr, sqr_ptr, num_tokens, stride_xt, K,
    HC3: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Program (token block m, split s): x[m-block, chunk] @ fn[:, chunk].T with an fp32
    matrix-core dot (IEEE), plus sum x^2 over the chunk, into mix[s, rows, :] and sqr[s, rows].
    Reads fn once per token block (V4.1's tiled kernel)."""
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


@triton.jit
def _post_stream(
    j: tl.constexpr, rows, rmask, h, xt, r0, r1, r2, r3, post_ptr, comb_ptr, out_ptr, fn_ptr,
    n, nmask, acc, sq, H: tl.constexpr,
):
    """Stream j of the post for a [BLOCK_M, CH] slice; stores it and adds its prenorm GEMM
    and squared-sum contributions (on the bf16-rounded value, as the reference)."""
    pj = tl.load(post_ptr + rows * 4 + j, mask=rmask, other=0.0)
    c0 = tl.load(comb_ptr + rows * 16 + 0 + j, mask=rmask, other=0.0)
    c1 = tl.load(comb_ptr + rows * 16 + 4 + j, mask=rmask, other=0.0)
    c2 = tl.load(comb_ptr + rows * 16 + 8 + j, mask=rmask, other=0.0)
    c3 = tl.load(comb_ptr + rows * 16 + 12 + j, mask=rmask, other=0.0)
    o = (c0[:, None] * r0 + c1[:, None] * r1 + c2[:, None] * r2 + c3[:, None] * r3
         + pj[:, None] * xt)
    ob = o.to(tl.bfloat16)
    tl.store(out_ptr + rows.to(tl.int64)[:, None] * (4 * H) + j * H + h[None, :], ob,
             mask=rmask[:, None])
    of = ob.to(tl.float32)
    f = tl.load(fn_ptr + n[None, :] * (4 * H) + j * H + h[:, None], mask=nmask[None, :],
                other=0.0)
    acc = tl.dot(of, f, acc, input_precision="ieee")
    sq += tl.sum(of * of, 1)
    return acc, sq


@triton.jit
def _fused_post_prenorm_kernel(
    x_ptr, res_ptr, post_ptr, comb_ptr, fn_ptr, out_ptr, mix_ptr, sqr_ptr,
    num_tokens, stride_xt, stride_rt, stride_rm,
    H: tl.constexpr, HC3: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CH: tl.constexpr,
):
    """Program (token block m, hidden slice s): the post for all four streams on hidden
    [s*CH, (s+1)*CH), stored to out, and its share of the prenorm GEMM into mix[s]/sqr[s]."""
    pm = tl.program_id(0)
    s = tl.program_id(1)
    rows = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < num_tokens
    r64 = rows.to(tl.int64)
    h = s * CH + tl.arange(0, CH)
    n = tl.arange(0, BLOCK_N)
    nmask = n < HC3
    m2 = rmask[:, None]
    xt = tl.load(x_ptr + r64[:, None] * stride_xt + h[None, :], mask=m2, other=0.0).to(tl.float32)
    rb = res_ptr + r64[:, None] * stride_rt + h[None, :]
    r0 = tl.load(rb, mask=m2, other=0.0).to(tl.float32)
    r1 = tl.load(rb + stride_rm, mask=m2, other=0.0).to(tl.float32)
    r2 = tl.load(rb + 2 * stride_rm, mask=m2, other=0.0).to(tl.float32)
    r3 = tl.load(rb + 3 * stride_rm, mask=m2, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for j in tl.static_range(4):
        acc, sq = _post_stream(j, rows, rmask, h, xt, r0, r1, r2, r3, post_ptr, comb_ptr,
                               out_ptr, fn_ptr, n, nmask, acc, sq, H)
    tl.store(mix_ptr + (s * num_tokens + rows)[:, None] * HC3 + n[None, :], acc,
             mask=rmask[:, None] & nmask[None, :])
    tl.store(sqr_ptr + s * num_tokens + rows, sq, mask=rmask)


# Per decode-sized token count (first row with m_max >= T): how to run the prenorm GEMM.
#   ("rows", CHUNK, BK, warps)            post kernel, then the per-token split-K GEMM
#   ("tiled", BLOCK_M, BLOCK_K, CHUNK, warps)  post kernel, then the tiled split-K GEMM
#   ("fused", BLOCK_M, CH, warps)         post and GEMM in one kernel
# and the big-fuse kernel's warps. Filled from test_mhc_fast.py; None: vLLM's path.
CONFIGS = [
    (128, ("fused", 16, 128, 4), 4),
]


def pick(t: int):
    for m_max, cfg, bf_warps in CONFIGS:
        if t <= m_max:
            return cfg, bf_warps
    return None


def _rows(t: torch.Tensor) -> torch.Tensor:
    return t if t.stride(-1) == 1 else t.contiguous()


def _prenorm(src, fn, cfg):
    """src [T, K] bf16 (rows may be strided), fn [hc3, K] fp32 -> partial mix [S, T, hc3],
    sqr [S, T]."""
    t, k = src.shape
    hc3 = fn.shape[0]
    dev = src.device
    if cfg[0] == "tiled":
        _, bm, bk, chunk, warps = cfg
        splits = k // chunk
        mix = torch.empty(splits, t, hc3, dtype=torch.float32, device=dev)
        sqr = torch.empty(splits, t, dtype=torch.float32, device=dev)
        _prenorm_tiled_kernel[(triton.cdiv(t, bm), splits)](
            src, fn, mix, sqr, t, src.stride(0), k, HC3=hc3, BLOCK_M=bm,
            BLOCK_N=triton.next_power_of_2(hc3), BLOCK_K=bk, CHUNK=chunk, num_warps=warps)
    else:
        _, chunk, bk, warps = cfg if cfg[0] == "rows" else ("rows", 512, 256, 4)
        splits = k // chunk
        mix = torch.empty(splits, t, hc3, dtype=torch.float32, device=dev)
        sqr = torch.empty(splits, t, dtype=torch.float32, device=dev)
        _prenorm_rows_kernel[(t, splits)](
            src, fn, mix, sqr, t, src.stride(0), k, HC3=hc3,
            R=triton.next_power_of_2(hc3), CHUNK=chunk, BK=bk, num_warps=warps)
    return mix, sqr


def _big_fuse(mix, sqr, hc_scale, hc_base, residual, hidden, rms_eps, hc_pre_eps,
              hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, warps):
    t = residual.shape[0]
    dev = residual.device
    post = torch.empty(t, HC, dtype=torch.float32, device=dev)
    comb = torch.empty(t, HC * HC, dtype=torch.float32, device=dev)
    layer_input = torch.empty(t, hidden, dtype=torch.bfloat16, device=dev)
    _pre_big_fuse_kernel[(t, hidden // BLOCK_H)](
        mix, sqr, hc_scale, hc_base, residual, post, comb, layer_input, t,
        residual.stride(0), residual.stride(1), rms_eps, hc_pre_eps, hc_sinkhorn_eps,
        hc_post_mult_value, H=hidden, HC=HC, N_SPLITS=mix.shape[0], SK_REPEAT=sinkhorn_repeat,
        BLOCK=BLOCK_H, num_warps=warps)
    return post, comb, layer_input


def fused_post_pre(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps,
                   hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                   choice=None):
    """mhc_post_torch then mhc_pre_torch on its output; returns residual_cur [.., hc, H],
    post_mix [.., hc, 1], comb_mix [.., hc, hc], layer_input [.., H] like vLLM's op.
    choice: (cfg, big-fuse warps), default pick(T)."""
    hc, hidden = residual.shape[-2], residual.shape[-1]
    outer = residual.shape[:-2]
    res = _rows(residual.reshape(-1, hc, hidden))
    t = res.shape[0]
    cfg, bf_warps = choice or pick(t)
    dev = residual.device
    xf = _rows(x.reshape(t, hidden))
    post_in = post_layer_mix.float().reshape(t, hc).contiguous()
    comb_in = comb_res_mix.float().reshape(t, hc * hc).contiguous()
    cur = torch.empty(t, hc, hidden, dtype=residual.dtype, device=dev)
    if cfg[0] == "fused":
        _, bm, ch, warps = cfg
        hc3 = fn.shape[0]
        splits = hidden // ch
        mix = torch.empty(splits, t, hc3, dtype=torch.float32, device=dev)
        sqr = torch.empty(splits, t, dtype=torch.float32, device=dev)
        _fused_post_prenorm_kernel[(triton.cdiv(t, bm), splits)](
            xf, res, post_in, comb_in, fn, cur, mix, sqr, t, xf.stride(0), res.stride(0),
            res.stride(1), H=hidden, HC3=hc3, BLOCK_M=bm,
            BLOCK_N=triton.next_power_of_2(hc3), CH=ch, num_warps=warps)
    else:
        _post_kernel[(t, hidden // BLOCK_H)](
            xf, res, post_in, comb_in, cur, xf.stride(0), res.stride(0), res.stride(1),
            H=hidden, HC=hc, BLOCK=BLOCK_H, num_warps=4)
        mix, sqr = _prenorm(cur.view(t, hc * hidden), fn, cfg)
    post, comb, layer_input = _big_fuse(
        mix, sqr, hc_scale, hc_base, cur, hidden, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
        hc_post_mult_value, sinkhorn_repeat, bf_warps)
    return (cur.view(*outer, hc, hidden), post.view(*outer, hc, 1),
            comb.view(*outer, hc, hc), layer_input.view(*outer, hidden))


def pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
        hc_post_mult_value, sinkhorn_repeat, choice=None):
    """mhc_pre_torch: post_mix [.., hc, 1], comb_mix [.., hc, hc], layer_input [.., H]."""
    hc, hidden = residual.shape[-2], residual.shape[-1]
    outer = residual.shape[:-2]
    res = _rows(residual.reshape(-1, hc, hidden))
    t = res.shape[0]
    cfg, bf_warps = choice or pick(t)
    if cfg[0] == "fused":
        cfg = ("tiled", cfg[1], 128, 512, cfg[3])
    src = res.view(t, hc * hidden) if res.is_contiguous() else res.reshape(t, hc * hidden)
    mix, sqr = _prenorm(src, fn, cfg)
    post, comb, layer_input = _big_fuse(
        mix, sqr, hc_scale, hc_base, res, hidden, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
        hc_post_mult_value, sinkhorn_repeat, bf_warps)
    return post.view(*outer, hc, 1), comb.view(*outer, hc, hc), layer_input.view(*outer, hidden)


def install() -> None:
    """Route vLLM's MHCFusedPostPreOp and MHCPreOp (forward_hip) here for the token counts
    in CONFIGS; anything else (more tokens, norm_weight, other shapes) keeps vLLM's path."""
    import vllm.model_executor.layers.mhc as m

    fused_stock = m.MHCFusedPostPreOp.forward_hip
    pre_stock = m.MHCPreOp.forward_hip

    def ok(residual, fn, norm_weight):
        hc, hidden = residual.shape[-2], residual.shape[-1]
        t = residual.numel() // (hc * hidden)
        return (norm_weight is None and hc == HC and hidden % 512 == 0 and t > 0
                and fn.dtype == torch.float32 and fn.is_contiguous() and pick(t) is not None)

    def fused_forward_hip(self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale,
                          hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                          sinkhorn_repeat, n_splits=1, tile_n=1, norm_weight=None,
                          norm_eps=0.0):
        if not ok(residual, fn, norm_weight):
            return fused_stock(self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale,
                               hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                               hc_post_mult_value, sinkhorn_repeat, n_splits, tile_n,
                               norm_weight, norm_eps)
        return fused_post_pre(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale,
                              hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                              hc_post_mult_value, sinkhorn_repeat)

    def pre_forward_hip(self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits=1,
                        norm_weight=None, norm_eps=0.0):
        if not ok(residual, fn, norm_weight):
            return pre_stock(self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                             hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits,
                             norm_weight, norm_eps)
        return pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                   hc_post_mult_value, sinkhorn_repeat)

    m.MHCFusedPostPreOp.forward_hip = fused_forward_hip
    m.MHCPreOp.forward_hip = pre_forward_hip
