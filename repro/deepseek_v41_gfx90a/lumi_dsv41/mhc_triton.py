"""V4.1's mHC (delayed pre and post) on gfx90a in two Triton kernels instead of torch.

Without aiter or TileLang (wrong on gfx90a, repro/dsv4_flash_debug), vLLM runs V4.1's
hyper-connections with the torch references in kernels/mhc/torch.py: about 40 small kernels
per call (gates, softmax, 20 Sinkhorn iterations, the collapse; the post's einsum), two
calls per layer, ~47 ms per decode step in the eager profile (job 23660212).

Here, per delayed pre: the fp32 prenorm GEMM and squared sum stay in torch (hipBLASLt; it
also serves prefill-sized batches), then one kernel computes the pre/post/comb gates,
Sinkhorn and the collapse with the carried pre-mix; it is V4's big-fuse kernel
(repro/dsv4_flash_debug/plugin/lumi_dsv4_debug/mhc_fix.py, served on V4) with the
delayed collapse. The post is one kernel. Same arithmetic as mhc_pre_delayed_torch and
mhc_post_torch, in fp32.
"""

import torch

from vllm.triton_utils import tl, triton

BLOCK_H = 512


@triton.jit
def _pre_delayed_kernel(
    mix_ptr, sqr_ptr, scale_ptr, base_ptr, res_ptr, carried_ptr,
    post_ptr, comb_ptr, next_pre_ptr, out_ptr,
    stride_rt, stride_rm, inv_k,
    rms_eps, pre_eps, sk_eps, post_mult,
    H: tl.constexpr, HC: tl.constexpr, SK_REPEAT: tl.constexpr, HAS_CARRIED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per (token, hidden block). Block 0 also writes the token's gates."""
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
        rms = tl.rsqrt(tl.load(sqr_ptr + t) * inv_k + rms_eps)
        m_pre = tl.load(mix_ptr + t * HC3 + j) * rms
        pre = tl.sigmoid(m_pre * tl.load(scale_ptr) + tl.load(base_ptr + j)) + pre_eps
        tl.store(next_pre_ptr + t * HC + j, pre)
        m_post = tl.load(mix_ptr + t * HC3 + HC + j) * rms
        post = tl.sigmoid(m_post * tl.load(scale_ptr + 1) + tl.load(base_ptr + HC + j))
        tl.store(post_ptr + t * HC + j, post * post_mult)
        cm = (tl.load(mix_ptr + t * HC3 + 2 * HC + jk) * rms * tl.load(scale_ptr + 2)
              + tl.load(base_ptr + 2 * HC + jk))
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
    assert hidden % BLOCK_H == 0 and fn.shape[0] == hc * (2 + hc)
    src = (residual.flatten(1) if x is None else x).float()
    mixes = src @ fn.t()  # [T, hc3] fp32
    sqrsum = src.square().sum(-1)  # [T]
    residual = _rows(residual)
    carried = pre_mix.float().contiguous() if pre_mix is not None else next_pre
    _pre_delayed_kernel[(t, hidden // BLOCK_H)](
        mixes, sqrsum, hc_scale, hc_base, residual, carried,
        post, comb, next_pre, layer_input,
        residual.stride(0), residual.stride(1), 1.0 / src.shape[-1],
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        H=hidden, HC=hc, SK_REPEAT=sinkhorn_repeat, HAS_CARRIED=pre_mix is not None,
        BLOCK=BLOCK_H, num_warps=4,
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
