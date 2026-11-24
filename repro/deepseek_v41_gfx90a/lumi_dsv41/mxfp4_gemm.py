"""MXFP4 weight GEMMs on gfx90a: where the time goes (diagnosis for a faster MoE kernel).

V4.1's experts are MXFP4 (e2m1 values, one e8m0 scale per 32 along K). gfx90a has no FP4
matrix unit, so triton_kernels' matmul_ogs ends in tl.dot_scaled(x bf16, w e2m1), which the
compiler expands into a software upcast and a bf16 dot. Its expert GEMMs run at 2-3x their
weight-read floor (gate/up 775 us at 96 tokens against ~350 us to read 566 MB; README).

One grouped GEMM over E experts, each [M, K] bf16 x [K, N] MXFP4 (bytes [E, K/2, N], two
values per byte along K, low nibble first; scales [E, N, K/32]), in four modes:
  0  tl.dot_scaled, as matmul_ogs
  1  explicit upcast: e2m1 -> bf16 bits by integer ops, the scale added to the exponent
     field (exact for normal results), then tl.dot
  2  bf16 weights converted beforehand ([E, K, N], 4x the bytes): the dot without upcast
  3  the FP4 bytes and scales loaded, no upcast or dot: the weight-read floor
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _upcast_e2m1(packed, scales, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    """packed [BLOCK_K // 2, BLOCK_N] uint8 (low nibble = even k), scales [BLOCK_N,
    BLOCK_K // 32] uint8 e8m0 -> [BLOCK_K, BLOCK_N] bf16."""
    b = packed.to(tl.int32)
    n = tl.join(b & 15, b >> 4)  # [K/2, N, 2]
    n = tl.permute(n, (0, 2, 1)).reshape(BLOCK_K, BLOCK_N)
    e = (n >> 1) & 3
    m = n & 1
    exp = tl.where(e == 0, 126, e + 126)
    mant = tl.where(e == 0, 0, m << 6)
    s = tl.trans(scales).to(tl.int32) - 127  # [K/32, N]
    exp = (exp.reshape(BLOCK_K // 32, 32, BLOCK_N) + s[:, None, :]).reshape(BLOCK_K, BLOCK_N)
    bits = tl.where((n & 7) != 0, (exp << 7) | mant, 0) | ((n & 8) << 12)
    return bits.to(tl.int16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _gemm_kernel(x_ptr, w_ptr, s_ptr, w16_ptr, y_ptr, M, N, K,
                 MODE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_K: tl.constexpr):
    pm, pn, ex = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    kk = tl.arange(0, BLOCK_K)
    kp = tl.arange(0, BLOCK_K // 2)
    ks = tl.arange(0, BLOCK_K // 32)
    xb = x_ptr + ex * M * K + rm[:, None] * K
    wb = w_ptr + ex * (K // 2) * N
    sb = s_ptr + ex * N * (K // 32) + rn[:, None] * (K // 32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        if MODE == 2:
            w16 = tl.load(w16_ptr + ex * K * N + (k0 + kk)[:, None] * N + rn[None, :])
            x = tl.load(xb + k0 + kk[None, :], mask=(rm < M)[:, None], other=0.0)
            acc = tl.dot(x, w16, acc)
        else:
            w = tl.load(wb + (k0 // 2 + kp)[:, None] * N + rn[None, :])
            sc = tl.load(sb + k0 // 32 + ks[None, :])
            if MODE == 3:
                acc += tl.sum(w.to(tl.float32), 0)[None, :] + tl.sum(sc.to(tl.float32), 1)[None, :]
            else:
                x = tl.load(xb + k0 + kk[None, :], mask=(rm < M)[:, None], other=0.0)
                if MODE == 0:
                    acc = tl.dot_scaled(x, None, "bf16", w, sc, "e2m1", acc=acc, fast_math=True)
                else:
                    acc = tl.dot(x, _upcast_e2m1(w, sc, BLOCK_K, BLOCK_N), acc)
    tl.store(y_ptr + ex * M * N + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16),
             mask=(rm < M)[:, None])


def grouped_gemm(x, w, scales, mode, w16=None, block_m=16, block_n=128, block_k=128,
                 num_warps=4, **launch):
    """x [E, M, K] bf16, w [E, K/2, N] uint8, scales [E, N, K/32] uint8 -> [E, M, N] bf16."""
    E, M, K = x.shape
    N = w.shape[-1]
    y = torch.empty(E, M, N, dtype=torch.bfloat16, device=x.device)
    _gemm_kernel[(triton.cdiv(M, block_m), triton.cdiv(N, block_n), E)](
        x, w, scales, w16 if w16 is not None else x, y, M, N, K, MODE=mode,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps, **launch)
    return y


def dequant(w, scales):
    """Reference: [E, K/2, N] uint8 + [E, N, K/32] e8m0 -> [E, K, N] fp32."""
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                       device=w.device)
    lo, hi = lut[(w & 15).long()], lut[(w >> 4).long()]
    E, K2, N = w.shape
    vals = torch.stack([lo, hi], dim=2).reshape(E, K2 * 2, N)  # k = 2i (lo), 2i+1 (hi)
    s = torch.exp2(scales.float() - 127).transpose(1, 2)  # [E, K/32, N]
    return vals * s.repeat_interleave(32, dim=1)
