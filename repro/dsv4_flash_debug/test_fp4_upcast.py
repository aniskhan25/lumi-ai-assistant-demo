"""Is Triton's MXFP4 upcast what bounds the expert GEMMs on gfx90a? (one GCD)

triton_kernels' matmul_ogs runs tl.dot_scaled(x bf16, w e2m1, scales) per K step; on gfx90a
(no FP4 or bf16 arithmetic) Triton expands it in software. test_moe_roofline.py (job
23908569) found the MXFP4 GEMMs at 0.5-0.7 TB/s on the routed weights against 1.0-1.3 TB/s
for the same GEMMs with bf16 weights. Here one plain GEMM, M = 16 rows (decode) against a
K = 4096 x N weight as wide as ~128 experts' gate/up, four ways:

  dot_scaled  tl.dot_scaled(x, None, "bf16", w, w_scale, "e2m1", fast_math=True), as matmul_ogs
  fp32 bits   each nibble's 3 magnitude bits into an fp32 pattern, one multiply by 2^(s-1)
              (the 2^126 rebias and the ue8m0 scale together, exact), the high 16 bits as
              bf16 (exact: an FP4 value has 2 significant bits), then tl.dot
  int bits    the bf16 pattern built with integer ops only (exponent add of the scale,
              a select for the 0.5 subnormal), then tl.dot
  bf16        bf16 weights, plain tl.dot: the memory-bound reference (4x the bytes)

The fp32/int variants never interleave the two nibbles of a byte: x's even and odd K columns
multiply the low and high nibbles separately. Each variant is checked against a torch
dequantisation; times are GPU us per call in graph replay, bandwidth on the weight bytes.

    python test_fp4_upcast.py
"""

import torch
import triton
import triton.language as tl

dev = "cuda"
M, K = 16, 4096
N = int(__import__("os").environ.get("FP4_N", 512 * 128))
torch.manual_seed(0)


@triton.jit
def _gemm(x_ptr, w_ptr, s_ptr, y_ptr, N, K: tl.constexpr, M: tl.constexpr,
          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, MODE: tl.constexpr):
    pn = tl.program_id(0)
    n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    m = tl.arange(0, M)
    acc = tl.zeros([M, BLOCK_N], dtype=tl.float32)
    PK: tl.constexpr = BLOCK_K // 2
    G: tl.constexpr = BLOCK_K // 32
    for k0 in range(0, K, BLOCK_K):
        if MODE == 3:  # bf16 weights [K, N]
            k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(x_ptr + m[:, None] * K + k[None, :])
            w = tl.load(w_ptr + k[:, None] * N + n[None, :])
            acc = tl.dot(x, w, acc)
        else:
            pk = k0 // 2 + tl.arange(0, PK)
            w = tl.load(w_ptr + pk[:, None] * N + n[None, :])  # [PK, BN] uint8, 2 k per byte
            g = k0 // 32 + tl.arange(0, G)
            s = tl.load(s_ptr + n[:, None] * (K // 32) + g[None, :])  # [BN, G] ue8m0
            if MODE == 0:
                k = k0 + tl.arange(0, BLOCK_K)
                x = tl.load(x_ptr + m[:, None] * K + k[None, :])
                acc = tl.dot_scaled(x, None, "bf16", w, s, "e2m1", acc=acc, fast_math=True)
            else:
                ke = k0 + 2 * tl.arange(0, PK)
                xe = tl.load(x_ptr + m[:, None] * K + ke[None, :])
                xo = tl.load(x_ptr + m[:, None] * K + ke[None, :] + 1)
                # scale per (k row, n): group = (2 * pk) // 32 = pk // 16
                st = tl.reshape(tl.broadcast_to(tl.trans(s)[:, None, :], [G, 16, BLOCK_N]),
                                [PK, BLOCK_N]).to(tl.int32)
                wi = w.to(tl.int32)
                lo = wi & 0xF
                hi = (wi >> 4) & 0xF
                if MODE == 1:
                    c = ((st - 1) << 23).to(tl.float32, bitcast=True)  # 2^(s-1)
                    flo = (((lo & 7) << 22) | ((lo & 8) << 28)).to(tl.float32, bitcast=True) * c
                    fhi = (((hi & 7) << 22) | ((hi & 8) << 28)).to(tl.float32, bitcast=True) * c
                    blo = (flo.to(tl.int32, bitcast=True) >> 16).to(tl.int16).to(tl.bfloat16, bitcast=True)
                    bhi = (fhi.to(tl.int32, bitcast=True) >> 16).to(tl.int16).to(tl.bfloat16, bitcast=True)
                else:
                    e = (st - 1) << 7  # bf16 exponent add for the scale
                    vlo = tl.where((lo & 6) == 0, tl.where((lo & 1) == 1, e, 0), ((lo & 7) << 6) + e)
                    vhi = tl.where((hi & 6) == 0, tl.where((hi & 1) == 1, e, 0), ((hi & 7) << 6) + e)
                    blo = (vlo | ((lo & 8) << 12)).to(tl.int16).to(tl.bfloat16, bitcast=True)
                    bhi = (vhi | ((hi & 8) << 12)).to(tl.int16).to(tl.bfloat16, bitcast=True)
                acc = tl.dot(xe, blo, acc)
                acc = tl.dot(xo, bhi, acc)
    tl.store(y_ptr + m[:, None] * N + n[None, :], acc)


E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                    device=dev)


def dequant(wq, s):
    lo, hi = (wq & 0xF).long(), (wq >> 4).long()
    w = torch.stack([E2M1[lo], E2M1[hi]], 1).reshape(K, N)  # k = 2 * pk + j
    scale = torch.exp2(s.float() - 127).t().repeat_interleave(32, 0)  # [K, N]
    return w * scale


def graph_us(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(5):
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1000 / reps)
    return best


x = (torch.randn(M, K, device=dev) * 0.5).bfloat16()
wq = torch.randint(0, 256, (K // 2, N), dtype=torch.uint8, device=dev)
s = torch.randint(118, 124, (N, K // 32), dtype=torch.uint8, device=dev)
wd = dequant(wq, s)
ref = x.float() @ wd
wb = wd.bfloat16()
fp4_bytes = wq.numel() + s.numel()
print(f"{torch.cuda.get_device_name()}  M={M} K={K} N={N}: MXFP4 {fp4_bytes / 2**20:.0f} MiB, "
      f"bf16 {wb.numel() * 2 / 2**20:.0f} MiB")
NAMES = {0: "dot_scaled", 1: "fp32 bits", 2: "int bits", 3: "bf16"}
for bn, bk in ((64, 128), (128, 128), (128, 256), (256, 128)):
    for mode, name in NAMES.items():
        y = torch.empty(M, N, device=dev)
        w = wb if mode == 3 else wq
        for warps in (4, 8):
            fn = lambda: _gemm[(N // bn,)](x, w, s, y, N, K=K, M=M, BLOCK_N=bn, BLOCK_K=bk,  # noqa: E731
                                          MODE=mode, num_warps=warps, num_stages=2)
            try:
                fn()
                err = ((y - ref).abs().max() / ref.abs().max()).item()
                us = graph_us(fn)
                nbytes = wb.numel() * 2 if mode == 3 else fp4_bytes
                print(f"  bn {bn:3d} bk {bk:3d} warps {warps}  {name:<10} {us:7.1f} us "
                      f"{nbytes / us / 1e6:5.2f} TB/s  err {err:.1e}", flush=True)
            except Exception as e:  # noqa: BLE001
                torch.cuda.synchronize()
                print(f"  bn {bn:3d} bk {bk:3d} warps {warps}  {name:<10} failed: {repr(e)[:90]}")
