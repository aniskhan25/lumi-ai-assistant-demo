"""qnorm_rope.py (DSV4_QNORM_ROPE) against vLLM's HIP op, on one GCD.

torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert vs qnorm_rope_kv_insert on
the same inputs (8 q heads padded to 16, 512-dim heads, 64 RoPE dims, fp8_ds_mla cache with
64-token blocks, some slots -1):
  q:     max |diff| and the share of bf16 elements that differ (padding heads must be 0);
  cache: bytes that differ in the FP8 part, the scales and the bf16 RoPE part, and the max
         difference after dequantising;
  time:  GPU us per call in graph replay at 2-128 tokens.

    python test_qnorm_rope.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "plugin"))
import vllm._custom_ops  # noqa: F401,E402  loads torch.ops._C
from lumi_dsv4_debug.qnorm_rope import qnorm_rope_kv_insert  # noqa: E402

dev = "cuda"
H, PAD, D, R, BS = 8, 16, 512, 64, 64
REC = 448 + 2 * R  # 576 bytes per token
STRIDE = -(-(BS * (REC + 8)) // 576) * 576
EPS = 1e-6
MAX_POS = 65536
torch.manual_seed(0)
inv = 1.0 / (10000 ** (torch.arange(0, R, 2, device=dev).float() / R))
ang = torch.arange(MAX_POS, device=dev).float()[:, None] * inv[None, :]
cos_sin = torch.cat([ang.cos(), ang.sin()], -1).contiguous()  # [max_pos, 64] fp32
stock = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
fails = []


def dequant(cache, slots):
    out = []
    for s in slots.tolist():
        if s < 0:
            continue
        blk, p = divmod(s, BS)
        rec = cache[blk, p * REC:(p + 1) * REC]
        sc = cache[blk, BS * REC + p * 8: BS * REC + p * 8 + 7].float()
        nope = rec[:448].view(torch.float8_e4m3fn).float() * torch.exp2(sc - 127).repeat_interleave(64)
        rope = rec[448:].view(torch.bfloat16).float()
        out.append(torch.cat([nope, rope]))
    return torch.stack(out)


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


print(torch.cuda.get_device_name())
for T in (1, 2, 7, 64, 128, 300):
    q = (torch.randn(T, H, D, device=dev) * 3).bfloat16()
    kv = (torch.randn(T, D, device=dev) * 2).bfloat16()
    kv[:, :64] *= 50  # one quant block with large values
    pos = torch.randint(0, MAX_POS, (T,), device=dev)
    nblocks = T // BS + 4
    slots = torch.randperm(nblocks * BS, device=dev)[:T]
    slots[::5] = -1
    c0 = torch.zeros(nblocks, STRIDE, dtype=torch.uint8, device=dev)
    c1 = c0.clone()
    q0 = stock(q.clone(), kv, c0, slots, pos, cos_sin, PAD, EPS, BS)
    q1 = qnorm_rope_kv_insert(q.clone(), kv, c1, slots, pos, cos_sin, PAD, EPS, BS)
    dq = (q0.float() - q1.float()).abs()
    pad_ok = bool((q1[:, H:] == 0).all()) and q0.shape == q1.shape
    nb = (c0 != c1)
    byte_frac = nb.float().mean().item()
    d0, d1 = dequant(c0, slots), dequant(c1, slots)
    dkv = ((d0 - d1).abs().max() / d0.abs().max()).item()
    rel_q = (dq.max() / q0.float().abs().max()).item()
    bad = not pad_ok or rel_q > 1e-2 or dkv > 1e-2
    if bad:
        fails.append(T)
    print(f"  {'FAIL' if bad else 'ok  '} T={T:<4} q: max |diff| {dq.max().item():.2e} (rel {rel_q:.1e}), "
          f"{(dq > 0).float().mean().item() * 100:.2f} % differ, padding zero {pad_ok}  | cache: "
          f"{byte_frac * 100:.3f} % bytes differ, dequantised rel diff {dkv:.1e}", flush=True)

print("\nus per call (graph replay)")
for T in (2, 16, 64, 128):
    q = torch.randn(T, H, D, device=dev).bfloat16()
    kv = torch.randn(T, D, device=dev).bfloat16()
    pos = torch.randint(0, MAX_POS, (T,), device=dev)
    nblocks = T // BS + 4
    slots = torch.randperm(nblocks * BS, device=dev)[:T]
    c = torch.zeros(nblocks, STRIDE, dtype=torch.uint8, device=dev)
    t0 = graph_us(lambda: stock(q, kv, c, slots, pos, cos_sin, PAD, EPS, BS))
    t1 = graph_us(lambda: qnorm_rope_kv_insert(q, kv, c, slots, pos, cos_sin, PAD, EPS, BS))
    print(f"  T={T:<4} HIP op {t0:6.1f} us  Triton {t1:6.1f} us  -> x44 per step: "
          f"{t0 * 44 / 1e3:.2f} -> {t1 * 44 / 1e3:.2f} ms")
print("FAILED:", fails or "none")
