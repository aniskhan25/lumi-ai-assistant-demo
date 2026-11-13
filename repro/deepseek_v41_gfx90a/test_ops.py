"""lumi_dsv41.ops (sbatch repro/deepseek_v41_gfx90a/run_ops_test.sh) vs 0.26's compiled kernel, with the V4 query norm on (the kernel's only
mode): the query path (norm, GPT-J RoPE, head padding) must match. One GCD."""
import torch
import vllm._custom_ops  # noqa: F401 - loads the _C ops

from lumi_dsv41.ops import qnorm_rope_kv_rope_quant_insert

torch.manual_seed(0)
dev = "cuda"
T, H, D, PAD, ROT, BS = 37, 64, 512, 128, 64, 32
for dtype in (torch.float32,):  # the kernel requires a float32 cos_sin_cache
    q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(T, D, device=dev, dtype=torch.bfloat16)
    pos = torch.randint(0, 4096, (T,), device=dev, dtype=torch.int64)
    theta = torch.rand(4096, ROT // 2, device=dev) * 6.28
    cos_sin = torch.cat([theta.cos(), theta.sin()], -1).to(dtype)
    page = -(-BS * 584 // 576) * 576
    k1 = torch.zeros(8, page, device=dev, dtype=torch.uint8)
    k2 = k1.clone()
    slots = torch.randperm(8 * BS, device=dev)[:T].to(torch.int64)
    ref = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q, kv, k1, slots, pos, cos_sin, PAD, 1e-6, BS)
    got = qnorm_rope_kv_rope_quant_insert(q, kv, k2, slots, pos, cos_sin, PAD, 1e-6, BS,
                                          True, False, True, False)
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    print(f"cos_sin {dtype}: q rel err {rel:.2e}, pad zero {bool((got[:, H:] == 0).all())}, "
          f"kv cache identical {bool(torch.equal(k1, k2))}, shapes {tuple(got.shape)} {tuple(ref.shape)}")
    assert rel < 1e-2 and torch.equal(k1, k2)
print("OPS OK")

# The Triton query kernel vs its torch reference (lumi_dsv41.ops._query_torch), V4.1's
# shapes: per-rank 8 heads (TP=8), padded widths, head dim 512, RoPE 64; then time per call.
from lumi_dsv41.ops import _query_kernel, _query_torch  # noqa: E402

bad = []
for cs_dtype in (torch.float32, torch.bfloat16):
    for T, H, W in ((1, 8, 8), (32, 8, 8), (32, 8, 16), (300, 64, 128)):
        q = torch.randn(T, H, 512, device=dev, dtype=torch.bfloat16)
        pos = torch.randint(0, 4096, (T,), device=dev, dtype=torch.int64)
        theta = torch.rand(4096, 32, device=dev) * 6.28
        cs = torch.cat([theta.cos(), theta.sin()], -1).to(cs_dtype)
        for norm in (False, True):
            for rope in (True, False):
                ref = _query_torch(q, pos, cs, W, 1e-6, norm, rope)
                out = torch.empty_like(ref)
                _query_kernel[(T, W)](q, out, pos, cs, q.stride(0), q.stride(1), cs.stride(0),
                                      1e-6, H=H, D=512, ROT=64, NORM=norm, ROPE=rope, num_warps=4)
                diff = (out.float() - ref.float()).abs().max().item()
                ok = diff <= 2 ** -6 * ref.float().abs().max().item() and bool((out[:, H:] == 0).all())
                if not ok:
                    bad.append((cs_dtype, T, H, W, norm, rope, diff))
                print(f"query T={T:3d} H={H:2d} W={W:3d} norm={norm:d} rope={rope:d} cs={cs_dtype}: "
                      f"max diff {diff:.1e} {'ok' if ok else 'MISMATCH'}")


def timed(fn, iters=200):
    g = torch.cuda.CUDAGraph()
    fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    g.replay()
    torch.cuda.synchronize()
    s.record()
    g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


q = torch.randn(32, 8, 512, device=dev, dtype=torch.bfloat16)
pos = torch.randint(0, 4096, (32,), device=dev, dtype=torch.int64)
cs = torch.cat([torch.rand(4096, 32, device=dev).cos(), torch.rand(4096, 32, device=dev).sin()], -1)
t_torch = timed(lambda: _query_torch(q, pos, cs, 8, 1e-6, False, True))
out = torch.empty_like(q)
t_tri = timed(lambda: _query_kernel[(32, 8)](q, out, pos, cs, q.stride(0), q.stride(1),
                                             cs.stride(0), 1e-6, H=8, D=512, ROT=64,
                                             NORM=False, ROPE=True, num_warps=4))
print(f"query at 32 tokens, 8 heads (graph replay, us/call): torch {t_torch:.1f}, Triton {t_tri:.1f}; "
      f"x 43 layers = {43 * (t_torch - t_tri) / 1e3:.2f} ms per step saved")
print("FAILED:", bad if bad else "none")
assert not bad
