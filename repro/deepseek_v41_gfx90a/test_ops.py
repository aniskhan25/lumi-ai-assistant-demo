"""lumi_dsv41.ops vs 0.26's compiled kernel, with the V4 query norm on (the kernel's only
mode): the query path (norm, GPT-J RoPE, head padding) must match. One GCD."""
import torch
import vllm._custom_ops  # noqa: F401 - loads the _C ops

from lumi_dsv41.ops import qnorm_rope_kv_rope_quant_insert

torch.manual_seed(0)
dev = "cuda"
T, H, D, PAD, ROT, BS = 37, 64, 512, 128, 64, 32
for dtype in (torch.float32, torch.bfloat16):
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
