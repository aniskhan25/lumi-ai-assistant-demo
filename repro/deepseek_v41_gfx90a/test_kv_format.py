"""Does 0.26's C++ SWA-cache encoder write the record upstream's readers expect on gfx90a?

Upstream reads the SWA cache with use_fnuz=current_platform.is_fp8_fnuz() (OCP on
gfx90a). Write the same K through 0.26's _C kernel (via lumi_dsv41.ops) and through
upstream's Triton OCP writer, compare bytes, then read the _C cache back with upstream's
gather. Position 0 with cos=1, sin=0 makes RoPE the identity. One GCD.
"""
import torch
import vllm._custom_ops  # noqa: F401 - loads the _C ops
from vllm.platforms import current_platform

from lumi_dsv41 import loader

loader.install()
from lumi_dsv41.ops import qnorm_rope_kv_rope_quant_insert  # noqa: E402
from vllm.models.deepseek_v41.common.ops.cache_utils import (  # noqa: E402
    dequantize_and_gather_k_cache, quantize_and_insert_k_cache)

torch.manual_seed(0)
dev = "cuda"
T, H, D, ROT = 50, 8, 512, 64
print("is_fp8_fnuz", current_platform.is_fp8_fnuz(), "fp8 dtype", current_platform.fp8_dtype())
for BS in (64, 128):
    NB = 4
    kv = torch.randn(T, D, device=dev, dtype=torch.bfloat16)
    q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    pos = torch.zeros(T, device=dev, dtype=torch.int64)
    cos_sin = torch.cat([torch.ones(8, ROT // 2), torch.zeros(8, ROT // 2)], -1).to(dev)
    slots = torch.arange(T, device=dev, dtype=torch.int64)
    c_cpp = torch.zeros(NB, BS, 584, device=dev, dtype=torch.uint8)
    c_tri = torch.zeros_like(c_cpp)
    qnorm_rope_kv_rope_quant_insert(q, kv, c_cpp, slots, pos, cos_sin, H, 1e-6, BS,
                                    False, False, True, False)
    quantize_and_insert_k_cache(kv, c_tri, slots, block_size=BS, use_fnuz=False)
    torch.cuda.synchronize()
    diff = (c_cpp != c_tri)
    print(f"BS={BS}: bytes differ {int(diff.sum())}/{c_cpp[:1].numel() * ((T + BS - 1) // BS)}"
          f" (per-byte-column first diffs: {diff.reshape(NB, -1).nonzero()[:6].tolist()})")
    out = torch.zeros(1, T, D, device=dev, dtype=torch.bfloat16)
    seq = torch.tensor([T], device=dev, dtype=torch.int32)
    bt = torch.arange(NB, device=dev, dtype=torch.int32).unsqueeze(0)
    for name, cache in (("cpp", c_cpp), ("triton", c_tri)):
        out.zero_()
        dequantize_and_gather_k_cache(out, cache, seq, None, bt, BS, 0, use_fnuz=False)
        rel = ((out[0].float() - kv.float()).norm() / kv.float().norm()).item()
        nope = ((out[0, :, :448].float() - kv[:, :448].float()).norm()
                / kv[:, :448].float().norm()).item()
        rope = ((out[0, :, 448:].float() - kv[:, 448:].float()).norm()
                / kv[:, 448:].float().norm()).item()
        print(f"  read {name} cache as OCP: rel err {rel:.3e} (nope {nope:.3e}, rope {rope:.3e})")
print("KVFMT DONE")
