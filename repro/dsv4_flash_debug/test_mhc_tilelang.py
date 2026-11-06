"""Which TileLang hyper-connection (mHC) kernel is wrong on gfx90a?

vLLM 0.26 runs four TileLang mHC ops on every ROCm GPU except gfx942, where it says they
"produce incorrect results". On gfx90a they made DeepSeek-V4-Flash emit garbage, and turning
them off fixed it (jobs 23494525, 23494641). This compares each one, on one GCD, with the
fallback vLLM itself uses when TileLang is off:

  mhc_pre_tilelang             vs  mhc_pre_torch
  mhc_post_tilelang            vs  mhc_post_torch
  mhc_fused_post_pre_tilelang  vs  mhc_post_torch + mhc_pre_torch (vLLM's native path)
  hc_head_fused_kernel_tilelang vs hc_head_triton, and both vs DeepSeek's reference hc_head

with DeepSeek-V4-Flash's real layer-5 and head parameters, and eps/iteration values from its
config. Random bf16 hidden states; several token counts.

    python test_mhc_tilelang.py <DeepSeek-V4-Flash snapshot dir>
"""

import json
import pathlib
import sys

import torch
from safetensors import safe_open

import vllm.model_executor.kernels.mhc.tilelang  # noqa: F401  registers *_tilelang ops
import vllm.model_executor.kernels.mhc.triton  # noqa: F401  registers hc_head_triton
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch

snapshot = pathlib.Path(sys.argv[1])
config = json.loads((snapshot / "config.json").read_text())
HC = config["hc_mult"]
H = config["hidden_size"]
RMS_EPS = config["rms_norm_eps"]
HC_EPS = config["hc_eps"]
ITERS = config["hc_sinkhorn_iters"]
POST_MULT = 2.0  # DeepseekV4DecoderLayer.hc_post_alpha
dev = torch.device("cuda")

index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]


def param(name: str) -> torch.Tensor:
    with safe_open(snapshot / index[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name).float().to(dev)


attn = [param(f"layers.5.hc_attn_{p}") for p in ("fn", "scale", "base")]
ffn = [param(f"layers.5.hc_ffn_{p}") for p in ("fn", "scale", "base")]
head = [param(f"hc_head_{p}") for p in ("fn", "scale", "base")]
print(f"device {torch.cuda.get_device_properties(0).gcnArchName}  hc={HC} hidden={H}  "
      f"fn {tuple(attn[0].shape)}  head_fn {tuple(head[0].shape)}")

failures = []


def compare(op: str, tokens: int, name: str, got: torch.Tensor, want: torch.Tensor, tol: float = 2e-2):
    got, want = got.float(), want.float()
    if got.shape != want.shape:
        failures.append(op)
        print(f"  FAIL {op:<18} T={tokens:<4} {name:<12} shape {tuple(got.shape)} vs {tuple(want.shape)}")
        return
    rel = ((got - want).norm() / want.norm().clamp_min(1e-12)).item()
    bad = not torch.isfinite(got).all() or rel > tol
    if bad:
        failures.append(op)
    print(f"  {'FAIL' if bad else 'ok  '} {op:<18} T={tokens:<4} {name:<12} rel err {rel:.2e}  "
          f"max abs {(got - want).abs().max().item():.2e}")


def reference_hc_head(x, fn, scale, base):  # DeepSeek inference/model.py ParallelHead.hc_head
    shape, dtype = x.shape, x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + RMS_EPS)
    mixes = torch.nn.functional.linear(x, fn) * rsqrt
    pre = torch.sigmoid(mixes * scale + base) + HC_EPS
    return torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1).to(dtype)


torch.manual_seed(0)
for tokens in (1, 7, 64, 300):
    residual = torch.randn(tokens, HC, H, dtype=torch.bfloat16, device=dev)
    x = torch.randn(tokens, H, dtype=torch.bfloat16, device=dev)
    args = (RMS_EPS, HC_EPS, HC_EPS, POST_MULT, ITERS)

    want = mhc_pre_torch(residual, *attn, *args)
    got = torch.ops.vllm.mhc_pre_tilelang(residual, *attn, *args, 1, None, 1e-6)
    for name, g, w in zip(("post_mix", "res_mix", "layer_input"), got, want):
        compare("mhc_pre", tokens, name, g, w)

    post_mix, res_mix, _ = want
    compare("mhc_post", tokens, "out",
            torch.ops.vllm.mhc_post_tilelang(x, residual, post_mix, res_mix),
            mhc_post_torch(x, residual, post_mix, res_mix))

    new_residual = mhc_post_torch(x, residual, post_mix, res_mix)
    want = (new_residual, *mhc_pre_torch(new_residual, *ffn, *args))
    got = torch.ops.vllm.mhc_fused_post_pre_tilelang(
        x, residual, post_mix, res_mix, *ffn, *args, 1, 1, None, 1e-6)
    for name, g, w in zip(("residual", "post_mix", "res_mix", "layer_input"), got, want):
        compare("mhc_fused_post_pre", tokens, name, g, w)

    ref = reference_hc_head(residual, *head)
    tl = torch.ops.vllm.hc_head_fused_kernel_tilelang(residual, *head, RMS_EPS, HC_EPS)
    tri = torch.empty(tokens, H, dtype=torch.bfloat16, device=dev)
    torch.ops.vllm.hc_head_triton(residual, *head, tri, H, RMS_EPS, HC_EPS, HC)
    compare("hc_head_tilelang", tokens, "vs ref", tl, ref)
    compare("hc_head_triton", tokens, "vs ref", tri, ref)

print("WRONG ON THIS GPU:", sorted(set(failures)) or "none")
