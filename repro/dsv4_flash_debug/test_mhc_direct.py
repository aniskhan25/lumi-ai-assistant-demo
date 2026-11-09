"""mhc_fix.mhc_pre_direct (plain function: torch prenorm GEMM + Triton big-fuse) vs vLLM's
mhc_pre_torch, with DeepSeek-V4-Flash's real layer-5 parameters, on one GCD.

    python test_mhc_direct.py <DeepSeek-V4-Flash snapshot dir>
"""
import json
import pathlib
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, str(pathlib.Path(__file__).parent / "plugin"))
from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch  # noqa: E402

from lumi_dsv4_debug.mhc_fix import mhc_pre_direct  # noqa: E402

snap = pathlib.Path(sys.argv[1])
cfg = json.loads((snap / "config.json").read_text())
index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
dev = torch.device("cuda")


def param(name):
    with safe_open(snap / index[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name).float().to(dev)


p = [param(f"layers.5.hc_attn_{k}") for k in ("fn", "scale", "base")]
args = (cfg["rms_norm_eps"], cfg["hc_eps"], cfg["hc_eps"], 2.0, cfg["hc_sinkhorn_iters"])
bad = 0
for t in (1, 7, 64, 300):
    r = torch.randn(t, cfg["hc_mult"], cfg["hidden_size"], device=dev).to(torch.bfloat16)
    for name, g, w in zip(("post", "comb", "input"), mhc_pre_direct(r, *p, *args),
                          mhc_pre_torch(r, *p, *args)):
        rel = ((g.float() - w.float()).norm() / w.float().norm()).item()
        ok = g.shape == w.shape and rel < 2e-2
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} T={t:<4} {name:<6} {tuple(g.shape)} rel err {rel:.1e}")
print("FAILED" if bad else "all ok")
