"""Do vLLM's ROCm skinny bf16 GEMMs (wvSplitK, LLMM1) handle non-contiguous inputs?

rocm_unquantized_gemm_impl picks wvSplitK for M <= 5 and LLMM1 for M == 1. The stock W8A8
path always makes its input contiguous while quantising it; DSV4_BF16_LINEAR passes the layer
input as is, and DeepSeek-V4 feeds some linears column slices of fused outputs.

    python test_skinny_gemm_strides.py
"""
import torch

from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

dev = torch.device("cuda")
torch.manual_seed(0)
for n, k in [(1536, 4096), (4096, 1024), (8192, 1024), (512, 4096)]:
    w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
    for m in (1, 2, 4, 5, 8, 32):
        wide = torch.randn(m, k + 512, device=dev, dtype=torch.bfloat16)
        x = wide[:, 256:256 + k]  # column slice: row stride k + 512
        ref = x.float() @ w.float().T
        for label, inp in (("contiguous", x.contiguous()), ("strided", x)):
            out = rocm_unquantized_gemm_impl(inp, w).float()
            rel = ((out - ref).norm() / ref.norm()).item()
            print(f"N={n} K={k} M={m:>2} {label:<10} rel err {rel:.2e} {'BAD' if rel > 1e-2 else ''}")
