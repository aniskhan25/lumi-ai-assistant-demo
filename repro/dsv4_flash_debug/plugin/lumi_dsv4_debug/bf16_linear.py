"""Run block-FP8 dense linears in bf16 on MI250X (DSV4_BF16_LINEAR=1).

The decode profile put DeepSeek-V4-Flash's W8A8 block-FP8 linears at about half of every
step (job 23530527). On gfx90a, which has no FP8 units, `_w8a8_triton_block_scaled_mm` is
20-90x slower than a bf16 GEMM of the same shape, and less accurate, because it also rounds
the activations to FP8 (bench_fp8_linear.py, job 23535213).

After vLLM's own weight processing, each block-quantised Fp8LinearMethod layer gets its weight
dequantised once to bf16 (exact: an e4m3 value times a power-of-two block scale fits bf16),
its scales dropped, and is then served by vLLM's ROCm bf16 GEMM dispatch (wvSplitK / LLMM1 for
small M, hipBLASLt otherwise). The dense weights take twice the memory.

DeepSeek-V4's wo_a never calls apply(): its ROCm einsum path dequantises it itself, and uses
the weight as is when the layer has no `weight_scale_inv`, which is what this leaves behind.
"""

import torch


def _dequant(weight: torch.Tensor, scale: torch.Tensor, block: list[int]) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import _upcast_e8m0_to_fp32

    n, k = weight.shape
    if scale.dtype == torch.float8_e8m0fnu:
        scale = _upcast_e8m0_to_fp32(scale)
    scale = scale.float().repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)[:n, :k]
    return (weight.float() * scale).to(torch.bfloat16)


def install() -> None:
    from vllm.logger import init_logger
    from vllm.model_executor.layers.quantization import fp8 as fp8_module
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

    log = init_logger("vllm.lumi_dsv4_debug")
    cls = fp8_module.Fp8LinearMethod
    if getattr(cls, "_lumi_bf16", False):
        return
    stock_process, stock_apply = cls.process_weights_after_loading, cls.apply
    gemm = dispatch_unquantized_gemm()
    state = {"layers": 0, "bytes": 0, "checks": 3}

    def process_weights_after_loading(self, layer):
        stock_process(self, layer)
        weight = getattr(layer, "weight", None)
        if (not self.block_quant or self.use_marlin or weight is None
                or weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)):
            return
        scale = getattr(layer, "weight_scale_inv", None)
        if scale is None:
            scale = getattr(layer, "weight_scale", None)
        if scale is None or weight.dim() != 2:
            return
        bf16 = _dequant(weight, scale, self.weight_block_size)
        x = None
        if state["checks"] > 0 and not getattr(layer, "is_bmm", False):
            x = torch.randn(4, weight.shape[1], dtype=torch.bfloat16, device=weight.device)
            want = stock_apply(self, layer, x).float()
        layer.weight = torch.nn.Parameter(bf16, requires_grad=False)
        for name in ("weight_scale_inv", "weight_scale", "input_scale"):
            if hasattr(layer, name):
                delattr(layer, name)
        layer._lumi_bf16 = True
        state["layers"] += 1
        state["bytes"] += bf16.numel()
        if x is not None:
            state["checks"] -= 1
            got = gemm(layer, x, layer.weight).float()
            rel = ((got - want).norm() / want.norm()).item()
            log.info("DSV4_BF16_LINEAR check %s: bf16 vs W8A8 rel diff %.2e (W8A8 rounds "
                     "activations to FP8: ~2.6e-2 expected)", tuple(bf16.shape), rel)
        n = state["layers"]
        if n & (n - 1) == 0:
            log.info("DSV4_BF16_LINEAR: %d block-FP8 linears now bf16 (+%.2f GiB vs FP8)",
                     n, state["bytes"] / 2**30)

    def apply(self, layer, x, bias=None):
        if getattr(layer, "_lumi_bf16", False):
            return gemm(layer, x, layer.weight, bias)
        return stock_apply(self, layer, x, bias)

    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply = apply
    cls._lumi_bf16 = True
