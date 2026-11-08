"""Run block-FP8 dense linears in bf16 on MI250X (DSV4_BF16_LINEAR=1).

The decode profile put DeepSeek-V4-Flash's W8A8 block-FP8 linears at about half of every
step (job 23530527). On gfx90a, which has no FP8 units, `_w8a8_triton_block_scaled_mm` is
20-90x slower than a bf16 GEMM of the same shape (bench_fp8_linear.py, job 23535213).

After vLLM's own weight processing, each block-quantised Fp8LinearMethod layer gets its weight
dequantised once to bf16 (exact: an e4m3 value times a power-of-two block scale fits bf16),
its scales dropped, and is then served by vLLM's ROCm bf16 GEMM dispatch (wvSplitK / LLMM1 for
small M, hipBLASLt otherwise). The dense weights take twice the memory.

The activations are still rounded to FP8 per 128-group first, as the stock W8A8 path does
(DSV4_BF16_LINEAR_ACT=fp8, the default). The model needs that: with bf16 activations, which
are closer to the exact product, GSM8K fell from 98 % to 90.5 % with 10/200 garbage answers
(job 23535577); with the rounding it is back to 98 %, 0 garbage (job 23535685).
DSV4_BF16_LINEAR_ACT=bf16 skips the rounding.

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
    import os

    from vllm.logger import init_logger
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
    from vllm.model_executor.layers.quantization import fp8 as fp8_module
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

    log = init_logger("vllm.lumi_dsv4_debug")
    cls = fp8_module.Fp8LinearMethod
    if getattr(cls, "_lumi_bf16", False):
        return
    stock_process, stock_apply = cls.process_weights_after_loading, cls.apply
    gemm = dispatch_unquantized_gemm()
    # Round the activations to FP8 per 128-group (fp32 scales) as the stock W8A8 path does,
    # then dequantise and run the bf16 GEMM: stock numerics up to bf16 rounding of q*s.
    act_fp8 = os.environ.get("DSV4_BF16_LINEAR_ACT", "fp8") != "bf16"
    state = {"layers": 0, "bytes": 0, "max_diff": 0.0}

    def fake_quant(x2d, group):
        q, s = per_token_group_quant_fp8(x2d, group, use_ue8m0=False)
        m, k = x2d.shape
        return (q.float().view(m, k // group, group) * s.unsqueeze(-1)).view(m, k).to(x2d.dtype)

    def run(self, layer, x, bias):
        if not act_fp8:
            return gemm(layer, x, layer.weight, bias)
        x2d = fake_quant(x.reshape(-1, x.shape[-1]), self.weight_block_size[1])
        return gemm(layer, x2d, layer.weight, bias).view(*x.shape[:-1], layer.weight.shape[0])

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
        if not getattr(layer, "is_bmm", False):  # every layer: one small GEMM each at load
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
            got = run(self, layer, x, None).float()
            rel = ((got - want).norm() / want.norm()).item()
            state["max_diff"] = max(state["max_diff"], rel)
            if rel > (1e-2 if act_fp8 else 4e-2):  # fp8: ~2.7e-3; bf16 act: ~2.6e-2
                log.warning("DSV4_BF16_LINEAR outlier %s %s: rel diff vs W8A8 %.2e",
                            getattr(layer, "prefix", "?"), tuple(bf16.shape), rel)
        n = state["layers"]
        if n & (n - 1) == 0 or n % 100 == 0:
            log.info("DSV4_BF16_LINEAR: %d block-FP8 linears now bf16 (+%.2f GiB vs FP8), "
                     "max rel diff vs W8A8 so far %.2e%s", n, state["bytes"] / 2**30,
                     state["max_diff"], ", FP8-rounded activations" if act_fp8 else "")

    def apply(self, layer, x, bias=None):
        if getattr(layer, "_lumi_bf16", False):
            return run(self, layer, x, bias)
        return stock_apply(self, layer, x, bias)

    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply = apply
    cls._lumi_bf16 = True
