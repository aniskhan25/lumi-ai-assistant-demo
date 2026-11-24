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

from vllm.triton_utils import tl, triton

FP8_MAX = 448.0  # float8_e4m3fn


@triton.jit
def _fp8_round_kernel(x_ptr, out_ptr, stride_xm, stride_om, eps, GROUP: tl.constexpr):
    """One (row, group): scale to the e4m3 range as vLLM's per-token-group quantiser does
    (scale = max(amax, eps) / 448, fp32), round to e4m3 precision in registers, scale back,
    write bf16. Rounding: 3 mantissa bits, round-to-nearest-even; below 2^-6 (e4m3's smallest
    normal) the grid is the subnormal spacing 2^-9."""
    m = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * GROUP + tl.arange(0, GROUP)
    x = tl.load(x_ptr + m * stride_xm + offs).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x)), eps) / 448.0
    v = tl.minimum(tl.maximum(x / s, -448.0), 448.0)
    bits = v.to(tl.int32, bitcast=True)
    bits = (bits + 0x7FFFF + ((bits >> 20) & 1)) & -1048576  # keep 3 of 23 mantissa bits, RNE
    normal = bits.to(tl.float32, bitcast=True)
    sub = ((v * 512.0 + 12582912.0) - 12582912.0) / 512.0  # RNE to multiples of 2^-9
    q = tl.where(tl.abs(v) < 0.015625, sub, normal)
    tl.store(out_ptr + m * stride_om + offs, (q * s).to(tl.bfloat16))


def fp8_round_triton(x2d: torch.Tensor, group: int, eps: float = 1e-10) -> torch.Tensor:
    """x2d (M, K) bf16 -> bf16 rounded through FP8 e4m3 per `group` columns, in one kernel."""
    m, k = x2d.shape
    assert k % group == 0 and x2d.stride(-1) == 1
    out = torch.empty(m, k, dtype=torch.bfloat16, device=x2d.device)
    if m:
        _fp8_round_kernel[(m, k // group)](x2d, out, x2d.stride(0), out.stride(0), eps,
                                           GROUP=group, num_warps=1)
    return out


def fp8_round_torch(x2d: torch.Tensor, group: int) -> torch.Tensor:
    """The same through vLLM's quantiser and a separate dequantise (two kernels + temporaries)."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    q, s = per_token_group_quant_fp8(x2d, group, use_ue8m0=False)
    m, k = x2d.shape
    return (q.float().view(m, k // group, group) * s.unsqueeze(-1)).view(m, k).to(x2d.dtype)


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
    # Fused kernel by default (bit-identical, 3-5x faster, job 23548678); DSV4_FP8_ROUND=torch: vLLM
    fake_quant = fp8_round_torch if os.environ.get("DSV4_FP8_ROUND") == "torch" else fp8_round_triton
    state = {"layers": 0, "bytes": 0, "max_diff": 0.0}

    # DSV4_FP8_FUSE=1 with DSV4_DENSE_GEMM: the rounding inside the Triton GEMM where its tile
    # allows (dense_gemm.gemm_fp8_round). Bit-identical, but 0.4-8 us slower per call than
    # rounding first (test_fp8_fuse.py, job 23974177): every program re-rounds the same
    # activation tile, which costs more than the launch saved. Off by default.
    fused = None
    if (act_fp8 and fake_quant is fp8_round_triton and os.environ.get("DSV4_DENSE_GEMM") == "1"
            and os.environ.get("DSV4_FP8_FUSE", "0") == "1"):
        from .dense_gemm import gemm_fp8_round as fused

    def run(self, layer, x, bias):
        if not act_fp8:
            return gemm(layer, x, layer.weight, bias)
        if fused is not None and bias is None:
            y = fused(x.reshape(-1, x.shape[-1]), layer.weight, self.weight_block_size[1])
            if y is not None:
                return y.view(*x.shape[:-1], layer.weight.shape[0])
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
