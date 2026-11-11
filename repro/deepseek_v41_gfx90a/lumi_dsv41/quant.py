"""DeepSeek-V4.1 quantization on vLLM 0.26, registered as `deepseek_v41_fp8`.

Upstream's DeepseekV4FP8Config (vendored deepseek_v41/quant_config.py) with two layer
kinds routed to methods 0.26 has:
- Dense linears are MXFP8: e4m3 weights with one e8m0 scale per 32x32 block. That is
  block-FP8 with 32x32 blocks and power-of-two scales, which 0.26's Fp8LinearMethod
  loads and runs (as for V4's 128x128 blocks); upstream uses a newer ModelOpt MXFP8
  method. The V4 plugin's DSV4_BF16_LINEAR fast path applies unchanged.
- MXFP4 experts: 0.26's Mxfp4MoEMethod takes no activation_key, so experts take bf16
  activations, as 0.26's own V4 does (GSM8K 97.8 % on V4-Flash).
"""

from vllm.model_executor.layers.fused_moe import RoutedExperts, UnquantizedFusedMoEMethod
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config

NAME = "deepseek_v41_fp8"


@register_quantization_config(NAME)
class DeepseekV41FP8Config(DeepseekV4FP8Config):
    @classmethod
    def get_name(cls):
        return NAME

    def _skipped(self, prefix: str) -> bool:
        return is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        )

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase) and self.weight_block_size == [32, 32]:
            if self._skipped(prefix):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)
        if isinstance(layer, RoutedExperts) and self.expert_dtype == "fp4":
            if self._skipped(prefix):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return Mxfp4MoEMethod(layer.moe_config)
        return super().get_quant_method(layer, prefix)
