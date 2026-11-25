"""DeepSeek V4.1, language model only (lumi_dsv41 override of upstream's package __init__).

Upstream's __init__ imports the vision-language wrapper and the DSpark drafter eagerly;
neither is ported to LAIF's vLLM 0.26. The plugin registers
vllm.models.deepseek_v41.amd.model:DeepseekV41LLMForCausalLM directly.
"""

from .quant_config import DeepseekV4FP8Config

__all__ = ["DeepseekV4FP8Config"]
