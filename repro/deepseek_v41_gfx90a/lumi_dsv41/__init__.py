"""vLLM 0.26 plugin: upstream DeepSeek-V4.1 (AMD path, language model only) on LUMI-G.

Loaded by vLLM's `vllm.general_plugins` entry point (../lumi_dsv41-0.1.dist-info) in
every vLLM process. ../sitecustomize.py installs the module loader (loader.py) at
interpreter start; register() adds the engine shims and the model, config and
quantization registrations. Put the parent directory on PYTHONPATH.
"""

ARCH = "DeepseekV41ForCausalLM"
MODEL = "vllm.models.deepseek_v41.amd.model:DeepseekV41LLMForCausalLM"


def register() -> None:
    from . import shims

    shims.install()

    import vllm.transformers_utils.configs as configs
    from vllm import ModelRegistry
    from vllm.model_executor.models import config as model_configs
    from vllm.transformers_utils import config as hf_configs

    configs._CLASS_TO_MODULE["DeepseekV41Config"] = "vllm.transformers_utils.configs.deepseek_v41"
    hf_configs._CONFIG_REGISTRY["deepseek_v41"] = "DeepseekV41Config"
    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, MODEL)
    model_configs.MODELS_CONFIG_MAP[ARCH] = _config_class(model_configs)


def _config_class(model_configs):
    base = model_configs.VerifyAndUpdateConfig

    class DeepseekV41ForCausalLMConfig(base):
        @staticmethod
        def verify_and_update_model_config(model_config) -> None:
            # Upstream routes "fp8" to deepseek_v41.quant_config for this model type;
            # 0.26 only knows V4's, so V4.1's gets its own method name (quant.py).
            for hf in (model_config.hf_config, model_config.hf_text_config):
                quant = getattr(hf, "quantization_config", None)
                if quant is not None and quant.get("quant_method") == "fp8":
                    quant["quant_method"] = "deepseek_v41_fp8"

        @staticmethod
        def verify_and_update_config(vllm_config) -> None:
            if vllm_config.speculative_config is not None:
                raise NotImplementedError("DeepSeek-V4.1 on vLLM 0.26: MTP/DSpark not ported")
            if vllm_config.cache_config.enable_prefix_caching:
                raise NotImplementedError(
                    "DeepSeek-V4.1 on vLLM 0.26: pass --no-enable-prefix-caching")
            if not vllm_config.use_v2_model_runner:
                # Engram reads each request's previous tokens from the model state,
                # which only Model Runner V2 calls.
                raise ValueError("DeepSeek-V4.1 needs Model Runner V2: VLLM_USE_V2_MODEL_RUNNER=1")

    return DeepseekV41ForCausalLMConfig
