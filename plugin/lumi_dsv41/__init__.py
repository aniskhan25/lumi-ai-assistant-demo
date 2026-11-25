"""vLLM 0.26 plugin: DeepSeek-V4.1-Flash (upstream AMD path, language model only) on LUMI-G
(MI250X, gfx90a).

Loaded by vLLM's `vllm.general_plugins` entry point (../lumi_dsv41-0.1.dist-info) in every
vLLM process. ../sitecustomize.py installs the module loader (loader.py) at interpreter
start, so the vendored V4.1 modules (../vendor) resolve; register() adds the model, config
and quantization registrations and the gfx90a kernels. Put the parent directory on
PYTHONPATH.
"""

ARCH = "DeepseekV41ForCausalLM"
MODEL = "lumi_dsv41.model:DeepseekV41ForCausalLM"
DSPARK_ARCH = "DSparkDraftModel"
DSPARK_MODEL = "lumi_dsv41.dspark:DSparkDeepseekV41ForCausalLM"


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
    # 0.26 names a DeepSeek target's in-checkpoint DSpark drafter "DSparkDraftModel"
    # (config/speculative.py); this plugin only loads for V4.1, so point it at V4.1's.
    ModelRegistry.register_model(DSPARK_ARCH, DSPARK_MODEL)
    model_configs.MODELS_CONFIG_MAP[ARCH] = _config_class(model_configs)
    from . import quant  # noqa: F401 - registers the deepseek_v41_fp8 method

    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        from . import bf16_linear, custom_ar, moe_small, router_guard

        bf16_linear.install()  # block-FP8 linears as bf16 GEMMs (no FP8 units on gfx90a)
        custom_ar.install()  # vLLM's custom all-reduce, coherent on gfx90a
        moe_small.install()  # MoE routing blocks and expert GEMM tiles for V4.1's shapes
        router_guard.install()  # NaN pad rows under CUDA graphs must not reach the router


def _config_class(model_configs):
    base = model_configs.VerifyAndUpdateConfig

    class DeepseekV41ForCausalLMConfig(base):
        @staticmethod
        def verify_and_update_model_config(model_config) -> None:
            # Upstream routes "fp8" to deepseek_v41.quant_config for this model type;
            # 0.26 only knows V4's, so V4.1's gets its own method name (quant.py).
            for hf in (model_config.hf_config, model_config.hf_text_config):
                # Language model only: no vision tower, so no image-sentinel routing.
                hf.vision_n_layers = 0
                quant = getattr(hf, "quantization_config", None)
                if quant is not None and quant.get("quant_method") == "fp8":
                    quant["quant_method"] = "deepseek_v41_fp8"

        @staticmethod
        def verify_and_update_config(vllm_config) -> None:
            # Upstream's default "auto" lets the indexer pick its cache format (fp8);
            # 0.26 defaults to "bf16". (0.26 types the field without "auto".)
            attention = vllm_config.attention_config
            object.__setattr__(attention, "indexer_kv_dtype", "auto")
            # Upstream resolves this in VllmConfig.__post_init__ (factor * max_model_len
            # sizes the sparse-MLA prefill workspace).
            if getattr(attention, "sparse_mla_prefill_workspace_factor", None) is None:
                factor = 5 if vllm_config.model_config.max_model_len <= 256 * 1024 else 2
                object.__setattr__(attention, "sparse_mla_prefill_workspace_factor", factor)
            spec = vllm_config.speculative_config
            if spec is not None and spec.method != "dspark":
                raise NotImplementedError(
                    f"DeepSeek-V4.1 on vLLM 0.26: speculative method {spec.method!r}; "
                    "only its DSpark drafter (method dspark) is ported")
            import os

            if vllm_config.cache_config.enable_prefix_caching:
                # Prefix caching without replay (circular_buffer.py). V4.1's 43 sliding-window
                # layers in 15 groups of 32-token blocks make 0.26's dense retention hash and
                # queue every sliding-window block, so a cached prefix is evicted after a few
                # hundred thousand tokens of other prefill. Keep sliding-window tails once per
                # 4096 tokens and at each prompt's end. Set here, in the API process before
                # the engine core starts; an explicit setting wins.
                os.environ.setdefault("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "4096")
            if not vllm_config.use_v2_model_runner:
                # Engram reads each request's previous tokens from the model state,
                # which only Model Runner V2 calls.
                raise ValueError("DeepSeek-V4.1 needs Model Runner V2: VLLM_USE_V2_MODEL_RUNNER=1")

    return DeepseekV41ForCausalLMConfig
