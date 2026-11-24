"""vLLM 0.26 plugin: upstream DeepSeek-V4.1 (AMD path, language model only) on LUMI-G.

Loaded by vLLM's `vllm.general_plugins` entry point (../lumi_dsv41-0.1.dist-info) in
every vLLM process. ../sitecustomize.py installs the module loader (loader.py) at
interpreter start; register() adds the engine shims and the model, config and
quantization registrations. Put the parent directory on PYTHONPATH.
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

    import os

    if os.environ.get("DSV41_MOE_PACK") == "1":
        # V4 plugin's moe_small: pack_bitmatrix with 16-row blocks (bit-identical; vLLM
        # launches 512-row blocks whatever the batch). Its expert GEMM tiles are for V4's
        # TP-sharded experts, so only this part is used.
        from lumi_dsv4_debug import moe_small

        moe_small.install_pack()
    if os.environ.get("DSV41_MOE_TILES") == "1":
        # V4's moe_small matmul_ogs wrapper with V4.1's tiles: whole experts under EP
        # (48 per rank; gate/up 5120 -> 4608, down 2304 -> 5120). test_moe_matmul.py (job
        # 23857141, graph replay, cold weights): gate/up 143 -> 38 us at 1 token, 154 -> 72
        # at 4, 513 -> 431 at 32, 1006 -> 827 at 128; down 112 -> 32 at 1, 66 -> 48 at 4,
        # flat from 16. Same output as the default tiles (split-K reorders sums: <= 2e-3).
        from lumi_dsv4_debug import moe_small

        # Up to 1024 tokens: DSpark verifies 6 tokens per sequence, so decode batches reach
        # 96-768 rows; block_n 128 / split_k 1 beats the default by 13-16 % on gate/up there
        # (job 23871479; 192: 1068 -> 933 us, 384: 1722 -> 1458), down 0-4 %. Prefill
        # chunks (2048) keep the library's choice.
        # Prefill chunks too (job 23900870): 2048 tokens gate/up 1780 -> 1548 us, down
        # 1002 -> 981; 1024 and 1536 alike.
        moe_small.GATE_UP = [(1, (32, 2)), (2, (32, 4)), (8, (128, 2)), (8192, (128, 1))]
        moe_small.DOWN = [(2, (32, 2)), (4, (128, 2)), (8192, (128, 1))]
        moe_small.install_matmul()
    if os.environ.get("DSV41_GATE_GEMM") == "1":
        # V4's GATE_GEMM (GateLinear -> score_gemm.score_mm, bf16 MFMA, fp32 logits) with
        # tiles for V4.1's router, 5120 -> 384. test_router_gemm.py (job 23857491): 37 us and
        # rel err 2-3e-3 (bf16 F.linear + cast) -> 13-21 us, 2e-7, at 1-128 tokens.
        from lumi_dsv4_debug import moe_small, score_gemm

        score_gemm.CONFIGS[384] = [
            (16, (16, 16, 128, 8, 4)), (24, (16, 64, 128, 8, 4)), (32, (32, 16, 128, 8, 4)),
            (96, (16, 64, 128, 8, 4)), (128, (64, 16, 128, 4, 4))]
        moe_small.install_gate()


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
                if os.environ.get("DSV41_PREFIX_CACHING") == "0":
                    raise NotImplementedError(
                        "DeepSeek-V4.1: DSV41_PREFIX_CACHING=0 but prefix caching is on; "
                        "pass --no-enable-prefix-caching")
                # Without replay (see circular_buffer.py). V4.1's 43 sliding-window layers
                # in 15 groups of 32-token blocks make dense retention hash and queue every
                # sliding-window block: a cached prefix is evicted after 200-400k tokens of
                # other prefill, not the pool's 2.24M (README, agentic load). Keep their
                # tails once per 4096 tokens and at each prompt's end. Set here, in the API
                # process before the engine core starts, so a plain `vllm serve` gets it;
                # DSV41_RETENTION=dense keeps 0.26's default, or set the variable yourself.
                retention = os.environ.get("DSV41_RETENTION", "4096")
                if retention != "dense":
                    os.environ.setdefault("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", retention)
            if not vllm_config.use_v2_model_runner:
                # Engram reads each request's previous tokens from the model state,
                # which only Model Runner V2 calls.
                raise ValueError("DeepSeek-V4.1 needs Model Runner V2: VLLM_USE_V2_MODEL_RUNNER=1")

    return DeepseekV41ForCausalLMConfig
