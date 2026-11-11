"""Functions the loader runs on 0.26 engine-side modules right after they execute.

Grafts of upstream helpers come from graft.py; the rest are hand-written where upstream's
definition depends on engine features 0.26 lacks. Does not import vllm at the top.
"""

from . import graft


def _engram_dp(module) -> None:
    """Upstream shards the Engram tables over DP replicas; we run without DP."""

    def get_engram_dp_group():
        return None

    def get_engram_dp_size() -> int:
        return 1

    module.get_engram_dp_group = get_engram_dp_group
    module.get_engram_dp_size = get_engram_dp_size


def _kv_cache_interface(module) -> None:
    """KVCacheLayout (vendored module, new since 0.26) and CircularBufferSpec.

    The spec's allocation behaviour is registered with 0.26's KV-cache manager in
    circular_buffer.register(), from the plugin's register().
    """
    from dataclasses import dataclass

    from vllm.v1.kv_cache_layout import KVCacheLayout

    module.KVCacheLayout = KVCacheLayout

    @dataclass(frozen=True, kw_only=True)
    class CircularBufferSpec(module.AttentionSpec):
        """One block per request holding the raw keys of the token group still being
        compressed; `block_size` is the ring capacity (upstream kv_cache_interface)."""

        head_size_v: int = 0

        @property
        def real_page_size_bytes(self) -> int:
            # Keys only (head_size_v=0); 0.26's AttentionSpec counts K and V.
            from vllm.utils.torch_utils import get_dtype_size

            return self.block_size * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)

        @property
        def block_table_token_alignment(self):
            return None

        def max_memory_usage_bytes(self, vllm_config) -> int:
            return self.page_size_bytes

        def max_num_blocks_per_req(self, vllm_config, max_len: int) -> int:
            return 1

        def is_uniform_with_collection(self, kv_cache_specs) -> bool:
            return all(isinstance(s, CircularBufferSpec) for s in kv_cache_specs.values())

        @property
        def prefix_cacheable(self) -> bool:
            return False

        @property
        def uses_slot_mapping(self) -> bool:
            return False

    # Specs are pickled between processes: make it resolvable by its public name.
    CircularBufferSpec.__module__ = module.__name__
    CircularBufferSpec.__qualname__ = "CircularBufferSpec"
    module.CircularBufferSpec = CircularBufferSpec
    graft.apply(module)


# Config fields upstream added after 0.26, as class-level defaults: instances that lack
# the field read these, and pickling to workers is unaffected. Values are upstream's
# defaults, except where an upstream default enables a feature that needs engine support
# 0.26 lacks (marked), or upstream resolves the value per platform (ROCm gfx90a here).
CONFIG_DEFAULTS = {
    "vllm.config.attention:AttentionConfig": {
        "hisparse_config": None,
        "indexer_sparse_logits": False,  # upstream: resolved; True only on SM100 + DeepGEMM
        "indexer_logits_dtype": "float32",  # upstream: resolved; bf16 only with MX scales
        "sparse_mla_prefill_workspace_factor": None,
        "sparse_mla_shard_q": True,
    },
    "vllm.config.cache:CacheConfig": {"swa_bounded_replay": False},  # upstream True: needs scheduler replay
    "vllm.config.compilation:CompilationConfig": {"cudagraph_decoder_replay": False},
    "vllm.config.kernel:KernelConfig": {
        "enable_jit_warmup": False,  # upstream True: 0.26's runner has no warmup registry
        "monokernel": False,
        "sparse_indexer_topk_backend": "auto",
    },
    "vllm.config.speculative:SpeculativeConfig": {"enable_adaptive_verification": False},
}


def _config_defaults(module) -> None:
    for target, fields in CONFIG_DEFAULTS.items():
        module_name, cls_name = target.split(":")
        if module_name != module.__name__:
            continue
        cls = getattr(module, cls_name)
        for name, value in fields.items():
            if name not in getattr(cls, "__dataclass_fields__", {}) and not hasattr(cls, name):
                setattr(cls, name, value)
    if module.__name__ == "vllm.config.attention":
        cls = module.AttentionConfig

        def resolve_indexer_kv_dtype(self, default):
            """Upstream: substitute `default` for "auto"."""
            return default if self.indexer_kv_dtype == "auto" else self.indexer_kv_dtype

        def resolve_indexer_logits_dtype(self, default):
            return default if self.indexer_logits_dtype == "auto" else self.indexer_logits_dtype

        cls.resolve_indexer_kv_dtype = resolve_indexer_kv_dtype
        cls.resolve_indexer_logits_dtype = resolve_indexer_logits_dtype
    if module.__name__ == "vllm.config.parallel":
        cls = module.ParallelConfig
        if not hasattr(cls, "pcp_shard_decode_requests"):
            cls.pcp_shard_decode_requests = property(
                lambda self: self.prefill_context_parallel_size > 1
                and self.decode_context_parallel_size == 1
            )


def _aiter_ops(module) -> None:
    """rocm_aiter_ops members V4.1's ROCm code calls. LAIF ships no aiter (gfx942/950
    only), where upstream's checks all return False and its kernels never run."""
    cls = module.rocm_aiter_ops

    def disabled(cls) -> bool:
        return False

    for name in ("is_rdna_aiter_enabled", "is_triton_sparse_mla_enabled",
                 "is_indexer_top_k_enabled"):
        if not hasattr(cls, name):
            setattr(cls, name, classmethod(disabled))
    if not hasattr(cls, "is_indexer_top_k_supported"):
        cls.is_indexer_top_k_supported = classmethod(lambda cls, **kwargs: False)
    if not hasattr(cls, "dsv4_indexer_prefers_native_top_k"):
        # Upstream verbatim (a shape heuristic, no aiter call).
        def dsv4_indexer_prefers_native_top_k(*, num_rows, num_columns, topk_tokens) -> bool:
            return topk_tokens == 512 and 0 < num_rows <= 384 and num_columns <= 1024 * 1024

        cls.dsv4_indexer_prefers_native_top_k = staticmethod(dsv4_indexer_prefers_native_top_k)
    for name in ("fused_qk_rmsnorm_group_quant", "gemm_a8w8_blockscale_bpreshuffle",
                 "indexer_top_k_decode", "indexer_top_k_prefill", "triton_sparse_mla_fwd",
                 "mhc_fused_post_pre_delayed_prefers_unfused"):
        if not hasattr(cls, name):
            def unavailable(*args, _name=name, **kwargs):
                raise NotImplementedError(f"rocm_aiter_ops.{_name}: aiter is not available on gfx90a")

            setattr(cls, name, staticmethod(unavailable))


def _rocm_platform(module) -> None:
    cls = module.RocmPlatform
    if not hasattr(cls, "enable_multi_stream_overlap"):
        # Upstream's ROCm gate allows side streams only inside graph capture; keep
        # every overlap off on gfx90a (the interface default).
        cls.enable_multi_stream_overlap = classmethod(lambda cls, aux_stream_list, attn_metadata: False)


def _envs(module) -> None:
    """Upstream environment variables (envs.py), with upstream's defaults."""
    import os

    added = {
        "VLLM_SPARSE_INDEXER_MAX_DECODE_LOGITS_MB":
            lambda: int(os.getenv("VLLM_SPARSE_INDEXER_MAX_DECODE_LOGITS_MB", "2048")),
        "VLLM_TRITON_JIT_WARMUP_NUM_THREADS":
            lambda: int(os.getenv("VLLM_TRITON_JIT_WARMUP_NUM_THREADS", "4")),
        "VLLM_USE_FUSED_MQA_TOPK": lambda: bool(int(os.getenv("VLLM_USE_FUSED_MQA_TOPK", "1"))),
    }
    for name, fn in added.items():
        module.environment_variables.setdefault(name, fn)


HOOKS = {name: [graft.apply] for name in graft.GRAFTS}
HOOKS["vllm._aiter_ops"] = [_aiter_ops]
HOOKS["vllm.platforms.rocm"].append(_rocm_platform)
HOOKS["vllm.envs"] = [_envs]
for _target in CONFIG_DEFAULTS:
    HOOKS.setdefault(_target.split(":")[0], []).append(_config_defaults)
HOOKS.setdefault("vllm.config.parallel", []).append(_config_defaults)
HOOKS["vllm.distributed.parallel_state"] = [_engram_dp]
HOOKS["vllm.distributed"] = [_engram_dp]
HOOKS["vllm.v1.kv_cache_interface"] = [_kv_cache_interface]
