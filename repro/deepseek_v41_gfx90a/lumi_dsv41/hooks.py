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
    _spec_state_properties(module)


def _spec_state_properties(module) -> None:
    """Upstream's per-state view of a page (tokens_per_state, num_states,
    state_content_size_bytes), in terms of 0.26's compress_ratio (see kv_specs.py)."""
    from vllm.utils.torch_utils import get_dtype_size

    def tokens_per_state(self):
        return getattr(self, "compress_ratio", 1)

    def num_states(self):
        return self.block_size // tokens_per_state(self)

    def state_content_size_bytes(self):
        if (getattr(self, "cache_dtype_str", None) == "fp8_ds_mla"
                and getattr(self, "model_version", None) == "deepseek_v4"):
            from . import kv_specs

            return kv_specs.FP8_DS_MLA_STATE_BYTES
        head_size_v = getattr(self, "head_size_v", self.head_size)
        if isinstance(self, (module.MLAAttentionSpec, module.SlidingWindowMLASpec)):
            head_size_v = 0  # one latent vector per state, no separate V
        return (self.head_size + head_size_v) * get_dtype_size(self.dtype)

    base = module.KVCacheSpec
    for name, fn in (("tokens_per_state", tokens_per_state), ("num_states", num_states),
                     ("state_content_size_bytes", state_content_size_bytes)):
        if not hasattr(base, name):
            setattr(base, name, property(fn))


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


def _mhc_no_tilelang(module) -> None:
    """TileLang mHC is wrong on gfx90a: its HIP codegen drops barriers in the
    warp-specialised pre stage (found on V4-Flash, repro/dsv4_flash_debug). Upstream turns
    it off only on gfx942; do the same on gfx90a, so the torch/Triton fallbacks run
    (V4's DSV4_TORCH_MHC=1)."""
    import os

    from vllm.platforms import current_platform

    if current_platform.is_rocm() and os.environ.get("DSV41_TILELANG_MHC") != "1":
        module.HAS_TILELANG_MHC = False


def _mhc_kernels(module) -> None:
    """0.26's kernels/mhc package re-exports its own torch fallbacks; upstream's adds the
    delayed pre (kernels/mhc/torch.py, vendored)."""
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch

    module.mhc_pre_delayed_torch = mhc_pre_delayed_torch


def _fp8_utils(module) -> None:
    """Upstream's get_fp8_block_weight_scale also recognises a Quark per-block scheme
    0.26 does not have; block-FP8 layers otherwise keep `weight_scale_inv`."""
    if not hasattr(module, "get_fp8_block_weight_scale"):
        module.get_fp8_block_weight_scale = lambda layer: getattr(layer, "weight_scale_inv", None)


def _multi_stream_utils(module) -> None:
    """Upstream's execute_in_parallel(default_first=...) orders the default stream's
    work only when multi-stream runs; the plugin keeps it off (enable=False), where both
    run sequentially, default first."""
    stock = module.execute_in_parallel

    def execute_in_parallel(*args, default_first: bool = False, **kwargs):
        return stock(*args, **kwargs)

    module.execute_in_parallel = execute_in_parallel


def _group_rings(module) -> None:
    """0.26's DeepSeek-V4 grouping (group_and_unify_kv_cache_specs) knows only the MLA
    and sliding-window MLA specs and drops anything else; V4.1's compressor keeps its open
    token group in a CircularBufferSpec ring. Give the rings their own uniform group(s)."""
    from collections import defaultdict

    from vllm.v1.kv_cache_interface import CircularBufferSpec, UniformTypeKVCacheSpecs

    stock = module.group_and_unify_kv_cache_specs

    def group_and_unify_kv_cache_specs(kv_cache_spec):
        rings = {n: s for n, s in kv_cache_spec.items() if isinstance(s, CircularBufferSpec)}
        groups = stock({n: s for n, s in kv_cache_spec.items() if n not in rings})
        if groups is None or not rings:
            return groups
        by_size = defaultdict(dict)
        for name, spec in rings.items():
            by_size[spec.block_size][name] = spec
        for specs in by_size.values():
            uniform = UniformTypeKVCacheSpecs.from_specs(specs)
            assert uniform is not None
            groups.append(uniform)
        return groups

    module.group_and_unify_kv_cache_specs = group_and_unify_kv_cache_specs
    # The packer pads every non-MLA group to the MLA layer-tuple page sizes and splits it
    # into tuples; nothing in it depends on the group being sliding-window MLA except
    # one assertion. Let the ring groups through it.
    _source_patch(
        module, "_get_kv_cache_groups_uniform_groups",
        "isinstance(spec, SlidingWindowMLASpec)",
        "isinstance(spec, (SlidingWindowMLASpec, CircularBufferSpec))",
        extra_globals={"CircularBufferSpec": CircularBufferSpec},
    )


def _source_patch(module, name, old, new, extra_globals=None) -> None:
    """Re-define module.name (or module.Class.method) from its own source with one
    exact text replacement."""
    import inspect
    import textwrap

    owner_name, _, attr = name.rpartition(".")
    owner = getattr(module, owner_name) if owner_name else module
    target = owner.__dict__[attr] if owner_name else getattr(module, attr)
    source = textwrap.dedent(inspect.getsource(target))
    if source.count(old) != 1:
        raise RuntimeError(f"lumi_dsv41: cannot patch {module.__name__}.{name}: "
                           f"{old!r} found {source.count(old)} times")
    module.__dict__.update(extra_globals or {})
    scope = {}
    exec(compile(source.replace(old, new), inspect.getsourcefile(module), "exec"),
         module.__dict__, scope)
    setattr(owner, attr, scope[attr])


def _debug_kv_groups(module) -> None:
    """DSV41_DEBUG_KV=1: log the KV cache groups and any layer left out of them."""
    import os

    if os.environ.get("DSV41_DEBUG_KV") != "1":
        return
    stock = module.get_kv_cache_groups

    def get_kv_cache_groups(vllm_config, kv_cache_spec):
        specs = dict(kv_cache_spec)
        groups = stock(vllm_config, kv_cache_spec)
        grouped = {n for g in groups for n in g.layer_names}
        for i, g in enumerate(groups):
            print(f"[DSV41_DEBUG_KV] group {i}: {type(g.kv_cache_spec).__name__} "
                  f"{len(g.layer_names)} layers {g.layer_names[:4]}...", flush=True)
        for name, spec in specs.items():
            print(f"[DSV41_DEBUG_KV] spec {name}: {type(spec).__name__} block={spec.block_size} "
                  f"page={spec.page_size_bytes} window={getattr(spec, 'sliding_window', None)} "
                  f"ratio={getattr(spec, 'compress_ratio', None)}", flush=True)
            if name not in grouped:
                print(f"[DSV41_DEBUG_KV] UNGROUPED {name}: {spec}", flush=True)
        return groups

    module.get_kv_cache_groups = get_kv_cache_groups


def _metadata_builders(module) -> None:
    """Upstream's AttentionGroup passes block_table_width to builders that declare
    requires_block_table_width (V4.1's indexer). Pass the width of the block table 0.26's
    Model Runner V2 actually builds for a full-attention group, cdiv(max_model_len,
    block_size) per KV block, since the builder copies rows of it into its buffer."""
    from vllm.utils.math_utils import cdiv

    cls = module.AttentionGroup

    def create_metadata_builders(self, vllm_config, device, kernel_block_size=None,
                                 num_metadata_builders: int = 1):
        spec = self.kv_cache_spec
        spec_builder = (spec.copy_with_new_block_size(kernel_block_size)
                        if kernel_block_size is not None else spec)
        builder_cls = self.backend.get_builder_cls()
        kwargs = {}
        if getattr(builder_cls, "requires_block_table_width", False):
            width = cdiv(vllm_config.model_config.max_model_len, spec.block_size)
            if kernel_block_size is not None:
                width *= spec.block_size // kernel_block_size
            kwargs["block_table_width"] = width
        self.metadata_builders = [
            builder_cls(spec_builder, self.layer_names, vllm_config, device, **kwargs)
            for _ in range(num_metadata_builders)
        ]

    cls.create_metadata_builders = create_metadata_builders


def _kv_cache_shapes(module) -> None:
    """Upstream backends no longer implement get_kv_cache_shape: the worker derives a
    [blocks, head slots, states, state content] page from the spec. 0.26's reshape asks
    the backend; fall back to upstream's layout when the backend does not answer. Here
    block_size is already the kernel block in states (storage_block_size)."""
    from vllm.utils.torch_utils import get_dtype_size

    def _lumi_kv_cache_shape(backend, spec, num_blocks, block_size, num_kv_heads,
                             head_size, cache_dtype_str="auto"):
        try:
            return backend.get_kv_cache_shape(num_blocks, block_size, num_kv_heads,
                                              head_size, cache_dtype_str=cache_dtype_str)
        except NotImplementedError:
            content = spec.state_content_size_bytes
            dtype_size = get_dtype_size(spec.dtype)
            assert content % dtype_size == 0, (spec, content)
            return (num_blocks, num_kv_heads, block_size, content // dtype_size)

    module._lumi_kv_cache_shape = _lumi_kv_cache_shape
    _source_patch(module, "_reshape_kv_cache",
                  "kv_cache_shape = group.backend.get_kv_cache_shape(",
                  "kv_cache_shape = _lumi_kv_cache_shape(group.backend, kv_cache_spec, ")


def _backend_block_dim(module) -> None:
    """Upstream backends define no get_kv_cache_shape; their [B, H, N, C] pages have the
    block index first (see _kv_cache_shapes)."""
    cls = module.AttentionBackend
    stock = cls.get_kv_cache_block_dim.__func__

    def get_kv_cache_block_dim(cls, *args, **kwargs) -> int:
        try:
            return stock(cls, *args, **kwargs)
        except NotImplementedError:
            return 0

    cls.get_kv_cache_block_dim = classmethod(get_kv_cache_block_dim)


HOOKS = {name: [graft.apply] for name in graft.GRAFTS}
HOOKS["vllm.v1.attention.backend"].append(_backend_block_dim)
HOOKS["vllm.v1.worker.gpu.attn_utils"] = [_kv_cache_shapes]
HOOKS["vllm.v1.worker.utils"] = [_metadata_builders]
HOOKS["vllm.v1.core.kv_cache_utils"] = [_group_rings, _debug_kv_groups]
HOOKS["vllm.utils.multi_stream_utils"] = [_multi_stream_utils]
HOOKS["vllm.model_executor.layers.quantization.utils.fp8_utils"] = [_fp8_utils]
HOOKS["vllm.model_executor.kernels.mhc"] = [_mhc_kernels]
HOOKS["vllm._aiter_ops"] = [_aiter_ops]
HOOKS["vllm.platforms.rocm"].append(_rocm_platform)
HOOKS["vllm.envs"] = [_envs]
for _target in CONFIG_DEFAULTS:
    HOOKS.setdefault(_target.split(":")[0], []).append(_config_defaults)
HOOKS.setdefault("vllm.config.parallel", []).append(_config_defaults)
HOOKS["vllm.distributed.parallel_state"] = [_engram_dp]
HOOKS["vllm.distributed"] = [_engram_dp]
HOOKS["vllm.v1.kv_cache_interface"] = [_kv_cache_interface]

def _indexer_op(module) -> None:
    """Register the ROCm sparse indexer op without aiter, as upstream does on gfx950:
    every step inside it has a non-aiter path (Triton cache insert and gather, compiled
    top-k, torch/Triton MQA logits); V4's INDEXER_GATE finding."""
    import torch

    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import direct_register_custom_op

    if not current_platform.is_rocm():
        return
    try:
        torch.ops.vllm.rocm_aiter_sparse_attn_indexer
        return
    except (AttributeError, RuntimeError):
        pass
    direct_register_custom_op(
        op_name="rocm_aiter_sparse_attn_indexer",
        op_func=module.rocm_aiter_sparse_attn_indexer,
        mutates_args=["topk_indices_buffer", "candidate_blocks"],
        fake_impl=module.rocm_aiter_sparse_attn_indexer_fake,
        dispatch_key=current_platform.dispatch_key,
    )


def _indexer_gate(module) -> None:
    """Upstream's SparseAttnIndexer.forward_hip runs the op only with aiter or on gfx950
    and raises otherwise; on gfx90a make the identical call (op registered above)."""
    import torch

    cls = module.SparseAttnIndexer
    stock = cls.forward_hip

    def forward_hip(self, hidden_states, q_quant, k, weights):
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.platforms.rocm import on_gfx950

        if rocm_aiter_ops.is_enabled() or on_gfx950():
            return stock(self, hidden_states, q_quant, k, weights)
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor)
        import vllm.v1.attention.ops.rocm_aiter_mla_sparse  # noqa: F401 - registers the op

        return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
            hidden_states,
            module._encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_quant,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=self.skip_k_cache_insert,
            compress_ratio=self.compress_ratio,
            candidate_blocks=self.candidate_blocks,
            candidate_block_size=self.candidate_block_size,
            candidate_write=self.candidate_write,
            use_pcp=self.use_pcp,
            pcp_shard_decode_requests=self.pcp_shard_decode_requests,
            cache_layout=self.cache_layout,
            dcp_rank=self.dcp_rank,
            dcp_world_size=self.dcp_world_size,
            cp_interleave=self._parallel_config.cp_kv_cache_interleave_size,
        )

    cls.forward_hip = forward_hip


def _v41_attention(module) -> None:
    """V4.1's query skips V4's per-head norm; 0.26's fused kernel always applies it.
    Route the call to ops.py (KV side still on 0.26's kernel; query in PyTorch)."""
    from . import ops

    _source_patch(
        module, "DeepseekV4Attention._fused_qnorm_rope_kv_insert",
        "torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(",
        "_lumi_qnorm_rope_kv_rope_quant_insert(",
        extra_globals={"_lumi_qnorm_rope_kv_rope_quant_insert": ops.qnorm_rope_kv_rope_quant_insert},
    )


# Functions run on vendored (upstream) modules right after they execute.
VENDORED_HOOKS = {
    "vllm.model_executor.layers.mhc": [_mhc_no_tilelang],
    "vllm.v1.attention.ops.rocm_aiter_mla_sparse": [_indexer_op],
    "vllm.model_executor.layers.sparse_attn_indexer": [_indexer_gate],
    "vllm.models.deepseek_v41.attention": [_v41_attention],
}
