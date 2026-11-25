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
    warp-specialised pre stage (found on V4-Flash). Upstream turns
    it off only on gfx942; do the same on gfx90a, so the torch/Triton fallbacks run
    """
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        module.HAS_TILELANG_MHC = False


def _mhc_triton(module) -> None:
    """The delayed pre and the post in Triton (mhc_triton.py) where neither aiter nor
    TileLang serves them, instead of the torch references."""
    if module.HAS_AITER_MHC or module.HAS_TILELANG_MHC:
        return
    from .mhc_triton import mhc_post_triton, mhc_pre_delayed_triton

    def pre_forward_hip(self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix=None,
                        x=None, norm_weight=None, norm_eps=1e-6, sublayer_out=None,
                        post_layer_mix=None, comb_res_mix=None):
        residual = self._maybe_post(residual, sublayer_out, post_layer_mix, comb_res_mix)
        post, comb, layer_input, next_pre = mhc_pre_delayed_triton(
            residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
            hc_post_mult_value, sinkhorn_repeat, pre_mix=pre_mix, x=x)
        return (residual, post, comb, module._apply_mhc_norm(layer_input, norm_weight, norm_eps),
                next_pre)

    def post_forward_hip(self, x, residual, post_layer_mix, comb_res_mix):
        return mhc_post_triton(x, residual, post_layer_mix, comb_res_mix)

    module.MHCPreDelayedOp.forward_hip = pre_forward_hip
    module.MHCPostOp.forward_hip = post_forward_hip


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
    # Upstream layers unpack their page view in bind_kv_cache() (e.g. [B, H=1, N, C] ->
    # [B, N, C]); 0.26 assigns layer.kv_cache directly. Call the method when present.
    module._lumi_bind_layer = lambda layer, kv_cache: (
        layer.bind_kv_cache(kv_cache) if hasattr(layer, "bind_kv_cache")
        else setattr(layer, "kv_cache", kv_cache))
    _source_patch(module, "bind_kv_cache",
                  "forward_context[layer_name].kv_cache = kv_cache",
                  "_lumi_bind_layer(forward_context[layer_name], kv_cache)")


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


def _dspark_confidence_head(module) -> None:
    """Upstream's V4.1 drafter imports DSparkConfidenceHead from qwen3_dspark; 0.26 has
    only DSparkMarkovHead (same signature as upstream's). The head serves adaptive
    verification, which 0.26's DSpark speculator does not do, so lumi_dsv41/dspark.py
    turns it off; this stand-in satisfies the import and refuses to be built."""
    if hasattr(module, "DSparkConfidenceHead"):
        return

    class DSparkConfidenceHead:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("DSpark confidence head: not on vLLM 0.26")

    module.DSparkConfidenceHead = DSparkConfidenceHead


def _config_replace(module) -> None:
    """vllm.config.utils.replace rebuilds a config from its instance __dict__ and raises
    for any key that is not a dataclass field. The plugin stores upstream-only settings as
    plain attributes on 0.26's AttentionConfig (indexer_kv_dtype,
    sparse_mla_prefill_workspace_factor; __init__.py verify_and_update_config), so the
    DSpark loader's replace(attention_config, ...) failed. Skip non-fields
    when rebuilding, then carry them over to the new instance."""
    import dataclasses

    # Fields the plugin sets to values 0.26's type does not admit (validated again on
    # rebuild): indexer_kv_dtype "auto" (0.26's Literal lacks it).
    outside_type = {"indexer_kv_dtype": "auto"}

    def replace(dataclass_instance, /, **kwargs):
        cls = type(dataclass_instance)
        names = {f.name for f in dataclasses.fields(cls)}
        current = dataclass_instance.__dict__
        carry = {k: v for k, v in current.items() if k not in kwargs and (
            k not in names or outside_type.get(k, object()) == v)}
        init = {k: v for k, v in current.items()
                if k in names and k not in carry and module.is_init_field(cls, k)}
        init.update(kwargs)
        new = cls(**init)
        for k, v in carry.items():  # upstream-only attributes and values the plugin set
            object.__setattr__(new, k, v)
        return new

    module.replace = replace


def _expert_placement(module) -> None:
    """A per-layer expert -> EP rank placement instead of the linear one (48 consecutive
    experts a rank), from expert_placement.json: MoE layer name -> 384 ranks, balanced on
    router counts measured under agentic GSM8K-style load; layers not in it (the DSpark
    drafter's) stay linear. DSV41_EXPERT_PLACEMENT=<json> uses another file, =none the
    linear placement.

    Each layer ends in an all-reduce that waits for its slowest rank, and the router's hot
    experts are concentrated in a different rank per layer: timed on the served kernels, a
    balanced placement cuts the MoE critical path of a 2048-token chunk from 146.6 to
    120.6 ms. The map is the one ExpertMapManager
    hands both the weight loader (global -> local expert id; others are skipped) and the
    Triton MoE path (remap_topk_to_local), so overriding it before weights load is enough.
    Local ids follow ascending global ids."""
    import json
    import os

    path = os.environ.get("DSV41_EXPERT_PLACEMENT",
                          os.path.join(os.path.dirname(__file__), "expert_placement.json"))
    if path in ("", "none"):
        return
    import torch

    import vllm.model_executor.layers.fused_moe.expert_map_manager as emm

    placement = json.load(open(path))
    manager_cls = emm.ExpertMapManager
    if not getattr(manager_cls, "_lumi_placement_hook", False):
        stock_calc = manager_cls._calculate_expert_maps

        def _calculate_expert_maps(self):
            stock_calc(self)
            ranks = getattr(self, "_lumi_placement", None)
            if ranks is None or self._expert_map is None:
                return
            assert len(ranks) == self.global_num_experts and self.num_fused_shared_experts == 0
            mine = [e for e, r in enumerate(ranks) if r == self.ep_rank]
            assert len(mine) == self._local_num_experts, (len(mine), self._local_num_experts)
            emap = torch.full((self.global_num_experts,), -1, dtype=torch.int32)
            emap[mine] = torch.arange(len(mine), dtype=torch.int32)
            self._expert_map = emap.to(self._expert_map.device)
            if self._expert_mask is not None:
                self._expert_mask[: self.global_num_experts] = emap > -1

        manager_cls._calculate_expert_maps = _calculate_expert_maps
        manager_cls._lumi_placement_hook = True

    cls = module.RoutedExperts
    stock_init = cls.__init__

    def __init__(self, *args, **kwargs):
        stock_init(self, *args, **kwargs)
        ranks = placement.get(self.layer_name)
        if ranks is not None and self.expert_map_manager.expert_map is not None:
            self.expert_map_manager._lumi_placement = ranks
            self.expert_map_manager._calculate_expert_maps()
            self.update_expert_map_info()

    cls.__init__ = __init__


HOOKS = {name: [graft.apply] for name in graft.GRAFTS}
HOOKS["vllm.config.utils"].append(_config_replace)
HOOKS["vllm.model_executor.models.qwen3_dspark"] = [_dspark_confidence_head]
HOOKS["vllm.v1.attention.backend"].append(_backend_block_dim)
HOOKS["vllm.v1.worker.gpu.attn_utils"] = [_kv_cache_shapes]
HOOKS["vllm.v1.worker.utils"] = [_metadata_builders]
HOOKS["vllm.model_executor.layers.fused_moe.routed_experts"] = [_expert_placement]
HOOKS["vllm.v1.core.kv_cache_utils"] = [_group_rings]
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

def _indexer_graph_pad(module) -> None:
    """Under PIECEWISE graphs 0.26's Model Runner V2 pads the token count to the graph
    size but not the request count (upstream pads requests too). The decode indexer then
    unpacks top-k rows for the real requests only (15) and stores them into the padded
    token slice (16): shape error, engine dead. Store the real rows and
    mark the pad tokens' rows invalid (-1). Must run before _indexer_op registers it."""
    _source_patch(
        module, "rocm_aiter_sparse_attn_indexer",
        "            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (\n"
        "                topk_indices\n"
        "            )\n",
        "            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (\n"
        "                topk_indices\n"
        "            )\n"
        "            topk_indices_buffer[topk_indices.shape[0] : num_decode_tokens] = -1\n",
    )


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


def _chunked_mqa_logits(module) -> None:
    """The indexer's torch MQA-logits fallback materialises a dense fp32 [heads, rows, kv]
    score in one go (4 GiB for an 8k-row prefill chunk at 12k context: OOM next to the
    KV cache). Rows are independent: compute them in chunks of ~512 MiB of scores, the
    same arithmetic per row."""
    import torch

    stock = module.fp8_mqa_logits_torch

    def fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
        m, h, n = q.shape[0], q.shape[1], kv[0].shape[0]
        rows = max(1, (512 << 20) // max(1, h * n * 4))
        if rows >= m:
            return stock(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
        out = torch.empty(m, n, dtype=torch.float32, device=q.device)
        for start in range(0, m, rows):
            end = min(m, start + rows)
            out[start:end] = stock(q[start:end], kv, weights[start:end],
                                   cu_seqlen_ks[start:end], cu_seqlen_ke[start:end])
        return out

    module.fp8_mqa_logits_torch = fp8_mqa_logits_torch


def _prefill_stages(module) -> None:
    """Launch the ROCm sparse-attention prefill kernel with num_stages=1, 2 warps and the
    16-wide matrix instruction. Its wrapper leaves Triton's default pipelining (2 stages),
    which makes it 2.2x slower on gfx90a; with these settings a 2048-query chunk at 16k
    context takes 1.77 ms instead of 5.99, same output."""
    if module._ON_GFX950:
        return
    _source_patch(
        module, "_rocm_sparse_attn_prefill_ragged_triton",
        "        num_warps=num_warps,\n",
        "        num_warps=2,\n        num_stages=1,\n        matrix_instr_nonkdim=16,\n"
        "        waves_per_eu=1,\n",
    )


def _prefill_logits_triton(module) -> None:
    """The indexer's prefill logits (fp8_mqa_logits_torch, the no-aiter fallback of
    rocm_fp8_mqa_logits) in one Triton kernel (mqa_logits.py) instead of torch einsum +
    elementwise passes + a reduction (~10x). Replaces the row-chunked version above."""
    from .mqa_logits import fp8_mqa_logits_triton

    module.fp8_mqa_logits_torch = fp8_mqa_logits_triton


# (max_num_seqs, num_speculative_tokens), set by the model's __init__ (model.py).
SERVING_LIMITS: tuple[int, int] | None = None


def _decode_rows_bound(module) -> None:
    """Upstream bounds the decode-logits workspace rows by max_num_seqs through
    get_current_vllm_config(), which under 0.26 does not return the serving config in the
    profiling forward: the bound falls back to max_num_batched_tokens and gfx90a's
    [heads, rows, max_model_len] fp32 reservation becomes 32 GiB (8192 rows, 32k context)
    instead of 128 MiB. Use the limits captured when the model was built."""
    stock = module._max_decode_logits_rows

    def _max_decode_logits_rows(num_batched_tokens: int) -> int:
        if SERVING_LIMITS is None:
            return stock(num_batched_tokens)
        max_num_seqs, num_spec = SERVING_LIMITS
        return min(num_batched_tokens, max_num_seqs * (1 + num_spec))

    module._max_decode_logits_rows = _max_decode_logits_rows


def _decode_inv_rope(module) -> None:
    """rocm_sparse_attn_decode reports every row as inverse-RoPE'd when given
    inv_rope_positions (it returns output.shape[0]), but only the gfx942/gfx950 split-K
    reduce applies it: the fallback kernel for other GPUs returns before that. On gfx90a
    the caller (DeepseekV41 ROCm attention) then skips its standalone inverse RoPE for the
    decode rows, so every decode token reaches wo_a with a rotated RoPE tail: prefill is
    right, decode drifts (decode and prefill logprobs of the same tokens differed by up to
    23 nats, GSM8K 3 %). Withhold the positions there: the call then
    returns 0 and the caller's rocm_inverse_rope_rows_ covers the decode rows too."""
    if module._ON_GFX942 or module._ON_GFX950:
        return
    stock = module.rocm_sparse_attn_decode

    def rocm_sparse_attn_decode(*args, **kwargs):
        kwargs.pop("inv_rope_positions", None)
        kwargs.pop("inv_rope_cos_sin_cache", None)
        assert kwargs.get("output_mxfp8") is None, "MXFP8 decode output needs gfx950"
        return stock(*args, **kwargs)

    module.rocm_sparse_attn_decode = rocm_sparse_attn_decode




def _sparse_decode(module) -> None:
    """Decode sparse attention with sparse_decode.py on gfx90a, at every batch size
    (2.1-14x the one-pass fallback, 1.6-6x upstream's split-K path at 1-128 queries).
    Calls the stock function for what it does not cover: FNUZ caches, MXFP8 output, fused
    inverse RoPE (withheld on gfx90a by _decode_inv_rope anyway), more than 16 heads, other
    dims."""
    if module._ON_GFX942 or module._ON_GFX950:
        return
    from .sparse_decode import sparse_decode

    stock = module._rocm_sparse_attn_decode_ragged_triton
    names = ("q", "main_cache", "main_indices", "main_indptr", "scale", "attn_sink",
             "nope_head_dim", "rope_head_dim", "extra_cache", "extra_indices", "extra_indptr",
             "out")

    def decode(*args, **kwargs):
        a = dict(zip(names, args))
        a.update(kwargs)
        q, out = a["q"], a.get("out")
        if (module.current_platform.is_fp8_fnuz() or a.get("out_mxfp8") is not None
                or a.get("inv_rope_positions") is not None or a.get("extra_cache_nan_free")
                or a["nope_head_dim"] != 448 or a["rope_head_dim"] != 64 or q.shape[1] > 16
                or q.stride(-1) != 1 or (out is not None and out.stride(-1) != 1)):
            return stock(*args, **kwargs)
        ei, ep = a.get("extra_indices"), a.get("extra_indptr")
        return sparse_decode(
            q, a["main_cache"], module._as_int32_contiguous_1d(a["main_indices"]),
            module._as_int32_contiguous_1d(a["main_indptr"]), a["scale"], a["attn_sink"],
            extra_cache=a.get("extra_cache"),
            extra_indices=None if ei is None else module._as_int32_contiguous_1d(ei),
            extra_indptr=None if ep is None else module._as_int32_contiguous_1d(ep),
            out=out,
        )

    module._rocm_sparse_attn_decode_ragged_triton = decode


def _paged_logits_unshuffle(module) -> None:
    """V4.1's indexer K writer (indexer_k_store.py) stores values on ROCm in the 16x16-tiled
    ("SHUFFLE") layout, which prefill's gather and aiter's Preshuffle decode kernel read.
    Without aiter, decode falls back to fp8_paged_mqa_logits_torch, which reads the values
    row-major: permuted key bytes, wrong top-k once a context has more compressed
    positions than index_topk (decode diverged from prefill from ~3k tokens on, job
    23643490). Un-tile each gathered page: within a block, byte b of position p is at
    [p // 16, b // 16, p % 16, b % 16]. Scales (fp32, one per position) are not tiled."""
    import torch
    import torch.nn.functional as F

    stock = module.fp8_paged_mqa_logits_torch
    fp8 = module.FP8_DTYPE
    tile = 16

    def fp8_paged_mqa_logits_torch(q, kv_cache, weights, context_lens, block_tables,
                                   max_model_len):
        batch, next_n, _, dim = q.shape
        block = kv_cache.shape[1]
        if next_n != 1 or block <= 1:  # untiled caches (the writer tiles iff block > 1)
            return stock(q, kv_cache, weights, context_lens, block_tables, max_model_len)
        assert block % tile == 0 and dim % tile == 0
        logits = torch.full((batch, max_model_len), float("-inf"), device=q.device,
                            dtype=torch.float32)
        lens = context_lens.reshape(batch, -1)[:, -1].tolist()
        flat = kv_cache.view(kv_cache.shape[0], -1)
        for i, seq_len in enumerate(lens):
            pages = -(-seq_len // block)
            cache = flat[block_tables[i, :pages].long()]
            values = (cache[:, : block * dim]
                      .view(pages, block // tile, dim // tile, tile, tile)
                      .permute(0, 1, 3, 2, 4).reshape(pages * block, dim))
            scales = cache[:, block * dim:].contiguous().view(torch.float32).reshape(-1)
            score = F.relu(F.linear(values.view(fp8).float(), q[i, 0].float()))
            score = (score * weights[i][None, :]).sum(dim=1) * scales
            logits[i, :seq_len] = score[:seq_len]
        return logits

    module.fp8_paged_mqa_logits_torch = fp8_paged_mqa_logits_torch


def _paged_logits_triton(module) -> None:
    """Decode logits off gfx942/950 without aiter: one batched Triton launch with no host
    sync (paged_logits.py) instead of the per-sequence torch loop above, so the indexer can
    sit inside a CUDA graph and does not grow with the batch."""
    if module._ON_GFX942 or module._ON_GFX950:
        return
    from .paged_logits import fp8_paged_mqa_logits_triton

    stock = module.rocm_fp8_paged_mqa_logits

    def rocm_fp8_paged_mqa_logits(q_fp8, kv_cache_fp8, weights, context_lens, block_tables,
                                  schedule_metadata, max_model_len, **kw):
        from vllm._aiter_ops import rocm_aiter_ops

        if rocm_aiter_ops.is_enabled() or q_fp8.shape[1] != 1:
            return stock(q_fp8, kv_cache_fp8, weights, context_lens, block_tables,
                         schedule_metadata, max_model_len, **kw)
        # The writer tiles iff block_size > 1 (indexer_k_store.py).
        shuffle = kv_cache_fp8.shape[1] > 1
        return fp8_paged_mqa_logits_triton(q_fp8, kv_cache_fp8, weights, context_lens,
                                           block_tables, max_model_len, shuffle)

    module.rocm_fp8_paged_mqa_logits = rocm_fp8_paged_mqa_logits


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


def _v41_dspark(module) -> None:
    """The DSpark drafter writes its context KV through the same fused op with upstream's
    11 arguments; 0.26's takes 9. Route it to ops.py as the attention."""
    from . import ops

    _source_patch(
        module, "_insert_context_kv",
        "torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(",
        "_lumi_qnorm_rope_kv_rope_quant_insert(",
        extra_globals={"_lumi_qnorm_rope_kv_rope_quant_insert": ops.qnorm_rope_kv_rope_quant_insert},
    )


# Functions run on vendored (upstream) modules right after they execute.
def _empty_prefill_chunks(module) -> None:
    """Async scheduling with --max-num-batched-tokens 2048 crashed in
    V4.1's ROCm _forward_prefill: a chunk of 22 prefills whose CPU query and sequence
    lengths were all 0 gives chunk_M = 0, and view(22, 0, -1) fails. A request with no
    query tokens writes no output rows, so its chunk can be skipped."""
    cls = module.DeepseekSparseSWAMetadata
    stock = cls.get_prefill_chunk_plan

    def get_prefill_chunk_plan(self, *args, **kwargs):
        return [c for c in stock(self, *args, **kwargs) if c[3] > 0]

    cls.get_prefill_chunk_plan = get_prefill_chunk_plan


VENDORED_HOOKS = {
    "vllm.v1.attention.backends.mla.sparse_swa": [_empty_prefill_chunks],
    "vllm.model_executor.layers.mhc": [_mhc_no_tilelang, _mhc_triton],
    "vllm.v1.attention.ops.rocm_aiter_mla_sparse": [
        _indexer_graph_pad, _indexer_op, _chunked_mqa_logits, _prefill_logits_triton,
        _decode_rows_bound, _decode_inv_rope,
        _paged_logits_unshuffle, _paged_logits_triton, _sparse_decode,
        _prefill_stages],
    "vllm.model_executor.layers.sparse_attn_indexer": [_indexer_gate],
    "vllm.models.deepseek_v41.attention": [_v41_attention],
    "vllm.models.deepseek_v41.amd.dspark": [_v41_dspark],
}
