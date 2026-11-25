"""Build vLLM 0.26 KV-cache specs from upstream's spec arguments.

Upstream describes compressed caches with `tokens_per_state` and an explicit
`state_content_bytes`; 0.26 has `compress_ratio` and derives the bytes per state from
`cache_dtype_str`/`model_version` (fp8_ds_mla + deepseek_v4: 584 B). On gfx90a V4.1
stores V4's records (its 528 B MXFP8 record is SM100 only), so the pages are identical
and the translation is a rename plus checks. 0.26's merge() rebuilds specs from its own
fields, which is why specs are translated rather than subclassed.

The loader swaps the names in vendored modules for proxies (install_in): calling one
builds a 0.26 spec; isinstance/issubclass and attributes go to the 0.26 class.
"""

FP8_DS_MLA_STATE_BYTES = 584
# Upstream fields with no 0.26 counterpart: value at which dropping them changes nothing.
_DROP_IF_DEFAULT = {
    "max_tp_shards": None,  # planner hint for TP-replicated caches; 0.26 sizes per rank
    "bounded_replay": False,
    "block_stride_alignment": None,
    "non_causal_multi_token_decode": False,
    "topk_buffer_width": None,
    "is_index_group_leader": False,
}
_DROP_ALWAYS = {"max_tp_shards", "cache_role", "topk_buffer_width", "is_index_group_leader"}


def _translate(cls_name: str, kwargs: dict) -> dict:
    kw = dict(kwargs)
    for name, default in _DROP_IF_DEFAULT.items():
        if name in kw:
            value = kw.pop(name)
            if name not in _DROP_ALWAYS and value != default:
                raise NotImplementedError(
                    f"{cls_name}({name}={value!r}) has no vLLM 0.26 equivalent")
    for name in _DROP_ALWAYS:
        kw.pop(name, None)
    tokens = kw.pop("tokens_per_state", 1)
    if tokens != 1:
        kw["compress_ratio"] = tokens
    content = kw.pop("state_content_bytes", None)
    if content is not None:
        native = (FP8_DS_MLA_STATE_BYTES
                  if kw.get("cache_dtype_str") == "fp8_ds_mla"
                  and kw.get("model_version") == "deepseek_v4" else None)
        if content != native:
            raise NotImplementedError(
                f"{cls_name}: {content} B per state differs from vLLM 0.26's layout "
                f"({native} B for cache_dtype_str={kw.get('cache_dtype_str')!r})")
    return kw


class _SpecProxy(type):
    def __call__(cls, *args, **kwargs):
        return cls._real(*args, **_translate(cls._real.__name__, kwargs))

    def __instancecheck__(cls, obj):
        return isinstance(obj, cls._real)

    def __subclasscheck__(cls, sub):
        return issubclass(sub, cls._real)

    def __getattr__(cls, name):
        return getattr(cls._real, name)


def _proxy(real):
    return _SpecProxy(real.__name__, (), {"_real": real, "__module__": real.__module__})


_PROXIES = {}


def install_in(module) -> None:
    """Replace 0.26 spec classes a vendored module imported with translating proxies."""
    import vllm.v1.kv_cache_interface as kvi

    for name in ("MLAAttentionSpec", "SlidingWindowMLASpec"):
        real = getattr(kvi, name)
        if module.__dict__.get(name) is real:
            if name not in _PROXIES:
                _PROXIES[name] = _proxy(real)
            module.__dict__[name] = _PROXIES[name]
