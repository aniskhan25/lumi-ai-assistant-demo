"""Upstream's CircularBufferSpec and CircularBufferManager, ported to vLLM 0.26.

V4.1's compressor keeps the raw keys of the token group that is still open in a
ring: one block per request for its whole lifetime, never prefix-cached. Upstream's core
reads that from spec properties that 0.26's core lacks, so the behaviour lives in a
manager registered through 0.26's KVCacheSpecRegistry instead.
"""

import sys
from collections.abc import Sequence
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger("vllm.lumi_dsv41")  # vLLM configures only "vllm.*" loggers


@dataclass(frozen=True, kw_only=True)
class CircularBufferSpec(AttentionSpec):
    """One block per request; `block_size` is the ring capacity."""

    head_size_v: int = 0

    @property
    def real_page_size_bytes(self) -> int:
        # Keys only (upstream head_size_v=0); 0.26's AttentionSpec counts K and V.
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


class CircularBufferManager(FullAttentionManager):
    """Claims the ring's single block per request; prefix caching disabled."""

    def _claim_ring_block(self, request_id: str) -> list[KVCacheBlock]:
        req_blocks = self.req_to_blocks[request_id]
        if req_blocks:
            return []
        new_blocks = self.block_pool.get_new_blocks(1)
        req_blocks.extend(new_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(block.block_id for block in new_blocks)
        return new_blocks

    def get_num_blocks_to_allocate(self, request_id: str, *args, **kwargs) -> int:
        return 0 if self.req_to_blocks.get(request_id) else 1

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        return self._claim_ring_block(request_id)

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        self._claim_ring_block(request_id)

    @classmethod
    def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids, *args, **kwargs):
        return tuple([] for _ in kv_cache_group_ids), 0

    def cache_blocks(self, request, num_tokens: int, retention_interval=None) -> None:
        return

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        return

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        return

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        return 0

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0


def register() -> None:
    """Add the ring to 0.26's built-in spec registration.

    0.26 registers its built-in specs only while the registry is empty
    (KVCacheSpecRegistry._ensure_registered), and the engine core calls
    register_all_kvcache_specs directly, through a name it imported before any plugin
    ran (vllm/v1/engine/core.py). Registering ours first would make the registry
    non-empty and silently drop every built-in, so wrap that function, wherever it
    is bound, to append the ring after them.
    """
    import vllm.v1.core.single_type_kv_cache_manager as managers
    import vllm.v1.kv_cache_spec_registry as registry
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    def add_ring() -> None:
        KVCacheSpecRegistry.register(
            CircularBufferSpec,
            CircularBufferManager,
            uniform_type_base_spec=CircularBufferSpec,
        )

    builtin = managers.register_all_kvcache_specs
    if not getattr(builtin, "_lumi_dsv41", False):

        def register_all_kvcache_specs(vllm_config):
            builtin(vllm_config)
            add_ring()

        register_all_kvcache_specs._lumi_dsv41 = True
        managers.register_all_kvcache_specs = register_all_kvcache_specs
        engine_core = sys.modules.get("vllm.v1.engine.core")
        if engine_core is not None:
            engine_core.register_all_kvcache_specs = register_all_kvcache_specs

    if registry._REGISTRY_KVCACHESPEC_LIST:  # built-ins already in: add directly
        add_ring()
