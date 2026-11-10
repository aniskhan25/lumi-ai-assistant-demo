"""Add the few names upstream's V4.1 code imports from vLLM 0.26's engine-side modules.

Engine modules (configs, distributed, platforms, KV-cache specs, the attention backend
base, the loader) stay 0.26's. Where upstream only added a self-contained helper, its
definition is grafted: the statement is taken from upstream's file (../vendor/graft,
byte-for-byte upstream) and executed in 0.26's module, so it sees 0.26's globals.

install() runs in register(), before anything imports vllm.models.deepseek_v41.
"""

import ast
import importlib
import pathlib

from . import loader

GRAFT_ROOT = loader.VENDOR / "graft" / "vllm"

# module -> names whose upstream top-level definition is executed in 0.26's module
GRAFTS = {
    "vllm.config.multimodal": ["MultiModalDummyOptions"],
    "vllm.distributed.device_communicators.shm_broadcast": ["SHM_PATH", "check_shm_free_space"],
    "vllm.distributed.kv_transfer.kv_connector.utils": ["kv_connector_may_load_foreign_kv"],
    "vllm.forward_context": ["in_piecewise_cudagraph"],
    "vllm.model_executor.layers.fusion.quant_activation": ["get_input_quant_key"],
    "vllm.model_executor.layers.quantization.utils.fp8_utils": ["get_fp8_block_weight_scale"],
    "vllm.model_executor.model_loader.weight_utils": ["make_attention_sink"],
    "vllm.platforms.rocm": ["_ON_RDNA", "get_cdna_version"],
    "vllm.utils.cpu_resource_utils": ["get_available_memory"],
    "vllm.utils.deep_gemm": ["mqa_logits_mx_inputs", "mqa_logits_use_mx_scales"],
    "vllm.v1.attention.backend": ["max_decode_query_len"],
    "vllm.v1.kv_cache_interface": ["get_block_table_width"],
}


def _defined(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {x.id for t in targets for x in ast.walk(t) if isinstance(x, ast.Name)}
    return set()


def graft(module_name: str, names: list[str]) -> None:
    module = importlib.import_module(module_name)
    path = GRAFT_ROOT.joinpath(*module_name.split(".")[1:]).with_suffix(".py")
    tree = ast.parse(path.read_text(), filename=str(path))
    wanted = [n for n in names if n not in module.__dict__]
    for name in wanted:
        nodes = [n for n in tree.body if name in _defined(n)]
        if not nodes:
            raise LookupError(f"{name} is not defined at the top of {path}")
        code = compile(ast.Module(body=nodes[-1:], type_ignores=[]), str(path), "exec")
        exec(code, module.__dict__)


def _distributed() -> None:
    """Engram tables are sharded over DP replicas upstream; we run without DP."""
    import vllm.distributed as distributed
    import vllm.distributed.parallel_state as parallel_state

    def get_engram_dp_group():
        return None

    def get_engram_dp_size() -> int:
        return 1

    for module in (parallel_state, distributed):
        module.get_engram_dp_group = get_engram_dp_group
        module.get_engram_dp_size = get_engram_dp_size


def _kv_cache() -> None:
    import vllm.v1.kv_cache_interface as kv_cache_interface
    from vllm.v1.kv_cache_layout import KVCacheLayout  # vendored (new since 0.26)

    from . import circular_buffer

    kv_cache_interface.KVCacheLayout = KVCacheLayout
    kv_cache_interface.CircularBufferSpec = circular_buffer.CircularBufferSpec
    circular_buffer.register()


_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    loader.install()
    for module_name, names in GRAFTS.items():
        graft(module_name, names)
    _distributed()
    _kv_cache()
    _installed = True
