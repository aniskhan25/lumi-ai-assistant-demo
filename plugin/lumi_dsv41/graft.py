"""Execute single upstream definitions inside vLLM 0.26's engine-side modules.

Engine modules (configs, distributed, platforms, KV-cache specs, the attention backend
base, the loader) stay 0.26's. Where upstream only added self-contained helpers, each
statement is taken from upstream's file (../vendor/graft, byte-for-byte upstream) and run
in 0.26's module, so it sees 0.26's globals. Names it refers to that 0.26's module lacks
are grafted first from the same file: its imports and helper definitions.

The loader applies these right after each module executes (post-import hook), so import
order does not matter. Does not import vllm.
"""

import ast
import builtins
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
GRAFT_ROOT = ROOT / "vendor" / "graft" / "vllm"

GRAFTS = {
    "vllm.config.multimodal": ["MultiModalDummyOptions"],
    "vllm.config.utils": ["CompileFactors", "get_compile_factors"],
    "vllm.distributed.device_communicators.shm_broadcast": ["SHM_PATH", "check_shm_free_space"],
    "vllm.distributed.kv_transfer.kv_connector.utils": ["kv_connector_may_load_foreign_kv"],
    "vllm.forward_context": ["in_piecewise_cudagraph"],
    "vllm.model_executor.layers.fusion.quant_activation": ["get_input_quant_key"],
    "vllm.model_executor.model_loader.weight_utils": ["make_attention_sink"],
    "vllm.platforms.rocm": ["_ON_RDNA", "get_cdna_version"],
    "vllm.utils.cpu_resource_utils": ["get_available_memory"],
    "vllm.utils.deep_gemm": [
        "fp8_fp4_paged_sparse_mqa_logits", "fp8_fp4_sparse_mqa_logits",
        "get_paged_mqa_page_sizes", "get_paged_sparse_mqa_logits_metadata",
        "get_sparse_mqa_logits_metadata", "has_deep_gemm_sparse_mqa",
        "mqa_logits_mx_inputs", "mqa_logits_use_mx_scales", "native_next_n_supported",
    ],
    "vllm.v1.attention.backend": ["max_decode_query_len"],
    "vllm.v1.kv_cache_interface": ["get_block_table_width"],
}

_trees: dict[pathlib.Path, ast.Module] = {}


def _tree(path: pathlib.Path) -> ast.Module:
    if path not in _trees:
        _trees[path] = ast.parse(path.read_text(), filename=str(path))
    return _trees[path]


def _provides(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {x.id for t in targets for x in ast.walk(t) if isinstance(x, ast.Name)}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in node.names}
    return set()


def _graft_name(module, path, name, seen, required) -> None:
    if name in module.__dict__ or name in seen or hasattr(builtins, name):
        return
    seen.add(name)
    tree = _tree(path)
    nodes = [n for n in tree.body if name in _provides(n)]
    if not nodes:
        if required:
            raise LookupError(f"{name} is not defined at the top of {path}")
        return
    node = nodes[-1]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        aliases = [a for a in node.names if (a.asname or a.name).split(".")[0] == name]
        node = type(node)(**{**{f: getattr(node, f) for f in node._fields}, "names": aliases})
        ast.fix_missing_locations(ast.copy_location(node, nodes[-1]))
    else:
        for dep in {x.id for x in ast.walk(node) if isinstance(x, ast.Name)} - {name}:
            try:
                _graft_name(module, path, dep, seen, required=False)
            except Exception:  # noqa: BLE001 - a local that shadows a top-level name
                pass
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), module.__dict__)


def apply(module) -> None:
    names = GRAFTS.get(module.__name__)
    if not names:
        return
    path = GRAFT_ROOT.joinpath(*module.__name__.split(".")[1:]).with_suffix(".py")
    for name in names:
        _graft_name(module, path, name, set(), required=True)
