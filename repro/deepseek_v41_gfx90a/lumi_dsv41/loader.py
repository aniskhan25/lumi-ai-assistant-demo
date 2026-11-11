"""Serve the vendored upstream modules (../vendor/vllm) in place of, or next to, vLLM 0.26's.

Installed at interpreter start by ../sitecustomize.py, before anything imports vllm, so
every process (API server, engine core, workers, spawned helpers) sees the same modules.
It does not import vllm itself.

- A `vllm.*` module with a file under vendor/vllm is loaded from there. If 0.26 has the
  same module (a "shadow"), names that only 0.26's version defines stay reachable through
  a module `__getattr__` that loads 0.26's copy on first use, under a private name. That
  keeps 0.26 importers working where upstream moved helpers (e.g. the context-parallel
  ones in v1/attention/ops/common.py) without executing 0.26's copy otherwise.
- A module in hooks.HOOKS is 0.26's, with functions run on it right after it executes:
  grafts of upstream helpers (graft.py) and a few hand-written additions (hooks.py).
- A module in STUBS is an empty placeholder: upstream imports it on NVIDIA, CPU, vision
  or optional paths only. Any name from it is a class that raises when used.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
# Our own edits of whole vendored files (e.g. a language-model-only package __init__);
# searched before vendor/.
OVERRIDES = pathlib.Path(__file__).resolve().parent / "overrides"
PRIVATE = "_lumi_dsv41_v026."

STUBS = frozenset({
    "vllm._cpu_ops",
    "vllm.model_executor.kernels.linear.mxfp8.pytorch",
    "vllm.model_executor.layers.fused_moe.moe_output",
    "vllm.model_executor.layers.rocm_paged_mxfp4_indexer",
    "vllm.models.deepseek_v4.amd.mega_moe",
    "vllm.models.deepseek_v4.common.vision",
    "vllm.models.deepseek_v4.cpu",
    "vllm.models.deepseek_v4.cpu.cpu_mla",
    "vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer",
    "vllm.v1.attention.ops.rocm_paged_mxfp4_indexer",
    "vllm.v1.attention.ops.pcp",
    "vllm.v1.attention.ops.shardq",
    "vllm.models.deepseek_v41.amd.mono_decode",  # opt-in monokernel (--monokernel, gfx942/950)
})


class _Unported:
    """Placeholder for a name from a stubbed module: subclassable, unusable."""

    _where = "?"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(f"{self._where} is not ported to LAIF's vLLM 0.26 (lumi_dsv41)")

    def __class_getitem__(cls, item):
        return cls


def _placeholder(module: str, name: str):
    if name.isupper():  # a constant, e.g. a kernel handle: falsy, never called
        return None
    return type(name, (_Unported,), {"_where": f"{module}.{name}", "__module__": module})


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = module.__name__
        module.__path__ = []  # also acts as a package, e.g. vllm.models.deepseek_v4.cpu

        def __getattr__(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            value = _placeholder(name, attr)
            setattr(module, attr, value)
            return value

        module.__getattr__ = __getattr__


class _ShadowLoader(importlib.machinery.SourceFileLoader):
    """Upstream source; 0.26-only names fall back to 0.26's copy, loaded lazily."""

    def __init__(self, fullname, path, old_origin):
        super().__init__(fullname, path)
        self.old_origin = old_origin

    def exec_module(self, module):
        super().exec_module(module)
        if self.old_origin is None:
            return
        own = module.__dict__.get("__getattr__")
        old_origin, fullname = self.old_origin, module.__name__

        def __getattr__(attr):
            if own is not None:
                try:
                    return own(attr)
                except AttributeError:
                    pass
            if attr.startswith("__"):
                raise AttributeError(attr)
            old = _load_old(fullname, old_origin)
            try:
                value = getattr(old, attr)
            except AttributeError:
                raise AttributeError(f"module {fullname!r} has no attribute {attr!r}") from None
            setattr(module, attr, value)
            return value

        module.__getattr__ = __getattr__


def _load_old(fullname: str, origin: str) -> types.ModuleType:
    private = PRIVATE + fullname
    if private not in sys.modules:
        spec = importlib.util.spec_from_file_location(private, origin)
        old = importlib.util.module_from_spec(spec)
        # Relative imports inside 0.26's copy resolve against the real package.
        old.__package__ = fullname.rpartition(".")[0]
        sys.modules[private] = old
        spec.loader.exec_module(old)
    return sys.modules[private]


class VendorFinder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self._old_root = None

    def _old_path(self, fullname: str) -> str | None:
        """Where vLLM 0.26 has this module, without importing it."""
        if self._old_root is None:
            spec = importlib.machinery.PathFinder.find_spec("vllm")
            self._old_root = pathlib.Path(spec.origin).parent if spec else pathlib.Path("/nonexistent")
        rel = pathlib.Path(*fullname.split(".")[1:])
        for p in (self._old_root / rel.with_suffix(".py"), self._old_root / rel / "__init__.py"):
            if p.exists():
                return str(p)
        return None

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("vllm."):
            return None
        if fullname in STUBS:
            return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package=True)
        from . import hooks  # noqa: PLC0415 - imported lazily, it does not import vllm

        if fullname in hooks.HOOKS:
            return self._hooked(fullname, path, target, hooks.HOOKS[fullname])
        rel = fullname.split(".")[1:]
        base = VENDOR.joinpath("vllm", *rel)
        override = OVERRIDES.joinpath("vllm", *rel)
        init, single = base / "__init__.py", base.with_suffix(".py")
        if (override / "__init__.py").exists():
            init = override / "__init__.py"
        elif override.with_suffix(".py").exists():
            single = override.with_suffix(".py")
        if init.exists():
            # New packages only (vendor never carries a 0.26 package's __init__).
            locations = [str(base)]
            old = self._old_path(fullname)
            if old and old.endswith("__init__.py"):
                locations.append(str(pathlib.Path(old).parent))
            loader = _ShadowLoader(fullname, str(init), None)
            return importlib.util.spec_from_file_location(
                fullname, init, loader=loader, submodule_search_locations=locations)
        if single.exists():
            loader = _ShadowLoader(fullname, str(single), self._old_path(fullname))
            return importlib.util.spec_from_file_location(fullname, single, loader=loader)
        return None


    def _hooked(self, fullname, path, target, functions):
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                spec.loader = _HookLoader(spec.loader, functions)
                return spec
        return None


class _HookLoader(importlib.abc.Loader):
    def __init__(self, loader, functions):
        self.loader, self.functions = loader, functions

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        for function in self.functions:
            function(module)


def install() -> None:
    if not any(isinstance(f, VendorFinder) for f in sys.meta_path):
        loaded = [m for m in sys.modules if m.startswith("vllm.")
                  and (VENDOR.joinpath("vllm", *m.split(".")[1:]).with_suffix(".py").exists())]
        if loaded:
            raise RuntimeError(f"lumi_dsv41: shadowed modules already imported: {loaded}")
        sys.meta_path.insert(0, VendorFinder())
