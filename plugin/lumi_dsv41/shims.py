"""Engine-side registrations that need vLLM importable (run from register()).

Missing names in 0.26's engine modules are added by the loader's post-import hooks
(hooks.py, graft.py) as those modules load; this only registers behaviour with 0.26's
registries.
"""

from . import loader

_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    loader.install()
    from . import circular_buffer

    circular_buffer.register()
    _installed = True
