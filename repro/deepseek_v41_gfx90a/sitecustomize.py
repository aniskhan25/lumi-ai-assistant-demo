"""Runs at every interpreter start while this directory is on PYTHONPATH.

Installs lumi_dsv41's module loader before anything imports vllm, so every process,
including spawned ones that unpickle the DeepSeek-V4.1 config before plugins load,
resolves the vendored modules. Shadows the container's sitecustomize (Ubuntu's apport
hook) and chains to it.
"""

import os
import runpy

from lumi_dsv41 import loader

loader.install()

_SYSTEM = "/usr/lib/python3.12/sitecustomize.py"
if os.path.exists(_SYSTEM):
    runpy.run_path(_SYSTEM)
