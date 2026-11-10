"""Bring-up step 1: the plugin loads and the V4.1 model class imports on vLLM 0.26.

    python check_import.py      # inside the container, plugin dir on PYTHONPATH
Prints the first failure with its traceback; exit code 0 when the class resolves.
"""

import importlib
import sys
import traceback


def main() -> int:
    import lumi_dsv41
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    from vllm import ModelRegistry

    assert lumi_dsv41.ARCH in ModelRegistry.get_supported_archs(), "plugin did not register"
    module_name, cls_name = lumi_dsv41.MODEL.split(":")
    try:
        cls = getattr(importlib.import_module(module_name), cls_name)
    except Exception:
        traceback.print_exc()
        return 1
    print(f"OK: {cls.__module__}.{cls.__qualname__}")
    from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

    print(f"OK: {DeepseekV41Config.model_type}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
