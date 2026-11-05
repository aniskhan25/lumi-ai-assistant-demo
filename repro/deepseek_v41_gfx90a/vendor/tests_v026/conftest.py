# Not upstream: the one fixture these tests take from vLLM's own tests/conftest.py.
import pytest


@pytest.fixture
def default_vllm_config():
    from vllm.config import VllmConfig, set_current_vllm_config

    config = VllmConfig()
    with set_current_vllm_config(config):
        yield config

# `from tests.kernels.utils import ...` resolves to ../tests_helpers.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests_helpers"))
