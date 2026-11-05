# The two helpers test_fused_inv_rope_fp8_quant.py takes from vLLM 0.26's
# tests/kernels/utils.py, verbatim (the full module pulls in much more).
import torch


def bf16_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Representable-step distance between two bf16 tensors.

    Reinterprets the bf16 bit patterns under the IEEE-754 total ordering so
    that adjacent representable values differ by exactly 1.
    """

    def key(t: torch.Tensor) -> torch.Tensor:
        u = t.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
        return torch.where(u >= 0x8000, 0xFFFF - u, u + 0x8000)

    return (key(a) - key(b)).abs()


def fp8_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Representable-step distance between two 8-bit fp8 tensors.

    Reinterprets the fp8 bytes under a sign-magnitude total ordering so that
    adjacent representable values differ by exactly 1. Inputs must already share
    the same fp8 encoding (e.g. both FP8_STORE_DTYPE).
    """

    def key(t: torch.Tensor) -> torch.Tensor:
        u = t.contiguous().view(torch.uint8).to(torch.int64)
        return torch.where(u >= 0x80, 0xFF - u, u + 0x80)

    return (key(a) - key(b)).abs()
