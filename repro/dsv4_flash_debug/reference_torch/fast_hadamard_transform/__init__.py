"""Plain-PyTorch stand-in for Dao-AILab's fast_hadamard_transform (CUDA only).

hadamard_transform(x, scale) = scale * (x @ H), with H the Sylvester (natural-order)
Hadamard matrix of size x.shape[-1], a power of two, as the CUDA library computes.
"""

import torch


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    n = x.shape[-1]
    assert n & (n - 1) == 0, "size must be a power of two"
    y = x.float().reshape(-1, n)
    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        h *= 2
    return (y * scale).reshape(x.shape).to(x.dtype)
