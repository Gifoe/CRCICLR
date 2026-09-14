"""Fixed standard temporal RoPE for full 16-dimensional attention heads."""
from __future__ import annotations

import torch


def apply_rope(value: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Apply full-dimensional RoPE to [B, H, L, D], where D is even."""
    if value.ndim != 4 or value.shape[-1] % 2:
        raise ValueError(f"RoPE requires [B,H,L,even-D], got {tuple(value.shape)}")
    _, _, length, dimension = value.shape
    half = dimension // 2
    positions = torch.arange(length, dtype=value.dtype, device=value.device)
    inverse = base ** (-torch.arange(half, dtype=value.dtype, device=value.device) / half)
    angle = positions[:, None] * inverse[None, :]
    cosine, sine = angle.cos()[None, None], angle.sin()[None, None]
    even, odd = value[..., 0::2], value[..., 1::2]
    output = torch.empty_like(value)
    output[..., 0::2] = even * cosine - odd * sine
    output[..., 1::2] = even * sine + odd * cosine
    return output
