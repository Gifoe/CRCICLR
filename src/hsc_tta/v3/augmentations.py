from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class NuisanceConfig:
    amplitude_scale: float = 1.02
    gaussian_std: float = 0.005
    time_shift_patches: int = 1
    frequency_mask_fraction: float = 0.05
    seed: int = 3407


class TokenNuisanceAugmenter:
    """Deterministic mild nuisances on channel/patch CBraMod representations.

    Scaling/noise approximate amplitude and sensor noise; patch rolling preserves chronology locally;
    a contiguous feature-band mask is a conservative representation-space proxy for frequency masking.
    """

    names = ("amplitude_scaling", "gaussian_noise", "time_shift", "frequency_mask")

    def __init__(self, config: NuisanceConfig = NuisanceConfig()):
        self.config = config

    def apply(self, tokens: torch.Tensor, name: str, *, seed_offset: int = 0) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError("tokens must be [sample, channel, patch, feature]")
        x = tokens.clone(); c = self.config
        if name == "amplitude_scaling":
            return x * c.amplitude_scale
        if name == "gaussian_noise":
            generator = torch.Generator(device=x.device).manual_seed(c.seed + seed_offset)
            return x + torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) * c.gaussian_std
        if name == "time_shift":
            return torch.roll(x, shifts=c.time_shift_patches, dims=2)
        if name == "frequency_mask":
            width = max(1, int(round(x.shape[-1] * c.frequency_mask_fraction)))
            start = int(np.random.default_rng(c.seed + seed_offset).integers(0, x.shape[-1] - width + 1))
            x[..., start:start+width] = 0
            return x
        raise ValueError(f"unknown nuisance: {name}")

    def all(self, tokens: torch.Tensor, *, seed_offset: int = 0) -> dict[str, torch.Tensor]:
        return {name: self.apply(tokens, name, seed_offset=seed_offset) for name in self.names}
