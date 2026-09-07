"""Single-trial relation/residual decomposition over the frozen EEGNet carrier."""
from __future__ import annotations

import torch
from torch import nn

from eegnet_carrier import EEGNetCarrier


class OffsetEstimator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(64, 32)
        self.act = nn.GELU()
        self.out = nn.Linear(32, 64)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(self.hidden(h)))


def projector() -> nn.Sequential:
    return nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.LayerNorm(32))


class DecompositionEEGNet(nn.Module):
    def __init__(self, channels: int = 62) -> None:
        super().__init__()
        self.carrier = EEGNetCarrier(channels)
        self.offset = OffsetEstimator()
        self.rel_projector = projector()
        self.res_projector = projector()
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(64, 2)

    def decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.carrier(x)
        offset = self.offset(h)
        h_rel = h - offset
        z_rel = self.rel_projector(h_rel)
        z_res = self.res_projector(h)
        return h, offset, h_rel, z_rel, z_res

    def classify_components(self, z_rel: torch.Tensor, z_res: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(torch.cat([z_rel, z_res], dim=1)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h, offset, h_rel, z_rel, z_res = self.decompose(x)
        return self.classify_components(z_rel, z_res), h, offset, h_rel, z_rel, z_res
