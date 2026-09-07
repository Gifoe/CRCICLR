"""Canonical EEGNet feature carrier, without its original classifier head."""
from __future__ import annotations

import torch
from torch import nn


class EEGNetCarrier(nn.Module):
    """Exact canonical EEGNet through the 64-D LayerNorm embedding."""

    def __init__(self, channels: int = 62) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(1, 8, (1, 64), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.spatial = nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.25)
        self.depth = nn.Conv2d(16, 16, (1, 16), padding="same", groups=16, bias=False)
        self.point = nn.Conv2d(16, 16, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.25)
        self.embedding = nn.Sequential(nn.Linear(496, 64), nn.ELU(), nn.LayerNorm(64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.bn1(self.temporal(x))
        x = self.drop1(self.pool1(torch.nn.functional.elu(self.bn2(self.spatial(x)))))
        x = self.drop2(self.pool2(torch.nn.functional.elu(self.bn3(self.point(self.depth(x))))))
        return self.embedding(x.flatten(1))


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
