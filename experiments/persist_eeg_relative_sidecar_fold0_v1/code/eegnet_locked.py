"""Exact canonical EEGNet used as a hash-locked, fully frozen base predictor."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class EEGNet(nn.Module):
    def __init__(self, channels: int = 62, samples: int = 1000) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(1, 8, (1, 64), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.spatial = nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(.25)
        self.depth = nn.Conv2d(16, 16, (1, 16), padding="same", groups=16, bias=False)
        self.point = nn.Conv2d(16, 16, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(.25)
        self.embedding = nn.Sequential(nn.Linear(16 * (samples // 4 // 8), 64), nn.ELU(), nn.LayerNorm(64))
        self.head = nn.Linear(64, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        v = x.unsqueeze(1)
        v = self.bn1(self.temporal(v))
        v = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(v)))))
        v = self.drop2(self.pool2(F.elu(self.bn3(self.point(self.depth(v))))))
        return self.embedding(v.flatten(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.forward_features(x)
        return self.head(h), h


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
