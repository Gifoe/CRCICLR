from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardEEGNet(nn.Module):
    def __init__(self, f1: int = 8, f2: int = 16, dropout: float = 0.25):
        super().__init__()
        self.temporal = nn.Conv2d(1, f1, (1, 64), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.spatial = nn.Conv2d(f1, 2 * f1, (62, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(2 * f1)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.depth = nn.Conv2d(2 * f1, 2 * f1, (1, 16), padding="same", groups=2 * f1, bias=False)
        self.point = nn.Conv2d(2 * f1, f2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        self.embedding = nn.Sequential(nn.Linear(f2 * 31, 64), nn.ELU(), nn.LayerNorm(64))
        self.head = nn.Linear(64, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        value = x.unsqueeze(1)
        value = self.bn1(self.temporal(value))
        value = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(value)))))
        value = self.drop2(self.pool2(F.elu(self.bn3(self.point(self.depth(value))))))
        return self.embedding(value.flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class ShallowConvNet(nn.Module):
    def __init__(self, filters: int = 40, dropout: float = 0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, filters, (1, 25), bias=False)
        self.spatial = nn.Conv2d(filters, filters, (62, 1), groups=1, bias=False)
        self.bn = nn.BatchNorm2d(filters)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Sequential(nn.Linear(filters * 61, 64), nn.ELU(), nn.LayerNorm(64))
        self.head = nn.Linear(64, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        value = self.bn(self.spatial(self.temporal(x.unsqueeze(1))))
        value = torch.square(value)
        value = torch.log(torch.clamp(self.pool(value), min=1e-6))
        return self.embedding(self.dropout(value).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build(configuration: dict) -> nn.Module:
    if configuration["architecture"] == "eegnet":
        return StandardEEGNet(
            int(configuration["f1"]),
            int(configuration["f2"]),
            float(configuration["dropout"]),
        )
    if configuration["architecture"] == "shallow":
        return ShallowConvNet(int(configuration["filters"]), float(configuration["dropout"]))
    raise ValueError(configuration)
