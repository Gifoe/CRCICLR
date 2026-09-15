"""Architecturally isolated LiteBN ablations."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def normalizer(kind: str, channels: int) -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    groups = 4 if channels in (8, 16) else 8
    return nn.GroupNorm(groups, channels)


class LiteBNAblation(nn.Module):
    VARIANTS = (
        "B1_SINGLE_SCALE_TEMPORAL",
        "B2_SPATIAL_D1",
        "B3_ONE_STAGE_BACKEND",
        "B4_BN_TO_GN",
    )

    def __init__(self, channels: int, classes: int, variant: str):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        kernels = (63,) if variant == "B1_SINGLE_SCALE_TEMPORAL" else (15, 63, 127)
        spatial_channels = 8 if variant == "B2_SPATIAL_D1" else 16
        kind = "gn" if variant == "B4_BN_TO_GN" else "bn"
        self.temporal = nn.ModuleList([
            nn.Conv2d(1, 8, (1, kernel), padding="same", bias=False) for kernel in kernels
        ])
        self.temporal_norm = nn.ModuleList([normalizer(kind, 8) for _ in kernels])
        self.spatial = nn.ModuleList([
            nn.Conv2d(8, spatial_channels, (channels, 1), groups=8, bias=False) for _ in kernels
        ])
        self.spatial_norm = nn.ModuleList([normalizer(kind, spatial_channels) for _ in kernels])
        concatenated = len(kernels) * spatial_channels
        self.depth1 = nn.Conv2d(concatenated, concatenated, (1, 15), groups=concatenated, padding="same", bias=False)
        self.point1 = nn.Conv2d(concatenated, 64, 1, bias=False)
        self.norm1 = normalizer(kind, 64)
        if variant == "B3_ONE_STAGE_BACKEND":
            self.depth2 = None
            self.point2 = None
            self.norm2 = None
        else:
            self.depth2 = nn.Conv2d(64, 64, (1, 31), groups=64, padding="same", bias=False)
            self.point2 = nn.Conv2d(64, 64, 1, bias=False)
            self.norm2 = normalizer(kind, 64)
        self.pool = nn.AdaptiveAvgPool2d((1, 8))
        self.embedding = nn.Sequential(nn.Linear(512, 64), nn.ELU(), nn.LayerNorm(64))
        self.drop = nn.Dropout(.25)
        self.head = nn.Linear(64, classes)

    def forward(self, x):
        value = x.unsqueeze(1)
        branches = []
        for temporal, temporal_norm, spatial, spatial_norm in zip(
            self.temporal, self.temporal_norm, self.spatial, self.spatial_norm
        ):
            branch = F.elu(temporal_norm(temporal(value)))
            branch = F.elu(spatial_norm(spatial(branch)))
            branch = F.avg_pool2d(branch, (1, 4))
            branches.append(F.dropout(branch, .20, self.training))
        value = torch.cat(branches, dim=1)
        value = F.elu(self.norm1(self.point1(self.depth1(value))))
        value = F.dropout(F.avg_pool2d(value, (1, 2)), .15, self.training)
        if self.depth2 is not None:
            value = F.elu(self.norm2(self.point2(self.depth2(value))))
            value = F.dropout(F.avg_pool2d(value, (1, 2)), .15, self.training)
        representation = self.drop(self.embedding(self.pool(value).flatten(1)))
        return self.head(representation), representation
