"""Scientifically isolated LiteBN v2 architecture variants.

The full model is intentionally not implemented here: the user explicitly
requested that only B1--B4 be trained. The official frozen LiteBN is used only
as a non-matched reference during evaluation.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LiteBNAblationV2(nn.Module):
    VARIANTS = (
        "B1_SAME_SCALE_63",
        "B2_SCALE_COLLAPSE",
        "B3_SINGLE_SPATIAL_BASIS",
        "B4_ONE_STAGE_BACKEND",
    )

    def __init__(self, channels: int, classes: int, variant: str):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        kernels = (63, 63, 63) if variant == "B1_SAME_SCALE_63" else (15, 63, 127)
        self.temporal = nn.ModuleList([
            nn.Conv2d(1, 8, (1, kernel), padding="same", bias=False)
            for kernel in kernels
        ])
        self.temporal_norm = nn.ModuleList([nn.BatchNorm2d(8) for _ in kernels])

        if variant == "B3_SINGLE_SPATIAL_BASIS":
            self.spatial_basis = nn.ModuleList([
                nn.Conv2d(8, 8, (channels, 1), groups=8, bias=False)
                for _ in kernels
            ])
            self.spatial_expand = nn.ModuleList([
                nn.Conv2d(8, 16, 1, bias=False) for _ in kernels
            ])
            self.spatial = None
        else:
            self.spatial = nn.ModuleList([
                nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False)
                for _ in kernels
            ])
            self.spatial_basis = None
            self.spatial_expand = None
        self.spatial_norm = nn.ModuleList([nn.BatchNorm2d(16) for _ in kernels])

        self.depth1 = nn.Conv2d(48, 48, (1, 15), groups=48, padding="same", bias=False)
        self.point1 = nn.Conv2d(48, 64, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(64)
        if variant == "B4_ONE_STAGE_BACKEND":
            self.depth2 = None
            self.point2 = None
            self.norm2 = None
        else:
            self.depth2 = nn.Conv2d(64, 64, (1, 31), groups=64, padding="same", bias=False)
            self.point2 = nn.Conv2d(64, 64, 1, bias=False)
            self.norm2 = nn.BatchNorm2d(64)

        self.pool = nn.AdaptiveAvgPool2d((1, 8))
        self.embedding = nn.Sequential(nn.Linear(512, 64), nn.ELU(), nn.LayerNorm(64))
        self.drop = nn.Dropout(.25)
        self.head = nn.Linear(64, classes)

    def forward(self, x):
        value = x.unsqueeze(1)
        branches = []
        for index, (temporal, temporal_norm, spatial_norm) in enumerate(zip(
            self.temporal, self.temporal_norm, self.spatial_norm
        )):
            branch = F.elu(temporal_norm(temporal(value)))
            if self.variant == "B3_SINGLE_SPATIAL_BASIS":
                # No normalization or activation between basis and expansion.
                branch = self.spatial_expand[index](self.spatial_basis[index](branch))
            else:
                branch = self.spatial[index](branch)
            branch = F.elu(spatial_norm(branch))
            branch = F.avg_pool2d(branch, (1, 4))
            branches.append(F.dropout(branch, .20, self.training))

        if self.variant == "B2_SCALE_COLLAPSE":
            z_bar = torch.stack(branches, dim=0).mean(dim=0)
            value = torch.cat((z_bar, z_bar, z_bar), dim=1)
        else:
            value = torch.cat(branches, dim=1)
        if value.shape[1] != 48:
            raise RuntimeError(f"backend input width must be 48, got {value.shape[1]}")

        value = F.elu(self.norm1(self.point1(self.depth1(value))))
        value = F.dropout(F.avg_pool2d(value, (1, 2)), .15, self.training)
        if self.depth2 is not None:
            value = F.elu(self.norm2(self.point2(self.depth2(value))))
            value = F.dropout(F.avg_pool2d(value, (1, 2)), .15, self.training)
        representation = self.drop(self.embedding(self.pool(value).flatten(1)))
        return self.head(representation), representation
