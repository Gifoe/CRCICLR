"""Minimal R2EEG-v1 repair that restores fixed-montage spatial identity."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class R2EEGSpatialRepair(nn.Module):
    """Same shared temporal encoder as v1, followed by learned indexed mixing."""
    def __init__(self, channels: int = 62) -> None:
        super().__init__()
        self.channels = int(channels)
        # Temporal stack intentionally matches R2EEG-v1 definition layer-for-layer.
        self.block1 = nn.Sequential(nn.Conv1d(1, 24, 25, stride=2, padding=12, bias=False), nn.GroupNorm(6, 24), nn.GELU())
        self.dw2 = nn.Conv1d(24, 24, 15, dilation=2, padding=14, groups=24, bias=False)
        self.pw2 = nn.Conv1d(24, 32, 1, bias=False); self.norm2 = nn.GroupNorm(8, 32); self.pool2 = nn.AvgPool1d(4, 4)
        self.dw3 = nn.Conv1d(32, 32, 15, dilation=4, padding=28, groups=32, bias=False)
        self.pw3 = nn.Conv1d(32, 48, 1, bias=False); self.norm3 = nn.GroupNorm(8, 48); self.pool3 = nn.AdaptiveAvgPool1d(8)
        # 48 independent feature groups, two learned filters per feature across electrode index.
        self.spatial = nn.Conv2d(48, 96, kernel_size=(self.channels, 1), groups=48, bias=False)
        self.spatial_norm = nn.GroupNorm(12, 96)
        self.tdw = nn.Conv1d(96, 96, 3, padding=1, groups=96, bias=False)
        self.tpw = nn.Conv1d(96, 96, 1, bias=False); self.tnorm = nn.GroupNorm(12, 96); self.carrier_norm = nn.LayerNorm(96)
        self.zrel = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.LayerNorm(64))
        self.zres = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.LayerNorm(64))
        self.dropout = nn.Dropout(0.25); self.head = nn.Linear(128, 2)

    def temporal_features(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        if (c, t) != (self.channels, 1000): raise RuntimeError(f"expected [B,{self.channels},1000], received {tuple(x.shape)}")
        v = self.block1(x.reshape(b*c, 1, t))
        v = self.pool2(F.gelu(self.norm2(self.pw2(self.dw2(v)))))
        v = self.pool3(F.gelu(self.norm3(self.pw3(self.dw3(v)))))
        return v.reshape(b, c, 48, 8)

    def forward_latents(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.temporal_features(x)
        g = F.gelu(self.spatial_norm(self.spatial(h.permute(0, 2, 1, 3)).squeeze(2)))
        t = F.gelu(self.tnorm(self.tpw(self.tdw(g))))
        h96 = self.carrier_norm((g + t).mean(dim=2))
        zrel, zres = self.zrel(h96), self.zres(h96)
        return zrel, zres, torch.cat((zrel, zres), dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, z = self.forward_latents(x)
        return self.head(self.dropout(z)), z


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
