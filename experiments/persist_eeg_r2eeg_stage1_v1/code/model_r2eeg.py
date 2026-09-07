"""R2EEG: a cache-level, non-EEGNet prospective relation backbone."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class R2EEG(nn.Module):
    """Specified R2EEG encoder; it contains no BatchNorm or attention layer."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=25, stride=2, padding=12, bias=False),
            nn.GroupNorm(6, 24), nn.GELU(),
        )
        self.dw2 = nn.Conv1d(24, 24, kernel_size=15, dilation=2, padding=14, groups=24, bias=False)
        self.pw2 = nn.Conv1d(24, 32, kernel_size=1, bias=False)
        self.norm2 = nn.GroupNorm(8, 32)
        self.pool2 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.dw3 = nn.Conv1d(32, 32, kernel_size=15, dilation=4, padding=28, groups=32, bias=False)
        self.pw3 = nn.Conv1d(32, 48, kernel_size=1, bias=False)
        self.norm3 = nn.GroupNorm(8, 48)
        self.pool3 = nn.AdaptiveAvgPool1d(8)
        self.mix1 = nn.Linear(96, 48, bias=True)
        self.mix2 = nn.Linear(48, 48, bias=True)
        self.tdw = nn.Conv1d(96, 96, kernel_size=3, padding=1, groups=96, bias=False)
        self.tpw = nn.Conv1d(96, 96, kernel_size=1, bias=False)
        self.tnorm = nn.GroupNorm(12, 96)
        self.zrel = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.LayerNorm(64))
        self.zres = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.LayerNorm(64))
        self.dropout = nn.Dropout(0.25)
        self.head = nn.Linear(128, 2)

    def forward_latents(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, t = x.shape
        if c != self.channels or t != 1000:
            raise RuntimeError(f"R2EEG expected [B,{self.channels},1000], got {tuple(x.shape)}")
        value = x.reshape(b * c, 1, t)
        value = self.block1(value)
        value = self.pool2(F.gelu(self.norm2(self.pw2(self.dw2(value)))))
        value = self.pool3(F.gelu(self.norm3(self.pw3(self.dw3(value)))))
        # [B, C, 48, 8], with the per-bin channel mixer shared across channels.
        h = value.reshape(b, c, 48, 8)
        mean_channel = h.mean(dim=1, keepdim=True)
        q = torch.cat((h, h - mean_channel), dim=2).permute(0, 1, 3, 2)
        update = self.mix2(F.gelu(self.mix1(q))).permute(0, 1, 3, 2)
        u = h + 0.5 * update
        g = torch.cat((u.mean(dim=1), u.std(dim=1, unbiased=False)), dim=1)  # [B,96,8]
        temporal = F.gelu(self.tnorm(self.tpw(self.tdw(g))))
        h96 = (temporal + g).mean(dim=2)
        z_rel = self.zrel(h96)
        z_res = self.zres(h96)
        return z_rel, z_res, torch.cat((z_rel, z_res), dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, z_full = self.forward_latents(x)
        return self.head(self.dropout(z_full)), z_full


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
