"""Exact W0R0 candidate for the Linux-only seed-0 screen."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class W0R0(nn.Module):
    """LiteBN W0 width with C/S/M and the ordered eight-bin R0 readout."""

    def __init__(self, channels: int, classes: int, stem_cls, mixer_cls):
        super().__init__()
        self.stem = stem_cls(channels, 8, 16, 64)

        self.channel_mlp = nn.Sequential(
            nn.Linear(2, 8), nn.GELU(), nn.Linear(8, 1)
        )
        self.lambda_channel = nn.Parameter(torch.zeros((), dtype=torch.float32))

        self.scale_mlp = nn.Sequential(
            nn.Linear(48, 24), nn.GELU(), nn.Linear(24, 3)
        )
        self.lambda_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))

        self.mixers = nn.ModuleList(
            [mixer_cls(64, dilation) for dilation in (1, 2, 4)]
        )
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.embedding = nn.Sequential(
            nn.Linear(512, 64),
            nn.ELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.25),
        )
        self.head = nn.Linear(64, classes)

    def _channel_gate(self, value: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(value.square().mean(dim=-1) + 1e-8)
        diff = value[..., 1:] - value[..., :-1]
        diff_rms = torch.sqrt(diff.square().mean(dim=-1) + 1e-8)
        raw = self.channel_mlp(torch.stack((rms, diff_rms), dim=-1)).squeeze(-1)
        factor = 1.0 + torch.tanh(self.lambda_channel) * torch.tanh(raw)
        return value * factor.unsqueeze(-1)

    def _scale_gate(self, branches: list[torch.Tensor]) -> list[torch.Tensor]:
        descriptor = torch.cat(
            [branch.mean(dim=(2, 3)) for branch in branches], dim=1
        )
        alpha = F.softmax(self.scale_mlp(descriptor), dim=1)
        amplitude = torch.tanh(self.lambda_scale)
        return [
            branch
            * (1.0 + amplitude * (3.0 * alpha[:, index] - 1.0)).view(
                -1, 1, 1, 1
            )
            for index, branch in enumerate(branches)
        ]

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self._channel_gate(value)
        branches = self._scale_gate(self.stem.branch_outputs(value))
        value = self.stem.temporal_blocks(branches).squeeze(2)
        for mixer in self.mixers:
            value = mixer(value)
        representation = self.embedding(self.pool(value).flatten(1))
        return self.head(representation), representation

