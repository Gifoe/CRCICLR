"""Exact historical LiteBN with identity-initialized C/M residuals."""
from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalResidualMixer(nn.Module):
    def __init__(self, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            64, 64, kernel_size=9, padding=4 * dilation,
            dilation=dilation, groups=64, bias=True,
        )
        self.norm = nn.LayerNorm(64)
        self.expand = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.10)
        self.contract = nn.Linear(128, 64)
        self.gamma = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.depthwise(value).transpose(1, 2)
        update = self.norm(update)
        update = self.contract(self.dropout(F.gelu(self.expand(update))))
        return value + self.gamma * update.transpose(1, 2)


class LiteBNCMGA(nn.Module):
    """Frozen exact LiteBN plus trainable C/M residual parameters only."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        self.channel_mlp = nn.Sequential(
            nn.Linear(2, 8), nn.GELU(), nn.Linear(8, 1)
        )
        self.lambda_channel = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.mixers = nn.ModuleList(
            [TemporalResidualMixer(dilation) for dilation in (1, 2, 4)]
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def residual_parameters(self) -> list[nn.Parameter]:
        return [parameter for name, parameter in self.named_parameters()
                if not name.startswith("base.") and parameter.requires_grad]

    def _channel_residual(self, value: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(value.square().mean(dim=-1) + 1e-8)
        difference = value[..., 1:] - value[..., :-1]
        diff_rms = torch.sqrt(difference.square().mean(dim=-1) + 1e-8)
        raw = self.channel_mlp(
            torch.stack((rms, diff_rms), dim=-1)
        ).squeeze(-1)
        factor = 1.0 + torch.tanh(self.lambda_channel) * torch.tanh(raw)
        return value * factor.unsqueeze(-1)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self._channel_residual(value).unsqueeze(1)
        branches = []
        for temporal, temporal_norm, spatial, spatial_norm in zip(
            self.base.temporal,
            self.base.temporal_norm,
            self.base.spatial,
            self.base.spatial_norm,
        ):
            branch = F.elu(temporal_norm(temporal(value)))
            branch = F.elu(spatial_norm(spatial(branch)))
            branch = F.avg_pool2d(branch, (1, 4))
            branches.append(F.dropout(branch, 0.20, self.base.training))
        value = torch.cat(branches, dim=1)
        value = F.dropout(
            F.avg_pool2d(
                F.elu(self.base.norm1(self.base.point1(self.base.depth1(value)))),
                (1, 2),
            ),
            0.15,
            self.base.training,
        )
        value = F.dropout(
            F.avg_pool2d(
                F.elu(self.base.norm2(self.base.point2(self.base.depth2(value)))),
                (1, 2),
            ),
            0.15,
            self.base.training,
        ).squeeze(2)
        for mixer in self.mixers:
            value = mixer(value)
        representation = self.base.drop(
            self.base.embedding(self.base.pool(value.unsqueeze(2)).flatten(1))
        )
        return self.base.head(representation), representation


def tensor_mapping_sha256(mapping: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(mapping.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def frozen_base_sha256(model: LiteBNCMGA) -> str:
    return tensor_mapping_sha256(model.base.state_dict())

