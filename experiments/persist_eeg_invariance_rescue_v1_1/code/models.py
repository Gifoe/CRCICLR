from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import firwin


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor):
        return -ctx.strength * gradient, None


@dataclass
class ModelOutput:
    logits: torch.Tensor
    features: torch.Tensor
    projection: torch.Tensor | None = None
    domain_logits: torch.Tensor | None = None
    router_weights: torch.Tensor | None = None
    expert_features: torch.Tensor | None = None


class ControlledEEGNet(nn.Module):
    """EEGNet with a capacity-matched subject head for A0/A1."""

    def __init__(self, n_subjects: int, embedding_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        f1, f2 = 8, 16
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
        self.embedding = nn.Sequential(
            nn.Linear(f2 * 31, embedding_dim), nn.ELU(), nn.LayerNorm(embedding_dim)
        )
        self.head = nn.Linear(embedding_dim, 2)
        self.subject_head = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ELU(), nn.Dropout(0.1), nn.Linear(64, n_subjects)
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        value = x.unsqueeze(1)
        value = self.bn1(self.temporal(value))
        value = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(value)))))
        value = self.drop2(self.pool2(F.elu(self.bn3(self.point(self.depth(value))))))
        return self.embedding(value.flatten(1))

    def forward(self, x: torch.Tensor, grl_strength: float = 0.0) -> ModelOutput:
        features = self.forward_features(x)
        subject_input = GradientReversal.apply(features, grl_strength) if grl_strength else features
        return ModelOutput(
            logits=self.head(features),
            features=features,
            domain_logits=self.subject_head(subject_input),
        )


class SamePadTemporal(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, groups: int = 1):
        super().__init__()
        left = (kernel - 1) // 2
        right = kernel - 1 - left
        self.pad = nn.ZeroPad2d((left, right, 0, 0))
        self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel), groups=groups, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class EEGDGNet(nn.Module):
    """Clean-room multi-scale, source-routed EEG-DG instantiation."""

    def __init__(self, n_subjects: int, embedding_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.temporal = nn.ModuleList([SamePadTemporal(1, 4, kernel) for kernel in (8, 16, 32, 64)])
        self.temporal_bn = nn.BatchNorm2d(16)
        self.spatial = nn.Conv2d(16, 32, (62, 1), groups=16, bias=False)
        self.spatial_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.separable = nn.ModuleList([SamePadTemporal(32, 32, kernel, groups=32) for kernel in (2, 4, 8, 16)])
        self.sep_point = nn.ModuleList([nn.Conv2d(32, 32, 1, bias=False) for _ in range(4)])
        self.sep_bn = nn.BatchNorm2d(128)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        self.shared = nn.Sequential(
            nn.Linear(128 * 31, embedding_dim), nn.ReLU(), nn.LayerNorm(embedding_dim)
        )
        self.router = nn.Linear(embedding_dim, n_subjects)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.ReLU()) for _ in range(n_subjects)]
        )
        self.head = nn.Linear(embedding_dim, 2)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        value = x.unsqueeze(1)
        value = torch.cat([branch(value) for branch in self.temporal], dim=1)
        value = self.temporal_bn(value)
        value = self.drop1(self.pool1(F.relu(self.spatial_bn(self.spatial(value)))))
        branches = [point(depth(value)) for point, depth in zip(self.sep_point, self.separable)]
        value = torch.cat(branches, dim=1)
        value = self.drop2(self.pool2(F.relu(self.sep_bn(value))))
        shared = self.shared(value.flatten(1))
        domain_logits = self.router(shared)
        weights = F.softmax(domain_logits, dim=1)
        expert_stack = torch.stack([expert(shared) for expert in self.experts], dim=1)
        weighted = torch.sum(weights.unsqueeze(-1) * expert_stack, dim=1)
        return weighted, domain_logits, weights, expert_stack

    def forward(self, x: torch.Tensor, grl_strength: float = 0.0) -> ModelOutput:
        del grl_strength
        features, domain_logits, weights, expert_stack = self.forward_features(x)
        return ModelOutput(
            logits=self.head(features),
            features=features,
            domain_logits=domain_logits,
            router_weights=weights,
            expert_features=expert_stack,
        )


class FixedFIRFilterBank(nn.Module):
    def __init__(self, bands: Sequence[Sequence[float]], sampling_rate: float = 250.0):
        super().__init__()
        taps = [
            firwin(65, [float(low), float(high)], pass_zero=False, fs=sampling_rate, window="hann")
            for low, high in bands
        ]
        self.register_buffer("filters", torch.tensor(np.asarray(taps)[:, None, :], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, times = x.shape
        value = x.reshape(batch * channels, 1, times)
        value = F.conv1d(value, self.filters, stride=4, padding=32)
        value = value.reshape(batch, channels, self.filters.shape[0], value.shape[-1])
        return value.permute(0, 2, 1, 3).contiguous()


class MixedDepthwise(nn.Module):
    def __init__(self, channels: int, kernels: Sequence[int]):
        super().__init__()
        base = channels // len(kernels)
        sizes = [base] * len(kernels)
        for index in range(channels - sum(sizes)):
            sizes[index] += 1
        self.sizes = sizes
        self.branches = nn.ModuleList(
            [SamePadTemporal(size, size, kernel, groups=size) for size, kernel in zip(sizes, kernels)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(x, self.sizes, dim=1)
        return torch.cat([branch(chunk) for branch, chunk in zip(self.branches, chunks)], dim=1)


class SCLDGNNet(nn.Module):
    """Clean-room SCLDGN with a fixed 4--40 Hz nine-band front end."""

    def __init__(
        self,
        n_subjects: int,
        bands: Sequence[Sequence[float]],
        embedding_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        del n_subjects
        self.filterbank = FixedFIRFilterBank(bands)
        self.mix1 = MixedDepthwise(9, (15, 31, 63, 125))
        self.reduce1 = nn.Conv2d(9, 32, (1, 2), stride=(1, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.mix2 = MixedDepthwise(32, (15, 31, 63, 125))
        self.reduce2 = nn.Conv2d(32, 32, (1, 2), stride=(1, 2), bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.spatial = nn.Conv2d(32, 128, (62, 1), groups=32, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AvgPool2d((1, 10), stride=(1, 10))
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Sequential(
            nn.Linear(128 * 6, embedding_dim), nn.ReLU(), nn.LayerNorm(embedding_dim)
        )
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.ReLU(), nn.Linear(128, 64)
        )
        self.head = nn.Linear(embedding_dim, 2)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.filterbank(x)
        value = F.relu(self.bn1(self.reduce1(self.mix1(value))))
        value = F.relu(self.bn2(self.reduce2(self.mix2(value))))
        value = self.dropout(self.pool(F.relu(self.bn3(self.spatial(value)))))
        features = self.embedding(value.flatten(1))
        projection = F.normalize(self.projection(features), dim=1)
        return features, projection

    def forward(self, x: torch.Tensor, grl_strength: float = 0.0) -> ModelOutput:
        del grl_strength
        features, projection = self.forward_features(x)
        return ModelOutput(logits=self.head(features), features=features, projection=projection)


def method_family(method_id: str) -> str:
    if method_id.startswith("A"):
        return "A"
    if method_id.startswith("B"):
        return "B"
    if method_id.startswith("C"):
        return "C"
    raise ValueError(method_id)


def build_model(method_id: str, n_subjects: int, config: Mapping[str, Any]) -> nn.Module:
    family = method_family(method_id)
    if family == "A":
        return ControlledEEGNet(n_subjects=n_subjects, embedding_dim=int(config["embedding_dim"]))
    if family == "B":
        return EEGDGNet(n_subjects=n_subjects, embedding_dim=int(config["embedding_dim"]))
    return SCLDGNNet(
        n_subjects=n_subjects,
        bands=config["scldgn_filter_bands_hz"],
        embedding_dim=int(config["embedding_dim"]),
    )


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def roster(config: Mapping[str, Any]) -> list[str]:
    methods = ["A0_TASK_ONLY_EEGNET"]
    for value in config["grl_candidate_grid"]:
        methods.append(f"A1_SUBJECT_GRL_EEGNET_L{int(round(float(value) * 1000)):04d}")
    methods.extend([
        "B0_EEG_DG_TASK_ONLY",
        "B1_EEG_DG_FULL",
        "C0_SCLDGN_TASK_ONLY",
        "C1_SCLDGN_FULL",
    ])
    return methods


def primary_pairs(config: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    # V1.1 selects GRL strength separately per development fold by train-only
    # inner CV.  This compatibility helper uses the first declared candidate;
    # scientific code reads HYPERPARAM_SELECTION.csv for the actual pair.
    primary = int(round(float(config["grl_candidate_grid"][0]) * 1000))
    return {
        "A_SUBJECT_GRL_EEGNET": ("A0_TASK_ONLY_EEGNET", f"A1_SUBJECT_GRL_EEGNET_L{primary:04d}"),
        "B_EEG_DG": ("B0_EEG_DG_TASK_ONLY", "B1_EEG_DG_FULL"),
        "C_SCLDGN": ("C0_SCLDGN_TASK_ONLY", "C1_SCLDGN_FULL"),
    }


def grl_lambda(method_id: str) -> float:
    if "_L" not in method_id:
        return 0.0
    return int(method_id.rsplit("_L", 1)[1]) / 1000.0
