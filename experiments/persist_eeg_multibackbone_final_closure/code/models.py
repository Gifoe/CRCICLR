"""Independently implemented B1--B4 models for the prospective closure.

Every model exposes a native penultimate representation through
``forward_features`` and a single final linear classifier in ``head``.  No
hidden layer can be selected after the PERSIST audit.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_CHANNELS = 58
INPUT_SAMPLES = 1000
N_CLASSES = 2


class FBCNet(nn.Module):
    """FBCNet-family fixed filter-bank + learned spatial + log-variance model.

    The public FBCNet formulation consumes nine pre-filtered 4-Hz bands,
    learns band-specific spatial filters, and applies segmented log-variance.
    Here the same filter bank is an architecture-internal deterministic FFT
    mask so the frozen WBCIC preprocessing/cache is not changed.
    """

    def __init__(self, spatial_filters: int = 8, windows: int = 4, dropout: float = 0.25):
        super().__init__()
        self.bands = tuple((low, low + 4) for low in range(4, 40, 4))
        self.spatial_filters = int(spatial_filters)
        self.windows = int(windows)
        frequencies = torch.fft.rfftfreq(INPUT_SAMPLES, d=1.0 / 250.0)
        masks = [((frequencies >= low) & (frequencies < high)).float() for low, high in self.bands]
        self.register_buffer("frequency_masks", torch.stack(masks), persistent=True)
        self.spatial_weight = nn.Parameter(
            torch.empty(len(self.bands), self.spatial_filters, INPUT_CHANNELS)
        )
        nn.init.xavier_uniform_(self.spatial_weight)
        self.bn = nn.BatchNorm1d(len(self.bands) * self.spatial_filters)
        self.dropout = nn.Dropout(float(dropout))
        self.representation_dim = len(self.bands) * self.spatial_filters * self.windows
        self.head = nn.Linear(self.representation_dim, N_CLASSES)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        _check_input(x, "FBCNet")
        # FFT remains float32 under mixed precision; gradients are needed only
        # for the learned spatial weights, not for input data.
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft(x.float(), dim=-1)
            weight = self.spatial_weight.float()
            weight = weight / weight.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            # The nine supports are disjoint and occupy only 144 of 501 FFT
            # bins. Contract each support separately instead of multiplying
            # all band weights across all frequencies and masking afterwards;
            # this is algebraically identical and about an order faster.
            spatial = spectrum.new_zeros(
                (len(x), len(self.bands), self.spatial_filters, spectrum.shape[-1])
            )
            for band in range(len(self.bands)):
                support = self.frequency_masks[band].bool()
                spatial[:, band, :, support] = torch.einsum(
                    "bcf,mc->bmf",
                    spectrum[:, :, support],
                    weight[band].to(spectrum.dtype),
                )
            signal = torch.fft.irfft(spatial, n=INPUT_SAMPLES, dim=-1)
        value = signal.reshape(len(x), -1, INPUT_SAMPLES)
        value = self.dropout(value * torch.sigmoid(self.bn(value)))
        value = value.reshape(len(x), value.shape[1], self.windows, INPUT_SAMPLES // self.windows)
        feature = torch.log(value.var(dim=-1, unbiased=False).clamp(1e-6, 1e6))
        return feature.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class EEGConformer(nn.Module):
    """Convolutional patch tokenizer followed by a Transformer encoder."""

    def __init__(
        self,
        d_model: int = 64,
        depth: int = 3,
        heads: int = 4,
        dropout: float = 0.25,
        temporal_kernel: int = 25,
    ):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(1, 40, (1, int(temporal_kernel)), bias=False),
            nn.Conv2d(40, 40, (INPUT_CHANNELS, 1), bias=False),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), stride=(1, 15)),
            nn.Dropout(float(dropout)),
            nn.Conv2d(40, int(d_model), (1, 1), bias=False),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(heads),
            dim_feedforward=4 * int(d_model),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(depth), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(int(d_model))
        self.representation_dim = int(d_model)
        self.head = nn.Linear(self.representation_dim, N_CLASSES)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        _check_input(x, "EEGConformer")
        token = self.patch(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        return self.norm(self.encoder(token).mean(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class DeepConvNet(nn.Module):
    """DeepConvNet-family temporal-spatial CNN with a native pooled readout."""

    def __init__(self, base_filters: int = 25, dropout: float = 0.5):
        super().__init__()
        f = int(base_filters)
        self.first_temporal = nn.Conv2d(1, f, (1, 10), bias=False)
        self.first_spatial = nn.Conv2d(f, f, (INPUT_CHANNELS, 1), bias=False)
        self.first_bn = nn.BatchNorm2d(f)
        self.blocks = nn.ModuleList()
        current = f
        for output in (2 * f, 4 * f, 8 * f):
            self.blocks.append(
                nn.Sequential(
                    nn.Dropout(float(dropout)),
                    nn.Conv2d(current, output, (1, 10), bias=False),
                    nn.BatchNorm2d(output),
                    nn.ELU(),
                    nn.MaxPool2d((1, 3), stride=(1, 3)),
                )
            )
            current = output
        self.first_pool = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.final_norm = nn.LayerNorm(current)
        self.representation_dim = current
        self.head = nn.Linear(self.representation_dim, N_CLASSES)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        _check_input(x, "DeepConvNet")
        value = self.first_temporal(x.unsqueeze(1))
        value = self.first_pool(F.elu(self.first_bn(self.first_spatial(value))))
        for block in self.blocks:
            value = block(value)
        return self.final_norm(value.mean(dim=(-2, -1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class CoTARBlock(nn.Module):
    """Centralized token aggregation and redistribution block."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        core_dim = max(16, int(d_model) // 4)
        self.local = nn.Linear(d_model, d_model)
        self.core = nn.Linear(d_model, core_dim)
        self.merge = nn.Linear(d_model + core_dim, d_model)
        self.redistribute = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = F.gelu(self.local(x))
        score = self.core(local)
        weight = score.softmax(dim=1)
        center = (score * weight).sum(dim=1, keepdim=True).expand(-1, x.shape[1], -1)
        update = self.redistribute(F.gelu(self.merge(torch.cat((x, center), dim=-1))))
        value = self.norm1(x + update)
        return self.norm2(value + self.ffn(value))


class TeCh(nn.Module):
    """Faithful clean-room TeCh reimplementation from the public paper.

    Input adaptation is exactly [B,C,T] -> [B,T,C].  Stochastic augmentation
    is disabled in evaluation and audit.  The pooled channel+temporal vector
    immediately before ``head`` is the fixed audit representation.
    """

    def __init__(
        self,
        d_model: int = 64,
        channel_depth: int = 2,
        temporal_depth: int = 2,
        patch_len: int = 20,
        dropout: float = 0.2,
        train_jitter: float = 0.0,
    ):
        super().__init__()
        self.patch_len = int(patch_len)
        self.train_jitter = float(train_jitter)
        self.channel_embed = nn.Linear(INPUT_SAMPLES, int(d_model))
        self.temporal_patch = nn.Conv2d(
            1,
            int(d_model),
            kernel_size=(INPUT_CHANNELS, self.patch_len),
            stride=(1, self.patch_len),
            bias=False,
        )
        self.channel_blocks = nn.ModuleList(
            [CoTARBlock(int(d_model), float(dropout)) for _ in range(int(channel_depth))]
        )
        self.temporal_blocks = nn.ModuleList(
            [CoTARBlock(int(d_model), float(dropout)) for _ in range(int(temporal_depth))]
        )
        self.output_norm = nn.LayerNorm(int(d_model))
        self.representation_dim = int(d_model)
        self.head = nn.Linear(self.representation_dim, N_CLASSES)

    @staticmethod
    def _position(token: torch.Tensor) -> torch.Tensor:
        length, dim = token.shape[1], token.shape[2]
        position = torch.arange(length, device=token.device, dtype=token.dtype)[:, None]
        rate = torch.exp(
            torch.arange(0, dim, 2, device=token.device, dtype=token.dtype)
            * (-math.log(10000.0) / dim)
        )
        value = torch.zeros((length, dim), device=token.device, dtype=token.dtype)
        value[:, 0::2] = torch.sin(position * rate)
        value[:, 1::2] = torch.cos(position * rate[: value[:, 1::2].shape[1]])
        return token + value[None]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        _check_input(x, "TeCh")
        time_channel = x.transpose(1, 2)  # sole input adaptation [B,T,C]
        if self.training and self.train_jitter > 0:
            time_channel = time_channel + torch.randn_like(time_channel) * self.train_jitter
        channel_token = self._position(self.channel_embed(time_channel.transpose(1, 2)))
        temporal_token = self.temporal_patch(time_channel.transpose(1, 2).unsqueeze(1))
        temporal_token = self._position(temporal_token.squeeze(2).transpose(1, 2))
        for block in self.channel_blocks:
            channel_token = block(channel_token)
        for block in self.temporal_blocks:
            temporal_token = block(temporal_token)
        return self.output_norm(channel_token.mean(1) + temporal_token.mean(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


MODEL_TYPES = {
    "FBCNet": FBCNet,
    "EEGConformer": EEGConformer,
    "DeepConvNet": DeepConvNet,
    "TeCh": TeCh,
}


def build_model(backbone: str, config: Mapping[str, Any]) -> nn.Module:
    if backbone not in MODEL_TYPES:
        raise KeyError(f"Unknown prospective backbone: {backbone}")
    return MODEL_TYPES[backbone](**dict(config.get("model", {})))


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _check_input(x: torch.Tensor, name: str) -> None:
    if x.ndim != 3 or tuple(x.shape[1:]) != (INPUT_CHANNELS, INPUT_SAMPLES):
        raise ValueError(f"{name} expects (batch,58,1000), received {tuple(x.shape)}")
