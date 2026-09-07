from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FBCNet(nn.Module):
    """Canonical FBCNet family: fixed 4-Hz bands, learned spatial filters, log-variance."""

    def __init__(self, channels: int, samples: int = 1000, sfreq: float = 250.0, spatial_filters: int = 8, windows: int = 4, dropout: float = 0.25):
        super().__init__(); self.channels = channels; self.samples = samples; self.windows = windows
        self.bands = tuple((low, low + 4) for low in range(4, 40, 4))
        frequencies = torch.fft.rfftfreq(samples, d=1.0 / sfreq)
        self.register_buffer("frequency_masks", torch.stack([((frequencies >= low) & (frequencies < high)) for low, high in self.bands]), persistent=True)
        self.spatial_weight = nn.Parameter(torch.empty(len(self.bands), spatial_filters, channels)); nn.init.xavier_uniform_(self.spatial_weight)
        width = len(self.bands) * spatial_filters
        self.bn = nn.BatchNorm1d(width); self.dropout = nn.Dropout(dropout)
        self.representation_dim = width * windows; self.head = nn.Linear(self.representation_dim, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft(x.float(), dim=-1); weight = self.spatial_weight.float(); weight = weight / weight.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            spatial = spectrum.new_zeros((len(x), len(self.bands), weight.shape[1], spectrum.shape[-1]))
            for band in range(len(self.bands)):
                support = self.frequency_masks[band]
                spatial[:, band, :, support] = torch.einsum("bcf,mc->bmf", spectrum[:, :, support], weight[band].to(spectrum.dtype))
            signal = torch.fft.irfft(spatial, n=self.samples, dim=-1)
        value = signal.reshape(len(x), -1, self.samples); value = self.dropout(value * torch.sigmoid(self.bn(value)))
        feature = torch.log(value.reshape(len(x), value.shape[1], self.windows, self.samples // self.windows).var(-1, unbiased=False).clamp(1e-6, 1e6))
        return feature.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class TCNBlock(nn.Module):
    def __init__(self, dim: int, dilation: int, dropout: float):
        super().__init__(); pad = dilation * 2
        self.conv1 = nn.Conv1d(dim, dim, 3, padding=pad, dilation=dilation); self.bn1 = nn.BatchNorm1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, 3, padding=pad, dilation=dilation); self.bn2 = nn.BatchNorm1d(dim); self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-1]; y = self.drop(F.elu(self.bn1(self.conv1(x)))[..., :size]); y = self.drop(F.elu(self.bn2(self.conv2(y)))[..., :size]); return F.elu(x + y)


class ATCNet(nn.Module):
    """ATCNet family with EEG convolutional tokenizer, window attention, and dilated TCN."""

    def __init__(self, channels: int, samples: int = 1000, f1: int = 16, depth_multiplier: int = 2, dropout: float = 0.3):
        super().__init__(); f2 = f1 * depth_multiplier
        self.temporal = nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False)
        self.spatial = nn.Conv2d(f1, f2, (channels, 1), groups=f1, bias=False)
        self.bn1 = nn.BatchNorm2d(f2); self.drop1 = nn.Dropout(dropout)
        self.sep_dw = nn.Conv2d(f2, f2, (1, 16), padding=(0, 8), groups=f2, bias=False); self.sep_pw = nn.Conv2d(f2, f2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(f2); self.drop2 = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(f2, 4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(f2); self.tcn = nn.Sequential(TCNBlock(f2, 1, dropout), TCNBlock(f2, 2, dropout))
        self.representation_dim = f2; self.head = nn.Linear(f2, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop1(F.elu(self.bn1(self.spatial(self.temporal(x[:, None]))))); y = F.avg_pool2d(y, (1, 8))
        y = self.drop2(F.elu(self.bn2(self.sep_pw(self.sep_dw(y))))); y = F.avg_pool2d(y, (1, 8)).squeeze(2).transpose(1, 2)
        window = min(12, y.shape[1]); starts = sorted(set((0, max(0, (y.shape[1] - window) // 2), max(0, y.shape[1] - window))))
        outputs = []
        for start in starts:
            token = y[:, start:start + window]; attended, _ = self.attn(token, token, token, need_weights=False); token = self.norm(token + attended)
            outputs.append(self.tcn(token.transpose(1, 2))[:, :, -1])
        return torch.stack(outputs).mean(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class EEGInceptionMI(nn.Module):
    """EEG-Inception MI family using multi-scale temporal branches and spatial depthwise filters."""

    def __init__(self, channels: int, samples: int = 1000, base: int = 8, dropout: float = 0.3):
        super().__init__(); kernels = (64, 32, 16)
        self.temporal = nn.ModuleList([nn.Conv2d(1, base, (1, k), padding=(0, k // 2), bias=False) for k in kernels])
        self.spatial = nn.ModuleList([nn.Conv2d(base, base * 2, (channels, 1), groups=base, bias=False) for _ in kernels])
        first = base * 2 * len(kernels); self.bn1 = nn.BatchNorm2d(first); self.drop1 = nn.Dropout(dropout)
        second_base = 16; self.second = nn.ModuleList([nn.Conv1d(first, second_base, k, padding=k // 2, bias=False) for k in (32, 16, 8)])
        self.bn2 = nn.BatchNorm1d(second_base * 3); self.drop2 = nn.Dropout(dropout); self.representation_dim = second_base * 3; self.head = nn.Linear(self.representation_dim, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        for temporal, spatial in zip(self.temporal, self.spatial):
            y = spatial(temporal(x[:, None])); parts.append(y[..., :x.shape[-1]])
        y = self.drop1(F.elu(self.bn1(torch.cat(parts, 1)))); y = F.avg_pool2d(y, (1, 4)).squeeze(2)
        length = y.shape[-1]; y = torch.cat([branch(y)[..., :length] for branch in self.second], 1)
        y = self.drop2(F.elu(self.bn2(y))); y = F.avg_pool1d(y, 4)
        return y.mean(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_model(name: str, channels: int, config: dict[str, object]) -> nn.Module:
    model_config = dict(config.get("model", {}))
    if name == "FBCNet": return FBCNet(channels, **model_config)
    if name == "ATCNet": return ATCNet(channels, **model_config)
    if name == "EEGInceptionMI": return EEGInceptionMI(channels, **model_config)
    raise KeyError(name)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
