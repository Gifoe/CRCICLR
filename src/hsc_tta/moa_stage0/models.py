from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .lifting import lifting_operators
from .operators import OperatorView


@dataclass(frozen=True)
class TorchOperator:
    operator_id: str
    family: str
    A: torch.Tensor
    B: torch.Tensor
    coefficients: torch.Tensor
    active: torch.Tensor
    centroid: torch.Tensor
    reference: torch.Tensor
    interpolation: torch.Tensor
    L: torch.Tensor
    R: torch.Tensor

    def to(self, device: torch.device | str) -> "TorchOperator":
        return TorchOperator(
            self.operator_id, self.family,
            *(getattr(self, key).to(device) for key in ("A", "B", "coefficients", "active", "centroid", "reference", "interpolation", "L", "R")),
        )


def _weighted_centroid(weights: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    denominator = np.abs(weights).sum(axis=1, keepdims=True)
    denominator[denominator == 0] = 1.0
    return np.abs(weights) @ coordinates / denominator


def make_torch_operator(view: OperatorView, electrode_coordinates: np.ndarray, canonical_centers: np.ndarray, alpha: float) -> TorchOperator:
    coefficients = np.asarray(view.electrode_coefficients, float)
    positive = np.clip(coefficients, 0, None)
    negative = np.clip(-coefficients, 0, None)
    active = _weighted_centroid(positive, electrode_coordinates)
    reference = _weighted_centroid(negative, electrode_coordinates)
    centroid = _weighted_centroid(coefficients, electrode_coordinates)
    squared = ((canonical_centers[:, None, :] - centroid[None, :, :]) ** 2).sum(axis=-1)
    interpolation = np.exp(-squared / (2 * 0.45 ** 2))
    interpolation /= interpolation.sum(axis=1, keepdims=True) + 1e-12
    values = lifting_operators(view.B, alpha)
    convert = lambda value: torch.as_tensor(value, dtype=torch.float32)
    return TorchOperator(
        view.operator_id, view.operator_family, convert(view.A), convert(view.B), convert(coefficients),
        convert(active), convert(centroid), convert(reference), convert(interpolation), convert(values["L"]), convert(values["R"]),
    )


class SharedTransformer(nn.Module):
    def __init__(self, hidden_dim: int = 256, layers: int = 6, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        block = nn.TransformerEncoderLayer(
            hidden_dim, heads, dim_feedforward=hidden_dim * 8, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(hidden_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)


class Stage0Transformer(nn.Module):
    METHODS = {"B2", "B3", "B4", "B5", "B6", "B7", "B8"}

    def __init__(
        self, method: str, num_classes: int, canonical_dim: int = 32, patch_size: int = 160,
        hidden_dim: int = 256, layers: int = 6, heads: int = 8, dropout: float = 0.1,
    ):
        super().__init__()
        if method not in self.METHODS:
            raise ValueError(f"unknown Stage-0 method {method}")
        self.method, self.patch_size, self.hidden_dim = method, patch_size, hidden_dim
        self.signal_projection = nn.Linear(patch_size, hidden_dim)
        feature_dimension = {"B2": 3, "B3": 3, "B4": 11, "B5": 3, "B6": canonical_dim, "B7": 3, "B8": canonical_dim}[method]
        self.operator_projection = nn.Linear(feature_dimension, hidden_dim)
        self.component_embedding = nn.Embedding(64, hidden_dim)
        self.patch_embedding = nn.Embedding(32, hidden_dim)
        self.core = SharedTransformer(hidden_dim, layers, heads, dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))
        self.family_order = ("dense_subset", "sparse_subset", "bipolar", "polarity", "rereference")

    def _representation(self, source_y: torch.Tensor, operator: TorchOperator) -> tuple[torch.Tensor, torch.Tensor]:
        observed = torch.einsum("oc,bct->bot", operator.A, source_y) * 1e6
        if self.method == "B2":
            return observed, operator.active
        if self.method == "B3":
            return observed, operator.centroid
        if self.method == "B4":
            family = torch.zeros((observed.shape[1], len(self.family_order)), device=observed.device)
            family[:, self.family_order.index(operator.family)] = 1.0
            return observed, torch.cat((operator.active, operator.reference, family), dim=1)
        if self.method == "B5":
            canonical = torch.einsum("kc,bct->bkt", operator.interpolation, observed)
            # Canonical centers are represented by weighted source centroids; no signed functional is exposed.
            coordinates = operator.interpolation @ operator.centroid
            return canonical, coordinates
        if self.method == "B6":
            return observed, operator.B
        lifted = torch.einsum("kc,bct->bkt", operator.L, observed)
        if self.method == "B7":
            # Component identity is supplied separately; coordinates are a neutral zero feature.
            return lifted, torch.zeros((lifted.shape[1], 3), device=lifted.device)
        return lifted, operator.R

    def forward_representation(self, signal: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if signal.shape[-1] % self.patch_size:
            signal = signal[..., : signal.shape[-1] - signal.shape[-1] % self.patch_size]
        patches = signal.unfold(-1, self.patch_size, self.patch_size)
        batch, channels, patch_count, _ = patches.shape
        tokens = self.signal_projection(patches)
        tokens = tokens + self.operator_projection(features)[None, :, None, :]
        component_ids = torch.arange(channels, device=signal.device)
        patch_ids = torch.arange(patch_count, device=signal.device)
        tokens = tokens + self.component_embedding(component_ids)[None, :, None, :] + self.patch_embedding(patch_ids)[None, None, :, :]
        encoded = self.core(tokens.reshape(batch, channels * patch_count, self.hidden_dim))
        return self.classifier(encoded.mean(dim=1))

    def forward(self, source_y: torch.Tensor, operator: TorchOperator) -> torch.Tensor:
        signal, features = self._representation(source_y, operator)
        return self.forward_representation(signal, features)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
