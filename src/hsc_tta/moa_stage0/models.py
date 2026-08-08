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
    # Optional source-reference map. v2 uses this to make Y0=C Yraw explicit;
    # None preserves the historical v1 identity behavior for old checkpoints.
    source_reference: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "TorchOperator":
        return TorchOperator(
            self.operator_id, self.family,
            *(getattr(self, key).to(device) for key in ("A", "B", "coefficients", "active", "centroid", "reference", "interpolation", "L", "R")),
            self.source_reference.to(device) if self.source_reference is not None else None,
        )


def _weighted_centroid(weights: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    denominator = np.abs(weights).sum(axis=1, keepdims=True)
    denominator[denominator == 0] = 1.0
    return np.abs(weights) @ coordinates / denominator


def make_torch_operator(
    view: OperatorView,
    electrode_coordinates: np.ndarray,
    canonical_centers: np.ndarray,
    alpha: float,
    source_reference: np.ndarray | None = None,
) -> TorchOperator:
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
        convert(source_reference) if source_reference is not None else None,
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
        temporal_stem: bool = False,
    ):
        super().__init__()
        if method not in self.METHODS:
            raise ValueError(f"unknown Stage-0 method {method}")
        self.method, self.patch_size, self.hidden_dim = method, patch_size, hidden_dim
        self.signal_projection = nn.Linear(patch_size, hidden_dim)
        feature_dimension = {"B2": 3, "B3": 3, "B4": 11, "B5": 3, "B6": canonical_dim, "B7": 3, "B8": canonical_dim}[method]
        self.feature_dimension = feature_dimension
        # Per-row LayerNorm is fit-free and cannot use test-set statistics.
        self.operator_feature_norm = nn.LayerNorm(feature_dimension)
        self.operator_projection = nn.Linear(feature_dimension, hidden_dim)
        self.component_embedding = nn.Embedding(64, hidden_dim)
        self.patch_embedding = nn.Embedding(32, hidden_dim)
        self.use_component_embedding = method in {"B7", "B8"}
        self.temporal_stem = (
            nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=25, padding=12), nn.GELU(),
                nn.Conv1d(8, 1, kernel_size=25, padding=12),
            ) if temporal_stem else nn.Identity()
        )
        self.core = SharedTransformer(hidden_dim, layers, heads, dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))
        self.family_order = ()

    def _representation(self, source_y: torch.Tensor, operator: TorchOperator) -> tuple[torch.Tensor, torch.Tensor]:
        if operator.source_reference is None:
            source_observation = source_y
        else:
            source_observation = torch.einsum("rc,bct->brt", operator.source_reference, source_y)
        observed = torch.einsum("oc,bct->bot", operator.A, source_observation) * 1e6
        if self.method == "B2":
            return observed, operator.active
        if self.method == "B3":
            return observed, operator.centroid
        if self.method == "B4":
            # Continuous, row-local metadata only. No family/category one-hot
            # can leak test-only operator labels into B4.
            positive = torch.clamp(operator.coefficients, min=0)
            negative = torch.clamp(-operator.coefficients, min=0)
            scale = float(operator.coefficients.shape[1])
            metadata = torch.cat((
                positive.sum(dim=1, keepdim=True), negative.sum(dim=1, keepdim=True),
                (positive > 1e-8).sum(dim=1, keepdim=True) / scale,
                (negative > 1e-8).sum(dim=1, keepdim=True) / scale,
                (torch.abs(operator.coefficients) > 1e-8).sum(dim=1, keepdim=True) / scale,
            ), dim=1)
            return observed, torch.cat((operator.active, operator.reference, metadata), dim=1)
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
        batch_size, channels, time = signal.shape
        signal = self.temporal_stem(signal.reshape(batch_size * channels, 1, time)).reshape(batch_size, channels, time)
        patches = signal.unfold(-1, self.patch_size, self.patch_size)
        batch, channels, patch_count, _ = patches.shape
        tokens = self.signal_projection(patches)
        features = self.operator_feature_norm(features)
        tokens = tokens + self.operator_projection(features)[None, :, None, :]
        component_ids = torch.arange(channels, device=signal.device)
        patch_ids = torch.arange(patch_count, device=signal.device)
        if self.use_component_embedding:
            tokens = tokens + self.component_embedding(component_ids)[None, :, None, :]
        tokens = tokens + self.patch_embedding(patch_ids)[None, None, :, :]
        encoded = self.core(tokens.reshape(batch, channels * patch_count, self.hidden_dim))
        return self.classifier(encoded.mean(dim=1))

    def forward(self, source_y: torch.Tensor, operator: TorchOperator) -> torch.Tensor:
        signal, features = self._representation(source_y, operator)
        return self.forward_representation(signal, features)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
