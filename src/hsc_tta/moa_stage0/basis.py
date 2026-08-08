from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping

import numpy as np


DEFAULT_ALIASES = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
}


def normalize_electrode_name(name: str, aliases: Mapping[str, str] | None = None) -> str:
    """Normalize syntax only; physical aliases must be supplied explicitly."""
    normalized = re.sub(r"[^A-Z0-9]", "", str(name).strip().upper())
    mapping = dict(DEFAULT_ALIASES)
    if aliases:
        mapping.update({str(k).upper(): str(v).upper() for k, v in aliases.items()})
    return mapping.get(normalized, normalized)


def standard_1020_coordinates(names: Iterable[str], aliases: Mapping[str, str] | None = None) -> dict[str, np.ndarray]:
    """Return unit-sphere MNE standard-1020 coordinates with strict coverage."""
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    positions = montage.get_positions()["ch_pos"]
    lookup = {normalize_electrode_name(key): np.asarray(value, float) for key, value in positions.items()}
    output: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for raw_name in names:
        name = normalize_electrode_name(raw_name, aliases)
        if name not in lookup:
            missing.append(name)
            continue
        value = lookup[name]
        norm = float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm <= 0:
            missing.append(name)
            continue
        output[name] = value / norm
    if missing:
        raise ValueError(f"canonical coordinates missing for: {sorted(set(missing))}")
    return output


def _fibonacci_cap(count: int) -> np.ndarray:
    # Deterministic upper/head sphere centers, independent of datasets and labels.
    index = np.arange(count, dtype=float) + 0.5
    z = 1.0 - 1.65 * index / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    angle = np.pi * (3.0 - np.sqrt(5.0)) * index
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))


@dataclass(frozen=True)
class CanonicalBasis:
    centers: np.ndarray
    bandwidth: float
    grid_mean: np.ndarray
    inverse_sqrt_gram: np.ndarray
    ridge: float
    grid_size: int

    @property
    def dimension(self) -> int:
        return int(self.centers.shape[0])

    @classmethod
    def fixed(cls, dimension: int = 32, bandwidth: float = 0.55, grid_size: int = 2048, ridge: float = 1e-10) -> "CanonicalBasis":
        if dimension <= 0 or grid_size < dimension:
            raise ValueError("invalid canonical basis dimensions")
        centers = _fibonacci_cap(dimension)
        grid = _fibonacci_cap(grid_size)
        raw = cls._rbf(grid, centers, bandwidth)
        grid_mean = raw.mean(axis=0)
        centered = raw - grid_mean
        gram = centered.T @ centered
        values, vectors = np.linalg.eigh(gram + ridge * np.eye(dimension))
        if float(values.min()) <= 0:
            raise ValueError("canonical Gram matrix is not positive definite")
        inverse_sqrt = (vectors * values[None, :] ** -0.5) @ vectors.T
        return cls(centers, float(bandwidth), grid_mean, inverse_sqrt, float(ridge), int(grid_size))

    @staticmethod
    def _rbf(points: np.ndarray, centers: np.ndarray, bandwidth: float) -> np.ndarray:
        squared = ((np.asarray(points)[:, None, :] - np.asarray(centers)[None, :, :]) ** 2).sum(axis=-1)
        return np.exp(-squared / (2.0 * bandwidth * bandwidth))

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        raw = self._rbf(np.atleast_2d(points), self.centers, self.bandwidth)
        return (raw - self.grid_mean) @ self.inverse_sqrt_gram

    def audit(self) -> dict[str, object]:
        grid = _fibonacci_cap(self.grid_size)
        evaluated = self.evaluate(grid)
        gram = evaluated.T @ evaluated
        return {
            "dimension": self.dimension,
            "bandwidth": self.bandwidth,
            "ridge": self.ridge,
            "grid_size": self.grid_size,
            "constant_mode_max_abs": float(np.abs(evaluated.mean(axis=0)).max()),
            "gram_identity_relative_error": float(np.linalg.norm(gram - np.eye(self.dimension)) / np.sqrt(self.dimension)),
            "gram_eigenvalue_min": float(np.linalg.eigvalsh(gram).min()),
            "gram_eigenvalue_max": float(np.linalg.eigvalsh(gram).max()),
            "centers": self.centers.tolist(),
        }
