"""Cross-fitted mixed-effects bank and exact residual-span HardRandom."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


EPS = 1e-8


@dataclass(frozen=True)
class EffectSnapshot:
    subjects: tuple[str, ...]
    labels: tuple[int, ...]
    b: np.ndarray
    c: np.ndarray
    residual: np.ndarray
    counts: np.ndarray
    eta: np.ndarray
    rho: np.ndarray
    valid: np.ndarray


def _ordered(values: Iterable[object]) -> tuple:
    return tuple(sorted(set(values)))


class MixedEffectsBank:
    """Immutable bank built only from the explicitly supplied training rows."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, subjects: np.ndarray, row_ids: np.ndarray | None = None):
        self.features = np.asarray(features, np.float64)
        self.labels_raw = np.asarray(labels, np.int64)
        self.subjects_raw = np.asarray(subjects).astype(str)
        self.row_ids = np.arange(len(self.features), dtype=np.int64) if row_ids is None else np.asarray(row_ids, np.int64)
        if self.features.ndim != 2 or len(self.features) != len(self.labels_raw) or len(self.features) != len(self.subjects_raw):
            raise ValueError("bank arrays are not row aligned")
        if len(np.unique(self.row_ids)) != len(self.row_ids):
            raise ValueError("bank row ids are not unique")
        if not np.isfinite(self.features).all():
            raise ValueError("bank contains non-finite features")
        self.subjects = tuple(sorted(np.unique(self.subjects_raw).tolist()))
        self.labels = tuple(sorted(np.unique(self.labels_raw).astype(int).tolist()))
        self.subject_index = {value: idx for idx, value in enumerate(self.subjects)}
        self.label_index = {value: idx for idx, value in enumerate(self.labels)}
        shape = (len(self.subjects), len(self.labels))
        self.counts = np.zeros(shape, np.int64)
        self.sums = np.zeros(shape + (self.features.shape[1],), np.float64)
        self.class_counts = np.zeros(len(self.labels), np.int64)
        self.class_sums = np.zeros((len(self.labels), self.features.shape[1]), np.float64)
        for z, y, s in zip(self.features, self.labels_raw, self.subjects_raw):
            si, yi = self.subject_index[s], self.label_index[int(y)]
            self.counts[si, yi] += 1
            self.sums[si, yi] += z
            self.class_counts[yi] += 1
            self.class_sums[yi] += z
        self.rho = np.asarray([
            np.median(self.counts[:, yi][self.counts[:, yi] > 0]) for yi in range(len(self.labels))
        ], np.float64)
        if (self.rho <= 0).any():
            raise RuntimeError("degenerate shrinkage rho")
        self.full = self.snapshot()
        self._fit_residual_subspace()

    def snapshot(self, exclude_position: int | None = None) -> EffectSnapshot:
        counts = self.counts.copy()
        sums = self.sums.copy()
        class_counts = self.class_counts.copy()
        class_sums = self.class_sums.copy()
        if exclude_position is not None:
            if not 0 <= int(exclude_position) < len(self.features):
                raise IndexError(exclude_position)
            pos = int(exclude_position)
            si = self.subject_index[self.subjects_raw[pos]]
            yi = self.label_index[int(self.labels_raw[pos])]
            counts[si, yi] -= 1
            sums[si, yi] -= self.features[pos]
            class_counts[yi] -= 1
            class_sums[yi] -= self.features[pos]
        valid = counts > 0
        cell_mean = sums / np.maximum(counts[..., None], 1)
        class_mean = class_sums / np.maximum(class_counts[:, None], 1)
        residual = cell_mean - class_mean[None, :, :]
        eta = counts / (counts + self.rho[None, :])
        shrunk = eta[..., None] * residual
        shrunk[~valid] = 0.0
        denom = valid.sum(1, keepdims=True)
        b = shrunk.sum(1) / np.maximum(denom, 1)
        c = shrunk - b[:, None, :]
        c[~valid] = 0.0
        return EffectSnapshot(self.subjects, self.labels, b, c, shrunk, counts, eta, self.rho.copy(), valid)

    def anchor_snapshot(self, row_id: int) -> EffectSnapshot:
        positions = np.flatnonzero(self.row_ids == int(row_id))
        if len(positions) != 1:
            raise KeyError(f"anchor row not in bank: {row_id}")
        return self.snapshot(int(positions[0]))

    def direction(self, row_id: int, target_subject: str, *, factorized: bool = True) -> np.ndarray:
        pos = int(np.flatnonzero(self.row_ids == int(row_id))[0])
        source = self.subjects_raw[pos]
        label = int(self.labels_raw[pos])
        if str(target_subject) == source:
            raise ValueError("target subject equals source")
        snap = self.snapshot(pos)
        si, ti = self.subject_index[source], self.subject_index[str(target_subject)]
        if factorized:
            return (snap.b[ti] - snap.b[si]).astype(np.float32)
        yi = self.label_index[label]
        if not snap.valid[si, yi] or not snap.valid[ti, yi]:
            raise RuntimeError("invalid class-conditional cell")
        return (snap.residual[ti, yi] - snap.residual[si, yi]).astype(np.float32)

    def _fit_residual_subspace(self) -> None:
        centered = self.full.b - self.full.b.mean(0, keepdims=True)
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        keep = singular > max(singular[0] * 1e-6 if len(singular) else 0.0, EPS)
        if not np.any(keep):
            raise RuntimeError("degenerate subject-main-effect bank")
        self.basis = vt[keep].T
        # Coordinates have singular / sqrt(n-1) scale.  A small deterministic
        # shrinkage floor makes rank-deficient covariance numerically stable.
        scale = singular[keep] / np.sqrt(max(len(centered) - 1, 1))
        floor = max(float(np.median(scale)) * 1e-3, EPS)
        self.whiten_scale = np.maximum(scale, floor)
        pair_norms = []
        for left in range(len(self.full.b)):
            for right in range(left + 1, len(self.full.b)):
                pair_norms.append(self.whitened_norm(self.full.b[right] - self.full.b[left]))
        self.norm_radius = float(np.quantile(pair_norms, 0.95)) if pair_norms else 0.0

    def whitened_coordinates(self, delta: np.ndarray) -> np.ndarray:
        return (np.asarray(delta, np.float64) @ self.basis) / self.whiten_scale

    def whitened_norm(self, delta: np.ndarray) -> float:
        return float(np.linalg.norm(self.whitened_coordinates(delta)))

    def hard_random(self, structured: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        target_norm = self.whitened_norm(structured)
        coordinates = rng.normal(size=len(self.whiten_scale))
        coordinates *= target_norm / max(float(np.linalg.norm(coordinates)), EPS)
        # Reverse whitening and map through the residual SVD basis.
        return ((coordinates * self.whiten_scale) @ self.basis.T).astype(np.float32)

    def audit_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for si, subject in enumerate(self.subjects):
            b_norm = float(np.linalg.norm(self.full.b[si]))
            for yi, label in enumerate(self.labels):
                residual_energy = float(np.dot(self.full.residual[si, yi], self.full.residual[si, yi]))
                b_energy = float(np.dot(self.full.b[si], self.full.b[si]))
                rows.append({
                    "subject_id": subject,
                    "label": label,
                    "count": int(self.full.counts[si, yi]),
                    "rho": float(self.rho[yi]),
                    "eta": float(self.full.eta[si, yi]),
                    "norm_b": b_norm,
                    "norm_c": float(np.linalg.norm(self.full.c[si, yi])),
                    "residual_energy": residual_energy,
                    "main_effect_energy_fraction": b_energy / max(residual_energy, EPS),
                })
        return rows


def detach_bank_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device).detach()
    tensor.requires_grad_(False)
    return tensor

