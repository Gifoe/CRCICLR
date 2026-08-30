"""Cross-fitted class-centred subject style and Bures transport geometry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from common import KNN_K, SUPPORT_QUANTILE, stable_seed


def _sym(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value, np.float64) + np.asarray(value, np.float64).T) * 0.5


def _sqrt_psd(value: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = _sym(value)
    eig, vec = np.linalg.eigh(value)
    eig = np.maximum(eig, floor)
    root = _sym((vec * np.sqrt(eig)) @ vec.T)
    inv = _sym((vec * (1.0 / np.sqrt(eig))) @ vec.T)
    return root, inv, eig


def bures_map(c_source: np.ndarray, c_target: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Return the symmetric Gaussian OT/Bures linear map C_s -> C_t."""
    source_root, source_inv, _ = _sqrt_psd(c_source, floor)
    middle = _sym(source_root @ _sym(c_target) @ source_root)
    middle_root, _, _ = _sqrt_psd(middle, floor)
    return _sym(source_inv @ middle_root @ source_inv)


def _cov(value: np.ndarray, mean: np.ndarray, dim: int, floor: float) -> np.ndarray:
    value = np.asarray(value, np.float64)
    if len(value) <= 1:
        return np.eye(dim, dtype=np.float64) * floor
    centered = value - mean
    raw = _sym((centered.T @ centered) / max(1, len(value) - 1))
    # Numerical round-off can make a rank-deficient sample covariance have a
    # tiny negative eigenvalue.  Apply the declared floor at construction so
    # every stored cell is positive definite, not only matrices passed later
    # to a square-root routine.
    eig, vec = np.linalg.eigh(raw)
    eig = np.maximum(eig, float(floor))
    return _sym((vec * eig) @ vec.T)


def anchor_excluded_indices(features: np.ndarray, row_ids: np.ndarray, position: int) -> np.ndarray:
    """Exclude the original row id and exact duplicate anchor rows."""
    features = np.asarray(features, np.float64)
    row_ids = np.asarray(row_ids)
    position = int(position)
    # A byte-view gives exact duplicate detection in O(n) scalar comparisons;
    # the previous row-wise ``all(isclose(...))`` made every kNN query O(n*d)
    # in Python and dominated the source grid.
    contiguous = np.ascontiguousarray(features)
    key_dtype = np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    keys = contiguous.view(key_dtype).reshape(-1)
    same_id = row_ids == row_ids[position]
    duplicate = keys == keys[position]
    keep = ~(same_id | duplicate)
    keep[position] = False
    return np.flatnonzero(keep)


def anchor_excluded_neighbors(features: np.ndarray, row_ids: np.ndarray, k: int = KNN_K) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized anchor-excluded nearest-neighbor indices and distances."""
    value = np.asarray(features, np.float32); ids = np.asarray(row_ids); n = len(value); k = min(int(k), max(1, n - 1))
    contiguous = np.ascontiguousarray(value); key_dtype = np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1])); keys = contiguous.view(key_dtype).reshape(-1)
    out_idx = np.full((n, k), -1, np.int64); out_dist = np.full((n, k), np.inf, np.float64)
    for start in range(0, n, 256):
        stop = min(n, start + 256); block = value[start:stop]
        # Squared distances avoid a second pass over the full matrix.
        distance = ((block[:, None, :] - value[None, :, :]) ** 2).sum(axis=2, dtype=np.float32)
        exclude = (ids[start:stop, None] == ids[None, :]) | (keys[start:stop, None] == keys[None, :])
        distance[exclude] = np.inf
        near = np.argpartition(distance, kth=k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(distance, near, axis=1)
        order = np.argsort(vals, axis=1, kind="stable")
        out_idx[start:stop] = np.take_along_axis(near, order, axis=1)
        out_dist[start:stop] = np.sqrt(np.take_along_axis(vals, order, axis=1).astype(np.float64))
    return out_idx, out_dist


def knn_mean_distance(features: np.ndarray, row_ids: np.ndarray, position: int, k: int = KNN_K) -> float:
    _, distance = anchor_excluded_neighbors(features, row_ids, max(k, 1))
    values = distance[int(position)]
    values = values[np.isfinite(values)]
    return float(values[: min(k, len(values))].mean()) if len(values) else float("inf")


def class_support_radius(features: np.ndarray, labels: np.ndarray, row_ids: np.ndarray, quantile: float = SUPPORT_QUANTILE) -> dict[int, float]:
    result: dict[int, float] = {}
    _, distances = anchor_excluded_neighbors(features, row_ids, KNN_K)
    for label in sorted(np.unique(labels).tolist()):
        positions = np.flatnonzero(np.asarray(labels) == label)
        values = np.asarray([np.mean(distances[int(p)][np.isfinite(distances[int(p)])]) if np.isfinite(distances[int(p)]).any() else np.inf for p in positions], np.float64)
        finite = values[np.isfinite(values)]
        result[int(label)] = float(np.quantile(finite, quantile)) if len(finite) else float("inf")
    return result


@dataclass(frozen=True)
class Cell:
    mean: np.ndarray
    cov: np.ndarray
    count: int


class BuresBank:
    """A two-way cross-fitted bank built from one source-training partition."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, subjects: np.ndarray, row_ids: np.ndarray, *, dataset: str, fold: int, seed: int):
        self.features = np.asarray(features, np.float64)
        self.labels = np.asarray(labels, np.int64)
        self.subjects = np.asarray(subjects).astype(str)
        self.row_ids = np.asarray(row_ids, np.int64)
        self.dataset, self.fold, self.seed = dataset, int(fold), int(seed)
        self.dim = int(self.features.shape[1])
        self.subject_list = tuple(sorted(np.unique(self.subjects).tolist()))
        self.class_list = tuple(sorted(np.unique(self.labels).tolist()))
        self.half = np.asarray([stable_seed(dataset, fold, seed, s, int(y), int(r)) % 2 for s, y, r in zip(self.subjects, self.labels, self.row_ids)], np.int8)
        self.cell: dict[tuple[str, int, int], Cell] = {}
        self.class_cell: dict[tuple[int, int], Cell] = {}
        self._subject_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, int]] = {}
        self._map_cache: dict[tuple[str, str, int, int], np.ndarray] = {}
        # Target affinity is queried for every alpha candidate.  Cache the
        # target class cloud and the inverse covariance once per cross-fit
        # half; recomputing an eigendecomposition inside that loop made a
        # source unit needlessly expensive without changing the statistic.
        self._target_affinity_cache: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
        self._cell_positions: dict[tuple[str, int, int], np.ndarray] = {}
        self._build_cells()
        positive = []
        for key, value in self.class_cell.items():
            if value.count > 1:
                eig = np.linalg.eigvalsh(value.cov)
                positive.extend(eig[eig > 1e-10].tolist())
        self.pool_floor = float(1e-4 * (np.median(positive) if positive else 1.0))
        all_mean = self.features.mean(0)
        self.pool_cov = _sym(_cov(self.features, all_mean, self.dim, self.pool_floor))
        self.class_mean = {label: self._class_cell(label, None).mean for label in self.class_list}
        self.class_cov = {label: self._class_cell(label, None).cov for label in self.class_list}
        self.radius = class_support_radius(self.features, self.labels, self.row_ids)

    def _make_cell(self, idx: np.ndarray) -> Cell:
        if len(idx):
            mean = self.features[idx].mean(0)
            cov = _cov(self.features[idx], mean, self.dim, self.pool_floor if hasattr(self, "pool_floor") else 1e-8)
            return Cell(mean, cov, int(len(idx)))
        return Cell(np.zeros(self.dim), np.eye(self.dim), 0)

    def _build_cells(self) -> None:
        for subject in self.subject_list:
            for label in self.class_list:
                for half in (0, 1):
                    idx = np.flatnonzero((self.subjects == subject) & (self.labels == label) & (self.half == half))
                    self._cell_positions[(subject, int(label), half)] = idx
                    self.cell[(subject, int(label), half)] = self._make_cell(idx)
        for label in self.class_list:
            for half in (0, 1):
                idx = np.flatnonzero((self.labels == label) & (self.half == half))
                self.class_cell[(int(label), half)] = self._make_cell(idx)

    def _class_cell(self, label: int, half: int | None) -> Cell:
        if half in (0, 1):
            value = self.class_cell[(int(label), int(half))]
            if value.count:
                return value
        idx = np.flatnonzero(self.labels == int(label))
        return self._make_cell(idx)

    def _cell_for(self, subject: str, label: int, half: int, *, exclude_row: int | None = None) -> Cell:
        value = self.cell[(str(subject), int(label), int(half))]
        positions = self._cell_positions[(str(subject), int(label), int(half))]
        excluded = exclude_row is not None and bool(np.any(self.row_ids[positions] == self.row_ids[int(exclude_row)]))
        if value.count >= 2 and not excluded:
            return value
        idx = positions if exclude_row is None else positions[self.row_ids[positions] != self.row_ids[int(exclude_row)]]
        if len(idx) == 0 and exclude_row is not None:
            idx = np.flatnonzero((self.subjects == str(subject)) & (self.labels == int(label)) & (self.row_ids != self.row_ids[int(exclude_row)]))
        if len(idx):
            return self._make_cell(idx)
        pooled = self._class_cell(int(label), half)
        return pooled if pooled.count else self._class_cell(int(label), None)

    def subject_style(self, subject: str, half: int, *, exclude_row: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
        key = (str(subject), int(half))
        if exclude_row is None and key in self._subject_cache:
            return self._subject_cache[key]
        means, covs, counts = [], [], []
        for label in self.class_list:
            cell = self._cell_for(subject, int(label), half, exclude_row=exclude_row)
            global_cell = self._class_cell(int(label), half)
            if cell.count:
                means.append(cell.mean - global_cell.mean)
                covs.append(cell.cov)
                counts.append(cell.count)
        if not means:
            result = np.zeros(self.dim), self.pool_cov.copy(), 0
        else:
            m = np.mean(np.stack(means), axis=0)
            raw_cov = _sym(np.mean(np.stack(covs), axis=0))
            n_eff = int(max(1, round(np.mean(counts))))
            shrink = self.dim / (n_eff + self.dim)
            cov = _sym((1.0 - shrink) * raw_cov + shrink * self.pool_cov)
            eig, vec = np.linalg.eigh(cov)
            eig = np.maximum(eig, self.pool_floor)
            cov = _sym((vec * eig) @ vec.T)
            result = m, cov, n_eff
        if exclude_row is None:
            self._subject_cache[key] = result
        return result

    def style(self, position: int, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        position = int(position)
        source = str(self.subjects[position]); label = int(self.labels[position]); opposite = 1 - int(self.half[position])
        # Opposite-half cross-fitting already excludes the anchor.  Avoiding an
        # anchor-specific recomputation here also makes the bank cacheable per
        # subject/half while preserving the exact cross-fit rule.
        ms, cs, ns = self.subject_style(source, opposite)
        mt, ct, nt = self.subject_style(str(target), opposite)
        key = (source, str(target), opposite, label)
        if key not in self._map_cache:
            self._map_cache[key] = bures_map(cs, ct, self.pool_floor)
        return self.class_mean[label], ms, mt, self._map_cache[key], ct

    def endpoint(self, position: int, target: str) -> np.ndarray:
        mu, ms, mt, amap, _ = self.style(position, target)
        z = self.features[int(position)]
        return mu + mt + amap @ (z - mu - ms)

    def displacement(self, position: int, target: str) -> np.ndarray:
        return self.endpoint(position, target) - self.features[int(position)]

    def target_mean_cov(self, position: int, target: str) -> tuple[np.ndarray, np.ndarray]:
        label = int(self.labels[int(position)]); opposite = 1 - int(self.half[int(position)])
        mean_style, cov, _ = self.subject_style(str(target), opposite)
        return self.class_mean[label] + mean_style, cov

    def target_affinity_stats(self, position: int, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        """Return target same-class values and cached Gaussian quantities.

        The target Gaussian is estimated from the opposite cross-fit half of
        the anchor, so ``half`` is part of the cache key.  The returned tuple
        is ``(target_values, mean, precision, logdet)``.  ``None`` denotes a
        target subject with no same-class rows (which is an invalid
        candidate, never a reason to fall back to validation data).
        """
        position = int(position)
        label = int(self.labels[position])
        target = str(target)
        opposite = 1 - int(self.half[position])
        key = (target, label, opposite)
        if key in self._target_affinity_cache:
            return self._target_affinity_cache[key]
        target_mask = (self.subjects == target) & (self.labels == label)
        target_values = np.asarray(self.features[target_mask], np.float64)
        if not len(target_values):
            return None
        mean, cov = self.target_mean_cov(position, target)
        _, inv, _ = _sqrt_psd(cov, self.pool_floor)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0 or not np.isfinite(logdet):
            return None
        result = (target_values, np.asarray(mean, np.float64), np.asarray(inv, np.float64), float(logdet))
        self._target_affinity_cache[key] = result
        return result

    def whitened_norm(self, value: np.ndarray) -> float:
        root, inv, _ = _sqrt_psd(self.pool_cov, self.pool_floor)
        del root
        return float(np.linalg.norm(inv @ np.asarray(value, np.float64)))

    def audit_rows(self) -> list[dict[str, object]]:
        rows = []
        for subject in self.subject_list:
            for half in (0, 1):
                m, cov, n = self.subject_style(subject, half)
                eig = np.linalg.eigvalsh(cov)
                rows.append({"subject_id": subject, "half": half, "n_effective": n, "norm_m": float(np.linalg.norm(m)), "cov_min_eig": float(eig.min()), "cov_trace": float(np.trace(cov)), "lambda_s": float(self.dim / (n + self.dim))})
        return rows


def matched_random_displacement(structured: np.ndarray, bank: BuresBank, rng: np.random.Generator) -> np.ndarray:
    """Draw a signed vector in the observed one-dimensional displacement span.

    A signed draw is isotropic in the one-dimensional span of the observed
    structured displacement.  This construction guarantees exact Euclidean and
    pooled-whitened norm matching even when the residual covariance is
    anisotropic; the matching audit records both errors.
    """
    value = np.asarray(structured, np.float64)
    sign = -1.0 if int(rng.integers(0, 2)) else 1.0
    return (sign * value).astype(np.float32)


def target_affinity(features: np.ndarray, labels: np.ndarray, subjects: np.ndarray, position: int, candidate: np.ndarray, bank: BuresBank, target: str) -> tuple[float, float, float, float]:
    label = int(labels[int(position)]); target_mask = (np.asarray(subjects).astype(str) == str(target)) & (np.asarray(labels) == label)
    target_values = np.asarray(features, np.float64)[target_mask]
    if not len(target_values):
        return float("nan"), float("nan"), float("nan"), float("nan")
    anchor = np.asarray(features, np.float64)[int(position)]
    k = min(KNN_K, len(target_values))
    before_dist = float(np.sort(np.linalg.norm(target_values - anchor[None], axis=1))[:k].mean())
    after_dist = float(np.sort(np.linalg.norm(target_values - candidate[None], axis=1))[:k].mean())
    cached = bank.target_affinity_stats(int(position), str(target))
    if cached is None:
        return before_dist, after_dist, float("nan"), float("nan")
    _, mean, inv, logdet = cached
    def nll(value: np.ndarray) -> float:
        diff = np.asarray(value, np.float64) - mean
        return float(0.5 * (diff @ inv @ inv @ diff + logdet))
    return before_dist, after_dist, nll(anchor), nll(candidate)


__all__ = ["BuresBank", "bures_map", "anchor_excluded_indices", "anchor_excluded_neighbors", "knn_mean_distance", "class_support_radius", "matched_random_displacement", "target_affinity"]
