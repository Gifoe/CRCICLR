"""Fixed-complexity TRAIN-only linear coordinate-transform families.

All matrices use row-vector convention. High-dimensional orthogonal and
low-rank maps act in a TRAIN-derived at-most-32-dimensional orthonormal support
and are identity on its complement; no D-by-D affine matrix is fitted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.utils.extmath import randomized_svd


RANKS = (1, 2, 4, 8)


def _basis(pool: np.ndarray, max_rank: int = 32) -> np.ndarray:
    centered = np.asarray(pool - pool.mean(axis=0), dtype=np.float32)
    rank = min(max_rank, centered.shape[0] - 1, centered.shape[1])
    if rank < 1:
        raise RuntimeError("insufficient TRAIN observations for canonical support")
    _, singular, vt = randomized_svd(centered, n_components=rank, n_iter=5, random_state=0)
    keep = singular > max(float(singular[0]) * 1e-6, 1e-8)
    basis = vt[keep].T.astype(np.float32)
    if not basis.shape[1]:
        raise RuntimeError("canonical support has zero rank")
    return basis


def _covariance(x: np.ndarray, ridge: float = 1e-5) -> np.ndarray:
    v = np.asarray(x - x.mean(axis=0), dtype=np.float64)
    return (v.T @ v / max(len(v) - 1, 1) + ridge * np.eye(x.shape[1]))


def _spectral_power(cov: np.ndarray, exponent: float) -> np.ndarray:
    eig, vec = np.linalg.eigh(cov)
    return (vec * np.maximum(eig, 1e-8) ** exponent) @ vec.T


def _fixed_sign_eig(cov: np.ndarray) -> np.ndarray:
    eig, vec = np.linalg.eigh(cov)
    vec = vec[:, np.argsort(eig)[::-1]]
    for column in range(vec.shape[1]):
        point = int(np.argmax(np.abs(vec[:, column])))
        if vec[point, column] < 0:
            vec[:, column] *= -1
    return vec


@dataclass
class Transform:
    family: str
    mode: str
    source_mean: np.ndarray
    target_mean: np.ndarray
    diagonal: np.ndarray | None = None
    basis: np.ndarray | None = None
    support_map: np.ndarray | None = None
    paired_class_centroids_used: bool = False

    def apply(self, x: np.ndarray) -> np.ndarray:
        if self.family == "identity":
            return np.asarray(x, dtype=np.float32)
        centered = np.asarray(x - self.target_mean, dtype=np.float32)
        if self.family == "translation":
            return (centered + self.source_mean).astype(np.float32)
        if self.family == "diagonal_affine":
            return (centered * self.diagonal + self.source_mean).astype(np.float32)
        if self.basis is None or self.support_map is None:
            raise RuntimeError("missing TRAIN-derived canonical support")
        local = centered @ self.basis
        delta = (local @ (self.support_map - np.eye(self.support_map.shape[0]))) @ self.basis.T
        return (centered + delta + self.source_mean).astype(np.float32)

    def complexity(self) -> dict:
        shift = float(np.linalg.norm(self.source_mean - self.target_mean))
        if self.family == "identity":
            change = 0.0
        elif self.family == "translation":
            change = shift
        elif self.family == "diagonal_affine":
            change = float(np.linalg.norm(self.diagonal - 1))
        else:
            change = float(np.linalg.norm(self.support_map - np.eye(len(self.support_map))))
        return {"deviation_from_identity": change, "translation_norm": shift,
                "support_rank": 0 if self.basis is None else int(self.basis.shape[1]),
                "paired_class_centroids_used": self.paired_class_centroids_used}


def fit_unsupervised(source: np.ndarray, target: np.ndarray, family: str,
                     *, support_pool: np.ndarray | None = None) -> Transform:
    """Fit distributional moments only; no subject/class pairing or labels."""
    source, target = np.asarray(source, dtype=np.float32), np.asarray(target, dtype=np.float32)
    if source.shape[1] != target.shape[1] or min(len(source), len(target)) < 3:
        raise RuntimeError("insufficient or incompatible unlabeled TRAIN statistics")
    ms, mt = source.mean(0), target.mean(0)
    if family == "identity":
        return Transform(family, "UNSUPERVISED", np.zeros_like(ms), np.zeros_like(mt))
    if family == "translation":
        return Transform(family, "UNSUPERVISED", ms, mt)
    if family == "diagonal_affine":
        vs, vt = source.var(0), target.var(0)
        ridge = max(float(np.median(vs + vt)) * .1, 1e-6)
        diagonal = np.clip(np.sqrt((vs + ridge) / (vt + ridge)), .5, 2).astype(np.float32)
        return Transform(family, "UNSUPERVISED", ms, mt, diagonal=diagonal)
    pool = np.concatenate((source, target)) if support_pool is None else support_pool
    basis = _basis(pool, max_rank=min(32, len(source) - 1, len(target) - 1))
    xs, xt = (source - ms) @ basis, (target - mt) @ basis
    covs, covt = _covariance(xs), _covariance(xt)
    k = basis.shape[1]
    if family == "orthogonal_translation":
        # Matching covariance eigenspaces is label-free and sign-deterministic.
        es, et = _fixed_sign_eig(covs), _fixed_sign_eig(covt)
        mapping = et @ es.T
        if np.max(np.abs(mapping.T @ mapping - np.eye(k))) > 1e-5:
            raise RuntimeError("unsupervised Procrustes map not orthogonal")
        return Transform(family, "UNSUPERVISED", ms, mt, basis=basis, support_map=mapping.astype(np.float32))
    if family.startswith("lowrank_residual_affine_r"):
        rank = int(family.rsplit("r", 1)[-1])
        if rank not in RANKS:
            raise ValueError(family)
        # Row-vector covariance coloring, regularized, then truncated residual.
        color = _spectral_power(covt, -.5) @ _spectral_power(covs, .5)
        delta = color - np.eye(k)
        u, singular, vt = np.linalg.svd(delta, full_matrices=False)
        use = min(rank, k)
        mapping = np.eye(k) + (u[:, :use] * singular[:use]) @ vt[:use]
        return Transform(family, "UNSUPERVISED", ms, mt, basis=basis, support_map=mapping.astype(np.float32))
    raise KeyError(family)


def fit_oracle(source_paired: np.ndarray, target_paired: np.ndarray, family: str,
               *, unlabeled_support_pool: np.ndarray) -> Transform:
    """TRAIN subject-class centroid correspondence; oracle diagnostic only."""
    source, target = np.asarray(source_paired, dtype=np.float32), np.asarray(target_paired, dtype=np.float32)
    if source.shape != target.shape or len(source) < 4:
        raise RuntimeError("oracle TRAIN correspondence shape invalid")
    ms, mt = source.mean(0), target.mean(0)
    if family == "identity":
        return Transform(family, "LABEL_ASSISTED_ORACLE", np.zeros_like(ms), np.zeros_like(mt),
                         paired_class_centroids_used=True)
    if family == "translation":
        return Transform(family, "LABEL_ASSISTED_ORACLE", ms, mt, paired_class_centroids_used=True)
    if family == "diagonal_affine":
        xs, ys = target - mt, source - ms
        numerator = (xs * ys).sum(axis=0)
        denominator = (xs * xs).sum(axis=0)
        ridge = max(float(np.median(denominator)) * .1, 1e-6)
        diag = np.clip((numerator + ridge) / (denominator + ridge), .5, 2).astype(np.float32)
        return Transform(family, "LABEL_ASSISTED_ORACLE", ms, mt, diagonal=diag,
                         paired_class_centroids_used=True)
    basis = _basis(unlabeled_support_pool, max_rank=min(32, len(source) - 1, len(target) - 1))
    xt, ys = (target - mt) @ basis, (source - ms) @ basis
    k = basis.shape[1]
    if family == "orthogonal_translation":
        u, _, vt = np.linalg.svd(xt.T @ ys, full_matrices=False)
        mapping = u @ vt
        if np.max(np.abs(mapping.T @ mapping - np.eye(k))) > 1e-5:
            raise RuntimeError("oracle Procrustes map not orthogonal")
    elif family.startswith("lowrank_residual_affine_r"):
        rank = int(family.rsplit("r", 1)[-1])
        if rank not in RANKS:
            raise ValueError(family)
        ridge = max(float(np.trace(xt.T @ xt)) / k * .1, 1e-6)
        delta = np.linalg.solve(xt.T @ xt + ridge * np.eye(k), xt.T @ (ys - xt))
        u, singular, vt = np.linalg.svd(delta, full_matrices=False)
        use = min(rank, k)
        mapping = np.eye(k) + (u[:, :use] * singular[:use]) @ vt[:use]
    else:
        raise KeyError(family)
    return Transform(family, "LABEL_ASSISTED_ORACLE", ms, mt,
                     basis=basis, support_map=mapping.astype(np.float32), paired_class_centroids_used=True)


def families() -> tuple[str, ...]:
    return ("identity", "translation", "diagonal_affine", "orthogonal_translation") + tuple(
        f"lowrank_residual_affine_r{r}" for r in RANKS)
