"""Shrinkage error-covariance aggregation for a compact expert pool."""

from __future__ import annotations

import numpy as np


def weights(probability: np.ndarray, labels: np.ndarray, shrinkage: float) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    labels = np.asarray(labels, dtype=float)
    residual = probability - labels[:, None]
    covariance = np.cov(residual, rowvar=False, ddof=1)
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - float(shrinkage)) * covariance + float(shrinkage) * diagonal
    covariance += np.eye(covariance.shape[0]) * 1e-6
    inverse = np.linalg.pinv(covariance)
    one = np.ones(covariance.shape[0])
    value = inverse @ one
    value /= float(one @ value)
    # Negative GLS weights are unstable under subject/session shift.  The
    # non-negative projection keeps this a conservative probability pool.
    value = np.maximum(value, 0.0)
    if value.sum() <= 0:
        value = np.ones_like(value)
    return value / value.sum()


def aggregate(probability: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.asarray(probability, dtype=float) @ np.asarray(weight, dtype=float)
