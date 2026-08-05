from __future__ import annotations

import numpy as np

from hsc_tta.contextual_risk.families import TPSFamily
from hsc_tta.contextual_risk.quantiles import higher_quantile


def inclusion_index_table(probabilities: np.ndarray) -> np.ndarray:
    """Return kappa(x,y) for every sample and possible class."""
    sets, _ = TPSFamily().build_sets(np.asarray(probabilities, dtype=float))
    table = np.full((len(sets), sets.shape[2]), 20, dtype=np.int16)
    for j in range(21):
        newly_included = sets[:, j, :] & (table == 20)
        table[newly_included] = j
    return table


def inclusion_indices(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    table = inclusion_index_table(probabilities)
    y = np.asarray(labels, dtype=int)
    if y.shape != (len(table),) or np.any((y < 0) | (y >= table.shape[1])):
        raise ValueError("invalid labels")
    return table[np.arange(len(y)), y]


def risk_curve_from_kappa(kappa: np.ndarray) -> np.ndarray:
    values = np.asarray(kappa, dtype=int)
    if values.ndim != 1 or len(values) == 0 or np.any((values < 0) | (values > 20)):
        raise ValueError("invalid inclusion indices")
    return np.asarray([np.mean(values > j) for j in range(21)], dtype=float)


def critical_index_from_kappa(kappa: np.ndarray, alpha: float) -> int:
    return int(np.clip(higher_quantile(np.asarray(kappa, dtype=float), 1.0 - alpha), 0, 20))


def risk_index_distribution(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    table = inclusion_index_table(p)
    distribution = np.zeros((len(p), 21), dtype=float)
    for j in range(21):
        distribution[:, j] = np.where(table == j, p, 0.0).sum(1)
    if not np.allclose(distribution.sum(1), 1.0, atol=1e-6):
        raise RuntimeError("risk-index distribution lost probability mass")
    return distribution


def risk_index_entropy(probabilities: np.ndarray) -> np.ndarray:
    distribution = risk_index_distribution(probabilities)
    return -(distribution * np.log(distribution + 1e-12)).sum(1)
