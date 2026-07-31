from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2 or p.shape[0] == 0:
        raise ValueError("probabilities must have shape (n_samples, n_classes), n_classes >= 2")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("probabilities must be finite and within [0, 1]")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each probability row must sum to 1")
    return p


def prediction_sets(probabilities: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    p = _validate_probabilities(probabilities)
    grid = np.asarray(lambdas, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or np.any(~np.isfinite(grid)) or np.any((grid < 0) | (grid > 1)):
        raise ValueError("lambdas must be a non-empty finite vector in [0, 1]")
    included = p[:, None, :] >= (1.0 - grid[None, :, None])
    argmax = p.argmax(axis=1)
    included[np.arange(p.shape[0])[:, None], np.arange(grid.size)[None, :], argmax[:, None]] = True
    return included


def evaluate_prediction_sets(probabilities: np.ndarray, labels: np.ndarray, lambdas: np.ndarray) -> list[dict[str, float]]:
    p = _validate_probabilities(probabilities)
    y = np.asarray(labels, dtype=int)
    if y.shape != (p.shape[0],) or np.any((y < 0) | (y >= p.shape[1])):
        raise ValueError("labels must match samples and class range")
    sets = prediction_sets(p, lambdas)
    pred = p.argmax(axis=1)
    classes = np.arange(p.shape[1])
    argmax_error = float(np.mean(pred != y))
    macro_f1 = float(f1_score(y, pred, labels=classes, average="macro", zero_division=0))
    balanced = float(balanced_accuracy_score(y, pred))
    rows = []
    for j, lam in enumerate(np.asarray(lambdas, dtype=float)):
        membership = sets[np.arange(y.size), j, y]
        sizes = sets[:, j, :].sum(axis=1)
        rows.append({
            "lambda": float(lam),
            "future_risk": float(np.mean(~membership)),
            "argmax_error": argmax_error,
            "average_set_size": float(np.mean(sizes)),
            "singleton_rate": float(np.mean(sizes == 1)),
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced,
        })
    return rows

