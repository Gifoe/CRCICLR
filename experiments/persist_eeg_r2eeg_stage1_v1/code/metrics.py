"""Metrics and representation-only diagnostics used after model selection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "BA": float(balanced_accuracy_score(y, pred)),
        "macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "recall_0": float(recall_score(y, pred, labels=[0], average=None, zero_division=0)[0]),
        "recall_1": float(recall_score(y, pred, labels=[1], average=None, zero_division=0)[0]),
    }


def geometry(source_z: np.ndarray, source_y: np.ndarray, future_z: np.ndarray, future_y: np.ndarray,
             future_logits: np.ndarray) -> dict[str, float]:
    def direction(z: np.ndarray, y: np.ndarray) -> np.ndarray:
        return z[y == 1].mean(0) - z[y == 0].mean(0)
    ds, dq = direction(source_z, source_y), direction(future_z, future_y)
    cosine = float(np.dot(ds, dq) / max(np.linalg.norm(ds) * np.linalg.norm(dq), 1e-12))
    means = [future_z[future_y == c].mean(0) for c in (0, 1)]
    separation = float(np.linalg.norm(means[1] - means[0]))
    scatter = float(np.mean(np.concatenate([np.sum((future_z[future_y == c] - means[c]) ** 2, axis=1) for c in (0, 1)])))
    labels = future_y.astype(int)
    margins = future_logits[np.arange(len(labels)), labels] - future_logits[np.arange(len(labels)), 1 - labels]
    return {"source_future_direction_cosine": cosine, "class_separation": separation,
            "within_class_scatter": scatter, "fisher": separation * separation / max(scatter, 1e-12),
            "classifier_logit_margin": float(np.mean(margins))}
