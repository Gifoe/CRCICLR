"""Classification and post-freeze geometry helpers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score


def score(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "BA": float(balanced_accuracy_score(y, pred)),
        "macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "recall_0": float(recall_score(y, pred, labels=[0], average=None, zero_division=0)[0]),
        "recall_1": float(recall_score(y, pred, labels=[1], average=None, zero_division=0)[0]),
    }


# The frozen Stage-1 loader imports these names at module import time.  This
# screening runner does not use its geometry implementation, but preserving the
# import surface lets it reuse only the immutable split/cache helpers.
classification_metrics = score


def geometry(source_z: np.ndarray, source_y: np.ndarray, future_z: np.ndarray, future_y: np.ndarray, future_logits: np.ndarray) -> dict[str, float]:
    ds = source_z[source_y == 1].mean(0) - source_z[source_y == 0].mean(0)
    dq = future_z[future_y == 1].mean(0) - future_z[future_y == 0].mean(0)
    return {"source_future_direction_cosine": float(ds @ dq / max(np.linalg.norm(ds) * np.linalg.norm(dq), 1e-12))}


def direction(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = z[y == 1].mean(axis=0) - z[y == 0].mean(axis=0)
    return d / max(float(np.linalg.norm(d)), 1e-12)


def representation_stats(records: dict[str, dict[str, np.ndarray]], key: str) -> dict[str, float]:
    centers = np.stack([row[key].mean(axis=0) for row in records.values()])
    between = float(np.mean(np.var(centers, axis=0)))
    within = float(np.mean(np.concatenate([np.mean((row[key] - row[key].mean(axis=0)) ** 2, axis=1) for row in records.values()])))
    separation = float(np.mean([np.linalg.norm(row[key][row["y"] == 1].mean(0) - row[key][row["y"] == 0].mean(0)) for row in records.values()]))
    return {"between_subject_center_variance": between, "within_subject_scatter": within, "mean_class_separation": separation}
