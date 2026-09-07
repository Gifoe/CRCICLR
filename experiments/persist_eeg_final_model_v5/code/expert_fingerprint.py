"""Train-only expert reliability fingerprints used by competence models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, log_loss

from common import sigmoid


def build(logits: np.ndarray, labels: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    subjects = np.asarray(subjects).astype(str)
    probability = sigmoid(logits)
    prediction = (logits >= 0).astype(int)
    rows = []
    for expert in range(logits.shape[1]):
        subject_ba = []
        for subject in np.unique(subjects):
            mask = subjects == subject
            subject_ba.append(balanced_accuracy_score(labels[mask], prediction[mask, expert]))
        rows.append(
            [
                balanced_accuracy_score(labels, prediction[:, expert]),
                np.mean(prediction[labels == 1, expert] == 1),
                np.mean(prediction[labels == 0, expert] == 0),
                log_loss(labels, probability[:, expert], labels=[0, 1]),
                np.mean((probability[:, expert] - labels) ** 2),
                np.mean(subject_ba),
                np.std(subject_ba, ddof=0),
            ]
        )
    return np.asarray(rows, dtype=np.float32)
