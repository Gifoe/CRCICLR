"""Leak-free expert-correctness and pairwise competence targets."""

from __future__ import annotations

import numpy as np


def multilabel_correctness(expert_logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return ((np.asarray(expert_logits) >= 0).astype(int) == np.asarray(labels, dtype=int)[:, None]).astype(np.float32)


def pairwise_preference(expert_logits: np.ndarray, labels: np.ndarray, left: int, right: int):
    predictions = (np.asarray(expert_logits) >= 0).astype(int)
    eligible = predictions[:, int(left)] != predictions[:, int(right)]
    target = predictions[eligible, int(left)] == np.asarray(labels, dtype=int)[eligible]
    return eligible, target.astype(np.int8)
