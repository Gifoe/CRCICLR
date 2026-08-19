"""Small parameter-count-bounded expert attention pooling utility."""

from __future__ import annotations

import numpy as np


def pool(expert_probability: np.ndarray, competence_score: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    score = np.asarray(competence_score, dtype=float) / max(float(temperature), 1e-6)
    score -= score.max(axis=1, keepdims=True)
    weight = np.exp(score)
    weight /= weight.sum(axis=1, keepdims=True)
    return np.sum(weight * np.asarray(expert_probability, dtype=float), axis=1)
