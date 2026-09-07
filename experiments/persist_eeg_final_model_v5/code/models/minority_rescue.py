"""Conservative baseline-error and minority-rescue utilities."""

from __future__ import annotations

import numpy as np


def eligibility(data, scope: str) -> np.ndarray:
    votes = (np.asarray(data.expert_logits) >= 0).astype(int)
    vote_count = votes.sum(axis=1)
    majority_size = np.maximum(vote_count, votes.shape[1] - vote_count)
    opposed = np.any(votes != np.asarray(data.current_prediction)[:, None], axis=1)
    if scope == "opposed":
        return opposed
    if scope == "three_two":
        return opposed & (majority_size == 3)
    if scope == "nonunanimous":
        return opposed & (majority_size < votes.shape[1])
    if scope == "uncertain_opposed":
        return opposed & (np.abs(np.asarray(data.current_probability) - 0.5) <= 0.20)
    raise ValueError(scope)


def switched_prediction(
    data,
    error_probability: np.ndarray,
    *,
    scope: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible = eligibility(data, scope)
    switch = eligible & (np.asarray(error_probability, dtype=float) >= float(threshold))
    prediction = np.asarray(data.current_prediction, dtype=int).copy()
    prediction[switch] = 1 - prediction[switch]
    probability = np.asarray(data.current_probability, dtype=float).copy()
    expert_probability = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(data.expert_logits), -40.0, 40.0)))
    for label in (0, 1):
        mask = switch & (prediction == label)
        if not mask.any():
            continue
        supporting = (data.expert_logits[mask] >= 0).astype(int) == label
        values = expert_probability[mask]
        numerator = np.sum(values * supporting, axis=1)
        denominator = np.maximum(supporting.sum(axis=1), 1)
        pooled = numerator / denominator
        probability[mask] = np.maximum(pooled, 0.5001) if label == 1 else np.minimum(pooled, 0.4999)
    return np.clip(probability, 1e-7, 1 - 1e-7), prediction, switch
