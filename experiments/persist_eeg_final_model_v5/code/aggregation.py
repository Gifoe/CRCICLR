from __future__ import annotations

import numpy as np


def anchored_postprocess(
    base_probability: np.ndarray,
    base_prediction: np.ndarray,
    candidate_probability: np.ndarray,
    expert_logits: np.ndarray,
    *,
    alpha: float,
    gate: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    base_probability = np.asarray(base_probability, dtype=float)
    candidate_probability = np.asarray(candidate_probability, dtype=float)
    probability = base_probability + float(alpha) * (candidate_probability - base_probability)
    votes = (np.asarray(expert_logits) >= 0).astype(int)
    majority = np.maximum(votes.sum(axis=1), votes.shape[1] - votes.sum(axis=1))
    if gate == "all":
        eligible = np.ones(len(probability), dtype=bool)
    elif gate == "not_unanimous":
        eligible = majority < votes.shape[1]
    elif gate == "max_disagreement":
        eligible = majority == (votes.shape[1] // 2 + 1)
    elif gate == "current_uncertain_010":
        eligible = np.abs(base_probability - 0.5) <= 0.10
    elif gate == "current_uncertain_020":
        eligible = np.abs(base_probability - 0.5) <= 0.20
    else:
        raise ValueError(gate)
    prediction = np.asarray(base_prediction, dtype=int).copy()
    prediction[eligible] = (probability[eligible] >= float(threshold)).astype(int)
    probability = np.where(eligible, probability, base_probability)
    return np.clip(probability, 1e-7, 1 - 1e-7), prediction
