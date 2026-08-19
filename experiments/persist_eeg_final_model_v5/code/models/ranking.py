"""Pairwise expert-ranking feature construction."""

from __future__ import annotations

import numpy as np


def pair_examples(x: np.ndarray, expert_logits: np.ndarray, labels: np.ndarray | None = None):
    x = np.asarray(x, dtype=np.float32)
    logits = np.asarray(expert_logits, dtype=np.float32)
    rows, targets, trial_indices, pairs = [], [], [], []
    expert_count = logits.shape[1]
    for left in range(expert_count):
        for right in range(left + 1, expert_count):
            differing = (logits[:, left] >= 0) != (logits[:, right] >= 0)
            indices = np.flatnonzero(differing)
            if not len(indices):
                continue
            pair_one_hot = np.zeros((len(indices), expert_count * (expert_count - 1) // 2), dtype=np.float32)
            pair_position = sum(expert_count - item - 1 for item in range(left)) + right - left - 1
            pair_one_hot[:, pair_position] = 1.0
            token = np.column_stack(
                [
                    logits[indices, left],
                    logits[indices, right],
                    np.abs(logits[indices, left]),
                    np.abs(logits[indices, right]),
                    logits[indices, left] - logits[indices, right],
                    pair_one_hot,
                ]
            )
            rows.append(np.column_stack([x[indices], token]))
            trial_indices.append(indices)
            pairs.extend([(left, right)] * len(indices))
            if labels is not None:
                targets.append(((logits[indices, left] >= 0).astype(int) == np.asarray(labels)[indices]).astype(int))
    matrix = np.concatenate(rows, axis=0)
    target = np.concatenate(targets, axis=0) if targets else None
    return matrix, target, np.concatenate(trial_indices), np.asarray(pairs, dtype=int)


def scores_from_pair_probability(
    row_count: int,
    expert_count: int,
    trial_indices: np.ndarray,
    pairs: np.ndarray,
    probability_left_correct: np.ndarray,
) -> np.ndarray:
    score = np.zeros((row_count, expert_count), dtype=float)
    count = np.zeros((row_count, expert_count), dtype=float)
    for index, (trial, pair, probability) in enumerate(zip(trial_indices, pairs, probability_left_correct)):
        del index
        left, right = map(int, pair)
        score[int(trial), left] += float(probability)
        score[int(trial), right] += 1.0 - float(probability)
        count[int(trial), left] += 1.0
        count[int(trial), right] += 1.0
    return np.divide(score, np.maximum(count, 1.0))
