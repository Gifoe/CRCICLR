from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


class ScientificStop(RuntimeError):
    """Expected preregistered scientific stop (process exit code zero)."""


class TechnicalBlock(RuntimeError):
    """Technical inability to execute the frozen protocol (exit code two)."""


def fold_roles(evaluation_fold: int, n_folds: int = 5) -> tuple[list[int], int, int]:
    if not 0 <= evaluation_fold < n_folds:
        raise ValueError("invalid evaluation fold")
    validation_fold = (evaluation_fold + 1) % n_folds
    training_folds = [fold for fold in range(n_folds) if fold not in (evaluation_fold, validation_fold)]
    return training_folds, validation_fold, evaluation_fold


def normalized_inverse_frequency(labels: Sequence[int], classes: Sequence[int]) -> dict[int, float]:
    values = np.asarray(labels, dtype=int)
    counts = {int(label): int(np.sum(values == label)) for label in classes}
    if any(count == 0 for count in counts.values()):
        raise ValueError("training folds omit a task class")
    raw = {label: 1.0 / count for label, count in counts.items()}
    scale = len(raw) / sum(raw.values())
    return {label: weight * scale for label, weight in raw.items()}


def class_balanced_risk(y_true: Sequence[int], y_pred: Sequence[int], weights: Mapping[int, float]) -> float:
    truth = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if truth.shape != pred.shape:
        raise ValueError("truth/prediction shape mismatch")
    sample_weights = np.asarray([weights[int(label)] for label in truth], dtype=float)
    return float(np.sum(sample_weights * (truth != pred)) / np.sum(sample_weights))


def winner_shares(risks: np.ndarray, model_names: Sequence[str], atol: float = 1e-12) -> dict[str, float]:
    values = np.asarray(risks, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(model_names):
        raise ValueError("risks must be subject x model")
    shares = np.zeros(len(model_names), dtype=float)
    for row in values:
        mask = np.isclose(row, np.nanmin(row), atol=atol, rtol=0.0)
        shares += mask / mask.sum()
    shares /= len(values)
    return {name: float(value) for name, value in zip(model_names, shares)}


def rescuable_error(best_fixed_correct: Sequence[bool], alternative_correct: np.ndarray) -> float:
    base = np.asarray(best_fixed_correct, dtype=bool)
    alternatives = np.asarray(alternative_correct, dtype=bool)
    if alternatives.ndim != 2 or alternatives.shape[0] != len(base):
        raise ValueError("alternative matrix shape mismatch")
    wrong = ~base
    if not np.any(wrong):
        return float("nan")
    return float(np.mean(np.any(alternatives[wrong], axis=1)))


def gate_q(
    *,
    probability_sane: bool,
    embedding_sane: bool,
    all_classes_present: bool,
    nonconstant_subject_rate: float,
    seed_ba_std: float,
    dataset_ba: float,
    median_subject_ba: float,
    cbramod_ba: float,
    positive_seed_count: int,
    folds_noncollapsed: bool,
    n_classes: int,
) -> dict[str, bool]:
    chance = 1.0 / n_classes
    return {
        "Q1": bool(probability_sane and embedding_sane),
        "Q2": bool(all_classes_present),
        "Q3": bool(nonconstant_subject_rate >= 0.95),
        "Q4": bool(seed_ba_std <= 0.05),
        "Q5": bool(dataset_ba >= chance + 0.08),
        "Q6": bool(median_subject_ba >= chance + 0.05),
        "Q7": bool(dataset_ba >= cbramod_ba - 0.15),
        "Q8": bool(positive_seed_count >= 4),
        "Q9": bool(folds_noncollapsed),
    }


def paired_subject_bootstrap(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    repetitions: int = 5000,
    seed: int = 20260810,
) -> tuple[float, float, float]:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    if num.shape != den.shape or num.ndim != 1:
        raise ValueError("bootstrap inputs must be aligned subject vectors")
    observed = float(np.mean(num) / np.mean(den))
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        draw = rng.integers(0, len(num), len(num))
        estimates[index] = np.mean(num[draw]) / np.mean(den[draw])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return observed, float(low), float(high)
