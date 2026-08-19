from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

from common import BOOTSTRAP_DRAWS, V5_SEED, stable_seed
from datasets import V5Dataset


def ece(labels: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    labels = np.asarray(labels, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (probability < upper)
        if upper == 1.0:
            mask |= probability == 1.0
        if mask.any():
            result += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probability[mask].mean()))
    return result


def probability_metrics(labels: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "NLL": float(log_loss(labels, np.column_stack([1 - probability, probability]), labels=[0, 1])),
        "Brier": float(np.mean((probability - labels) ** 2)),
        "ECE": ece(labels, probability),
    }


def subject_table(
    data: V5Dataset,
    method_id: str,
    prediction: np.ndarray,
    probability: np.ndarray,
    *,
    baseline: str = "current",
) -> pd.DataFrame:
    if baseline == "current":
        base_prediction, base_probability = data.current_prediction, data.current_probability
    elif baseline == "static":
        base_prediction, base_probability = data.static_prediction, data.static_probability
    else:
        raise ValueError(baseline)
    rows = []
    for subject in sorted(np.unique(data.subjects).tolist()):
        mask = data.subjects == subject
        metric = probability_metrics(data.labels[mask], prediction[mask], probability[mask])
        reference = probability_metrics(data.labels[mask], base_prediction[mask], base_probability[mask])
        rows.append(
            {
                "dataset": data.dataset_id,
                "method_id": method_id,
                "subject_id": subject,
                **metric,
                "baseline": baseline,
                "baseline_BA": reference["balanced_accuracy"],
                "delta_BA": metric["balanced_accuracy"] - reference["balanced_accuracy"],
                "OUTER_TEST_USED": False,
            }
        )
    return pd.DataFrame(rows)


def paired_subject_bootstrap(values: np.ndarray, method_id: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable_seed(V5_SEED, "subject_bootstrap", method_id))
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(
    data: V5Dataset,
    method_id: str,
    prediction: np.ndarray,
    probability: np.ndarray,
    outer_fold: np.ndarray,
    *,
    baseline: str = "current",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    prediction = np.asarray(prediction, dtype=int)
    probability = np.asarray(probability, dtype=float)
    subjects = subject_table(data, method_id, prediction, probability, baseline=baseline)
    deltas = subjects.delta_BA.to_numpy(float)
    ci_low, ci_high = paired_subject_bootstrap(deltas, f"{data.dataset_id}:{method_id}:{baseline}")
    base_prediction = data.current_prediction if baseline == "current" else data.static_prediction
    changed = prediction != base_prediction
    rescue = changed & (base_prediction != data.labels) & (prediction == data.labels)
    harm = changed & (base_prediction == data.labels) & (prediction != data.labels)
    global_metric = probability_metrics(data.labels, prediction, probability)
    fold_rows = []
    for fold in sorted(np.unique(outer_fold).tolist()):
        fold_subjects = set(data.subjects[np.asarray(outer_fold) == fold])
        part = subjects.loc[subjects.subject_id.isin(fold_subjects)]
        fold_rows.append(
            {
                "dataset": data.dataset_id,
                "method_id": method_id,
                "outer_fold": int(fold),
                "subjects": int(len(part)),
                "mean_subject_BA": float(part.balanced_accuracy.mean()),
                "Delta_BA": float(part.delta_BA.mean()),
                "OUTER_TEST_USED": False,
            }
        )
    folds = pd.DataFrame(fold_rows)
    oracle_correct = np.any((data.expert_logits >= 0).astype(int) == data.labels[:, None], axis=1)
    baseline_wrong = base_prediction != data.labels
    available = int(np.sum(baseline_wrong & oracle_correct))
    recovered = int(rescue.sum())
    row = {
        "dataset": data.dataset_id,
        "method_id": method_id,
        "baseline": baseline,
        "subjects": int(len(subjects)),
        "mean_subject_BA": float(subjects.balanced_accuracy.mean()),
        "Delta_BA": float(deltas.mean()),
        "CI95_L": ci_low,
        "CI95_U": ci_high,
        "median_subject_delta": float(np.median(deltas)),
        "positive_subject_fraction": float(np.mean(deltas > 0)),
        "nonnegative_subject_fraction": float(np.mean(deltas >= 0)),
        "worst_subject_delta": float(deltas.min()),
        "positive_fold_fraction": float(np.mean(folds.Delta_BA > 0)),
        "accuracy": global_metric["accuracy"],
        "macro_f1": float(subjects.macro_f1.mean()),
        "NLL": global_metric["NLL"],
        "Brier": global_metric["Brier"],
        "ECE": global_metric["ECE"],
        "switch_rate": float(changed.mean()),
        "rescue_count": recovered,
        "harm_count": int(harm.sum()),
        "rescue_precision": float(recovered / changed.sum()) if changed.any() else 0.0,
        "oracle_headroom_recovered": float(recovered / available) if available else 0.0,
        "OUTER_TEST_USED": False,
    }
    return row, subjects, folds
