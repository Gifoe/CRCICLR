from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

from common import BOOTSTRAP_REPETITIONS, V4_SEED, stable_seed
from datasets import ExpertDataset


def ece(labels: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    labels = np.asarray(labels, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (probability < upper)
        if upper == 1.0:
            mask |= probability == 1.0
        if mask.any():
            value += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probability[mask].mean()))
    return value


def probability_metrics(labels: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    labels = np.asarray(labels, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "NLL": float(log_loss(labels, np.column_stack([1 - probability, probability]), labels=[0, 1])),
        "Brier": float(np.mean((probability - labels) ** 2)),
        "ECE": ece(labels, probability),
    }


def subject_results(
    data: ExpertDataset,
    method_id: str,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    rows = []
    base_prediction = data.base_prediction
    base_probability = data.base_probability
    for subject in sorted(np.unique(data.subjects).tolist()):
        mask = data.subjects == subject
        metric = probability_metrics(data.labels[mask], prediction[mask], probability[mask])
        base = probability_metrics(data.labels[mask], base_prediction[mask], base_probability[mask])
        rows.append(
            {
                "dataset": data.dataset_id,
                "method_id": method_id,
                "subject_id": subject,
                **metric,
                "base_balanced_accuracy": base["balanced_accuracy"],
                "delta_BA_vs_B_STRONG": metric["balanced_accuracy"] - base["balanced_accuracy"],
                "OUTER_TEST_USED": False,
            }
        )
    return pd.DataFrame(rows)


def paired_subject_bootstrap(values: np.ndarray, method_id: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable_seed(V4_SEED, "BOOTSTRAP", method_id))
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPETITIONS, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_method(
    data: ExpertDataset,
    method_id: str,
    prediction: np.ndarray,
    probability: np.ndarray,
    outer_fold: np.ndarray | None = None,
    selected_expert: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    subjects = subject_results(data, method_id, prediction, probability)
    delta = subjects.delta_BA_vs_B_STRONG.to_numpy(dtype=float)
    lower, upper = paired_subject_bootstrap(delta, method_id)
    labels = data.labels
    base = data.base_prediction
    changed = np.asarray(prediction, dtype=int) != base
    rescue = changed & (base != labels) & (np.asarray(prediction, dtype=int) == labels)
    harm = changed & (base == labels) & (np.asarray(prediction, dtype=int) != labels)
    global_metrics = probability_metrics(labels, prediction, probability)
    row: dict[str, Any] = {
        "dataset": data.dataset_id,
        "method_id": method_id,
        "subjects": int(len(subjects)),
        "mean_subject_BA": float(subjects.balanced_accuracy.mean()),
        "Delta_BA_vs_B_STRONG": float(delta.mean()),
        "CI95_L": lower,
        "CI95_U": upper,
        "accuracy": global_metrics["accuracy"],
        "macro_f1": float(subjects.macro_f1.mean()),
        "NLL": global_metrics["NLL"],
        "Brier": global_metrics["Brier"],
        "ECE": global_metrics["ECE"],
        "positive_subject_fraction": float(np.mean(delta > 0)),
        "nonnegative_subject_fraction": float(np.mean(delta >= 0)),
        "worst_subject_delta": float(delta.min()),
        "switch_rate": float(changed.mean()),
        "rescue_count": int(rescue.sum()),
        "harm_count": int(harm.sum()),
        "rescue_precision": float(rescue.sum() / changed.sum()) if changed.any() else 0.0,
        "harm_rate": float(harm.sum() / changed.sum()) if changed.any() else 0.0,
        "OUTER_TEST_USED": False,
    }
    fold_rows = []
    if outer_fold is not None:
        for fold in sorted(np.unique(outer_fold).tolist()):
            mask = np.asarray(outer_fold) == fold
            fold_subject = subjects[subjects.subject_id.isin(set(data.subjects[mask]))]
            fold_rows.append(
                {
                    "dataset": data.dataset_id,
                    "method_id": method_id,
                    "outer_fold": int(fold),
                    "subjects": int(fold_subject.subject_id.nunique()),
                    "mean_subject_BA": float(fold_subject.balanced_accuracy.mean()),
                    "Delta_BA_vs_B_STRONG": float(fold_subject.delta_BA_vs_B_STRONG.mean()),
                    "OUTER_TEST_USED": False,
                }
            )
        row["positive_fold_fraction"] = float(np.mean([item["Delta_BA_vs_B_STRONG"] > 0 for item in fold_rows]))
    else:
        row["positive_fold_fraction"] = 0.0
    if selected_expert is not None:
        row["selected_expert_count"] = int(pd.Series(selected_expert).nunique())
    return row, subjects, pd.DataFrame(fold_rows)
