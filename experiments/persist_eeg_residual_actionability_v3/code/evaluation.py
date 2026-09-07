from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from v3_common import AUDIT_SEED, BOOTSTRAP_REPETITIONS, stable_seed


def binary_metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def subject_metric_table(
    trials: pd.DataFrame,
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    method_id: str,
    pool: str,
) -> pd.DataFrame:
    labels = trials.outcome_label.to_numpy(dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    reference = np.asarray(reference, dtype=int)
    rows: list[dict[str, Any]] = []
    for subject, indices in trials.groupby("subject_id", sort=True).indices.items():
        idx = np.asarray(indices, dtype=int)
        metrics = binary_metrics(labels[idx], prediction[idx])
        base = binary_metrics(labels[idx], reference[idx])
        rows.append(
            {
                "pool": pool,
                "method_id": method_id,
                "subject_id": str(subject),
                "trials": int(len(idx)),
                **metrics,
                "reference_balanced_accuracy": base["balanced_accuracy"],
                "reference_accuracy": base["accuracy"],
                "reference_macro_f1": base["macro_f1"],
                "delta_BA_vs_B6": metrics["balanced_accuracy"] - base["balanced_accuracy"],
                "delta_accuracy_vs_B6": metrics["accuracy"] - base["accuracy"],
                "delta_macro_f1_vs_B6": metrics["macro_f1"] - base["macro_f1"],
                "OUTER_TEST_USED": False,
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap(values: np.ndarray, *seed_parts: object) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired subject bootstrap requires finite subject values")
    rng = np.random.default_rng(stable_seed(AUDIT_SEED, *seed_parts))
    draws = rng.choice(values, size=(BOOTSTRAP_REPETITIONS, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def summarize_subject_table(subjects: pd.DataFrame, method_id: str, pool: str) -> dict[str, Any]:
    values = subjects.delta_BA_vs_B6.to_numpy(dtype=float)
    low, high = paired_bootstrap(values, method_id, pool)
    return {
        "pool": pool,
        "method_id": method_id,
        "subjects": int(len(subjects)),
        "mean_subject_BA": float(subjects.balanced_accuracy.mean()),
        "mean_subject_accuracy": float(subjects.accuracy.mean()),
        "mean_subject_macro_f1": float(subjects.macro_f1.mean()),
        "mean_subject_delta_BA_vs_B6": float(values.mean()),
        "median_subject_delta_BA_vs_B6": float(np.median(values)),
        "bootstrap_CI95_L": low,
        "bootstrap_CI95_U": high,
        "positive_subject_fraction": float(np.mean(values > 0)),
        "nonnegative_subject_fraction": float(np.mean(values >= 0)),
        "worst_subject_delta_BA_vs_B6": float(values.min()),
        "OUTER_TEST_USED": False,
    }


def evaluate_prediction(
    trials: pd.DataFrame,
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    method_id: str,
    pool: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    subjects = subject_metric_table(
        trials,
        prediction,
        reference,
        method_id=method_id,
        pool=pool,
    )
    return summarize_subject_table(subjects, method_id, pool), subjects


def by_session_class(
    trials: pd.DataFrame,
    prediction: np.ndarray,
    reference: np.ndarray,
    method_id: str,
) -> pd.DataFrame:
    labels = trials.outcome_label.to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for session, indices in trials.groupby("session_id").indices.items():
        idx = np.asarray(indices, dtype=int)
        metrics = binary_metrics(labels[idx], prediction[idx])
        base = binary_metrics(labels[idx], reference[idx])
        rows.append(
            {
                "method_id": method_id,
                "stratum_type": "session",
                "stratum": str(session),
                "trials": int(len(idx)),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "delta_metric_vs_B6": metrics["balanced_accuracy"] - base["balanced_accuracy"],
                "metric": "balanced_accuracy",
            }
        )
    for label in sorted(np.unique(labels)):
        idx = np.flatnonzero(labels == label)
        accuracy = float(np.mean(np.asarray(prediction)[idx] == labels[idx]))
        base = float(np.mean(np.asarray(reference)[idx] == labels[idx]))
        rows.append(
            {
                "method_id": method_id,
                "stratum_type": "class",
                "stratum": str(int(label)),
                "trials": int(len(idx)),
                "balanced_accuracy": np.nan,
                "delta_metric_vs_B6": accuracy - base,
                "metric": "class_recall",
            }
        )
    return pd.DataFrame(rows).assign(OUTER_TEST_USED=False)


def concentration_rows(
    trials: pd.DataFrame,
    subject_table: pd.DataFrame,
    rescue: np.ndarray,
    method_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positive_subject = np.maximum(subject_table.delta_BA_vs_B6.to_numpy(dtype=float), 0.0)
    subject_order = np.sort(positive_subject)[::-1]
    rescue = np.asarray(rescue, dtype=int)
    trial_order = np.sort(rescue)[::-1]
    for fraction in (0.01, 0.05, 0.10, 0.20):
        subject_n = max(1, int(np.ceil(fraction * len(subject_order))))
        trial_n = max(1, int(np.ceil(fraction * len(trial_order))))
        rows.extend(
            [
                {
                    "method_id": method_id,
                    "unit": "subject",
                    "top_fraction": fraction,
                    "units_selected": subject_n,
                    "fraction_of_positive_gain": (
                        float(subject_order[:subject_n].sum() / subject_order.sum())
                        if subject_order.sum() > 0
                        else np.nan
                    ),
                    "OUTER_TEST_USED": False,
                },
                {
                    "method_id": method_id,
                    "unit": "trial",
                    "top_fraction": fraction,
                    "units_selected": trial_n,
                    "fraction_of_positive_gain": (
                        float(trial_order[:trial_n].sum() / trial_order.sum())
                        if trial_order.sum() > 0
                        else np.nan
                    ),
                    "OUTER_TEST_USED": False,
                },
            ]
        )
    return rows
