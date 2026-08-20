from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import BOOTSTRAP_DRAWS, stable_seed


def ece(labels: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    labels = np.asarray(labels, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probability >= lower) & (probability < upper)
        if index == bins - 1:
            mask |= probability == 1.0
        if mask.any():
            result += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probability[mask].mean()))
    return result


def basic_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    prediction = (probability >= 0.5).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "NLL": float(log_loss(labels, probability, labels=[0, 1])),
        "Brier": float(np.mean((probability - labels) ** 2)),
        "ECE": ece(labels, probability),
    }


def subject_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject, group in predictions.groupby("subject_id", sort=True):
        row = basic_metrics(group.label.to_numpy(int), group.probability.to_numpy(float))
        rows.append(
            {
                "benchmark": str(group.benchmark.iloc[0]),
                "method_id": str(group.method_id.iloc[0]),
                "subject_id": str(subject),
                "outer_fold": int(group.outer_fold.iloc[0]),
                **row,
                "OUTER_TEST_USED": False,
            }
        )
    return pd.DataFrame(rows)


def _paired_ci(values: np.ndarray, key: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable_seed("v6-subject-bootstrap", key))
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(predictions: pd.DataFrame, reference: pd.DataFrame | None = None) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    required = {"benchmark", "method_id", "trial_uid", "subject_id", "outer_fold", "label", "probability"}
    if not required.issubset(predictions.columns) or predictions.trial_uid.duplicated().any():
        raise RuntimeError("Malformed V6 prediction table")
    subjects = subject_metrics(predictions)
    global_metrics = basic_metrics(predictions.label.to_numpy(int), predictions.probability.to_numpy(float))
    method = str(predictions.method_id.iloc[0])
    benchmark = str(predictions.benchmark.iloc[0])
    result: dict[str, object] = {
        "benchmark": benchmark,
        "method_id": method,
        "subjects": int(len(subjects)),
        "mean_subject_BA": float(subjects.balanced_accuracy.mean()),
        "accuracy": global_metrics["accuracy"],
        "macro_f1": float(subjects.macro_f1.mean()),
        "NLL": global_metrics["NLL"],
        "Brier": global_metrics["Brier"],
        "ECE": global_metrics["ECE"],
        "exploratory": True,
        "target_future_labels_used_for_fit": False,
        "OUTER_TEST_USED": False,
    }
    fold_rows = []
    for fold, group in subjects.groupby("outer_fold", sort=True):
        fold_rows.append(
            {
                "benchmark": benchmark,
                "method_id": method,
                "outer_fold": int(fold),
                "subjects": int(len(group)),
                "mean_subject_BA": float(group.balanced_accuracy.mean()),
                "OUTER_TEST_USED": False,
            }
        )
    folds = pd.DataFrame(fold_rows)
    if reference is not None:
        if reference.trial_uid.duplicated().any() or set(reference.trial_uid) != set(predictions.trial_uid):
            raise RuntimeError("Reference prediction coverage mismatch")
        aligned = reference.set_index("trial_uid").loc[predictions.trial_uid].reset_index()
        if not np.array_equal(aligned.label.to_numpy(int), predictions.label.to_numpy(int)):
            raise RuntimeError("Reference labels are not aligned")
        ref_subjects = subject_metrics(aligned)
        ref_ba = ref_subjects.set_index("subject_id").loc[subjects.subject_id, "balanced_accuracy"].to_numpy(float)
        delta = subjects.balanced_accuracy.to_numpy(float) - ref_ba
        ci_low, ci_high = _paired_ci(delta, f"{benchmark}:{method}:{aligned.method_id.iloc[0]}")
        subjects["reference_method_id"] = str(aligned.method_id.iloc[0])
        subjects["reference_BA"] = ref_ba
        subjects["delta_BA"] = delta
        fold_delta = subjects.groupby("outer_fold").delta_BA.mean()
        result.update(
            {
                "reference_method_id": str(aligned.method_id.iloc[0]),
                "Delta_BA": float(delta.mean()),
                "CI95_L": ci_low,
                "CI95_U": ci_high,
                "median_subject_delta": float(np.median(delta)),
                "positive_subject_fraction": float(np.mean(delta > 0)),
                "nonnegative_subject_fraction": float(np.mean(delta >= 0)),
                "worst_subject_delta": float(delta.min()),
                "positive_fold_fraction": float(np.mean(fold_delta > 0)),
                "positive_folds": int(np.sum(fold_delta > 0)),
            }
        )
        folds["Delta_BA"] = folds.outer_fold.map(fold_delta.to_dict())
    return result, subjects, folds
