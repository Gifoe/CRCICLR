"""Target-history-only subject-local heads in a frozen expert representation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def configurations() -> list[dict[str, Any]]:
    return [{"family": "logistic", "C": value} for value in (0.001, 0.01, 0.1, 1.0)]


def build(configuration: dict[str, Any], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "head",
                LogisticRegression(
                    C=float(configuration["C"]),
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def select_from_history(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sessions: np.ndarray,
    configurations_to_test: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select a head using S1->S2 and S2->S1 only."""
    records = []
    for order, configuration in enumerate(configurations_to_test):
        fold_ba, fold_nll = [], []
        for fit_session, validation_session in ((0, 1), (1, 0)):
            fit = sessions == fit_session
            validation = sessions == validation_session
            model = build(configuration, seed + order * 11 + fit_session)
            model.fit(embeddings[fit], labels[fit])
            probability = model.predict_proba(embeddings[validation])[:, 1]
            prediction = (probability >= 0.5).astype(int)
            fold_ba.append(float(balanced_accuracy_score(labels[validation], prediction)))
            fold_nll.append(float(log_loss(labels[validation], probability, labels=[0, 1])))
        records.append(
            {
                "configuration": configuration,
                "candidate_order": order,
                "history_cv_BA": float(np.mean(fold_ba)),
                "history_cv_worst_session_BA": float(np.min(fold_ba)),
                "history_cv_NLL": float(np.mean(fold_nll)),
            }
        )
    selected = max(
        records,
        key=lambda item: (
            item["history_cv_BA"],
            item["history_cv_worst_session_BA"],
            -item["history_cv_NLL"],
            -item["candidate_order"],
        ),
    )
    return dict(selected["configuration"]), records
