"""Compact subject-specific CSP classifier for legal prior-session adaptation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def spatial_filters(covariance: np.ndarray, labels: np.ndarray, pairs: int) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    class_covariance = [covariance[labels == label].mean(axis=0) for label in (0, 1)]
    channels = covariance.shape[1]
    scale = np.trace(class_covariance[0] + class_covariance[1]) / channels
    ridge = np.eye(channels) * max(float(scale) * 1e-4, 1e-8)
    _, vectors = eigh(class_covariance[1] + ridge, class_covariance[0] + class_covariance[1] + 2 * ridge)
    selected = np.r_[np.arange(int(pairs)), np.arange(channels - int(pairs), channels)]
    return np.asarray(vectors[:, selected], dtype=np.float64)


def features(covariance: np.ndarray, filters: np.ndarray) -> np.ndarray:
    variance = np.einsum("cf,ncd,df->nf", filters, covariance, filters, optimize=True)
    variance = np.maximum(variance, 1e-12)
    variance /= np.maximum(variance.sum(axis=1, keepdims=True), 1e-12)
    return np.log(variance).astype(np.float32)


def build_head(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "head",
                LogisticRegression(
                    C=float(c_value),
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def select_history_configuration(
    covariance: np.ndarray,
    labels: np.ndarray,
    sessions: np.ndarray,
    configurations: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = []
    for order, configuration in enumerate(configurations):
        fold_ba, fold_nll = [], []
        for train_session, validation_session in ((0, 1), (1, 0)):
            train = sessions == train_session
            validation = sessions == validation_session
            filters = spatial_filters(covariance[train], labels[train], int(configuration["pairs"]))
            x_train = features(covariance[train], filters)
            x_validation = features(covariance[validation], filters)
            model = build_head(float(configuration["C"]), seed + order * 13 + train_session)
            model.fit(x_train, labels[train])
            probability = model.predict_proba(x_validation)[:, 1]
            prediction = (probability >= 0.5).astype(int)
            fold_ba.append(float(balanced_accuracy_score(labels[validation], prediction)))
            fold_nll.append(float(log_loss(labels[validation], probability, labels=[0, 1])))
        records.append(
            {
                "configuration": dict(configuration),
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
