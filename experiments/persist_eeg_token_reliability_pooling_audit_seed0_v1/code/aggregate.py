"""Frozen-token pooling and subject-level linear-decoder evaluation.

This module never reads EEG files. All fitted transforms, including scaling,
are fitted only on supplied TRAIN source rows.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler


C_GRID = (.001, .01, .1, 1., 10.)
Q_GRID = (.10, .25, .50, .75)
TAU_GRID = (.25, .5, 1., 2., 4.)


def stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def rank_subset(score: np.ndarray, fraction: float) -> np.ndarray:
    if fraction not in Q_GRID or score.ndim != 1 or not np.isfinite(score).all():
        raise ValueError("invalid TRAIN-only rank score or subset fraction")
    count = max(1, int(np.ceil(len(score) * fraction)))
    return np.argsort(-score, kind="stable")[:count]


def zscore_tokens(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    std = float(value.std())
    if std <= 1e-12:
        raise ValueError("constant TRAIN token score")
    return (value - float(value.mean())) / std


def pool(z: np.ndarray, method: str, *, subset: np.ndarray | None = None,
         score: np.ndarray | None = None, tau: float | None = None) -> np.ndarray:
    if z.ndim != 3 or not np.isfinite(z).all():
        raise ValueError("expected finite [trial,token,d_model] tensor")
    if method == "MEAN_ALL":
        return np.mean(z, axis=1)
    if method == "FLATTEN_LINEAR":
        return z.reshape(len(z), -1)
    if method in ("ENERGY_TOPK", "RANDOM_TOPK", "RELIABLE_TOPK", "UTILITY_TOPK", "RELIABLE_UTILITY_TOPK"):
        if subset is None or len(subset) < 1 or len(subset) > z.shape[1]:
            raise ValueError("missing or invalid frozen subset")
        return np.mean(z[:, subset], axis=1)
    if method in ("RELIABILITY_WEIGHTED", "RU_WEIGHTED"):
        if score is None or tau not in TAU_GRID or len(score) != z.shape[1]:
            raise ValueError("missing frozen TRAIN score or temperature")
        weight = np.exp(tau * zscore_tokens(score))
        weight /= weight.sum()
        return np.einsum("nkd,k->nd", z, weight, optimize=True)
    if method in ("RU_TWO_STREAM", "ENERGY_TWO_STREAM", "RANDOM_TWO_STREAM"):
        if subset is None or not 0 < len(subset) < z.shape[1]:
            raise ValueError("invalid two-stream split")
        complement = np.ones(z.shape[1], dtype=bool)
        complement[subset] = False
        if complement.sum() == 0:
            raise ValueError("two-stream complement is empty")
        return np.concatenate((z[:, subset].mean(axis=1), z[:, complement].mean(axis=1)), axis=1)
    raise KeyError(method)


def fit_decoder(x: np.ndarray, y: np.ndarray, c: float) -> tuple[StandardScaler, LogisticRegression]:
    if c not in C_GRID or x.ndim != 2 or set(np.unique(y)) != {0, 1}:
        raise ValueError("invalid deterministic logistic decoder fit")
    scaler = StandardScaler(copy=True)
    scaled = scaler.fit_transform(x)
    model = LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=1000,
                               tol=1e-6, random_state=0)
    model.fit(scaled, y)
    return scaler, model


def predict_decoder(fitted: tuple[StandardScaler, LogisticRegression], x: np.ndarray) -> np.ndarray:
    scaler, model = fitted
    return model.predict_proba(scaler.transform(x))


def subject_metrics(y: np.ndarray, probability: np.ndarray, subjects: np.ndarray) -> list[dict[str, float | str | int]]:
    if probability.shape != (len(y), 2) or len(subjects) != len(y):
        raise ValueError("prediction inventory mismatch")
    rows = []
    for subject in sorted(set(subjects.tolist())):
        mask = subjects == subject
        if set(np.unique(y[mask])) != {0, 1}:
            raise ValueError(f"both MI classes required for subject-equal metrics: {subject}")
        p = probability[mask]
        pred = np.argmax(p, axis=1)
        rows.append({"subject": str(subject), "trials": int(mask.sum()),
                     "BA": float(balanced_accuracy_score(y[mask], pred)),
                     "macro_F1": float(f1_score(y[mask], pred, average="macro", zero_division=0)),
                     "NLL": float(log_loss(y[mask], np.clip(p, 1e-12, 1-1e-12), labels=[0, 1]))})
    return rows


def subject_equal(rows: list[dict]) -> dict[str, float]:
    if not rows:
        raise ValueError("no biological subjects")
    return {key: float(np.mean([row[key] for row in rows])) for key in ("BA", "macro_F1", "NLL")}


def train_loso_decoder(z: np.ndarray, y: np.ndarray, subjects: np.ndarray, sessions: np.ndarray,
                       *, method: str, c: float, subset: np.ndarray | None = None,
                       score: np.ndarray | None = None, tau: float | None = None) -> list[dict]:
    """Subject-grouped source-S1 fit / unseen TRAIN-subject-S2 evaluation.

    This helper is for methods whose subset/score has already been computed
    without the held subject. The caller must not pass a full-TRAIN score into
    this function for an unbiased score-selection validation.
    """
    rows = []
    for held in sorted(set(subjects.tolist())):
        fit = (subjects != held) & (sessions == 1)
        test = (subjects == held) & (sessions == 2)
        if not fit.any() or not test.any():
            raise ValueError("empty TRAIN LOSO source/held split")
        source = pool(z[fit], method, subset=subset, score=score, tau=tau)
        future = pool(z[test], method, subset=subset, score=score, tau=tau)
        fitted = fit_decoder(source, y[fit], c)
        scores = predict_decoder(fitted, future)
        rows.extend(subject_metrics(y[test], scores, subjects[test]))
    return rows


def paired_subject_bootstrap(a: dict[str, float], b: dict[str, float], *, draws: int = 20_000,
                             seed: int = 0) -> dict[str, float]:
    if draws < 20_000 or a.keys() != b.keys():
        raise ValueError("paired subject bootstrap requires 20000 draws and matched subject IDs")
    names = sorted(a)
    difference = np.asarray([a[name] - b[name] for name in names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(names), size=(draws, len(names)))
    replicates = difference[indices].mean(axis=1)
    return {"subjects": len(names), "draws": draws, "mean_difference": float(difference.mean()),
            "ci025": float(np.quantile(replicates, .025)), "ci975": float(np.quantile(replicates, .975))}
