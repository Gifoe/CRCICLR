from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


EPS = 1e-7


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-value))


def logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=float), EPS, 1 - EPS)
    return np.log(value) - np.log1p(-value)


@dataclass
class PopulationContext:
    scaler: StandardScaler
    model: LogisticRegression
    class_means: np.ndarray
    class_variances: np.ndarray


def fit_population(x: np.ndarray, y: np.ndarray, c: float, seed: int) -> PopulationContext:
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    model = LogisticRegression(
        C=float(c),
        class_weight="balanced",
        solver="liblinear",
        max_iter=4_000,
        random_state=int(seed),
    ).fit(z, y)
    means = np.stack([z[y == label].mean(axis=0) for label in (0, 1)])
    variances = np.stack([z[y == label].var(axis=0) + 0.05 for label in (0, 1)])
    return PopulationContext(scaler=scaler, model=model, class_means=means, class_variances=variances)


def _target_logistic(context: PopulationContext, x_history: np.ndarray, y_history: np.ndarray, c: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=float(c),
        class_weight="balanced",
        solver="liblinear",
        max_iter=4_000,
        random_state=int(seed),
    ).fit(context.scaler.transform(x_history), y_history)


def configurations(family: str) -> list[dict[str, Any]]:
    cs = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    if family == "population_linear":
        return [{"C": c} for c in cs]
    if family == "target_logistic":
        return [{"C": c} for c in cs]
    if family == "target_lda":
        return [{"shrinkage": value} for value in ("auto", 0.1, 0.3, 0.5, 0.7, 0.9)]
    if family == "history_calibrated":
        return [{"C": c} for c in (0.01, 0.1, 1.0, 10.0, 100.0)]
    if family == "cosine_prototype":
        return [{"temperature": value} for value in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)]
    if family == "shrinkage_prototype":
        return [
            {"target_weight": weight, "temperature": temperature}
            for weight in (0.25, 0.5, 0.75, 1.0)
            for temperature in (0.5, 1.0, 2.0, 4.0)
        ]
    if family in {"fusion_logistic", "fusion_lda"}:
        base = configurations("target_logistic") if family == "fusion_logistic" else configurations("target_lda")
        return [{**item, "target_weight": weight} for item in base for weight in (0.2, 0.4, 0.6, 0.8, 1.0)]
    raise ValueError(family)

def population_probability(context: PopulationContext, x: np.ndarray) -> np.ndarray:
    return context.model.predict_proba(context.scaler.transform(x))[:, 1]


def target_probability(
    family: str,
    configuration: dict[str, Any],
    context: PopulationContext,
    x_history: np.ndarray,
    y_history: np.ndarray,
    x_future: np.ndarray,
    seed: int,
) -> np.ndarray:
    z_history = context.scaler.transform(x_history)
    z_future = context.scaler.transform(x_future)
    pop_history = context.model.predict_proba(z_history)[:, 1]
    pop_future = context.model.predict_proba(z_future)[:, 1]
    if family == "target_logistic":
        model = _target_logistic(context, x_history, y_history, float(configuration["C"]), seed)
        return model.predict_proba(z_future)[:, 1]
    if family == "target_lda":
        shrinkage = configuration["shrinkage"]
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage).fit(z_history, y_history)
        return model.predict_proba(z_future)[:, 1]
    if family == "history_calibrated":
        calibration = LogisticRegression(
            C=float(configuration["C"]),
            class_weight="balanced",
            solver="liblinear",
            max_iter=4_000,
            random_state=int(seed),
        ).fit(logit(pop_history)[:, None], y_history)
        return calibration.predict_proba(logit(pop_future)[:, None])[:, 1]
    if family == "cosine_prototype":
        norm_history = z_history / np.maximum(np.linalg.norm(z_history, axis=1, keepdims=True), EPS)
        norm_future = z_future / np.maximum(np.linalg.norm(z_future, axis=1, keepdims=True), EPS)
        prototypes = np.stack([norm_history[y_history == label].mean(axis=0) for label in (0, 1)])
        prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), EPS)
        score = norm_future @ (prototypes[1] - prototypes[0])
        return sigmoid(float(configuration["temperature"]) * score)
    if family == "shrinkage_prototype":
        target_means = np.stack([z_history[y_history == label].mean(axis=0) for label in (0, 1)])
        weight = float(configuration["target_weight"])
        means = (1 - weight) * context.class_means + weight * target_means
        variance = context.class_variances.mean(axis=0)
        direction = (means[1] - means[0]) / variance
        offset = -0.5 * np.sum((means[1] ** 2 - means[0] ** 2) / variance)
        score = z_future @ direction + offset
        return sigmoid(float(configuration["temperature"]) * score)
    if family == "fusion_logistic":
        target = _target_logistic(context, x_history, y_history, float(configuration["C"]), seed).predict_proba(z_future)[:, 1]
        weight = float(configuration["target_weight"])
        return sigmoid((1 - weight) * logit(pop_future) + weight * logit(target))
    if family == "fusion_lda":
        target = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=configuration["shrinkage"]).fit(z_history, y_history).predict_proba(z_future)[:, 1]
        weight = float(configuration["target_weight"])
        return sigmoid((1 - weight) * logit(pop_future) + weight * logit(target))
    raise ValueError(family)
