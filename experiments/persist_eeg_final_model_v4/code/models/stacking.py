from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize


def configurations() -> list[dict[str, Any]]:
    return [
        {"l2": l2, "learn_scale": learn_scale, "session_specific": session_specific}
        for l2 in (0.01, 0.1, 1.0, 10.0)
        for learn_scale in (False, True)
        for session_specific in (False, True)
    ]


class MaskedPositiveLogitPool:
    """Positive, availability-normalized weights over six frozen KEEP runs."""

    def __init__(self, l2: float, learn_scale: bool, session_specific: bool):
        self.l2 = float(l2)
        self.learn_scale = bool(learn_scale)
        self.session_specific = bool(session_specific)
        self.parameters_: np.ndarray | None = None

    def _score(self, x: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        logits = np.asarray(x[:, :6], dtype=float)
        mask = np.asarray(x[:, 6:12], dtype=float)
        session = np.asarray(x[:, 12:14], dtype=float)
        groups = 2 if self.session_specific else 1
        theta = parameters[: 6 * groups].reshape(groups, 6)
        session_index = np.argmax(session, axis=1) if self.session_specific else np.zeros(len(x), dtype=int)
        selected_theta = theta[session_index]
        weight = np.exp(np.clip(selected_theta, -6.0, 6.0)) * mask
        pooled = np.sum(weight * logits, axis=1) / np.maximum(np.sum(weight, axis=1), 1e-8)
        cursor = 6 * groups
        scale = np.exp(np.clip(parameters[cursor], -2.0, 2.0)) if self.learn_scale else 1.0
        cursor += int(self.learn_scale)
        bias = parameters[cursor]
        return scale * pooled + bias

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MaskedPositiveLogitPool":
        labels = np.asarray(y, dtype=float)
        groups = 2 if self.session_specific else 1
        parameter_count = 6 * groups + int(self.learn_scale) + 1

        def objective(parameters: np.ndarray) -> float:
            z = np.clip(self._score(x, parameters), -40.0, 40.0)
            loss = float(np.mean(np.logaddexp(0.0, z) - labels * z))
            theta = parameters[: 6 * groups]
            loss += 0.5 * self.l2 * float(np.mean(theta**2))
            return loss

        result = minimize(
            objective,
            np.zeros(parameter_count, dtype=float),
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if not result.success and result.nit == 0:
            raise RuntimeError(f"Masked KEEP pool optimizer failed: {result.message}")
        self.parameters_ = np.asarray(result.x, dtype=float)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.parameters_ is None:
            raise RuntimeError("Masked KEEP pool is not fit")
        z = np.clip(self._score(x, self.parameters_), -40.0, 40.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p1, p1])


def build(configuration: dict[str, Any], seed: int) -> MaskedPositiveLogitPool:
    del seed
    return MaskedPositiveLogitPool(
        l2=float(configuration["l2"]),
        learn_scale=bool(configuration["learn_scale"]),
        session_specific=bool(configuration["session_specific"]),
    )


class MaskedPositiveProbabilityPool(MaskedPositiveLogitPool):
    """Positive, availability-normalized weights in probability space.

    This is the capacity-matched counterpart of the frozen equal-probability
    ensemble.  It is intentionally global (not trial-routed), so any gain is
    attributable to legal cross-subject stacking rather than a high-capacity
    gate.
    """

    def _score(self, x: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        logits = np.asarray(x[:, :6], dtype=float)
        mask = np.asarray(x[:, 6:12], dtype=float)
        session = np.asarray(x[:, 12:14], dtype=float)
        groups = 2 if self.session_specific else 1
        theta = parameters[: 6 * groups].reshape(groups, 6)
        session_index = np.argmax(session, axis=1) if self.session_specific else np.zeros(len(x), dtype=int)
        selected_theta = theta[session_index]
        weight = np.exp(np.clip(selected_theta, -6.0, 6.0)) * mask
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        pooled_probability = np.sum(weight * probability, axis=1) / np.maximum(np.sum(weight, axis=1), 1e-8)
        pooled_probability = np.clip(pooled_probability, 1e-7, 1 - 1e-7)
        pooled_logit = np.log(pooled_probability) - np.log1p(-pooled_probability)
        cursor = 6 * groups
        scale = np.exp(np.clip(parameters[cursor], -2.0, 2.0)) if self.learn_scale else 1.0
        cursor += int(self.learn_scale)
        bias = parameters[cursor]
        return scale * pooled_logit + bias


def build_probability(configuration: dict[str, Any], seed: int) -> MaskedPositiveProbabilityPool:
    del seed
    return MaskedPositiveProbabilityPool(
        l2=float(configuration["l2"]),
        learn_scale=bool(configuration["learn_scale"]),
        session_specific=bool(configuration["session_specific"]),
    )


def contextual_configurations() -> list[dict[str, Any]]:
    return [
        {"l2": l2, "context_mode": context_mode}
        for l2 in (0.01, 0.1, 1.0, 10.0)
        for context_mode in ("prediction", "session_prediction")
    ]


class ContextualMaskedPositiveLogitPool:
    def __init__(self, l2: float, context_mode: str):
        self.l2 = float(l2)
        self.context_mode = str(context_mode)
        if self.context_mode not in {"prediction", "session_prediction"}:
            raise ValueError(self.context_mode)
        self.parameters_: np.ndarray | None = None

    @property
    def groups(self) -> int:
        return 2 if self.context_mode == "prediction" else 4

    def _group_index(self, x: np.ndarray) -> np.ndarray:
        logits, mask = x[:, :6], x[:, 6:12]
        base = np.sum(logits * mask, axis=1) / np.maximum(mask.sum(axis=1), 1.0)
        prediction = (base >= 0).astype(int)
        if self.context_mode == "prediction":
            return prediction
        session = np.argmax(x[:, 12:14], axis=1)
        return 2 * session + prediction

    def _score(self, x: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        logits, mask = x[:, :6], x[:, 6:12]
        theta = parameters[: 6 * self.groups].reshape(self.groups, 6)
        selected = theta[self._group_index(x)]
        weight = np.exp(np.clip(selected, -6.0, 6.0)) * mask
        pooled = np.sum(weight * logits, axis=1) / np.maximum(weight.sum(axis=1), 1e-8)
        scale = np.exp(np.clip(parameters[6 * self.groups], -2.0, 2.0))
        bias = parameters[6 * self.groups + 1]
        return scale * pooled + bias

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ContextualMaskedPositiveLogitPool":
        labels = np.asarray(y, dtype=float)

        def objective(parameters: np.ndarray) -> float:
            z = np.clip(self._score(x, parameters), -40.0, 40.0)
            loss = float(np.mean(np.logaddexp(0.0, z) - labels * z))
            theta = parameters[: 6 * self.groups]
            return loss + 0.5 * self.l2 * float(np.mean(theta**2))

        result = minimize(
            objective,
            np.zeros(6 * self.groups + 2, dtype=float),
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if not result.success and result.nit == 0:
            raise RuntimeError(f"Contextual KEEP pool optimizer failed: {result.message}")
        self.parameters_ = np.asarray(result.x, dtype=float)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.parameters_ is None:
            raise RuntimeError("Contextual KEEP pool is not fit")
        z = np.clip(self._score(x, self.parameters_), -40.0, 40.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p1, p1])


def build_contextual(configuration: dict[str, Any], seed: int) -> ContextualMaskedPositiveLogitPool:
    del seed
    return ContextualMaskedPositiveLogitPool(
        l2=float(configuration["l2"]), context_mode=str(configuration["context_mode"])
    )
