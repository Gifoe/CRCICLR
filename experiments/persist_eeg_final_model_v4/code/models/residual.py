from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def configurations() -> list[dict[str, Any]]:
    return [
        {"l2": l2, "clip": clip}
        for l2 in (0.1, 1.0, 10.0, 100.0)
        for clip in (0.25, 0.5, 1.0, 2.0)
    ]


class AnchoredResidualLogit:
    """Small correction on top of the frozen B_STRONG logit.

    The base logit is an explicit offset. L2 shrinkage and a finite correction
    clip keep this family meaningfully different from unconstrained stacking.
    """

    def __init__(self, l2: float, clip: float):
        self.l2 = float(l2)
        self.clip = float(clip)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.coefficient_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, base_logit: np.ndarray) -> "AnchoredResidualLogit":
        values = self.scaler.fit_transform(self.imputer.fit_transform(x))
        labels = np.asarray(y, dtype=float)
        offset = np.asarray(base_logit, dtype=float)

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            raw_delta = beta[0] + values @ beta[1:]
            delta = self.clip * np.tanh(raw_delta / self.clip)
            z = np.clip(offset + delta, -40.0, 40.0)
            probability = 1.0 / (1.0 + np.exp(-z))
            loss = float(np.mean(np.logaddexp(0.0, z) - labels * z))
            loss += 0.5 * self.l2 * float(np.sum(beta[1:] ** 2)) / len(labels)
            derivative_clip = 1.0 - np.tanh(raw_delta / self.clip) ** 2
            residual = (probability - labels) * derivative_clip / len(labels)
            gradient = np.concatenate(
                [
                    [residual.sum()],
                    values.T @ residual + self.l2 * beta[1:] / len(labels),
                ]
            )
            return loss, gradient

        result = minimize(
            objective,
            np.zeros(values.shape[1] + 1, dtype=float),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if not result.success and result.nit == 0:
            raise RuntimeError(f"Residual optimizer failed: {result.message}")
        self.coefficient_ = np.asarray(result.x, dtype=float)
        return self

    def predict_proba(self, x: np.ndarray, base_logit: np.ndarray) -> np.ndarray:
        if self.coefficient_ is None:
            raise RuntimeError("Residual model is not fit")
        values = self.scaler.transform(self.imputer.transform(x))
        raw_delta = self.coefficient_[0] + values @ self.coefficient_[1:]
        delta = self.clip * np.tanh(raw_delta / self.clip)
        z = np.clip(np.asarray(base_logit, dtype=float) + delta, -40.0, 40.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p1, p1])


def build(configuration: dict[str, Any], seed: int) -> AnchoredResidualLogit:
    del seed
    return AnchoredResidualLogit(float(configuration["l2"]), float(configuration["clip"]))
