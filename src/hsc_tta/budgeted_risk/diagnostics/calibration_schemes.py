from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


S1 = "S1_ORIGINAL_3TRAIN_1CAL"
S2 = "S2_EXACT_2TRAIN_2CAL"
S3 = "S3_EXACT_2TRAIN_2CAL_SCALED"
S4 = "S4_CROSSFIT_POOLED_EXPLORATORY"
SCHEMES = (S1, S2, S3, S4)


@dataclass(frozen=True)
class FoldSplit:
    evaluation: tuple[int, ...]
    calibration: tuple[int, ...]
    training: tuple[int, ...]


def fold_split(scheme: str, evaluation_fold: int) -> FoldSplit:
    e = int(evaluation_fold) % 5
    if scheme == S1:
        return FoldSplit((e,), ((e + 1) % 5,), tuple((e + d) % 5 for d in (2, 3, 4)))
    if scheme in (S2, S3):
        return FoldSplit((e,), ((e + 1) % 5, (e + 2) % 5), ((e + 3) % 5, (e + 4) % 5))
    if scheme == S4:
        return FoldSplit((e,), tuple(i for i in range(5) if i != e), tuple())
    raise ValueError(f"unknown calibration scheme: {scheme}")


def conformal_rank(m: int, delta: float = .1) -> int:
    return int(math.ceil((int(m) + 1) * (1 - float(delta))))


def conformal_q(values: np.ndarray, delta: float = .1, *, sentinel: float = 20.) -> tuple[float, int]:
    scores = np.sort(np.asarray(values, dtype=float))
    k = conformal_rank(len(scores), delta)
    return (float(sentinel) if k > len(scores) else float(scores[k - 1])), k


def scale_design(frame) -> np.ndarray:
    return np.column_stack([
        np.asarray(frame["local_kappa_std"], float),
        np.asarray(frame["local_kappa_q90"], float) - np.asarray(frame["local_kappa_q50"], float),
        1. / np.sqrt(np.asarray(frame["effective_budget"], float) + 1.),
        np.asarray(frame["prefix_instability"], float),
    ])


def fit_nonnegative_scale(design: np.ndarray, residual: np.ndarray) -> np.ndarray:
    x = np.asarray(design, float)
    y = np.maximum(np.asarray(residual, float), 0.)
    if x.ndim != 2 or x.shape[1] != 4 or len(x) != len(y):
        raise ValueError("scale training arrays have incompatible shapes")

    def objective(a: np.ndarray) -> float:
        sigma = .25 + x @ a
        error = y - sigma
        pinball = np.maximum(.75 * error, -.25 * error).mean()
        return float(pinball + 1e-3 * np.square(a).sum())

    def subgradient(a: np.ndarray) -> np.ndarray:
        error = y - (.25 + x @ a)
        weight = np.where(error > 0, -.75, np.where(error < 0, .25, 0.))
        return (weight[:, None] * x).mean(0) + 2e-3 * a

    result = minimize(objective, np.zeros(4), jac=subgradient, method="SLSQP",
                      bounds=[(0., None)] * 4, options={"ftol": 1e-10, "maxiter": 500})
    if not result.success:
        # Deterministic derivative-free fallback for the rare exact-kink case.
        result = minimize(objective, np.zeros(4), method="Powell", bounds=[(0., None)] * 4,
                          options={"xtol": 1e-6, "ftol": 1e-8, "maxiter": 500})
    if not result.success:
        raise RuntimeError(f"scale optimization failed: {result.message}")
    return np.maximum(np.asarray(result.x, float), 0.)


def predict_scale(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    sigma = .25 + np.asarray(design, float) @ np.asarray(coefficients, float)
    if np.any(sigma <= 0):
        raise RuntimeError("scale must be strictly positive")
    return sigma


def sentinel_transition(raw_index, certified_index, sentinel: int = 20):
    return (np.asarray(raw_index) < sentinel) & (np.asarray(certified_index) == sentinel)


def is_outlier_driven(q_drop: float, sentinel_drop: float, gain_increase: float) -> bool:
    return bool(q_drop >= 2. or sentinel_drop >= .10 or gain_increase >= .05)
