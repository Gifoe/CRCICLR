from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


LEGACY_BOUND_VERSION = "empirical-bernstein-block-v1-legacy-diagnostic"
ACTIONS = ("no_tta", "t3a", "entropy_adapter")
SUBJECT_KEY = ("dataset", "seed", "episode_id", "subject_id", "alpha")
CURVE_KEY = (*SUBJECT_KEY[:-1], "action", "alpha")


def empirical_bernstein_bound(
    block_risks: np.ndarray, eta_within: float = 0.05
) -> dict[str, object]:
    """Legacy supplementary diagnostic; never a formal HSC-TTA target or score."""
    x = np.asarray(block_risks, dtype=float)
    if x.ndim != 1 or np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("block risks must be a finite vector within [0, 1]")
    if not 0 < eta_within < 1:
        raise ValueError("eta_within must be in (0, 1)")
    b = int(x.size)
    if b < 3:
        return {
            "mean": float(x.mean()) if b else float("nan"),
            "variance": float("nan"),
            "margin": float("nan"),
            "upper_risk": 1.0,
            "n_blocks": b,
            "status": "insufficient_future_blocks",
            "formula_version": LEGACY_BOUND_VERSION,
            "diagnostic_only": True,
            "block_risks": x.tolist(),
        }
    mean = float(x.mean())
    variance = float(x.var(ddof=1))
    log_term = math.log(3.0 / eta_within)
    margin = math.sqrt(2.0 * variance * log_term / b) + 3.0 * log_term / b
    return {
        "mean": mean,
        "variance": variance,
        "margin": margin,
        "upper_risk": float(np.clip(mean + margin, 0, 1)),
        "n_blocks": b,
        "status": "ok",
        "formula_version": LEGACY_BOUND_VERSION,
        "diagnostic_only": True,
        "block_risks": x.tolist(),
    }


def critical_index_from_curve(
    risks: Iterable[float], lambdas: Iterable[float], alpha: float
) -> int:
    """Return the first lambda index whose empirical future risk is <= alpha."""
    risk = np.asarray(list(risks), dtype=float)
    grid = np.asarray(list(lambdas), dtype=float)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if risk.ndim != 1 or grid.ndim != 1 or risk.size != grid.size or risk.size < 2:
        raise ValueError("risk and lambda curves must be aligned one-dimensional vectors")
    if np.any(~np.isfinite(risk)) or np.any((risk < 0) | (risk > 1)):
        raise ValueError("future risk must be finite and in [0, 1]")
    if np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
        raise ValueError("lambda grid must be finite and strictly increasing")
    if not np.isclose(grid[-1], 1.0):
        raise ValueError("lambda=1.0 full-set sentinel is required")
    if not np.isclose(risk[-1], 0.0):
        raise ValueError("full-set sentinel risk must be zero")
    if np.any(np.diff(risk) > 1e-12):
        raise ValueError("future-risk curve must be nonincreasing with lambda")
    return int(np.flatnonzero(risk <= alpha)[0])


def critical_index_table(curves: pd.DataFrame) -> pd.DataFrame:
    """Collapse complete future-risk curves to one critical-index row per full key."""
    required = {*CURVE_KEY, "lambda", "lambda_index", "future_risk"}
    missing = required - set(curves.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for key, group in curves.groupby(list(CURVE_KEY), sort=True, dropna=False):
        ordered = group.sort_values("lambda_index", kind="mergesort")
        indices = ordered["lambda_index"].to_numpy(int)
        if not np.array_equal(indices, np.arange(len(indices))):
            raise ValueError(f"lambda indices are incomplete for key={key}")
        alpha = float(ordered["alpha"].iloc[0])
        j = critical_index_from_curve(ordered["future_risk"], ordered["lambda"], alpha)
        row = dict(zip(CURVE_KEY, key if isinstance(key, tuple) else (key,)))
        row.update({"critical_index": j, "n_nontrivial_lambdas": len(indices) - 1})
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class CriticalIndexQuantile:
    q_alpha: float
    raw_quantile: float
    n_calibration_subjects: int
    order_k: int
    delta: float
    alpha: float
    n_nontrivial_lambdas: int
    provenance: str


def fit_actionwise_simultaneous_quantile(
    calibration_predictions: pd.DataFrame,
    *,
    delta: float = 0.10,
    n_nontrivial_lambdas: int,
    actions: tuple[str, ...] = ACTIONS,
) -> CriticalIndexQuantile:
    """Fit subject-level conformal correction, maximizing only across actions."""
    required = {*SUBJECT_KEY, "action", "critical_index", "predicted_critical_index"}
    missing = required - set(calibration_predictions.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if n_nontrivial_lambdas < 1:
        raise ValueError("n_nontrivial_lambdas must be positive")
    frame = calibration_predictions.copy()
    if frame["alpha"].nunique() != 1:
        raise ValueError("fit one alpha at a time")
    if frame[["dataset", "seed"]].drop_duplicates().shape[0] != 1:
        raise ValueError("fit one dataset/seed at a time")
    if frame.duplicated([*SUBJECT_KEY, "action"]).any():
        raise ValueError("duplicate subject/action prediction rows")
    expected = set(actions)
    for key, group in frame.groupby(list(SUBJECT_KEY), sort=False, dropna=False):
        found = set(group["action"])
        if found != expected:
            raise ValueError(f"calibration subject {key} has actions {sorted(found)}, expected {sorted(expected)}")
    for column in ("critical_index", "predicted_critical_index"):
        values = frame[column].to_numpy(float)
        if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > n_nontrivial_lambdas)):
            raise ValueError(f"{column} must be finite and in [0, L]")
    frame["_residual"] = frame["critical_index"] - frame["predicted_critical_index"]
    # This is the only maximum in the formal score: three actions, never lambdas.
    residuals = frame.groupby(list(SUBJECT_KEY), sort=True)["_residual"].max().to_numpy(float)
    m = int(residuals.size)
    if m == 0:
        raise ValueError("at least one calibration subject is required")
    k = int(math.ceil((m + 1) * (1 - delta)))
    if k > m:
        warnings.warn(
            "finite-sample order exceeds calibration size; using full-set index fallback",
            RuntimeWarning,
        )
        raw_q = float(n_nontrivial_lambdas)
        provenance = "conservative_full_set_k_gt_m"
    else:
        raw_q = float(np.sort(residuals)[k - 1])
        provenance = "finite_sample_subject_actionwise_order_statistic"
    return CriticalIndexQuantile(
        q_alpha=max(0.0, raw_q),
        raw_quantile=raw_q,
        n_calibration_subjects=m,
        order_k=k,
        delta=float(delta),
        alpha=float(frame["alpha"].iloc[0]),
        n_nontrivial_lambdas=int(n_nontrivial_lambdas),
        provenance=provenance,
    )


def apply_critical_index_certificate(
    predicted_critical_index: Iterable[float], quantile: CriticalIndexQuantile
) -> np.ndarray:
    predicted = np.asarray(list(predicted_critical_index), dtype=float)
    if np.any(~np.isfinite(predicted)):
        raise ValueError("predicted critical indices must be finite")
    return np.clip(
        np.ceil(predicted + quantile.q_alpha),
        0,
        quantile.n_nontrivial_lambdas,
    ).astype(int)

