from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


BOUND_VERSION = "empirical-bernstein-block-v1"


def empirical_bernstein_bound(block_risks: np.ndarray, eta_within: float = 0.05) -> dict[str, object]:
    x = np.asarray(block_risks, dtype=float)
    if x.ndim != 1 or np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("block risks must be a finite vector within [0, 1]")
    if not 0 < eta_within < 1:
        raise ValueError("eta_within must be in (0, 1)")
    b = int(x.size)
    if b < 3:
        return {"mean": float(x.mean()) if b else float("nan"), "variance": float("nan"), "margin": float("nan"), "upper_risk": 1.0, "n_blocks": b, "status": "insufficient_future_blocks", "formula_version": BOUND_VERSION, "block_risks": x.tolist()}
    mean = float(x.mean())
    variance = float(x.var(ddof=1))
    log_term = math.log(3.0 / eta_within)
    margin = math.sqrt(2.0 * variance * log_term / b) + 3.0 * log_term / b
    return {"mean": mean, "variance": variance, "margin": margin, "upper_risk": float(np.clip(mean + margin, 0, 1)), "n_blocks": b, "status": "ok", "formula_version": BOUND_VERSION, "block_risks": x.tolist()}


@dataclass(frozen=True)
class ConformalQuantile:
    q: float
    n_calibration_subjects: int
    order_k: int
    delta: float
    provenance: str


def fit_simultaneous_quantile(calibration_surface: pd.DataFrame, delta: float = 0.10) -> ConformalQuantile:
    required = {"subject_id", "upper_risk", "predicted_risk"}
    if not required.issubset(calibration_surface.columns):
        raise ValueError(f"missing columns: {sorted(required - set(calibration_surface.columns))}")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    residuals = calibration_surface.assign(_e=calibration_surface["upper_risk"] - calibration_surface["predicted_risk"]).groupby("subject_id", sort=True)["_e"].max().to_numpy(float)
    m = int(residuals.size)
    if m == 0:
        raise ValueError("at least one calibration subject is required")
    k = int(math.ceil((m + 1) * (1 - delta)))
    if k > m:
        warnings.warn("finite-sample order exceeds calibration size; using q=1", RuntimeWarning)
        q, provenance = 1.0, "conservative_k_gt_m"
    else:
        q, provenance = float(np.sort(residuals)[k - 1]), "finite_sample_higher_order_statistic"
    return ConformalQuantile(q=q, n_calibration_subjects=m, order_k=k, delta=delta, provenance=provenance)


def apply_certificate(predicted_risk: np.ndarray, quantile: ConformalQuantile) -> np.ndarray:
    return np.clip(np.asarray(predicted_risk, dtype=float) + quantile.q, 0.0, 1.0)

