"""Aggregation and fixed decision rule for LiteBN-StatsResidual."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TOL = 1e-12


def signs(values: np.ndarray) -> dict[str, int]:
    return {
        "positive": int((values > TOL).sum()),
        "negative": int((values < -TOL).sum()),
        "tied": int((np.abs(values) <= TOL).sum()),
    }


def paired_bootstrap(values: np.ndarray, seed: int = 0, resamples: int = 10_000) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return {
        "resamples": resamples,
        "seed": seed,
        "unit": "subject",
        "mean_delta_pp": float(values.mean()),
        "ci95_low_pp": float(np.quantile(draws, 0.025)),
        "ci95_high_pp": float(np.quantile(draws, 0.975)),
    }


def metric_summary(rows: pd.DataFrame, base_prefix: str = "B0", candidate_prefix: str = "StatsResidual") -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in ("BA", "macro_F1", "accuracy"):
        base = float(rows[f"{base_prefix}_{metric}"].mean())
        candidate = float(rows[f"{candidate_prefix}_{metric}"].mean())
        result[f"{base_prefix}_{metric}"] = base
        result[f"{candidate_prefix}_{metric}"] = candidate
        result[f"delta_{metric}_pp"] = 100.0 * (candidate - base)
    return result


def decide(outer_delta: float, fold_deltas: np.ndarray, subject_deltas: np.ndarray,
           heldout_delta: float, protocol_pass: bool) -> tuple[str, dict[str, bool]]:
    fold_signs, subject_signs = signs(fold_deltas), signs(subject_deltas)
    criteria = {
        "outer_delta_ge_0_20_pp": outer_delta >= 0.20 - TOL,
        "at_least_3_of_5_outer_folds_improve": fold_signs["positive"] >= 3,
        "worst_outer_fold_gt_minus_0_50_pp": float(fold_deltas.min()) > -0.50,
        "outer_subject_positive_ge_negative": subject_signs["positive"] >= subject_signs["negative"],
        "internal_heldout_delta_nonnegative": heldout_delta >= -TOL,
        "all_protocol_audits_pass": protocol_pass,
    }
    if not protocol_pass:
        return "STATSRES_PROTOCOL_FAIL", criteria
    if all(criteria.values()):
        return "STATSRES_WBCIC_STRONG_SIGNAL", criteria
    if outer_delta > TOL and heldout_delta >= -TOL:
        return "STATSRES_WBCIC_WEAK_POSITIVE", criteria
    if outer_delta > TOL and heldout_delta < -TOL:
        return "STATSRES_OUTER_HELDOUT_TRADEOFF", criteria
    return "STATSRES_NO_USEFUL_SIGNAL", criteria
