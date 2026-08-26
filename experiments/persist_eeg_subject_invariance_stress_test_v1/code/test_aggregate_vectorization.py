"""Regression checks for the aggregation-only bootstrap acceleration."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import aggregate


def _reference_mean(frame: pd.DataFrame, column: str, seed: int) -> np.ndarray:
    folds = sorted(frame.fold.unique().tolist())
    rng = np.random.default_rng(seed)
    lookup = {(int(row.fold), int(row.seed)): float(getattr(row, column)) for row in frame.itertuples()}
    draws = []
    for _ in range(aggregate.BOOTSTRAP_DRAWS):
        values = []
        for fold in rng.choice(folds, size=len(folds), replace=True):
            seeds = sorted(frame.loc[frame.fold == fold, "seed"].unique().tolist())
            for selected_seed in rng.choice(seeds, size=len(seeds), replace=True):
                values.append(lookup[(int(fold), int(selected_seed))])
        draws.append(float(np.mean(values)))
    return np.asarray(draws)


def _reference_slope(frame: pd.DataFrame, seed: int) -> np.ndarray:
    work = frame.copy()
    work["configuration"] = work.backbone.astype(str) + "|" + work.method.astype(str) + "|" + work["lambda"].map(lambda x: f"{x:g}")
    folds = sorted(work.fold.unique().tolist())
    configs = sorted(work.configuration.unique().tolist())
    lookup = work.set_index(["fold", "seed", "configuration"])[["identity_suppression_vs_ERM", "BA_delta_vs_ERM"]]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(aggregate.BOOTSTRAP_DRAWS):
        xs, ys = [], []
        sampled_configs = rng.choice(configs, size=len(configs), replace=True)
        for fold in rng.choice(folds, size=len(folds), replace=True):
            seeds = sorted(work.loc[work.fold == fold, "seed"].unique().tolist())
            for selected_seed in rng.choice(seeds, size=len(seeds), replace=True):
                for configuration in sampled_configs:
                    row = lookup.loc[(int(fold), int(selected_seed), str(configuration))]
                    xs.append(float(row.identity_suppression_vs_ERM))
                    ys.append(float(row.BA_delta_vs_ERM))
        value = aggregate.slope(np.asarray(xs), np.asarray(ys))
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values)


def test_vectorized_bootstraps_match_reference() -> None:
    old_draws = aggregate.BOOTSTRAP_DRAWS
    aggregate.BOOTSTRAP_DRAWS = 100
    try:
        mean_frame = pd.DataFrame(
            [{"fold": fold, "seed": seed, "value": fold * 0.1 + seed * 0.01} for fold in range(5) for seed in range(3)]
        )
        reference_mean = _reference_mean(mean_frame, "value", 41)
        accelerated_mean = aggregate.hierarchical_mean_ci(mean_frame, "value", 41)
        assert np.isclose(accelerated_mean["ci95"], np.quantile(reference_mean, [0.025, 0.975])).all()

        rows = []
        for backbone in ("eegnet", "eegconformer"):
            for method in ("DANN", "MMD"):
                for lam in (0.01, 0.1):
                    for fold in range(5):
                        for seed in range(3):
                            base = (fold + 1) * (seed + 2) * (1 if method == "DANN" else -1)
                            rows.append(
                                {
                                    "backbone": backbone,
                                    "method": method,
                                    "lambda": lam,
                                    "fold": fold,
                                    "seed": seed,
                                    "identity_suppression_vs_ERM": base * 0.01 + lam,
                                    "BA_delta_vs_ERM": base * -0.004 + lam * 0.03,
                                }
                            )
        slope_frame = pd.DataFrame(rows)
        reference_slopes = _reference_slope(slope_frame, 73)
        accelerated = aggregate.hierarchical_slope(slope_frame, 73)
        assert accelerated["bootstrap_draws_valid"] == len(reference_slopes)
        assert np.isclose(accelerated["ci95"], np.quantile(reference_slopes, [0.025, 0.975])).all()
    finally:
        aggregate.BOOTSTRAP_DRAWS = old_draws


if __name__ == "__main__":
    test_vectorized_bootstraps_match_reference()
    print("PASS")
