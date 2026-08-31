"""Fast protocol invariants for the CGR-Fuse implementation.

These tests are synthetic and do not read EEG or server caches.  The server
run remains the source of numerical evidence; this file guards the alignment,
convexity, subject-unit, and feature-boundary invariants that can otherwise
silently regress.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import cgrfuse


def _synthetic_agg(n_subjects: int = 10, samples_per_subject: int = 3) -> pd.DataFrame:
    n = n_subjects * samples_per_subject
    subjects = np.repeat([str(i) for i in range(n_subjects)], samples_per_subject)
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "subject": subjects,
            "label": np.arange(n) % 2,
            "p_keep": rng.uniform(0.15, 0.85, n),
            "p_amplify": rng.uniform(0.15, 0.85, n),
            "p_geometry": rng.uniform(0.15, 0.85, n),
            "s_vote": np.where(np.arange(n) % 3 == 0, 0.0, 2.0 / 3.0),
            "margin_keep": rng.normal(size=n),
        }
    )
    for action in ("keep", "amplify", "geometry"):
        for run_index in range(6):
            frame[f"margin_{action}_run{run_index}"] = rng.normal(size=n)
    return frame


def test_subject_folds_are_complete_and_disjoint() -> None:
    subjects = [str(i) for i in range(17)]
    folds = cgrfuse.subject_folds(subjects)
    assert set(folds) == set(subjects)
    assert len(set(folds.values())) == 5
    assert all(0 <= value < 5 for value in folds.values())


def test_constraints_are_convex_and_stable_keep_is_exact() -> None:
    agg = _synthetic_agg()
    n = len(agg)
    z = np.arange(n * 3, dtype=float).reshape(n, 3) / 10.0
    mu = np.ones((5, n, 2), dtype=float)
    out = cgrfuse.apply_constraints(agg, z, mu, eta=1.0, kappa=0.0)
    assert np.allclose(out["weights"].sum(axis=1), 1.0)
    assert np.all(out["weights"] >= -1e-12)
    p = agg[["p_keep", "p_amplify", "p_geometry"]].to_numpy(float)
    assert np.all(out["p_final"] >= p.min(axis=1) - 1e-12)
    assert np.all(out["p_final"] <= p.max(axis=1) + 1e-12)
    stable = out["stable"]
    strongest = (agg.margin_keep.to_numpy(float) >= 0).astype(np.int8)
    assert np.array_equal(out["prediction"][stable], strongest[stable])


def test_lcb_nonpositive_actions_force_keep() -> None:
    agg = _synthetic_agg()
    n = len(agg)
    out = cgrfuse.apply_constraints(
        agg,
        np.zeros((n, 3), dtype=float),
        np.zeros((5, n, 2), dtype=float),
        eta=1.0,
        kappa=0.5,
    )
    assert np.allclose(out["weights"], np.column_stack((np.ones(n), np.zeros((n, 2)))))


def test_stacking_is_subject_cross_fitted_and_uses_frozen_run_columns() -> None:
    agg = _synthetic_agg()
    pred = cgrfuse.crossfit_stacking_predictions(agg, ("KEEP", "AMPLIFY", "GEOMETRY"))
    assert pred.shape == (len(agg),)
    assert set(np.unique(pred)).issubset({0, 1})


def test_primary_feature_schema_excludes_identity_and_erase() -> None:
    bank = pd.DataFrame(
        {
            "dataset": "OpenBMI",
            "sample_id": ["a"] * 6,
            "subject": ["1"] * 6,
            "session": ["1"] * 6,
            "trial_index": [0] * 6,
            "fold": [0, 0, 1, 1, 2, 2],
            "seed": [0, 1, 0, 1, 0, 1],
            "router_fold": [0] * 6,
            "run_id": [f"f{f}s{s}" for f, s in ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))],
            "label": [1] * 6,
            "keep_logit_0": [0.0] * 6,
            "keep_logit_1": [1.0] * 6,
            "amplify_logit_0": [0.0] * 6,
            "amplify_logit_1": [0.5] * 6,
            "geometry_logit_0": [0.0] * 6,
            "geometry_logit_1": [0.2] * 6,
        }
    )
    _, features = cgrfuse.aggregate_bank(bank)
    assert features
    assert not any("subject" in name.lower() or "label" in name.lower() for name in features)
    assert not any("erase" in name.lower() for name in features)
