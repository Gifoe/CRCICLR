"""Small protocol tests for the TEA-EEG implementation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from tea_core import apply_mixture, build_target_context, legal_sample_columns  # noqa: E402
from run_stage import _random_gate  # noqa: E402


def _frame(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    d = pd.DataFrame({
        "subject": np.repeat(["1", "2"], n // 2), "session": "S1",
        "manifest_index": np.arange(n), "fold": 0, "seed": 0, "router_fold": 0,
        "y": rng.integers(0, 2, n), "margin_keep": rng.normal(size=n),
    })
    for a in ("keep", "amplify", "geometry", "erase"):
        d[f"margin_{a}"] = rng.normal(size=n)
        d[f"p_{a}"] = 1 / (1 + np.exp(-d[f"margin_{a}"]))
        d[f"pred_{a}"] = (d[f"margin_{a}"] >= 0).astype(int)
        d[f"confidence_{a}"] = np.maximum(d[f"p_{a}"], 1 - d[f"p_{a}"])
        d[f"entropy_{a}"] = 0.5
    for a in ("amplify", "geometry", "erase"):
        d[f"effect_{a}"] = 0
        d[f"dce_{a}"] = 0.0
        d[f"flip_{a}"] = 0
        d[f"delta_margin_{a}"] = 0.0
        d[f"delta_probability_{a}"] = 0.0
        d[f"confidence_change_{a}"] = 0.0
    d["baseline_error"] = 0
    d["action_disagreement_count"] = 0
    d["action_vote_fraction"] = 0.5
    d["action_margin_mean"] = 0.0
    d["action_margin_std"] = 0.0
    d["other_run_mean_margin_keep"] = 0.0
    d["other_run_mean_margin_amplify"] = 0.0
    d["other_run_mean_margin_geometry"] = 0.0
    d["other_run_mean_margin_erase"] = 0.0
    d["other_run_vote_keep"] = 0.5
    d["other_run_vote_amplify"] = 0.5
    d["other_run_vote_geometry"] = 0.5
    d["other_run_vote_erase"] = 0.5
    return d


def test_keep_fallback_is_exact():
    d = _frame()
    p, selected, _ = apply_mixture(d, np.full((len(d), 2), -1.0), np.ones((len(d), 2)), 2.0, 1.0, ("amplify", "geometry"))
    assert np.all(selected == "keep")
    assert np.array_equal(p, d.p_keep.to_numpy())


def test_uncertainty_is_conservative():
    d = _frame()
    mu = np.full((len(d), 2), 0.2)
    low = np.zeros_like(mu)
    high = np.full_like(mu, 0.5)
    _, s_low, _ = apply_mixture(d, mu, low, 1.0, 1.0, ("amplify", "geometry"))
    _, s_high, _ = apply_mixture(d, mu, high, 1.0, 1.0, ("amplify", "geometry"))
    assert np.sum(s_high != "keep") <= np.sum(s_low != "keep")


def test_context_has_no_query_block_rows():
    d = _frame()
    out, _ = build_target_context(d)
    assert out.target_block.nunique() == 5
    for (subject, block), group in out.groupby(["subject", "target_block"]):
        row_n = int(group.context_n.iloc[0])
        subject_n = int((out.subject == subject).sum())
        own_block_n = len(group)
        assert row_n == subject_n - own_block_n


def test_effects_and_identity_are_not_legal_inputs():
    d = _frame()
    cols = legal_sample_columns(d)
    assert not any("effect" in c or "dce" in c or "subject" in c or "fold" in c for c in cols)


def test_random_gate_matches_feasible_reference_counts():
    d = _frame()
    d["pred_amplify"] = (np.arange(len(d)) % 2).astype(int)
    d["pred_geometry"] = ((np.arange(len(d)) // 2) % 2).astype(int)
    reference = np.array(["amplify"] * 3 + ["geometry"] * 2 + ["keep"] * (len(d) - 5), dtype=object)
    selected = _random_gate(d, reference, ("amplify", "geometry"))
    assert int(np.sum(selected == "amplify")) == 3
    assert int(np.sum(selected == "geometry")) == 2
