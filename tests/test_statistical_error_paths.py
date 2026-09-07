import numpy as np
import pandas as pd
import pytest

from hsc_tta.certification import (
    apply_critical_index_certificate,
    critical_index_from_curve,
    critical_index_table,
    empirical_bernstein_bound,
    fit_actionwise_simultaneous_quantile,
)
from hsc_tta.selection import select_safe_action
from hsc_tta.simulation import evaluate_cpu_go, run_simulations
from test_action_selection import _candidates


@pytest.mark.parametrize(
    "risks,grid,alpha,message",
    [
        ([0.2, 0.0], [0.5, 1.0], 0.0, "alpha"),
        ([0.2], [1.0], 0.2, "aligned"),
        ([1.2, 0.0], [0.5, 1.0], 0.2, "future risk"),
        ([0.2, 0.0], [0.5, 0.5], 0.2, "strictly increasing"),
        ([0.2, 0.1], [0.5, 0.9], 0.2, "sentinel"),
        ([0.2, 0.1], [0.5, 1.0], 0.2, "sentinel risk"),
    ],
)
def test_critical_index_rejects_malformed_curves(risks, grid, alpha, message):
    with pytest.raises(ValueError, match=message):
        critical_index_from_curve(risks, grid, alpha)


def test_legacy_bound_validation_is_explicit():
    with pytest.raises(ValueError):
        empirical_bernstein_bound(np.array([np.nan]))
    with pytest.raises(ValueError):
        empirical_bernstein_bound(np.array([0.1, 0.2, 0.3]), eta_within=1.0)
    assert empirical_bernstein_bound(np.array([0.1, 0.2]))["diagnostic_only"]


def _calibration():
    rows = []
    for subject in range(20):
        for action in ("no_tta", "t3a", "entropy_adapter"):
            rows.append({"dataset": "hmc", "seed": 0, "episode_id": f"e{subject}", "subject_id": f"s{subject}", "alpha": 0.2, "action": action, "critical_index": 5, "predicted_critical_index": 4})
    return pd.DataFrame(rows)


def test_quantile_validation_and_nonfinite_apply():
    frame = _calibration()
    with pytest.raises(ValueError, match="delta"):
        fit_actionwise_simultaneous_quantile(frame, delta=1.0, n_nontrivial_lambdas=20)
    with pytest.raises(ValueError, match="positive"):
        fit_actionwise_simultaneous_quantile(frame, n_nontrivial_lambdas=0)
    with pytest.raises(ValueError, match="dataset/seed"):
        fit_actionwise_simultaneous_quantile(pd.concat([frame, frame.assign(seed=1)]), n_nontrivial_lambdas=20)
    with pytest.raises(ValueError, match="duplicate"):
        fit_actionwise_simultaneous_quantile(pd.concat([frame, frame.iloc[:1]]), n_nontrivial_lambdas=20)
    with pytest.raises(ValueError, match="actions"):
        fit_actionwise_simultaneous_quantile(frame[frame.action != "t3a"], n_nontrivial_lambdas=20)
    bad = frame.copy(); bad.loc[0, "critical_index"] = 21
    with pytest.raises(ValueError, match="critical_index"):
        fit_actionwise_simultaneous_quantile(bad, n_nontrivial_lambdas=20)
    q = fit_actionwise_simultaneous_quantile(frame, n_nontrivial_lambdas=20)
    with pytest.raises(ValueError, match="finite"):
        apply_critical_index_certificate([np.nan], q)


def test_incomplete_lambda_indices_rejected():
    frame = pd.DataFrame([{"dataset": "hmc", "seed": 0, "episode_id": "e", "subject_id": "s", "action": "no_tta", "alpha": 0.2, "lambda": 0.5, "lambda_index": 1, "future_risk": 0.0}])
    with pytest.raises(ValueError, match="incomplete"):
        critical_index_table(frame)


def test_selector_rejects_inconsistent_candidates():
    base = _candidates()
    cases = [
        (base.assign(action=["unknown", "t3a", "entropy_adapter"]), "unknown"),
        (pd.concat([base, base.iloc[:1]]), "one candidate"),
        (base.assign(n_classes=np.nan), "n_classes"),
        (base.assign(context_singleton_rate=2.0), "singleton"),
        (base.assign(adaptation_cost=[1, 1, 2]), "adaptation_cost"),
        (base.assign(nontrivial_candidate=False), "nontrivial_candidate"),
        (base.assign(certified_critical_index=21), "critical_critical|certified_critical_index"),
    ]
    for frame, message in cases:
        with pytest.raises(ValueError, match=message):
            select_safe_action(frame)


def test_simulation_and_cpu_go_error_paths(tmp_path):
    with pytest.raises(ValueError):
        run_simulations(tmp_path, repetitions=0)
    with pytest.raises(RuntimeError, match="500"):
        evaluate_cpu_go(pd.DataFrame(), 10, 20)
    good = pd.DataFrame({"alpha": [0.2] * 500, "csr_nonfull": [1.0] * 500, "selected_coverage": [0.91] * 500, "q_saturated": [False] * 500})
    assert evaluate_cpu_go(good, 500, 20)["csr_nonfull_alpha_0_20"] == 1.0
    with pytest.raises(RuntimeError, match="CSR"):
        evaluate_cpu_go(good.assign(csr_nonfull=0.0), 500, 20)
    with pytest.raises(RuntimeError, match="coverage"):
        evaluate_cpu_go(good.assign(selected_coverage=0.0), 500, 20)
    with pytest.raises(RuntimeError, match="saturates"):
        evaluate_cpu_go(good.assign(q_saturated=True), 500, 20)
