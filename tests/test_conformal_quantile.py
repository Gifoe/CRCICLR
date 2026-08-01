import pandas as pd
import pytest

from hsc_tta.certification import fit_actionwise_simultaneous_quantile


def _rows(n_subjects: int):
    rows = []
    for subject in range(n_subjects):
        for action, residual in zip(("no_tta", "t3a", "entropy_adapter"), (0.0, 0.5, 1.0)):
            rows.append({
                "dataset": "hmc", "seed": 0, "episode_id": f"e{subject}",
                "subject_id": f"s{subject}", "alpha": 0.2, "action": action,
                "critical_index": 5 + residual, "predicted_critical_index": 5,
            })
    return pd.DataFrame(rows)


def test_higher_quantile_is_subject_level_and_order_invariant():
    frame = _rows(20)
    first = fit_actionwise_simultaneous_quantile(frame, n_nontrivial_lambdas=20)
    second = fit_actionwise_simultaneous_quantile(frame.sample(frac=1, random_state=7), n_nontrivial_lambdas=20)
    assert first == second
    assert first.order_k == 19 and first.q_alpha == 1.0


def test_conservative_fallback_when_k_exceeds_m():
    with pytest.warns(RuntimeWarning, match="full-set"):
        result = fit_actionwise_simultaneous_quantile(_rows(2), n_nontrivial_lambdas=20)
    assert result.q_alpha == 20
    assert result.provenance == "conservative_full_set_k_gt_m"
