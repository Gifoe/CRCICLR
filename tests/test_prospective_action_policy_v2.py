from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "experiments" / "persist_eeg_prospective_action_policy_v2" / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import common  # noqa: E402
import data  # noqa: E402
import metrics  # noqa: E402
import policies  # noqa: E402


def test_frozen_subject_split_has_expected_counts_and_no_overlap() -> None:
    subjects = [str(value) for value in [*range(1, 6), *range(7, 17), *range(18, 55)]]
    holdout = {subject for subject in subjects if common.stable_unit(common.SPLIT_SALT, subject) < 0.25}
    exploration = set(subjects) - holdout
    assert len(subjects) == 52
    assert len(holdout) == 12
    assert len(exploration) == 40
    assert not holdout & exploration
    assert "24" in exploration


def test_cv_folds_are_rank_balanced() -> None:
    mapping = data._cv_folds([str(value) for value in range(40)])
    counts = pd.Series(mapping).value_counts().to_dict()
    assert counts == {0: 8, 1: 8, 2: 8, 3: 8, 4: 8}


def test_leave_one_out_excludes_target_value() -> None:
    frame = pd.DataFrame({"manifest_index": [1, 1, 1, 2, 2], "value": [1.0, 2.0, 4.0, 3.0, 9.0]})
    mean, std, count = data._leave_one_out(frame, "value")
    assert np.allclose(mean[:3], [3.0, 2.5, 1.5])
    assert np.allclose(mean[3:], [9.0, 3.0])
    assert np.array_equal(count, [2, 2, 2, 1, 1])
    assert np.all(std >= 0)


def _toy_frame() -> pd.DataFrame:
    rows = []
    # Two subjects, two balanced labels each. Candidate actions flip only the
    # second row of each subject.
    for subject in ("1", "2"):
        for index, label in enumerate((0, 1)):
            noop = 0
            flip = 1 if index == 1 else 0
            rows.append(
                {
                    "fold_id": 0,
                    "seed_id": 0,
                    "subject_id": subject,
                    "outcome_label": label,
                    "pred_noop": noop,
                    "pred_amplify": flip,
                    "pred_geometry": flip,
                    "pred_erase": flip,
                    "effect_amplify": int(flip == label) - int(noop == label),
                    "effect_geometry": int(flip == label) - int(noop == label),
                    "effect_erase": int(flip == label) - int(noop == label),
                    "flip_count": 3 if flip != noop else 0,
                    "target_baseline_error": int(noop != label),
                }
            )
    return pd.DataFrame(rows)


def test_action_selection_defaults_to_keep_and_obeys_menu() -> None:
    frame = _toy_frame()
    selected = metrics.select_actions(frame, [False, True, False, True], ("amplify", "geometry"))
    assert selected.tolist() == ["noop", "amplify", "noop", "amplify"]


def test_subject_balanced_metric_detects_rescue() -> None:
    frame = _toy_frame()
    selected = np.asarray(["noop", "amplify", "noop", "amplify"], dtype=object)
    result = metrics.policy_metrics(frame, selected, bootstrap_repetitions=50)
    assert result["mean_subject_delta_BA"] == 0.5
    assert result["rescue_precision"] == 1.0
    assert result["unsafe_intervention_rate"] == 0.0


def test_strong_gate_requires_nonzero_safe_gain() -> None:
    good = {
        "mean_subject_delta_BA": 0.011,
        "bootstrap_CI95_L": 0.001,
        "nonnegative_run_fraction": 1.0,
        "positive_run_fraction": 1.0,
        "action_rate": 0.05,
        "rescue_precision": 0.6,
        "unsafe_intervention_rate": 0.4,
        "recovered_oracle_headroom": 0.1,
    }
    assert policies.meets_strong_candidate(good)
    bad = dict(good, action_rate=0.0)
    assert not policies.meets_strong_candidate(bad)


def test_outer_audit_does_not_confuse_router_fold() -> None:
    common.require_false_outer(pd.DataFrame({"router_fold": [0, 1]}), "synthetic")
    try:
        common.require_false_outer(pd.DataFrame({"outer_test_used": [False, True]}), "synthetic")
    except RuntimeError:
        pass
    else:
        raise AssertionError("True outer-test marker was not rejected")
