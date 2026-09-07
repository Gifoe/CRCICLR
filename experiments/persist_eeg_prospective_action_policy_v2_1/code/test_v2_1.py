from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ensemble_baselines import build_ensemble_baselines
from equivalence_analysis import build_consensus_controls
from evaluate import paired_subject_bootstrap


def synthetic_frame() -> pd.DataFrame:
    prediction = np.array([0, 0, 1, 1, 0, 1], dtype=int)
    probability = np.array([0.20, 0.40, 0.60, 0.80, 0.25, 0.75], dtype=float)
    margin = np.log(probability / (1.0 - probability))
    frame = pd.DataFrame(
        {
            "manifest_index": [10, 10, 10, 10, 20, 20],
            "subject_id": ["1"] * 4 + ["2"] * 2,
            "session_id": ["s1"] * 6,
            "fold_id": [0, 0, 1, 1, 0, 1],
            "seed_id": [0, 1, 0, 1, 0, 0],
            "pred_noop": prediction,
            "p1_noop": probability,
            "margin_noop": margin,
            "outcome_label": [0, 0, 0, 0, 1, 1],
        }
    )
    other_vote = (
        frame.groupby("manifest_index").pred_noop.transform("sum").to_numpy() - prediction
    ) / (frame.groupby("manifest_index").pred_noop.transform("size").to_numpy() - 1)
    frame["other_run_base_majority"] = (other_vote >= 0.5).astype(int)
    frame["other_run_base_disagrees"] = (frame.other_run_base_majority != frame.pred_noop).astype(int)
    frame["pred_amplify"] = 1 - prediction
    frame["pred_geometry"] = 1 - prediction
    frame["pred_erase"] = prediction
    return frame


class EnsembleTests(unittest.TestCase):
    def test_leave_target_run_excludes_target(self) -> None:
        frame = synthetic_frame()
        methods = build_ensemble_baselines(frame)
        expected = np.array([2 / 3, 2 / 3, 1 / 3, 1 / 3, 1.0, 0.0])
        np.testing.assert_allclose(methods["B1_OTHER_RUN_HARD_MAJORITY"].probability, expected)

    def test_all_run_tie_is_class_one(self) -> None:
        frame = synthetic_frame()
        prediction = build_ensemble_baselines(frame)["B2_ALL_RUN_HARD_MAJORITY"].prediction
        np.testing.assert_array_equal(prediction[:4], np.ones(4, dtype=int))

    def test_confidence_weighting_is_label_free(self) -> None:
        frame = synthetic_frame()
        first = build_ensemble_baselines(frame)["B7_CONFIDENCE_WEIGHTED_KEEP_ENSEMBLE"]
        changed = frame.copy()
        changed["outcome_label"] = 1 - changed.outcome_label
        second = build_ensemble_baselines(changed)["B7_CONFIDENCE_WEIGHTED_KEEP_ENSEMBLE"]
        np.testing.assert_array_equal(first.prediction, second.prediction)
        np.testing.assert_allclose(first.probability, second.probability)


class EquivalenceTests(unittest.TestCase):
    def test_c2_c3_match_frozen_i003_sample_by_sample(self) -> None:
        frame = synthetic_frame()
        majority = frame.other_run_base_majority.to_numpy(dtype=int)
        baseline = frame.pred_noop.to_numpy(dtype=int)
        disagree = majority != baseline
        full_selected = np.full(len(frame), "noop", dtype=object)
        safe_selected = np.full(len(frame), "noop", dtype=object)
        full_selected[disagree] = "amplify"
        # Leave one disagreement action-masked in the protected-safe policy.
        safe_indices = np.flatnonzero(disagree)[:-1]
        safe_selected[safe_indices] = "geometry"
        full_prediction = baseline.copy()
        safe_prediction = baseline.copy()
        full_prediction[full_selected != "noop"] = majority[full_selected != "noop"]
        safe_prediction[safe_selected != "noop"] = majority[safe_selected != "noop"]
        policies = {
            "I003_CROSS_RUN_FULL": {"selected": full_selected, "prediction": full_prediction},
            "I003_CROSS_RUN_PROTECTED_SAFE": {"selected": safe_selected, "prediction": safe_prediction},
        }
        controls = build_consensus_controls(frame, policies)
        np.testing.assert_array_equal(
            controls["C2_ACTION_MASKED_DIRECT_CONSENSUS_FULL"].prediction,
            full_prediction,
        )
        np.testing.assert_array_equal(
            controls["C3_ACTION_MASKED_DIRECT_CONSENSUS_PROTECTED_SAFE"].prediction,
            safe_prediction,
        )


class BootstrapTests(unittest.TestCase):
    def test_subject_bootstrap_uses_subject_vector(self) -> None:
        low, high = paired_subject_bootstrap(np.full(7, 0.125), repetitions=1000, seed=7)
        self.assertAlmostEqual(low, 0.125, places=15)
        self.assertAlmostEqual(high, 0.125, places=15)

    def test_subject_bootstrap_is_deterministic(self) -> None:
        values = np.array([-0.1, 0.0, 0.2, 0.3])
        first = paired_subject_bootstrap(values, repetitions=2000, seed=11)
        second = paired_subject_bootstrap(values, repetitions=2000, seed=11)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

