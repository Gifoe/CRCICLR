"""Regression tests for the pre-result sampling lock."""
from __future__ import annotations

import unittest

import numpy as np

from sampling_core import select_pairs, select_pairs_from_norms, trial_view


class SamplingCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.a = np.arange(60, dtype=np.float32).reshape(30, 2) / 10
        self.y = np.asarray([0] * 10 + [1] * 10 + [0] * 5 + [1] * 5, np.int64)
        self.s = np.asarray(["1"] * 20 + ["2"] * 10)
        self.se = np.asarray([1] * 20 + [2] * 10, np.int64)
        self.kw = dict(model="EEGNet", task="OpenBMI_MI", fold=0, stage="temporal_bn", split="TRAIN")

    def test_reproducible_cap_and_isolation(self) -> None:
        x = select_pairs(self.a, self.y, self.s, self.se, **self.kw)
        z = select_pairs(self.a, self.y, self.s, self.se, **self.kw)
        np.testing.assert_array_equal(x.recipient, z.recipient)
        np.testing.assert_array_equal(x.donor, z.donor)
        streamed = select_pairs_from_norms(np.linalg.norm(self.a.astype(np.float64), axis=1),
                                           self.y, self.s, self.se, **self.kw)
        np.testing.assert_array_equal(x.recipient, streamed.recipient)
        np.testing.assert_array_equal(x.donor, streamed.donor)
        self.assertEqual(len(x.recipient), 26)
        self.assertTrue(all(row["selected_recipients"] <= 8 for row in x.coverage))
        self.assertTrue(np.all(self.s[x.recipient] == self.s[x.donor]))
        self.assertTrue(np.all(self.se[x.recipient] == self.se[x.donor]))
        self.assertTrue(np.all(self.y[x.recipient] != self.y[x.donor]))

    def test_nearest_norm_tie_breaks_by_original_row(self) -> None:
        a = np.asarray([[2.], [1.], [3.]], np.float32)
        y = np.asarray([0, 1, 1], np.int64)
        s = np.asarray(["A"] * 3)
        se = np.ones(3, np.int64)
        p = select_pairs(a, y, s, se, **self.kw)
        i = np.flatnonzero(p.recipient == 0)
        self.assertEqual(len(i), 1)
        self.assertEqual(int(p.donor[i[0]]), 1)

    def test_missing_class_is_reported_without_cross_subject_substitution(self) -> None:
        y = self.y.copy()
        y[20:] = 0
        p = select_pairs(self.a, y, self.s, self.se, **self.kw)
        rows = [r for r in p.coverage if r["subject"] == "2"]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["missing_eligible_donor_class"])
        self.assertEqual(rows[0]["pair_count"], 0)
        self.assertTrue(np.all(p.recipient < 20))

    def test_invalid_input_and_cap_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            select_pairs(self.a, self.y, self.s, self.se, cap=9, **self.kw)
        with self.assertRaises(ValueError):
            select_pairs(self.a, self.y, self.s, self.se, **(self.kw | {"split": "HELDOUT"}))

    def test_trial_view_never_uses_final_heldout(self) -> None:
        data = {
            "train_x": np.zeros((1, 2)), "train_y": np.asarray([0]), "train_subjects": np.asarray(["A"]),
            "future_train_x": np.ones((1, 2)), "future_train_y": np.asarray([1]),
            "future_train_subjects": np.asarray(["A"]),
            "outer_source_x": np.full((1, 2), 2.), "outer_source_y": np.asarray([0]),
            "outer_source_subjects": np.asarray(["B"]),
            "outer_future_x": np.full((1, 2), 3.), "outer_future_y": np.asarray([1]),
            "outer_future_subjects": np.asarray(["B"]),
            "source_session": 1, "future_session": 2,
            "final_heldout_x": np.full((1, 2), 999.),
        }
        train = trial_view(data, "TRAIN")
        outer = trial_view(data, "OUTER_DEVELOPMENT")
        np.testing.assert_array_equal(train.x[:, 0], [0, 1])
        np.testing.assert_array_equal(outer.x[:, 0], [2, 3])
        np.testing.assert_array_equal(train.sessions, [1, 2])
        np.testing.assert_array_equal(outer.sessions, [1, 2])
        with self.assertRaises(ValueError):
            trial_view(data, "FINAL")


if __name__ == "__main__":
    unittest.main()
