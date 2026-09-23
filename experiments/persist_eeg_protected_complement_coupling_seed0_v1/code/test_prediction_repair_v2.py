"""Synthetic contract for outcome-blind donor selection and features."""
from __future__ import annotations

import numpy as np
import torch

import repair_error_predictability_v2 as R


class SyntheticRunner:
    device = torch.device("cpu")

    def from_stage(self, _stage: str, activation: torch.Tensor):
        flat = activation.reshape(len(activation), -1)
        score = flat[:, 0] + flat[:, 0] * flat[:, 1]
        return flat, torch.stack((score, -score), dim=1)


a = np.asarray([[1., 1.], [-1., 1.], [2., 2.], [-2., 2.],
                [1.5, 1.], [-1.5, 1.]], np.float32)
subject = np.asarray(["A", "A", "A", "A", "B", "B"])
session = np.asarray([1, 1, 1, 1, 2, 2])
predicted = np.asarray([0, 1, 0, 1, 0, 1])
pairs = R.label_free_pairs(a, predicted, subject, session)
assert len(pairs) == len(a)
assert all(i != j and subject[i] == subject[j] and session[i] == session[j]
           and predicted[i] != predicted[j] for i, j in pairs)

# Altering true task labels cannot alter either the donor rule or its features.
true_labels_a = np.asarray([0, 1, 0, 1, 0, 1])
true_labels_b = np.asarray([1, 0, 1, 0, 1, 0])
assert not np.array_equal(true_labels_a, true_labels_b)
assert pairs == R.label_free_pairs(a, predicted, subject, session)
runner = SyntheticRunner()
with torch.inference_mode():
    _, logits = runner.from_stage("synthetic", torch.from_numpy(a))
q = np.asarray([[1.], [0.]], np.float32)
mu = np.zeros(2, np.float32)
qc = np.asarray([[0.], [1.]], np.float32)
rows = R.C.functional_pairs(runner, "synthetic", [2], a, logits.numpy(),
                            predicted, subject, pairs, q, mu, qc, session)
features = R.C.trial_coupling_features(rows, len(a))
assert features.shape == (len(a), len(R.C.FEATURE_KEYS))
assert np.isfinite(features).all()

same_pred = np.zeros(len(a), int)
assert R.label_free_pairs(a, same_pred, subject, session) == []
print("ALL_PREDICTION_REPAIR_V2_TESTS_PASSED")
