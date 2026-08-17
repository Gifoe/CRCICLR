from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FREEZE = load_module(
    "test_external_freeze",
    "experiments/persist_eeg_external_actionability_v1/code/freeze_protocol.py",
)
AUDIT = load_module(
    "test_external_actionability",
    "experiments/persist_eeg_external_actionability_v1/code/external_actionability_v1.py",
)


def test_frozen_subject_split_is_disjoint_and_exhaustive() -> None:
    split = FREEZE.split_subjects()
    primary = [set(split[key]) for key in (
        "task_head_train", "block_discovery", "confirmatory_calibration", "outer_test"
    )]
    assert [len(group) for group in primary] == [45, 30, 15, 19]
    assert len(set().union(*primary)) == 109
    assert all(not left.intersection(right) for index, left in enumerate(primary) for right in primary[index + 1 :])
    assert set(split["task_head_fit"]).isdisjoint(split["task_head_validation"])
    assert set(split["task_head_fit"]) | set(split["task_head_validation"]) == set(split["task_head_train"])


def test_balanced_accuracy_is_mean_per_class_recall() -> None:
    truth = np.asarray([0, 0, 1, 1, 1, 2])
    prediction = np.asarray([0, 1, 1, 1, 0, 2])
    expected = np.mean([0.5, 2 / 3, 1.0])
    assert abs(AUDIT.balanced_accuracy_score(truth, prediction) - expected) < 1e-12


def test_exact_matched_random_displacement_norm() -> None:
    rng = np.random.default_rng(7)
    residual = rng.normal(size=(31, 128))
    candidate_basis, _ = np.linalg.qr(rng.normal(size=(128, 8)))
    random_basis, _ = np.linalg.qr(rng.normal(size=(128, 8)))
    target = residual @ candidate_basis[:, :8] @ candidate_basis[:, :8].T
    matched = AUDIT.exact_matched_delta(residual, target, random_basis[:, :8])
    np.testing.assert_allclose(
        np.linalg.norm(matched, axis=1), np.linalg.norm(target, axis=1), atol=5e-6, rtol=0,
    )


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    adjusted = AUDIT.holm({"a": 0.01, "b": 0.03, "c": 0.02, "d": 0.9})
    order = ["a", "c", "b", "d"]
    values = [adjusted[key] for key in order]
    assert values == sorted(values)
    assert all(0 <= value <= 1 for value in values)
