import numpy as np
import pytest

from hsc_tta.fm_routing.core import (
    class_balanced_risk,
    fold_roles,
    gate_q,
    normalized_inverse_frequency,
    rescuable_error,
    winner_shares,
)


def test_fold_roles_are_disjoint_and_complete():
    for evaluation in range(5):
        training, validation, test = fold_roles(evaluation)
        assert test == evaluation
        assert validation == (evaluation + 1) % 5
        assert sorted(training + [validation, test]) == list(range(5))
        assert len(set(training + [validation, test])) == 5


def test_training_only_class_weights_and_risk():
    weights = normalized_inverse_frequency([0, 0, 1], [0, 1])
    assert weights[1] == pytest.approx(2 * weights[0])
    assert class_balanced_risk([0, 0, 1], [0, 1, 1], weights) == pytest.approx(0.25)


def test_tied_winners_split_weight():
    shares = winner_shares(np.array([[0.1, 0.1, 0.2], [0.2, 0.1, 0.1]]), ["a", "b", "c"])
    assert shares == pytest.approx({"a": 0.25, "b": 0.5, "c": 0.25})


def test_rescuable_error_uses_only_base_errors():
    assert rescuable_error([True, False, False], np.array([[False, False], [True, False], [False, False]])) == 0.5
    assert np.isnan(rescuable_error([True], np.array([[False, False]])))


def test_gate_q_is_strict_at_registered_thresholds():
    gates = gate_q(
        probability_sane=True, embedding_sane=True, all_classes_present=True,
        nonconstant_subject_rate=0.95, seed_ba_std=0.05, dataset_ba=0.28,
        median_subject_ba=0.25, cbramod_ba=0.4, positive_seed_count=4,
        folds_noncollapsed=True, n_classes=5,
    )
    assert all(gates.values())


def test_protected_shortcut_names_are_not_implemented():
    import hsc_tta.fm_routing as package
    root = package.__path__[0]
    import pathlib
    assert not (pathlib.Path(root) / "router.py").exists()
    assert not (pathlib.Path(root) / "abstention.py").exists()
    assert not (pathlib.Path(root) / "scout.py").exists()
