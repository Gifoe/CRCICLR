from __future__ import annotations

import json
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


def test_frozen_protocol_scope() -> None:
    p = json.loads((EXP / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))
    assert p["phase_a_seed"] == 0
    assert p["folds"] == [0, 1, 2, 3, 4]
    assert p["backbone"] == "canonical EEGNet"
    assert p["outer_sealed_access"] is False


def test_runner_has_no_python_hash_or_rescue() -> None:
    text = (EXP / "code" / "run_geosr.py").read_text(encoding="utf-8")
    # The protocol forbids Python's process-randomized built-in hash function,
    # while deterministic SHA helper names such as ``role_hash`` are allowed.
    assert "= hash(" not in text
    assert "scientific_rescue_performed" in text
    assert "GEOSR_FINAL_CONSTRUCTIVE_STOP" in text


def test_method_set_is_frozen() -> None:
    text = (EXP / "code" / "run_geosr.py").read_text(encoding="utf-8")
    for method in ("CANONICAL_ERM", "SUBJECT_BALANCED_ERM", "RANDOM_RANK", "LOSS_HARD", "GEO_ONLY", "GEOSR"):
        assert method in text


def test_inner_partition_is_disjoint_and_exhaustive() -> None:
    import sys

    sys.path.insert(0, str(EXP / "code"))
    try:
        import run_geosr as geosr
    finally:
        sys.path.pop(0)

    subjects = [str(i) for i in range(34)]
    held: list[str] = []
    for k in range(geosr.INNER_K):
        fit_subjects, held_subjects = geosr.inner_partition(subjects, "OpenBMI", 0, k, 0)
        assert set(fit_subjects).isdisjoint(held_subjects)
        held.extend(held_subjects)
    assert geosr.subj_sort(held) == geosr.subj_sort(subjects)
    assert len(held) == len(set(held))


def test_decision_runner_is_screening_subset_only() -> None:
    text = (EXP / "code" / "run_geosr_decision.py").read_text(encoding="utf-8")
    assert 'FOLDS = (0, 1)' in text
    assert '"SUBJECT_BALANCED_ERM", "RANDOM_RANK", "GEOSR"' in text
    assert 'complete_protocol_required_for_final_claim' in text
    assert 'outcome_labels_read_before_lock' in text
