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
