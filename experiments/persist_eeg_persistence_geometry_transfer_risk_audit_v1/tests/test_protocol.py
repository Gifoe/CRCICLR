from __future__ import annotations

import importlib.util
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_runner", EXP / "code" / "run_audit.py")
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def test_stable_seed_is_reproducible_and_not_python_hash() -> None:
    assert audit.stable_seed("descriptor", "OpenBMI", 0, 0) == audit.stable_seed("descriptor", "OpenBMI", 0, 0)
    assert "hash(" not in (EXP / "code" / "run_audit.py").read_text(encoding="utf-8")


def test_registered_scope_constants() -> None:
    assert audit.SEED == 0
    assert audit.FOLDS == (0, 1, 2, 3, 4)
    assert audit.DATASETS == ("OpenBMI", "WBCIC")
    assert audit.PER_GROUP == 32
    assert audit.Q_MIN == 16


def test_no_forbidden_outcome_construction_in_runner() -> None:
    text = (EXP / "code" / "run_audit.py").read_text(encoding="utf-8")
    assert "canonical_outcome_indices_materialized" in text
    assert "canonical_outcome_labels_read" in text
    assert "second_backbone_run" in text
