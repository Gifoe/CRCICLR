from __future__ import annotations

import json

import pytest

from hsc_tta.v2.access_guard import OldFinalAccessGuard
from hsc_tta.v2.splits import validate_v2_split


def test_old_final_access_guard(tmp_path):
    guard = OldFinalAccessGuard(tmp_path)
    tainted = tmp_path / "outputs/full_experiment/ALL_SUBJECT_DECISIONS.parquet"
    guard.assert_access(tainted, purpose="v1_oracle_diagnostic")
    with pytest.raises(PermissionError):
        guard.assert_access(tainted, purpose="development")
    freeze = tmp_path / "outputs/v2_joint_certified/freeze/V2_METHOD_FREEZE.json"
    freeze.parent.mkdir(parents=True)
    freeze.write_text(json.dumps({"methods_frozen": True}))
    guard.assert_access(tainted, purpose="exploratory_replication")


def test_nested_split_isolation():
    payload = {"dataset": "hmc", "meta_fit_subjects": [f"m{i}" for i in range(30)],
               "calibration_subjects": [f"c{i}" for i in range(14)],
               "outer_evaluation_subjects": [f"e{i}" for i in range(11)],
               "excluded_old_final_subjects": ["old"], "source_task_head_subjects": ["source"]}
    validate_v2_split(payload)
    payload["outer_evaluation_subjects"][0] = "m0"
    with pytest.raises(RuntimeError, match="overlap"):
        validate_v2_split(payload)


def test_cap_is_forbidden_from_v2_development():
    with pytest.raises(ValueError):
        validate_v2_split({"dataset": "cap"})
