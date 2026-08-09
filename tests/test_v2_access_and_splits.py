from __future__ import annotations

import json

import pandas as pd
import pytest

from hsc_tta.v2.access_guard import OldFinalAccessGuard
from hsc_tta.v2.splits import file_sha256, generate_v2_splits, validate_v2_split


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


def test_access_guard_reads_untainted_and_postfreeze_only(tmp_path):
    path = tmp_path / "ordinary.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(path)
    guard = OldFinalAccessGuard(tmp_path)
    assert guard.is_tainted("x/final_test_outcomes/y")
    assert not guard.is_tainted(path)
    assert guard.read_parquet(path, purpose="development").x.iloc[0] == 1
    tainted = tmp_path / "ALL_SUBJECT_DECISIONS.parquet"
    freeze = guard.freeze; freeze.parent.mkdir(parents=True)
    freeze.write_text(json.dumps({"methods_frozen": False}))
    with pytest.raises(PermissionError):
        guard.assert_access(tainted, purpose="exploratory_replication")


def test_split_leakage_and_size_guards():
    payload = {"dataset": "hmc", "meta_fit_subjects": [f"m{i}" for i in range(30)],
               "calibration_subjects": [f"c{i}" for i in range(14)],
               "outer_evaluation_subjects": [f"e{i}" for i in range(11)],
               "excluded_old_final_subjects": ["old"], "source_task_head_subjects": ["source"]}
    for field, value, match in (("outer_evaluation_subjects", "old", "old final"),
                                ("outer_evaluation_subjects", "source", "source task-head")):
        broken = {key: list(values) if isinstance(values, list) else values for key, values in payload.items()}
        broken[field][0] = value
        with pytest.raises(RuntimeError, match=match):
            validate_v2_split(broken)
    broken = {key: list(values) if isinstance(values, list) else values for key, values in payload.items()}
    broken["meta_fit_subjects"].pop()
    with pytest.raises(RuntimeError, match="size"):
        validate_v2_split(broken)


def test_generate_splits_is_deterministic_and_complete(tmp_path):
    for dataset, n_meta, n_cal in (("hmc", 35, 20), ("eegmmidb", 30, 15)):
        split_dir = tmp_path / "data/splits" / dataset
        split_dir.mkdir(parents=True)
        roles = {"meta_risk_train": [f"{dataset}:m{i}" for i in range(n_meta)],
                 "conformal_calibration": [f"{dataset}:c{i}" for i in range(n_cal)],
                 "task_head_train": [f"{dataset}:s{i}" for i in range(5)],
                 "final_test": [f"{dataset}:f{i}" for i in range(3)]}
        (split_dir / "seed_0.json").write_text(json.dumps({"roles": roles}))
    first = generate_v2_splits(tmp_path, seeds=(0,))
    assert first["files"] == 10
    path = tmp_path / "data/splits_v2_dev/hmc/seed_0/outer_fold_0.json"
    before = file_sha256(path)
    second = generate_v2_splits(tmp_path, seeds=(0,))
    assert second["files"] == 10 and file_sha256(path) == before
