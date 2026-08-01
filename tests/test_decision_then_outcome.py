import pandas as pd
import pytest

from hsc_tta.evaluation import join_decisions_and_outcomes, write_pre_outcome_decisions
from hsc_tta.freeze import create_freeze_manifest


def test_decisions_are_frozen_before_outcomes_and_join_on_full_key(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("alpha: 0.2\n")
    manifest_path = tmp_path / "manifest.json"
    manifest = create_freeze_manifest({"config": config}, git_commit="abc", metadata={}, output_path=manifest_path)
    decisions = pd.DataFrame([{
        "dataset": "hmc", "seed": 0, "subject_id": "s", "episode_id": "e",
        "alpha": 0.2, "selected_action": "no_tta", "selected_lambda": 0.8,
        "certified_critical_index": 5, "nontrivial_certified": True,
        "status": "certified", "selection_reason": "context utility",
        "freeze_hash": str(manifest["manifest_hash"]),
    }])
    decision_path = tmp_path / "decisions.parquet"
    lock = write_pre_outcome_decisions(decisions, decision_path, freeze_manifest_path=manifest_path, lock_path=tmp_path / "lock.json")
    assert lock["decision_sha256"]
    outcomes = pd.DataFrame([{"dataset": "hmc", "seed": 0, "subject_id": "s", "episode_id": "e", "alpha": 0.2, "true_future_risk": 0.1}])
    assert join_decisions_and_outcomes(decisions, outcomes).true_future_risk.iloc[0] == 0.1


def test_decision_writer_rejects_future_columns(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("x: 1\n")
    manifest_path = tmp_path / "manifest.json"
    manifest = create_freeze_manifest({"config": config}, git_commit="abc", metadata={}, output_path=manifest_path)
    bad = pd.DataFrame([{"future_risk": 0.1}])
    with pytest.raises(ValueError, match="future"):
        write_pre_outcome_decisions(bad, tmp_path / "d.parquet", freeze_manifest_path=manifest_path, lock_path=tmp_path / "l.json")


def test_decision_writer_rejects_wrong_freeze_hash(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("x: 1\n")
    manifest_path = tmp_path / "manifest.json"
    create_freeze_manifest({"config": config}, git_commit="abc", metadata={}, output_path=manifest_path)
    row = pd.DataFrame([{
        "dataset": "hmc", "seed": 0, "subject_id": "s", "episode_id": "e",
        "alpha": 0.2, "selected_action": "no_tta", "selected_lambda": 0.8,
        "certified_critical_index": 5, "nontrivial_certified": True,
        "status": "certified", "selection_reason": "context", "freeze_hash": "wrong",
    }])
    with pytest.raises(ValueError, match="active freeze"):
        write_pre_outcome_decisions(row, tmp_path / "d.parquet", freeze_manifest_path=manifest_path, lock_path=tmp_path / "l.json")
