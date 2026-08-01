import json

import pandas as pd
import pytest

from hsc_tta.evaluation import verify_final_test_gate, write_pre_outcome_decisions
from hsc_tta.freeze import create_freeze_manifest


def _decision(freeze_hash: str):
    return pd.DataFrame(
        [{
            "dataset": "hmc", "seed": 0, "subject_id": "s", "episode_id": "e",
            "alpha": 0.2, "selected_action": "no_tta", "selected_lambda": 0.8,
            "certified_critical_index": 5, "nontrivial_certified": True,
            "status": "certified", "selection_reason": "context utility",
            "freeze_hash": freeze_hash,
        }]
    )


def test_final_test_refuses_missing_freeze(tmp_path):
    with pytest.raises(RuntimeError, match="not frozen"):
        verify_final_test_gate(tmp_path / "missing.json", tmp_path / "d.parquet", tmp_path / "lock.json")


def test_final_test_refuses_changed_frozen_config(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("alpha: 0.2\n")
    manifest_path = tmp_path / "manifest.json"
    manifest = create_freeze_manifest({"config": config}, git_commit="abc", metadata={}, output_path=manifest_path)
    decision_path = tmp_path / "decisions.parquet"
    lock_path = tmp_path / "decision.lock.json"
    write_pre_outcome_decisions(_decision(str(manifest["manifest_hash"])), decision_path, freeze_manifest_path=manifest_path, lock_path=lock_path)
    verify_final_test_gate(manifest_path, decision_path, lock_path)
    config.write_text("alpha: 0.1\n")
    with pytest.raises(RuntimeError, match="freeze hash mismatch"):
        verify_final_test_gate(manifest_path, decision_path, lock_path)
