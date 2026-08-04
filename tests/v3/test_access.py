import json

import pytest

from hsc_tta.v3.access import AccessPhase, EpisodeAccessController


def test_four_phase_access_and_hash_guard(tmp_path):
    controller = EpisodeAccessController("s1", "cfg")
    assert controller.access_adapt("A") == "A"
    with pytest.raises(PermissionError): controller.access_future("V", "y")
    controller.begin_probe({"no_tta": "n", "t3a": "t"})
    assert controller.phase is AccessPhase.PROBE_PHASE and controller.access_probe("P") == "P"
    with pytest.raises(PermissionError, match="labels"): controller.access_probe("P", labels="hidden")
    path = tmp_path / "decision.json"
    decision = {"subject_id":"s1","selected_action":"t3a","action_state_hash":"t","config_hash":"cfg","lambda_index":3}
    digest = controller.freeze_decision(decision, path)
    assert digest and controller.phase is AccessPhase.FROZEN_DECISION
    assert controller.access_future("V", "y") == ("V", "y")
    assert json.loads(path.with_suffix(".json.freeze.json").read_text())["future_opened"] is True


def test_access_rejects_state_and_decision_tampering(tmp_path):
    controller = EpisodeAccessController("s", "cfg")
    with pytest.raises(ValueError): controller.begin_probe({})
    controller.begin_probe({"no_tta":"h"})
    path=tmp_path/"d.json"
    with pytest.raises(ValueError,match="state hash"): controller.freeze_decision(
        {"subject_id":"s","selected_action":"no_tta","action_state_hash":"bad","config_hash":"cfg","lambda_index":0},path)
    controller.freeze_decision({"subject_id":"s","selected_action":"no_tta","action_state_hash":"h","config_hash":"cfg","lambda_index":0},path)
    path.write_text("tampered")
    with pytest.raises(RuntimeError,match="changed"): controller.access_future("V","y")
