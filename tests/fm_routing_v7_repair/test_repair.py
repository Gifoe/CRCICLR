from __future__ import annotations

import json
import pathlib

import pytest

from hsc_tta.fm_routing_repair.core import PROTECTED_FLAGS, adapter_gate, guarded_target


REPO = pathlib.Path(__file__).resolve().parents[2]
OUTPUT = REPO / "outputs/fm_routing_v7_repair"
DELIVERY = REPO / "delivery/fm_routing_v7_repair"


def load(path: pathlib.Path):
    return json.loads(path.read_text())


def test_predecessor_verdict_and_terminal():
    decision = load(REPO / "delivery/fm_routing_v7/V7_STAGE0A_DECISION.json")
    assert decision["verdict"] == "V7_STAGE0A_STOP_MODEL_QUALIFICATION_FAILURE"
    assert decision["stopping_gate"] == "Gate Q"
    assert decision["terminal"] is True


def test_predecessor_oracle_absent():
    assert not (REPO / "outputs/fm_routing_v7/results/FULL_ORACLE_SUMMARY.csv").exists()


@pytest.mark.parametrize("relative", [
    "outputs/fm_routing_v7/x",
    "delivery/fm_routing_v7/x",
    "outputs/online_blockwise_v6/x",
    "delivery/online_blockwise_v6/x",
])
def test_predecessor_write_guard(relative):
    with pytest.raises(PermissionError):
        guarded_target(REPO, REPO / relative)


def test_freeze_precedes_adapter_audit():
    freeze = load(DELIVERY / "V7R_FREEZE.json")
    audit = load(OUTPUT / "audit/ADAPTER_FIDELITY.json")
    assert audit["freeze_hash"]
    assert audit["freeze_precedes_repaired_metrics"] is True
    assert freeze["post_hoc_development_repair"] is True


def test_model_and_checkpoint_pool_frozen():
    freeze = load(DELIVERY / "V7R_FREEZE.json")
    assert freeze["models"] == ["cbramod", "labram", "biot"]
    assert set(freeze["checkpoint_hashes"]) == {"cbramod", "labram", "biot"}


def test_only_two_readout_families_frozen():
    freeze = load(DELIVERY / "V7R_FREEZE.json")
    assert set(freeze["readout_families"]) == {"H0_GLOBAL_LOGREG", "H1_TOKEN_ATTENTION_POOL"}
    assert freeze["readout_families"]["H1_TOKEN_ATTENTION_POOL"]["parameter_cap"] == 100000


def test_adapter_evidence_and_no_performance_selection():
    audit = load(OUTPUT / "audit/ADAPTER_FIDELITY.json")
    assert all(item["evidence"] for item in audit["models"].values())
    assert audit["performance_metrics_read_for_adapter_selection"] is False
    assert all(item["performance_driven_selection"] is False for item in audit["models"].values())


def test_biot_checkpoint_channel_mismatch_is_explicit():
    audit = load(OUTPUT / "audit/ADAPTER_FIDELITY.json")
    text = audit["models"]["biot"]["fidelity_limit"]
    assert "C3-M2/C4-M1" in text and "C3-P3/C4-P4" in text
    assert audit["models"]["biot"]["dataset_fidelity"] == {"eegmmidb": True, "hmc": False}


def test_labram_bipolar_reference_mismatch_is_explicit():
    audit = load(OUTPUT / "audit/ADAPTER_FIDELITY.json")
    assert audit["models"]["labram"]["dataset_fidelity"] == {"eegmmidb": True, "hmc": False}
    assert "bipolar" in audit["models"]["labram"]["fidelity_limit"]


def test_structured_smoke_tokens_are_finite_and_ordered():
    audit = load(OUTPUT / "audit/ADAPTER_FIDELITY.json")
    smoke = audit["structured_smoke"]
    assert smoke["all_tokens_finite"] is True
    assert smoke["identity_and_order_recoverable"] is True
    assert smoke["backbone_requires_grad"] is False
    assert smoke["backbone_gradients"] is False
    assert len(smoke["rows"]) == 6
    assert all(row["token_shape"][0] == 1 for row in smoke["rows"])


def test_hmc_labram_subwindows_are_ordered():
    audit = load(OUTPUT / "audit/ADAPTER_FIDELITY.json")
    row = next(x for x in audit["structured_smoke"]["rows"] if x["dataset"] == "hmc" and x["model"] == "labram")
    ids = row["metadata"]["subwindow_id"]
    assert ids == sorted(ids)
    assert set(ids) == {0, 1, 2}


def test_biot_retains_sequence_before_mean():
    audit = load(OUTPUT / "audit/ADAPTER_FIDELITY.json")
    for row in audit["structured_smoke"]["rows"]:
        if row["model"] == "biot":
            assert row["token_shape"][1] > 1
            assert len(row["metadata"]["channel_or_group_id"]) == row["token_shape"][1]


def test_full_canonical_coverage_and_identity():
    coverage = load(OUTPUT / "audit/ADAPTER_FIDELITY.json")["coverage"]
    assert coverage["canonical_subjects"] == 155
    assert coverage["canonical_samples"] == 89474
    assert coverage["minimum_subject_coverage"] == 1.0
    assert coverage["minimum_sample_coverage"] >= 0.95
    assert coverage["sample_id_and_label_exact_match"] is True


def test_adapter_gate_function():
    models = {"a": {"official_fidelity": True, "performance_driven_selection": False}, "b": {"official_fidelity": False, "performance_driven_selection": False}}
    gates = adapter_gate(models, {"minimum_subject_coverage": 1.0, "minimum_sample_coverage": 1.0, "sample_id_and_label_exact_match": True}, {"all_tokens_finite": True, "identity_and_order_recoverable": True})
    assert gates == {"F1": False, "F2": True, "F3": True, "F4": True, "F5": True, "F6": True, "F7": True}


def test_scientific_stop_is_terminal_and_unique():
    state = load(OUTPUT / "RUN_STATE.json")
    decision = load(DELIVERY / "V7R_DECISION.json")
    assert state["state"] == "STOPPED"
    assert state["terminal"] is True
    assert state["verdict"] == decision["verdict"] == "V7R_STOP_ADAPTER_FIDELITY_FAILURE"
    assert decision["technical_block"] is False


def test_adapter_stop_prevents_downstream_results():
    forbidden = [
        OUTPUT / "heads/HEAD_SELECTION.csv",
        OUTPUT / "results/EXPERT_QUALIFICATION_GATE.csv",
        OUTPUT / "results/FULL_ORACLE_SUMMARY.csv",
        DELIVERY / "ORACLE_PROTOCOL_FREEZE.json",
    ]
    assert not any(path.exists() for path in forbidden)


def test_no_router_abstention_or_scout():
    forbidden = [
        REPO / "src/hsc_tta/fm_routing_repair/router.py",
        REPO / "src/hsc_tta/fm_routing_repair/abstention.py",
        REPO / "src/hsc_tta/fm_routing_repair/scout.py",
        OUTPUT / "router",
        OUTPUT / "abstention",
    ]
    assert not any(path.exists() for path in forbidden)


def test_protected_flags_false():
    state = load(OUTPUT / "RUN_STATE.json")
    decision = load(DELIVERY / "V7R_DECISION.json")
    assert all(state[flag] is False and decision[flag] is False for flag in PROTECTED_FLAGS)


def test_failures_file_contains_header_only():
    lines = (OUTPUT / "FAILURES.csv").read_text().strip().splitlines()
    assert lines == ["timestamp,phase,job,error"]


def test_manifest_confirms_old_v7_unchanged():
    manifest = load(DELIVERY / "DELIVERY_MANIFEST.json")
    assert manifest["old_v7_unchanged"] is True
    assert manifest["protected_flags_all_false"] is True
    assert manifest["router_abstention_scout_absent"] is True
