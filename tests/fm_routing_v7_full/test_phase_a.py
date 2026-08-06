from __future__ import annotations

import csv
import json
import pathlib

import pytest

from hsc_tta.fm_routing_full.compatibility import COMBINATION_ORDER, DATASET_ORDER, MODEL_ORDER, choose_core
from hsc_tta.fm_routing_full.core import guarded_target, sha256_file


@pytest.fixture(scope="module")
def repo() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def state(repo: pathlib.Path) -> dict:
    return json.loads((repo / "outputs/fm_routing_v7_full/RUN_STATE.json").read_text())


@pytest.fixture(scope="module")
def decision(repo: pathlib.Path) -> dict:
    return json.loads((repo / "delivery/fm_routing_v7_full/FINAL_DECISION.json").read_text())


@pytest.fixture(scope="module")
def matrix(repo: pathlib.Path) -> list[dict]:
    with (repo / "outputs/fm_routing_v7_full/audit/COMPATIBILITY_MATRIX.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_candidate_priority_is_preregistered() -> None:
    assert MODEL_ORDER == ["cbramod", "eegpt", "bendr", "brant", "eeg2rep", "neurogpt", "biot", "labram"]


def test_dataset_priority_is_preregistered() -> None:
    assert DATASET_ORDER == ["hmc", "eegmmidb", "sleepedffull", "bcic2a"]
    assert COMBINATION_ORDER[0] == ["hmc", "eegmmidb"]


def test_protocol_was_frozen_before_matrix(repo: pathlib.Path, state: dict) -> None:
    freeze = json.loads((repo / "outputs/fm_routing_v7_full/audit/COMPATIBILITY_PROTOCOL_FREEZE.json").read_text())
    assert freeze["frozen_before_compatibility_results"] is True
    assert freeze["freeze_hash"] == state["compatibility_protocol_hash"]
    assert state["completed_stages"].index("COMPATIBILITY_PROTOCOL_FROZEN") < state["completed_stages"].index("MODEL_DATASET_COMPATIBILITY_COMPLETE")


def test_predecessor_verdicts_exact(repo: pathlib.Path) -> None:
    audit = json.loads((repo / "outputs/fm_routing_v7_full/audit/PREDECESSORS.json").read_text())
    assert audit["exact_verdicts_match"] is True
    assert audit["old_v7"]["terminal"] is True
    assert audit["old_v7r"]["terminal"] is True


def test_predecessor_hashes_unchanged(repo: pathlib.Path, state: dict) -> None:
    assert all(sha256_file(repo / name) == digest for name, digest in state["predecessor_hashes"].items())


def test_historical_guard(repo: pathlib.Path) -> None:
    with pytest.raises(PermissionError):
        guarded_target(repo, repo / "outputs/fm_routing_v7/RUN_STATE.json")
    assert guarded_target(repo, repo / "outputs/fm_routing_v7_full/RUN_STATE.json").name == "RUN_STATE.json"


def test_matrix_is_complete(matrix: list[dict]) -> None:
    assert len(matrix) == 32
    assert {(row["model"], row["dataset"]) for row in matrix} == {(model, dataset) for model in MODEL_ORDER for dataset in DATASET_ORDER}


def test_matrix_has_no_performance_columns(matrix: list[dict]) -> None:
    forbidden = {"accuracy", "f1", "auroc", "loss", "risk", "performance", "prediction", "label"}
    assert not any(any(token in column.lower() for token in forbidden) for column in matrix[0])


def test_hmc_never_fakes_channel_semantics(matrix: list[dict]) -> None:
    compatible = [row["model"] for row in matrix if row["dataset"] == "hmc" and row["compatible"] == "True"]
    assert compatible == ["cbramod"]
    for row in matrix:
        if row["dataset"] == "hmc" and row["model"] in {"biot", "labram", "eegpt"}:
            assert row["A_COMP_3"] == "False"


def test_sleepedffull_absence_blocks_every_pair(matrix: list[dict]) -> None:
    assert all(row["compatible"] == "False" for row in matrix if row["dataset"] == "sleepedffull")


def test_no_core_can_be_selected(matrix: list[dict]) -> None:
    normalized = [{**row, "compatible": row["compatible"] == "True"} for row in matrix]
    assert choose_core(normalized) is None


def test_phase_a_gate_fails(decision: dict) -> None:
    assert decision["phase_a_gate"]["A3"] is False
    assert decision["phase_a_gate"]["A7"] is True
    assert not all(decision["phase_a_gate"].values())


def test_terminal_scientific_stop(state: dict, decision: dict) -> None:
    assert state["terminal"] is True and state["technical_block"] is False
    assert state["verdict"] == decision["verdict"] == "V7_STOP_NO_ADMISSIBLE_EXPERT_POOL"


def test_no_core_freeze(repo: pathlib.Path, decision: dict) -> None:
    assert decision["core_benchmark_freeze_created"] is False
    assert not (repo / "delivery/fm_routing_v7_full/CORE_BENCHMARK_FREEZE.json").exists()


@pytest.mark.parametrize("directory", ["expert_cache", "expert_heads", "oracle", "prefix_features", "routing", "abstention", "full_method", "formal_calibration", "internal_final", "external", "ablations"])
def test_no_downstream_directory(repo: pathlib.Path, directory: str) -> None:
    assert not (repo / "outputs/fm_routing_v7_full" / directory).exists()


def test_no_downstream_delivery_reports(repo: pathlib.Path) -> None:
    forbidden = ["EXPERT_PROTOCOL.md", "FULL_ORACLE_REPORT.md", "SIMPLE_ROUTING_REPORT.md", "PARES_METHOD.md", "INTERNAL_FINAL_REPORT.md", "EXTERNAL_REPLICATION_REPORT.md", "THEORY.md", "ABLATIONS.md", "ICLR_READINESS_ASSESSMENT.md"]
    assert all(not (repo / "delivery/fm_routing_v7_full" / name).exists() for name in forbidden)


def test_access_flags_remain_closed(state: dict, decision: dict) -> None:
    assert state["performance_metrics_read_for_selection"] is False
    assert state["protected_subjects_opened"] is False
    assert state["cap_opened"] is False
    assert all(value is False for value in decision["protected_access"].values())


def test_no_method_development(state: dict, decision: dict) -> None:
    assert state["router_developed"] is False
    assert state["abstention_developed"] is False
    assert state["full_method_developed"] is False
    assert decision["integrity"]["backbone_finetuning"] is False
    assert decision["integrity"]["evaluation_leakage"] is False


def test_checkpoint_and_code_provenance(repo: pathlib.Path) -> None:
    registry = json.loads((repo / "outputs/fm_routing_v7_full/audit/MODEL_REGISTRY.json").read_text())
    assert sorted(registry, key=lambda name: registry[name]["priority"]) == MODEL_ORDER
    assert all(spec["code_commit"] for spec in registry.values())
    assert registry["cbramod"]["checkpoint_sha256"] == "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"
    assert registry["biot"]["checkpoint_sha256"] == "40f55f5d23e83796495616c8145c8336fcff2b901c42e8ba5115223081c2ad70"
    assert registry["labram"]["checkpoint_sha256"] == "7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c"


def test_cbramod_smoke_is_frozen_and_deterministic(repo: pathlib.Path) -> None:
    smoke = json.loads((repo / "outputs/fm_routing_v7_full/audit/FORWARD_SMOKE.json").read_text())
    current = [row for row in smoke["rows"] if row["model"] == "cbramod" and row["dataset"] != "all"]
    assert len(current) == 3
    assert all(row["passed"] and row["backbone_frozen"] and row["repeat_max_abs_error"] == 0 for row in current)


def test_failures_file_has_header_only(repo: pathlib.Path) -> None:
    lines = (repo / "outputs/fm_routing_v7_full/FAILURES.csv").read_text().splitlines()
    assert len(lines) == 1


def test_manifest_has_hashes(repo: pathlib.Path) -> None:
    manifest = json.loads((repo / "delivery/fm_routing_v7_full/DELIVERY_MANIFEST.json").read_text())
    assert manifest["terminal"] is True
    assert manifest["file_count"] == len(manifest["files"])
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])
