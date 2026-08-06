from __future__ import annotations

import hashlib
import json
import pathlib

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
OUTPUT = REPO / "outputs/fm_routing_v7"
DELIVERY = REPO / "delivery/fm_routing_v7"


def test_v6_line_is_closed_before_v7():
    decision = json.loads((REPO / "delivery/online_blockwise_v6/V6_STAGE0_DECISION.json").read_text())
    state = json.loads((REPO / "outputs/online_blockwise_v6/RUN_STATE.json").read_text())
    assert decision["verdict"] == "STOPPED_NO_DYNAMIC_HEADROOM"
    assert decision["stopping_gate"] == "Gate A"
    assert state["terminal"] is True
    assert {"B1", "B2", "B3", "B4", "Gate B", "Gate C", "method development"} <= set(decision["later_stages_not_run"])


def test_protected_flags_remain_false():
    state = json.loads((OUTPUT / "RUN_STATE.json").read_text())
    for key in (
        "formal_calibration_opened", "internal_final_opened", "cap_opened",
        "sleep_edf_opened", "bcic2a_opened", "router_developed",
        "abstention_developed", "full_method_entered",
    ):
        assert state[key] is False


def test_model_pool_frozen_before_results_and_three_families():
    freeze = json.loads((DELIVERY / "V7_MODEL_POOL_FREEZE.json").read_text())
    provenance = json.loads((OUTPUT / "audit/MODEL_PROVENANCE.json").read_text())
    assert freeze["candidate_priority"] == ["LaBraM", "EEGPT", "BIOT", "BENDR"]
    assert freeze["model_order"] == ["cbramod", "labram", "biot"]
    families = {value["family"] for value in provenance["models"].values()}
    assert len(families) == 3
    assert "probe metrics" in freeze["unread_result_types"]


def test_checkpoint_hashes_match_frozen_files():
    paths = json.loads((OUTPUT / "audit/RESOLVED_MODEL_PATHS.json").read_text())
    expected = json.loads((OUTPUT / "audit/CHECKPOINT_HASHES.json").read_text())
    for name, path in paths.items():
        assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() == expected[name]


def test_adapters_are_frozen_and_label_independent():
    specs = json.loads((OUTPUT / "audit/ADAPTER_SPECS.json").read_text())
    assert set(specs) == {"cbramod", "labram", "biot"}
    serialized = json.dumps(specs).lower()
    assert "evaluation label" not in serialized
    assert specs["labram"]["sampling_rate"] == 200
    assert specs["biot"]["sampling_rate"] == 200
    assert len(specs["biot"]["eegmmidb_montages"]) == 16


def test_canonical_samples_cover_all_folds_and_models():
    canonical = pd.read_parquet(OUTPUT / "canonical/CANONICAL_SAMPLE_MANIFEST.parquet")
    coverage = pd.read_csv(OUTPUT / "canonical/MODEL_SAMPLE_COVERAGE.csv")
    assert canonical.duplicated(["dataset", "sample_id"]).sum() == 0
    assert canonical.groupby("dataset").outer_fold.nunique().eq(5).all()
    assert coverage.subject_coverage.eq(1.0).all()
    assert coverage.sample_coverage.eq(1.0).all()
    assert coverage.groupby("dataset").model.nunique().eq(3).all()


def test_embedding_cache_is_once_per_subject_model_and_reused_by_seeds():
    manifest = pd.read_parquet(OUTPUT / "embedding_cache/EMBEDDING_CACHE_MANIFEST.parquet")
    assert len(manifest) == 465
    assert manifest.duplicated(["dataset", "model", "subject_id"]).sum() == 0
    assert manifest.extraction_count.eq(1).all()
    assert "seed" not in manifest.columns


def test_probe_fold_roles_and_backbone_exclusion():
    probes = pd.read_parquet(OUTPUT / "probes/PROBE_MODEL_MANIFEST.parquet")
    hyper = pd.read_csv(OUTPUT / "probes/PROBE_HYPERPARAMETERS.csv")
    assert len(probes) == 2 * 3 * 5 * 5
    assert len(hyper) == len(probes)
    assert hyper.learning_rate.isin([0.0001, 0.0003, 0.001]).all()
    assert hyper.weight_decay.isin([0.0, 0.0001]).all()
    assert hyper.selected_epoch.between(1, 30).all()


def test_gate_q_failure_prevents_all_oracle_and_router_outputs():
    state = json.loads((OUTPUT / "RUN_STATE.json").read_text())
    assert state["terminal"] is True
    assert state["verdict"] == "V7_STAGE0A_STOP_MODEL_QUALIFICATION_FAILURE"
    assert state["stopping_gate"] == "Gate Q"
    forbidden = [
        OUTPUT / "results/FULL_SUBJECT_ORACLE_RESULTS.parquet",
        OUTPUT / "results/SPLIT_HALF_ORACLE_RESULTS.parquet",
        OUTPUT / "results/SUBJECT_SHUFFLE_NULL.parquet",
        OUTPUT / "router", OUTPUT / "abstention",
        REPO / "src/hsc_tta/fm_routing/router.py",
        REPO / "src/hsc_tta/fm_routing/abstention.py",
        REPO / "src/hsc_tta/fm_routing/scout.py",
    ]
    assert not any(path.exists() for path in forbidden)


def test_registered_repetition_counts_and_terminal_resume_guard():
    freeze = json.loads((DELIVERY / "V7_STAGE0A_FREEZE.json").read_text())
    assert freeze["subject_bootstrap"] == {"repetitions": 5000, "seed": 20260810}
    assert freeze["subject_shuffle_null"] == {"repetitions": 500, "seed": 20260811}
    source = (REPO / "scripts/fm_routing_v7/run_all.py").read_text()
    assert "terminal state retained" in source
    assert "partial execution cannot bypass" in source
