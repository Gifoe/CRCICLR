"""Prospectively freeze the PERSIST-EEG external actionability audit.

This command is intentionally outcome-blind.  It uses only public dataset
metadata and deterministic subject identifiers.  Run it before feature
extraction, task-head fitting, or any external task-performance evaluation.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = REPO_ROOT / "experiments" / "persist_eeg_external_actionability_v1"
OUT = EXP_ROOT / "outputs"
PROTOCOL_DIR = OUT / "protocol"
REFERENCE_COMMIT = "1eca3976d62d38fb4291e217ca06add484babd41"
IMPLEMENTATION_ID = "persist_external_actionability_v1_20260817"
SEED = 20260817


def sha_text(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_once(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(clean(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Prospective lock mismatch; refusing overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def current_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, timeout=5
        ).strip()
    except Exception:
        return None


def split_subjects() -> dict[str, list[str]]:
    subjects = [f"S{index:03d}" for index in range(1, 110)]
    rng = np.random.default_rng(SEED)
    ordered = [subjects[index] for index in rng.permutation(len(subjects))]
    counts = {
        "task_head_train": 45,
        "block_discovery": 30,
        "confirmatory_calibration": 15,
        "outer_test": 19,
    }
    result: dict[str, list[str]] = {}
    cursor = 0
    for role, count in counts.items():
        result[role] = sorted(ordered[cursor : cursor + count])
        cursor += count
    if cursor != 109 or len(set().union(*(set(v) for v in result.values()))) != 109:
        raise AssertionError("Subject split is not exhaustive and disjoint")
    train = result["task_head_train"]
    train_rng = np.random.default_rng(SEED + 1)
    train_order = [train[index] for index in train_rng.permutation(len(train))]
    result["task_head_fit"] = sorted(train_order[:38])
    result["task_head_validation"] = sorted(train_order[38:])
    return result


def main() -> int:
    outcome_markers = list((OUT / "results").glob("*")) if (OUT / "results").exists() else []
    required = [
        PROTOCOL_DIR / "EXTERNAL_AUDIT_PROTOCOL_LOCK.json",
        PROTOCOL_DIR / "EXTERNAL_DATASET_SELECTION_LOCK.json",
        PROTOCOL_DIR / "ACTIONABILITY_PROTOCOL_LOCK.json",
        PROTOCOL_DIR / "EXTERNAL_SPLIT_LOCK.json",
        PROTOCOL_DIR / "DEVELOPMENT_SCOPE_LOCK.json",
    ]
    if outcome_markers and not all(path.exists() for path in required):
        raise RuntimeError("External outcomes exist without all prospective locks")

    split = split_subjects()
    frozen_at = datetime.now(timezone.utc).isoformat()
    common = {
        "implementation_id": IMPLEMENTATION_ID,
        "reference_commit": REFERENCE_COMMIT,
        "code_commit_at_freeze": current_commit(),
        "seed": SEED,
        "frozen_before_external_outcomes": True,
        "frozen_at_utc": frozen_at,
    }

    dataset_lock = {
        **common,
        "selected_dataset": "EEGMMIDB",
        "selection_basis": "metadata, provenance, availability, and compute only; no external performance outcome inspected",
        "official_release": "PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0",
        "official_url": "https://physionet.org/content/eegmmidb/1.0.0/",
        "license": "ODC-By 1.0",
        "raw_root_expected": "D:/nips-temp/TotalP/P1/eegmmidb",
        "subjects": 109,
        "channels": 64,
        "sampling_rate_hz": 160,
        "target_runs": [4, 6, 8, 10, 12, 14],
        "context_runs": [4, 6],
        "future_runs": [8, 10, 12, 14],
        "classes": {
            "0": "left_fist_imagery",
            "1": "right_fist_imagery",
            "2": "both_fists_imagery",
            "3": "both_feet_imagery",
        },
        "expected_target_edf_count": 654,
        "persistent_nuisance": "subject-specific structure repeated across separately acquired task runs",
        "replication_scope": "repeated-run motor-imagery acquisition/task-condition shift",
        "explicit_limitations": [
            "EEGMMIDB has repeated runs, not documented independent sessions",
            "EEGMMIDB is single-site and single-device for this audit",
            "context/future runs repeat two motor-imagery paradigms; this is not a multisite, multidevice, or true multisession replication",
            "the EEGMMIDB encoder is trained within dataset and is not the OpenBMI frozen encoder",
        ],
        "candidate_registry": [
            {
                "dataset": "EEGMMIDB",
                "available": True,
                "selected": True,
                "reason": "109 subjects, repeated MI runs, clear labels, audited license, complete raw local copy, feasible on one GPU",
            },
            {
                "dataset": "OpenBMI",
                "available": True,
                "selected": False,
                "reason": "internal discovery dataset, therefore not an external replication",
            },
            {
                "dataset": "HMC sleep staging",
                "available": False,
                "selected": False,
                "reason": "not locally materialized and changes task/paradigm more radically than the pre-specified repeated-run MI question",
            },
            {
                "dataset": "CAP sleep staging",
                "available": False,
                "selected": False,
                "reason": "not locally materialized; introducing it would require a new download and a different task protocol",
            },
        ],
        "selection_is_single_dataset": True,
        "performance_based_dataset_selection": False,
    }

    split_lock = {
        **common,
        "dataset": "EEGMMIDB",
        "subject_disjoint": True,
        "trial_random_split_forbidden": True,
        "roles": {key: split[key] for key in (
            "task_head_train", "block_discovery", "confirmatory_calibration", "outer_test"
        )},
        "internal_task_head_split": {
            "fit": split["task_head_fit"],
            "validation": split["task_head_validation"],
        },
        "role_hashes": {
            key: sha_text(split[key]) for key in (
                "task_head_train", "block_discovery", "confirmatory_calibration", "outer_test",
                "task_head_fit", "task_head_validation"
            )
        },
        "outer_test_state": "OUTER_TEST_LOCKED",
        "outer_test_evaluation_authorized": False,
    }

    development_subjects = sorted(
        split["task_head_train"] + split["block_discovery"] + split["confirmatory_calibration"]
    )
    development_scope = {
        **common,
        "dataset": "EEGMMIDB",
        "allowed_roles": {
            "task_head_train": split["task_head_train"],
            "task_head_fit": split["task_head_fit"],
            "task_head_validation": split["task_head_validation"],
            "block_discovery": split["block_discovery"],
            "confirmatory_calibration": split["confirmatory_calibration"],
        },
        "allowed_subjects": development_subjects,
        "allowed_subjects_hash": sha_text(development_subjects),
        "outer_subject_count": 19,
        "outer_subject_hash_only": sha_text(split["outer_test"]),
        "outer_subject_ids_present": False,
        "runtime_must_not_open": "EXTERNAL_SPLIT_LOCK.json",
        "runtime_path_policy": "construct paths only for allowed subject IDs; never recursively enumerate the raw root",
        "scope_violation_terminal_state": "DATA_SCOPE_VIOLATION",
    }

    audit_lock = {
        **common,
        "authorization_rule": "DDA_B_PASS AND DDA_C_PASS => EXTERNAL_ACTIONABILITY_AUDIT_AUTHORIZED",
        "dda_frozen_terminal_state": "DDA_PARTIAL_MECHANISM_ONLY",
        "dda_a": {
            "status": "DDA_A_FAIL",
            "permanent_falsification": True,
            "external_audit_veto": False,
            "must_not_be_rerun_to_change_status": True,
        },
        "dda_b": "DDA_B_PASS",
        "dda_c": "DDA_C_PASS",
        "agdi_training_authorized_before_external_audit": False,
        "outer_test_state": "OUTER_TEST_LOCKED",
        "analysis_order": [
            "data_scope_audit", "feature_extraction", "task_head_fit", "block_discovery",
            "persistence", "signed_utility", "decision_dependence", "accuracy_actionability",
            "multiplicity_correction", "stability", "block_assignment", "terminal_decision",
        ],
        "supporting_cf_analysis_is_not_an_authorization_gate": True,
        "statistical_unit": "subject; run is a clustered repeated measure",
        "trial_level_inference_forbidden": True,
    }

    actionability_lock = {
        **common,
        "dataset": "EEGMMIDB",
        "representation": {
            "input": "per-trial 64-channel absolute and relative log-bandpower",
            "bands_hz": [[4, 8], [8, 12], [12, 16], [16, 20], [20, 30], [30, 40]],
            "feature_dim": 768,
            "encoder": "MLP 768-256-128 GELU with linear four-class head",
            "training_subjects": "task_head_fit only",
            "early_stopping_subjects": "task_head_validation only",
            "training_runs": [4, 6],
            "hyperparameter_grid": {
                "learning_rate": [0.0003, 0.001],
                "weight_decay": [0.0001, 0.001],
                "max_epochs": 80,
                "patience": 10,
                "batch_size": 256,
            },
            "selection_metric": "mean subject balanced accuracy on task_head_validation context runs",
            "one_seed_only": True,
        },
        "block_discovery": {
            "subjects": "block_discovery only",
            "method": "eigendecomposition of symmetrized cross-run subject-centroid covariance after discovery-run mean centering",
            "target_labels_used_for_centering": False,
            "candidate_blocks": [
                {"block": "P01_04", "components": [1, 4], "rank": 4},
                {"block": "P05_08", "components": [5, 8], "rank": 4},
                {"block": "P09_16", "components": [9, 16], "rank": 8},
                {"block": "P17_32", "components": [17, 32], "rank": 16},
            ],
            "candidate_count": 4,
        },
        "controls": {
            "random_draws": 100,
            "construction": "same-rank Haar subspace; per-sample intervention displacement norm exactly matched to candidate block",
            "random_seed_stream_frozen": True,
        },
        "inference": {
            "bootstrap_draws": 10000,
            "confidence_level": 0.95,
            "permutation": "subject-level sign flip; exact when feasible",
            "multiplicity": "Holm correction separately across four blocks for H1, H2, H3-finite, and H4",
        },
        "signed_utility_convention": {
            "u_abs": "CE_erase - CE_raw",
            "u_spec": "u_abs - mean(CE_random - CE_raw)",
            "candidate_harmful_direction": "negative",
            "protected_direction": "positive",
        },
        "gates": {
            "H1": "Holm-adjusted one-sided p<0.05 and LCB95 of candidate-minus-matched-random same-subject cross-run distance advantage >0",
            "H2": "Holm-adjusted one-sided p<0.05 and UCB95(u_spec)<0",
            "H3": "LCB95(local candidate/random)>1 AND Holm-adjusted one-sided p<0.05 AND LCB95(finite candidate/random)>1",
            "H4": "Holm-adjusted one-sided p<0.05, LCB95(delta_BA_specific)>0, and mean delta_BA_specific>=0.005",
            "H5": "all leave-one-subject-out means >0, all leave-one-future-run-out means >0, and >=60% subjects have nonnegative delta_BA_specific",
            "actionable_harmful": "H1 AND H2 AND H3 AND H4 AND H5",
        },
        "agdi_authorization": "at least one real block passes H1-H5",
        "no_target_terminal_state": "EXTERNAL_AUDIT_NO_ACTIONABLE_HARMFUL",
        "no_target_action": "STOP_AGDI_NO_ACTIONABLE_TARGET",
        "outer_test_state": "OUTER_TEST_LOCKED",
    }

    write_once(PROTOCOL_DIR / "EXTERNAL_DATASET_SELECTION_LOCK.json", dataset_lock)
    write_once(PROTOCOL_DIR / "EXTERNAL_SPLIT_LOCK.json", split_lock)
    write_once(PROTOCOL_DIR / "DEVELOPMENT_SCOPE_LOCK.json", development_scope)
    write_once(PROTOCOL_DIR / "EXTERNAL_AUDIT_PROTOCOL_LOCK.json", audit_lock)
    write_once(PROTOCOL_DIR / "ACTIONABILITY_PROTOCOL_LOCK.json", actionability_lock)
    print(json.dumps({"status": "PROSPECTIVE_LOCKS_FROZEN", "paths": [str(p) for p in required]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
