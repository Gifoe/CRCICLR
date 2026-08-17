"""Independent integrity/reproducibility audit for external actionability V1."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = REPO_ROOT / "experiments" / "persist_eeg_external_actionability_v1"
OUT = EXP_ROOT / "outputs"
RESULTS = OUT / "results"
PROTOCOL = OUT / "protocol"
TARGET = OUT / "OUTPUT_INTEGRITY_AUDIT.json"
REFERENCE_COMMIT = "1eca3976d62d38fb4291e217ca06add484babd41"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, timeout=5).strip()
    except Exception:
        return None


def main() -> int:
    required = [
        PROTOCOL / "EXTERNAL_AUDIT_PROTOCOL_LOCK.json",
        PROTOCOL / "EXTERNAL_DATASET_SELECTION_LOCK.json",
        PROTOCOL / "ACTIONABILITY_PROTOCOL_LOCK.json",
        PROTOCOL / "EXTERNAL_SPLIT_LOCK.json",
        PROTOCOL / "DEVELOPMENT_SCOPE_LOCK.json",
        PROTOCOL / "PREOUTCOME_DATA_AMENDMENT.json",
        PROTOCOL / "IMPLEMENTATION_REPAIR_LOG.json",
        PROTOCOL / "RAW_DATASET_INVENTORY.json",
        PROTOCOL / "DATA_SCOPE_AUDIT.json",
        PROTOCOL / "EMBEDDING_SCOPE_AUDIT.json",
        PROTOCOL / "CF_DATA_SCOPE_AUDIT.json",
        RESULTS / "TASK_HEAD_RESULT.json",
        RESULTS / "BLOCK_DISCOVERY_RESULT.json",
        RESULTS / "PERSISTENCE_RESULTS.csv",
        RESULTS / "SIGNED_UTILITY_RESULTS.csv",
        RESULTS / "DECISION_DEPENDENCE_RESULTS.csv",
        RESULTS / "ACTIONABILITY_RESULTS.csv",
        RESULTS / "BLOCK_ASSIGNMENTS.csv",
        RESULTS / "EXTERNAL_AUDIT_SUBJECT.csv",
        RESULTS / "EXTERNAL_AUDIT_RANDOM_SUBJECT.csv",
        RESULTS / "CF_RESCUE_HARM_SUBJECT.csv",
        RESULTS / "CF_RESCUE_HARM_RANDOM.csv",
        RESULTS / "CF_RESCUE_HARM_RESULT.json",
        OUT / "FINAL_DECISION.json",
        OUT / "scientific_report.md",
        EXP_ROOT / "README.md",
    ]
    missing = [str(path.relative_to(EXP_ROOT)).replace("\\", "/") for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required artifacts: {missing}")
    inventory = load_json(PROTOCOL / "RAW_DATASET_INVENTORY.json")
    scope = load_json(PROTOCOL / "DATA_SCOPE_AUDIT.json")
    embedding_scope = load_json(PROTOCOL / "EMBEDDING_SCOPE_AUDIT.json")
    cf_scope = load_json(PROTOCOL / "CF_DATA_SCOPE_AUDIT.json")
    final = load_json(OUT / "FINAL_DECISION.json")
    cf = load_json(RESULTS / "CF_RESCUE_HARM_RESULT.json")
    assignments = pd.read_csv(RESULTS / "BLOCK_ASSIGNMENTS.csv")
    persistence = pd.read_csv(RESULTS / "PERSISTENCE_RESULTS.csv")
    utility = pd.read_csv(RESULTS / "SIGNED_UTILITY_RESULTS.csv")
    decision = pd.read_csv(RESULTS / "DECISION_DEPENDENCE_RESULTS.csv")
    action = pd.read_csv(RESULTS / "ACTIONABILITY_RESULTS.csv")
    subjects = pd.read_csv(RESULTS / "EXTERNAL_AUDIT_SUBJECT.csv")
    random_subjects = pd.read_csv(RESULTS / "EXTERNAL_AUDIT_RANDOM_SUBJECT.csv")
    cf_subjects = pd.read_csv(RESULTS / "CF_RESCUE_HARM_SUBJECT.csv")
    cf_random = pd.read_csv(RESULTS / "CF_RESCUE_HARM_RANDOM.csv")
    numeric_frames = [persistence, utility, decision, action, subjects, random_subjects, cf_subjects, cf_random]
    checks = {
        "required_artifacts_present": not missing,
        "raw_target_edf_654": inventory.get("target_edf_count") == 654,
        "raw_target_tree_hash_present": len(str(inventory.get("tree_sha256", ""))) == 64,
        "data_scope_pass": scope.get("status") == "DATA_SCOPE_PASS",
        "development_subjects_90": scope.get("materialized_subject_count") == 90,
        "source_edf_open_count_540": scope.get("source_edf_open_count") == 540,
        "outer_signals_not_materialized": scope.get("outer_signals_materialized") is False,
        "outer_embeddings_not_materialized": embedding_scope.get("outer_embeddings_materialized") is False,
        "filtered_embedding_subjects_45": embedding_scope.get("subject_count") == 45,
        "external_four_preregistered_blocks": len(assignments) == 4 and set(assignments.block) == {"P01_04", "P05_08", "P09_16", "P17_32"},
        "external_subject_rows_60": len(subjects) == 60,
        "external_random_subject_rows_6000": len(random_subjects) == 6000,
        "cf_real_subject_rows_204": len(cf_subjects) == 204,
        "cf_random_subject_rows_20400": len(cf_random) == 20_400,
        "cf_100_draws_per_subject_bank": bool((cf_random.groupby(["run", "inner_fold", "subject"]).draw.nunique() == 100).all()),
        "all_primary_numeric_values_finite": all(np.isfinite(frame.select_dtypes("number")).all().all() for frame in numeric_frames),
        "agdi_authorization_matches_blocks": bool(final.get("agdi_training_authorized")) == bool(assignments.all_H1_H5.astype(bool).any()),
        "outer_test_locked": final.get("outer_test_state") == "OUTER_TEST_LOCKED" and final.get("outer_test_used") is False,
        "cf_does_not_change_dda_a": cf.get("frozen_dda_a_status") == "DDA_A_FAIL" and cf.get("dda_a_conclusion_changed") is False,
        "cf_not_authorization_gate": cf.get("authorization_gate") is False,
        "cf_loader_filtered": cf_scope.get("all_54_subject_manifest_materialized") is False and cf_scope.get("all_54_subject_h0_materialized") is False,
    }
    core = [path for path in required if path.is_file() and "cache" not in path.parts]
    payload = {
        "status": "EXTERNAL_ACTIONABILITY_OUTPUT_AUDIT_PASS" if all(checks.values()) else "EXTERNAL_ACTIONABILITY_OUTPUT_AUDIT_FAIL",
        "checks": checks,
        "terminal_state": final["terminal_state"],
        "next_action": final["next_action"],
        "reference_commit": REFERENCE_COMMIT,
        "audit_code_commit": git_commit(),
        "core_sha256": {str(path.relative_to(EXP_ROOT)).replace("\\", "/"): sha256(path) for path in core},
        "materialized_file_count_excluding_this_audit": sum(
            1 for path in EXP_ROOT.rglob("*") if path.is_file() and path.resolve() != TARGET.resolve()
        ),
    }
    write_json(TARGET, payload)
    reproducibility = {
        "reference_commit": REFERENCE_COMMIT,
        "final_code_commit_before_commit": git_commit(),
        "seed": 20260817,
        "python": sys.version,
        "platform": platform.platform(),
        "dataset_tree_sha256": inventory["tree_sha256"],
        "dataset_target_edf_count": inventory["target_edf_count"],
        "model_checkpoint": load_json(RESULTS / "TASK_HEAD_RESULT.json")["checkpoint"],
        "model_checkpoint_sha256": load_json(RESULTS / "TASK_HEAD_RESULT.json")["checkpoint_sha256"],
        "persistent_basis_sha256": load_json(RESULTS / "BLOCK_DISCOVERY_RESULT.json")["basis_sha256"],
        "split_role_hashes": load_json(PROTOCOL / "EXTERNAL_SPLIT_LOCK.json")["role_hashes"],
        "exact_commands": [
            "E:/Anaconda/python.exe experiments/persist_eeg_external_actionability_v1/code/extract_features.py inventory --raw-root D:/nips-temp/TotalP/P1/eegmmidb --workers 16",
            "E:/Anaconda/python.exe experiments/persist_eeg_external_actionability_v1/code/extract_features.py extract --raw-root D:/nips-temp/TotalP/P1/eegmmidb --workers 8 --resume",
            "D:/nips-temp/TotalP/P2/.conda/gpu-baseline-v1/python.exe experiments/persist_eeg_external_actionability_v1/code/external_actionability_v1.py train --device cuda",
            "D:/nips-temp/TotalP/P2/.conda/gpu-baseline-v1/python.exe experiments/persist_eeg_external_actionability_v1/code/external_actionability_v1.py embed --device cuda",
            "D:/nips-temp/TotalP/P2/.conda/gpu-baseline-v1/python.exe experiments/persist_eeg_external_actionability_v1/code/external_actionability_v1.py discover --device cpu",
            "D:/nips-temp/TotalP/P2/.conda/gpu-baseline-v1/python.exe experiments/persist_eeg_external_actionability_v1/code/external_actionability_v1.py audit --device cpu",
            "D:/nips-temp/TotalP/P2/.conda/gpu-baseline-v1/python.exe experiments/persist_eeg_external_actionability_v1/code/cf_rescue_harm.py --device cuda",
            "D:/nips-temp/TotalP/P2/.conda/gpu-baseline-v1/python.exe experiments/persist_eeg_external_actionability_v1/code/external_actionability_v1.py report --device cpu",
            "D:/nips-temp/TotalP/P2/.conda/gpu-baseline-v1/python.exe experiments/persist_eeg_external_actionability_v1/code/audit_outputs.py",
        ],
        "outer_test_used": False,
    }
    write_json(OUT / "REPRODUCIBILITY.json", reproducibility)
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"].endswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
