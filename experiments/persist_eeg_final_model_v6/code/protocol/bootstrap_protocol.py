"""Freeze V6 legal histories and reconstruct the frozen V5 evidence."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import LEADERBOARD, PROTOCOL, RESEARCH_LOG, ensure_directories, sha256_file, stage0_root, v5_output_root, wbcic_source_root, write_csv, write_json


def _load(path: Path, encoding: str = "utf-8") -> dict:
    return json.loads(path.read_text(encoding=encoding))


def run() -> None:
    ensure_directories()
    v5 = v5_output_root()
    result_path = v5 / "final_candidate" / "DEVELOPMENT_RESULTS.json"
    decision_path = v5 / "FINAL_DECISION.json"
    spec_path = v5 / "final_candidate" / "FINAL_MODEL_SPEC.json"
    results = _load(result_path)
    decision = _load(decision_path)
    if (
        results.get("OUTER_TEST_USED") is not False
        or decision.get("OUTER_TEST_USED") is not False
        or decision.get("selected_method") != "M13_CSP_AUGMENTED_REFIT4"
        or abs(float(results["WBCIC_development"]["mean_subject_BA"]) - 0.8177820891845283) > 1e-12
    ):
        raise RuntimeError("V5 reconstruction mismatch")
    write_json(
        PROTOCOL / "V5_RECONSTRUCTION.json",
        {
            "status": "V5_RECONSTRUCTED_EXACTLY",
            "selected_method": decision["selected_method"],
            "selected_model_name": decision["selected_model_name"],
            "OpenBMI": results["OpenBMI"],
            "WBCIC_development": results["WBCIC_development"],
            "source_hashes": {
                "DEVELOPMENT_RESULTS.json": sha256_file(result_path),
                "FINAL_DECISION.json": sha256_file(decision_path),
                "FINAL_MODEL_SPEC.json": sha256_file(spec_path),
            },
            "warning": "V5 OpenBMI used 52 subjects and pooled-session semantics; it is not a matched baseline for the new 54-subject S1-to-S2 protocol.",
            "exploratory": True,
            "OUTER_TEST_USED": False,
        },
    )

    open_path = stage0_root() / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json"
    open_payload = _load(open_path, "utf-8-sig")["openbmi"]
    w_path = wbcic_source_root() / "experiments" / "persist_eeg_wbcic_actionability_v2" / "outputs" / "protocol" / "DEVELOPMENT_SCOPE_LOCK.json"
    w_payload = _load(w_path)
    if (
        len(open_payload["subjects"]) != 54
        or len(w_payload["allowed_subjects"]) != 41
        or w_payload.get("outer_subject_ids_present") is not False
        or w_payload.get("runtime_must_not_open") != "OUTER_SPLIT_LOCK.json"
    ):
        raise RuntimeError("Development split lock violation")
    split_lock = {
        "OpenBMI": {
            "dataset": "NEMAR nm000273 MI offline/train",
            "subjects": list(map(str, open_payload["subjects"])),
            "folds": open_payload["folds"],
            "history_sessions": [1],
            "future_session": 2,
            "trials_per_subject_session": 100,
            "source_sha256": sha256_file(open_path),
        },
        "WBCIC": {
            "dataset": "NEMAR nm000348 authorized development",
            "allowed_subjects": list(map(str, w_payload["allowed_subjects"])),
            "audit_roles": w_payload["audit_roles"],
            "history_sessions": [1, 2],
            "future_session": 3,
            "sealed_outer_subject_ids_present": False,
            "source_sha256": sha256_file(w_path),
        },
        "exploratory": True,
        "OUTER_TEST_USED": False,
    }
    write_json(PROTOCOL / "DEVELOPMENT_SPLIT_LOCK.json", split_lock)
    write_json(
        PROTOCOL / "HISTORY_BUDGET_AUDIT.json",
        {
            "OpenBMI": {
                "target_history": "all 100 labeled MI trials from Session 1",
                "future_evaluation": "all 100 MI trials from Session 2",
                "K": 1,
                "future_labels_available_to_adapter": False,
            },
            "WBCIC": {
                "target_history": "all labeled MI trials from Sessions 1 and 2",
                "future_evaluation": "Session 3",
                "K": 2,
                "future_labels_available_to_adapter": False,
            },
            "matched_baseline_rule": "Every headline comparator receives the identical target-history labels used by V6.",
            "OUTER_TEST_USED": False,
        },
    )
    write_json(
        PROTOCOL / "LEGALITY_AUDIT.json",
        {
            "status": "LEGAL_DEVELOPMENT_PROTOCOL_LOCKED",
            "target_future_labels_used_for_training": False,
            "outcome_subjects_used_for_hyperparameter_selection": False,
            "WBCIC_outer_split_file_opened": False,
            "WBCIC_outer_subject_ids_loaded": False,
            "raw_WBCIC_root_enumerated": False,
            "all_results_exploratory": True,
            "OUTER_TEST_USED": False,
        },
    )
    write_json(
        PROTOCOL / "BASELINE_MATCHING_AUDIT.json",
        {
            "status": "INITIAL_BASELINE_AUDIT_CREATED",
            "required_controls": [
                "population linear head",
                "subject-specific last layer",
                "history calibration",
                "supervised prototypes",
                "shrinkage LDA",
                "history-conditioned fusion",
                "standard low-rank adapter without PERSIST",
                "FiLM/affine control",
                "geometry controls",
            ],
            "headline_reference_policy": "strongest legal control with identical history budget",
            "EEGNet_plus_5pp_is_secondary_only": True,
            "OUTER_TEST_USED": False,
        },
    )
    write_csv(
        LEADERBOARD / "BASELINE_EVOLUTION.csv",
        pd.DataFrame(
            [
                {
                    "stage": "V5_frozen_reference",
                    "benchmark": "WBCIC",
                    "method_id": "M13_CSP_AUGMENTED_REFIT4",
                    "mean_subject_BA": results["WBCIC_development"]["mean_subject_BA"],
                    "history_budget": "target S1/S2 labels",
                    "matched_to_V6": True,
                    "exploratory": True,
                    "OUTER_TEST_USED": False,
                }
            ]
        ),
    )
    ledger = RESEARCH_LOG / "HYPOTHESIS_LEDGER.md"
    ledger.write_text(
        "# V6 hypothesis ledger\n\n"
        "All entries are exploratory development analyses. WBCIC outer remains sealed.\n\n"
        "## Iteration 000 — protocol and baseline reconstruction\n\n"
        "- Previous result: V5 CS-LGS reached WBCIC BA 0.817782 (+1.099 pp versus W1); OpenBMI fell back unchanged.\n"
        "- Observed failure: frozen-output aggregation did not deliver a large dual-benchmark constructive effect.\n"
        "- Hypothesis: legal target-history labels can support representation-level subject adaptation beyond output stacking.\n"
        "- Change: use fold-compatible representations, S1→S2 OpenBMI, S1/S2→S3 WBCIC, and first lock strong matched controls.\n"
        "- Decision: KEEP as the mandatory starting audit; no performance claim yet.\n",
        encoding="utf-8",
    )
    write_json(
        PROTOCOL / "BOOTSTRAP_COMPLETE.json",
        {
            "status": "COMPLETE",
            "python": sys.version,
            "platform": platform.platform(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE.parents[2], text=True).strip(),
            "OUTER_TEST_USED": False,
        },
    )
    print("V6 protocol bootstrap complete", flush=True)


if __name__ == "__main__":
    run()
