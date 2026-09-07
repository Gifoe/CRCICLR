from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import PROTOCOL, ensure_directories, sha256_file, stage0_root, v6_outputs, wbcic_source_root, write_json


def run() -> None:
    ensure_directories()
    v6 = v6_outputs()
    decision_path = v6 / "FINAL_DECISION.json"
    development_path = v6 / "final_candidate" / "DEVELOPMENT_RESULTS.json"
    integrity_path = v6 / "protocol" / "FINAL_PREDICTION_INTEGRITY_AUDIT.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("terminal_state") != "V6_OPENBMI_TARGET_ONLY" or decision.get("OUTER_TEST_USED") is not False:
        raise RuntimeError("Authoritative V6 decision mismatch")
    write_json(PROTOCOL / "V6_RECONSTRUCTION.json", {
        "source_terminal_state": decision["terminal_state"],
        "source_commit": "c8413beff50912777f1f58180a99ffbdeff21637",
        "files": {
            str(decision_path): sha256_file(decision_path),
            str(development_path): sha256_file(development_path),
            str(integrity_path): sha256_file(integrity_path),
        },
        "existing_expert_oracle_headroom": {
            "OpenBMI": {"candidate_count": 23, "anchor_BA": 0.832037, "oracle_BA": 0.856852, "headroom_pp": 2.4815},
            "WBCIC": {"candidate_count": 39, "anchor_BA": 0.820817, "oracle_BA": 0.841806, "headroom_pp": 2.0989},
            "interpretation": "A new selector over V1-V6 experts cannot reach the V7 +5 pp target.",
            "used_to_tune_selector": False,
        },
        "OUTER_TEST_USED": False,
    })
    rows = []
    for benchmark, payload in decision["benchmarks"].items():
        rows.append({
            "benchmark": benchmark,
            "method_id": payload["strongest_information_matched_generic"],
            "mean_subject_BA": payload["strongest_information_matched_generic_BA"],
            "lock_source": str(decision_path),
            "target_future_labels_used_for_fit": False,
            "OUTER_TEST_USED": False,
        })
    pd.DataFrame(rows).to_csv(PROTOCOL / "BASELINE_LOCK.csv", index=False)
    write_json(PROTOCOL / "BASELINE_LOCK.json", {
        "OpenBMI": rows[0],
        "WBCIC": rows[1],
        "comparison_rule": "V7 is compared with the strongest fair information-matched baseline, not EEGNet.",
        "OUTER_TEST_USED": False,
    })
    split_path = stage0_root() / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json"
    scope_path = wbcic_source_root() / "experiments" / "persist_eeg_wbcic_actionability_v2" / "outputs" / "protocol" / "DEVELOPMENT_SCOPE_LOCK.json"
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    if scope.get("outer_subject_ids_present") is not False:
        raise RuntimeError("WBCIC development scope contains outer IDs")
    write_json(PROTOCOL / "HISTORY_LEGALITY.json", {
        "OpenBMI": {"history_sessions": [1], "future_session": 2, "history_labels_allowed": True, "future_labels_scoring_only_for_evaluation_subject": True},
        "WBCIC": {"history_sessions": [1, 2], "future_session": 3, "history_labels_allowed": True, "future_labels_scoring_only_for_evaluation_subject": True},
        "meta_training": "Future labels are legal only for non-outcome meta-training subjects.",
        "OUTER_TEST_USED": False,
    })
    write_json(PROTOCOL / "SPLIT_LOCK.json", {
        "OpenBMI_split": str(split_path),
        "OpenBMI_split_sha256": sha256_file(split_path),
        "WBCIC_development_scope": str(scope_path),
        "WBCIC_development_scope_sha256": sha256_file(scope_path),
        "folds": 5,
        "unit": "subject",
        "OUTER_TEST_USED": False,
    })
    write_json(PROTOCOL / "OUTER_LOCK.json", {
        "WBCIC_outer_split_file_opened": False,
        "WBCIC_outer_subjects_enumerated": False,
        "WBCIC_outer_raw_loaded": False,
        "WBCIC_outer_features_loaded": False,
        "WBCIC_outer_labels_loaded": False,
        "OUTER_TEST_USED": False,
    })
    write_json(PROTOCOL / "BOOTSTRAP_COMPLETE.json", {
        "required_files_present": True,
        "V6_reconstructed": True,
        "baseline_locked": True,
        "history_legality_locked": True,
        "subject_split_locked": True,
        "outer_locked": True,
        "OUTER_TEST_USED": False,
    })
    print("V7 protocol bootstrap complete; OUTER_TEST_USED=false", flush=True)


if __name__ == "__main__":
    run()
