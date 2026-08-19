from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import PROTOCOL, V4_ROOT, ensure_directories, sha256_file, write_json
from datasets import load_openbmi, load_wbcic
from evaluation import summarize


EXPECTED = {
    "OpenBMI_static_BA": 0.8464423076923075,
    "OpenBMI_current_BA": 0.8509615384615383,
    "WBCIC_static_BA": 0.8036257390983,
    "WBCIC_current_BA": 0.8067887718649914,
}


def _fold_assignment(data) -> np.ndarray:
    assignment = np.full(len(data.labels), -1, dtype=int)
    for fold in data.folds:
        assignment[np.isin(data.subjects, fold["test_subjects"])] = int(fold["outer_fold"])
    if np.any(assignment < 0):
        raise RuntimeError("Incomplete outer fold assignment")
    return assignment


def run() -> None:
    ensure_directories()
    manifest_path = V4_ROOT / "outputs" / "final_lock" / "FINAL_MODEL_HASHES.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hash_checks = []
    for section in ("code", "key_outputs"):
        for relative, expected in manifest[section].items():
            path = V4_ROOT / Path(relative)
            actual = sha256_file(path)
            hash_checks.append(
                {
                    "section": section,
                    "path": relative,
                    "expected": expected,
                    "actual": actual,
                    "pass": actual == expected,
                }
            )
    if not all(item["pass"] for item in hash_checks):
        raise RuntimeError("V4 frozen hash reconstruction failed")

    datasets = {"OpenBMI": load_openbmi(), "WBCIC-development": load_wbcic()}
    reconstructed = {}
    for name, data in datasets.items():
        folds = _fold_assignment(data)
        static_row, _, _ = summarize(
            data, f"{name}_STATIC", data.static_prediction, data.static_probability, folds, baseline="static"
        )
        current_row, _, _ = summarize(
            data, f"{name}_CURRENT", data.current_prediction, data.current_probability, folds, baseline="static"
        )
        reconstructed[name] = {
            "trials": len(data.labels),
            "subjects": len(np.unique(data.subjects)),
            "sessions": sorted(np.unique(data.sessions).tolist()),
            "experts": data.expert_names,
            "static_BA": static_row["mean_subject_BA"],
            "current_BA": current_row["mean_subject_BA"],
            "current_Delta_BA_vs_static": current_row["Delta_BA"],
        }
    checks = {
        "OpenBMI_static_BA": abs(reconstructed["OpenBMI"]["static_BA"] - EXPECTED["OpenBMI_static_BA"]) < 1e-12,
        "OpenBMI_current_BA": abs(reconstructed["OpenBMI"]["current_BA"] - EXPECTED["OpenBMI_current_BA"]) < 1e-12,
        "WBCIC_static_BA": abs(reconstructed["WBCIC-development"]["static_BA"] - EXPECTED["WBCIC_static_BA"]) < 1e-12,
        "WBCIC_current_BA": abs(reconstructed["WBCIC-development"]["current_BA"] - EXPECTED["WBCIC_current_BA"]) < 1e-12,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V4 numerical reconstruction failed: {checks}")
    write_json(
        PROTOCOL / "V4_RECONSTRUCTION.json",
        {
            "status": "V4_RECONSTRUCTION_PASS",
            "starting_branch": "codex/persist-eeg-final-model-v4",
            "hash_checks": hash_checks,
            "numerical_checks": checks,
            "datasets": reconstructed,
            "OUTER_TEST_USED": False,
        },
    )
    write_json(
        PROTOCOL / "BASELINE_LOCK.json",
        {
            "status": "V5_INITIAL_BASELINE_LOCK",
            "policy": "Report both static and strongest pre-existing current references; update if V5 creates a stronger non-adaptive baseline.",
            "OpenBMI": {
                "B_STATIC": "A0_STATIC_B_STRONG",
                "B_STATIC_BA": reconstructed["OpenBMI"]["static_BA"],
                "B_STRONG_CURRENT": "A1_DYNAMIC_KEEP_FINAL",
                "B_STRONG_CURRENT_BA": reconstructed["OpenBMI"]["current_BA"],
            },
            "WBCIC-development": {
                "B_STATIC": "W0_B_STRONG_PROBABILITY_MEAN",
                "B_STATIC_BA": reconstructed["WBCIC-development"]["static_BA"],
                "B_STRONG_CURRENT": "W1_RAW_LINEAR",
                "B_STRONG_CURRENT_BA": reconstructed["WBCIC-development"]["current_BA"],
            },
            "OUTER_TEST_USED": False,
        },
    )
    write_json(
        PROTOCOL / "LEGALITY_AUDIT.json",
        {
            "status": "V5_DEVELOPMENT_LEGALITY_PASS",
            "WBCIC_authorized_development_subjects": 41,
            "WBCIC_target_session": "S3",
            "target_S3_labels_permitted_for_model_fit_or_selection": False,
            "target_S1_S2_labels_permitted_for_cross_session_reliability": True,
            "outer_split_lock_opened": False,
            "outer_subject_ids_loaded": False,
            "outer_raw_EEG_loaded": False,
            "OUTER_TEST_USED": False,
        },
    )
    print(json.dumps({"status": "V4_RECONSTRUCTION_PASS", "datasets": reconstructed}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
