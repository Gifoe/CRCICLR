from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import (
    BASELINES, CACHE, PROTOCOL, V8_SEED, ensure_directories, sha256_file,
    stable_order, stage0_root, v7_outputs, wbcic_source_root, write_csv, write_json,
)


LOCKED = {
    "openbmi": {
        "benchmark": "OpenBMI_MI_S1_to_S2",
        "prefix": "OPENBMI",
        "subjects": 54,
        "history_sessions": [1],
        "future_session": 2,
        "method_id": "ANCHOR_BLEND__CONFORMER_NORM_FIXED_HEAD",
        "mean_subject_BA": 0.8377777777777777,
    },
    "wbcic": {
        "benchmark": "WBCIC_S1S2_to_S3_authorized_development",
        "prefix": "WBCIC",
        "subjects": 41,
        "history_sessions": [1, 2],
        "future_session": 3,
        "method_id": "ANCHOR_BLEND__CONFORMER_NORM_FIXED_HEAD",
        "mean_subject_BA": 0.8249672025129341,
    },
}


def _subject_ba(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, group in frame.groupby("subject_id", sort=False):
        rows.append({
            "subject_id": str(subject),
            "outer_fold": int(group.outer_fold.iloc[0]),
            "BA": float(balanced_accuracy_score(group.label, group.prediction)),
        })
    return pd.DataFrame(rows)


def _baseline(spec: dict) -> tuple[pd.DataFrame, Path]:
    path = v7_outputs() / "diagnostics" / f"{spec['prefix']}_CONFORMER_NORM_PREDICTIONS.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame.method_id.astype(str).eq(spec["method_id"])].copy()
    if frame.empty or frame.trial_uid.duplicated().any() or frame.OUTER_TEST_USED.astype(bool).any():
        raise RuntimeError(f"Malformed V7 baseline predictions: {path}")
    subjects = _subject_ba(frame)
    observed = float(subjects.BA.mean())
    if len(subjects) != spec["subjects"] or not np.isclose(observed, spec["mean_subject_BA"], atol=1e-12):
        raise RuntimeError((spec["benchmark"], len(subjects), observed, spec["mean_subject_BA"]))
    return subjects, path


def _partition(subjects: list[str], benchmark: str) -> tuple[list[str], list[str]]:
    ordered = stable_order(subjects, "V8_SEARCH_HOLDOUT", V8_SEED, benchmark)
    holdout_count = int(math.floor(0.25 * len(ordered) + 0.5))
    holdout = sorted(ordered[:holdout_count])
    search = sorted(ordered[holdout_count:])
    return search, holdout


def run() -> None:
    ensure_directories()
    decision_path = v7_outputs() / "FINAL_DECISION.json"
    report_path = v7_outputs() / "SCIENTIFIC_REPORT.md"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("terminal_state") != "V7_SCIENTIFIC_EXHAUSTION" or decision.get("OUTER_TEST_USED") is not False:
        raise RuntimeError("Authoritative V7 terminal decision mismatch")

    split_payload = {
        "seed": V8_SEED,
        "split_unit": "subject_id",
        "split_inputs": "subject IDs only; no labels, outcomes, features, or scores",
        "holdout_status": "internal V8 holdout, not independent confirmation",
        "search_time_holdout_outcomes_accessed": False,
        "OUTER_TEST_USED": False,
    }
    baseline_rows = []
    reconstruction = {
        "source_commit": "9feee89daa3fdcdae1602104a8b6ef5fc05afd64",
        "source_terminal_state": decision["terminal_state"],
        "source_files": {
            str(decision_path): sha256_file(decision_path),
            str(report_path): sha256_file(report_path),
        },
        "updated_comparison_rule": "V8 uses the strongest fair V7 generic, not the weaker V6 anchor.",
        "benchmarks": {},
        "OUTER_TEST_USED": False,
    }
    oracle_path = v7_outputs() / "diagnostics" / "NEW_BACKBONE_SUBJECT_ORACLE.csv"
    oracle = pd.read_csv(oracle_path)
    for key, spec in LOCKED.items():
        subjects, path = _baseline(spec)
        search, holdout = _partition(subjects.subject_id.astype(str).tolist(), key)
        if len(search) + len(holdout) != spec["subjects"] or set(search) & set(holdout):
            raise RuntimeError("Partition construction failure")
        split_payload[key] = {
            "benchmark": spec["benchmark"],
            "history_sessions": spec["history_sessions"],
            "future_session": spec["future_session"],
            "V8_SEARCH": search,
            "V8_INTERNAL_HOLDOUT": holdout,
            "search_subjects": len(search),
            "internal_holdout_subjects": len(holdout),
            "partition_uses_outcomes": False,
        }
        # Materialize a search-only view once at protocol construction.  Later
        # search programs never receive internal-holdout rows or labels.
        full_prediction_path = v7_outputs() / "diagnostics" / f"{spec['prefix']}_CONFORMER_NORM_PREDICTIONS.csv"
        full_predictions = pd.read_csv(full_prediction_path)
        search_predictions = full_predictions.loc[
            full_predictions.method_id.astype(str).eq(spec["method_id"])
            & full_predictions.subject_id.astype(str).isin(search)
        ].copy()
        if search_predictions.subject_id.astype(str).nunique() != len(search):
            raise RuntimeError(f"Incomplete search-only baseline materialization for {key}")
        search_predictions.to_parquet(
            CACHE / f"{spec['prefix']}_V7_LOCKED_GENERIC_SEARCH.parquet", index=False,
        )
        for fold in range(5):
            metadata_path = (
                v7_outputs() / "cache"
                / f"{spec['prefix']}_CONFORMER_NORM_FOLD_{fold}_METADATA.parquet"
            )
            metadata = pd.read_parquet(metadata_path)
            metadata.insert(0, "source_index", np.arange(len(metadata), dtype=np.int64))
            search_metadata = metadata.loc[metadata.subject_id.astype(str).isin(search)].copy()
            search_metadata["source_rows_total"] = len(metadata)
            if search_metadata.subject_id.astype(str).nunique() != len(search):
                raise RuntimeError(f"Incomplete search-only feature index for {key} fold {fold}")
            search_metadata.to_parquet(
                CACHE / f"{spec['prefix']}_SEARCH_ROWS_FOLD_{fold}.parquet", index=False,
            )
        selected_oracle = oracle.loc[oracle.benchmark.astype(str).str.lower().eq(key)]
        oracle_ba = float(selected_oracle.oracle_BA.mean())
        corrected_headroom = 100.0 * (oracle_ba - spec["mean_subject_BA"])
        reconstruction["benchmarks"][key] = {
            "V7_strongest_generic_method": spec["method_id"],
            "V7_strongest_generic_BA": spec["mean_subject_BA"],
            "V7_new_expert_subject_oracle_BA": oracle_ba,
            "V7_oracle_headroom_vs_strongest_generic_pp": corrected_headroom,
            "legacy_reported_headroom_used_weaker_V6_anchor": True,
            "preferred_V8_oracle_gate_BA": spec["mean_subject_BA"] + 0.08,
            "approximate_final_5pp_target_BA": spec["mean_subject_BA"] + 0.05,
        }
        baseline_rows.append({
            "benchmark": spec["benchmark"],
            "method_id": spec["method_id"],
            "mean_subject_BA": spec["mean_subject_BA"],
            "source_predictions": str(path),
            "source_predictions_sha256": sha256_file(path),
            "subjects": spec["subjects"],
            "information_matched": True,
            "target_future_labels_used_for_fit": False,
            "OUTER_TEST_USED": False,
        })

    write_json(PROTOCOL / "V7_RECONSTRUCTION.json", reconstruction)
    write_json(PROTOCOL / "BASELINE_LOCK.json", {
        "baseline_update_rule": "Promote any stronger fair information-matched generic discovered in V8.",
        "baselines": baseline_rows,
        "OUTER_TEST_USED": False,
    })
    write_csv(BASELINES / "BASELINE_LOCK.csv", pd.DataFrame(baseline_rows))
    write_json(PROTOCOL / "V8_SEARCH_SPLIT.json", split_payload)
    write_json(PROTOCOL / "V8_INTERNAL_HOLDOUT_LOCK.json", {
        "partition_file": str(PROTOCOL / "V8_SEARCH_SPLIT.json"),
        "partition_sha256": sha256_file(PROTOCOL / "V8_SEARCH_SPLIT.json"),
        "outcomes_opened_during_search": False,
        "unlock_token": "FROZEN_ONCE_20260821",
        "allowed_unlock_condition": "Phase A/B/C model, selector, PERSIST policy, and hyperparameters frozen",
        "evaluation_budget": "one opening only",
        "not_independent_confirmation": True,
        "OUTER_TEST_USED": False,
    })
    write_json(PROTOCOL / "HISTORY_LEGALITY.json", {
        "OpenBMI": {"history_sessions": [1], "future_session": 2},
        "WBCIC": {"history_sessions": [1, 2], "future_session": 3},
        "meta_training_query_labels": "legal only for V8_SEARCH training subjects",
        "search_outcome_future_labels": "scoring only; never fit transformations or policies",
        "internal_holdout_future_labels": "locked until one frozen evaluation",
        "OUTER_TEST_USED": False,
    })
    stage0_split = stage0_root() / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json"
    wbcic_scope = (
        wbcic_source_root() / "experiments" / "persist_eeg_wbcic_actionability_v2"
        / "outputs" / "protocol" / "DEVELOPMENT_SCOPE_LOCK.json"
    )
    scope = json.loads(wbcic_scope.read_text(encoding="utf-8"))
    if scope.get("outer_subject_ids_present") is not False:
        raise RuntimeError("WBCIC development scope contains outer subject IDs")
    write_json(PROTOCOL / "OUTER_LOCK.json", {
        "WBCIC_authorized_development_scope": str(wbcic_scope),
        "WBCIC_authorized_development_scope_sha256": sha256_file(wbcic_scope),
        "OpenBMI_source_split": str(stage0_split),
        "OpenBMI_source_split_sha256": sha256_file(stage0_split),
        "WBCIC_outer_split_file_opened": False,
        "WBCIC_outer_subjects_enumerated": False,
        "WBCIC_outer_raw_loaded": False,
        "WBCIC_outer_features_loaded": False,
        "WBCIC_outer_labels_loaded": False,
        "OUTER_TEST_USED": False,
    })
    print(json.dumps(reconstruction, indent=2), flush=True)
    print(
        "V8 protocol bootstrapped: search/holdout subject-only split locked; "
        "internal holdout outcomes unopened; WBCIC outer untouched.",
        flush=True,
    )


if __name__ == "__main__":
    run()
