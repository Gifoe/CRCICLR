from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from build_unique_trials import build_unique_trials, load_all_target_rows
from v3_common import (
    DATASET_ID,
    PROTOCOL,
    V21_CODE,
    V21_OUTPUTS,
    V2_OUTPUTS,
    canonical_hash,
    ensure_directories,
    markdown_table,
    sha256_file,
    write_json,
)


if str(V21_CODE) not in sys.path:
    sys.path.append(str(V21_CODE))
from ensemble_baselines import build_ensemble_baselines  # noqa: E402
from reconstruct_v2 import v2_policies_for_frame  # noqa: E402


EXPECTED_V2_POLICY_LOCK = "e679c7a955ccf3745bb35ce6c86a61c57705557f3eed8917b724b0e5613b5fd4"
METHODS = (
    "B0_TARGET_KEEP",
    "B4_ALL_RUN_PROBABILITY_MEAN",
    "B6_ALL_RUN_LOGIT_MEAN",
    "B7_CONFIDENCE_WEIGHTED_KEEP_ENSEMBLE",
    "I003_CROSS_RUN_FULL",
    "I003_CROSS_RUN_PROTECTED_SAFE",
)


def write_residual_action_spec() -> dict[str, Any]:
    ensure_directories()
    spec = {
        "status": "V3_RESIDUAL_ACTION_SPEC_FROZEN_BEFORE_ORACLE_OUTCOMES",
        "experiment_type": "exploratory discovery on 52 historical development subjects",
        "reference": {
            "method_id": "B6_ALL_RUN_LOGIT_MEAN",
            "margin": "arithmetic mean of all available frozen KEEP binary margins",
            "probability": "sigmoid(mean margin)",
            "prediction": "class 1 iff mean margin >= 0",
            "unit": "one row per dataset x subject x session x manifest trial",
        },
        "action_menus": {
            "PROTECTED_SAFE_GLOBAL": ["KEEP_ENSEMBLE", "ALL_AMPLIFY", "ALL_GEOMETRY"],
            "FULL_GLOBAL": ["KEEP_ENSEMBLE", "ALL_AMPLIFY", "ALL_GEOMETRY", "ALL_ERASE"],
            "PROTECTED_SAFE_SINGLE_REPLACEMENT": [
                "KEEP_ENSEMBLE",
                "each available r->AMPLIFY",
                "each available r->GEOMETRY",
            ],
            "FULL_SINGLE_REPLACEMENT": [
                "KEEP_ENSEMBLE",
                "each available r->AMPLIFY",
                "each available r->GEOMETRY",
                "each available r->ERASE",
            ],
        },
        "keep_only_menu": [
            "B6 all-run logit mean",
            "B4 all-run probability mean",
            "B2 all-run hard majority with tie->class1",
            "each leave-one-run KEEP logit mean",
            "each individual frozen KEEP expert",
        ],
        "oracle_priority": "KEEP first; if wrong, lexicographically first correct candidate within fixed menu",
        "primary_metric": "mean subject balanced accuracy and paired subject bootstrap Delta BA vs B6",
        "bootstrap_repetitions": 10000,
        "bootstrap_seed": 20260819,
        "headroom_gate": {
            "state_A_if": "strongest action oracle Delta BA < 0.005 OR top-20%-subject concentration >= 0.90 OR STATE C breadth criteria fail after generic-diversity exclusion",
            "state_B_if": "action oracle beats B6 but action-oracle minus KEEP-only-oracle Delta BA < 0.005",
            "state_C_all_required": {
                "strongest_action_oracle_delta_BA_min": 0.01,
                "action_oracle_minus_keep_only_oracle_delta_BA_min": 0.005,
                "combined_keep_plus_action_minus_keep_oracle_delta_BA_min": 0.005,
                "positive_subjects_min": 8,
                "positive_subject_fraction_min": 0.20,
                "positive_sessions_min": 2,
                "dominant_unique_action_rescue_fraction_max": 0.80,
                "dominant_unique_run_rescue_fraction_max": 0.80,
            },
        },
        "phase_8_plus_authorized_only_by": "STRUCTURAL_ACTION_RESIDUAL_EXISTS",
        "outer_test_authorized": False,
        "OUTER_TEST_USED": False,
    }
    spec["spec_sha256"] = canonical_hash(spec)
    write_json(PROTOCOL / "RESIDUAL_ACTION_SPEC.json", spec)
    return spec


def _target_run_subject_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    labels = frame.outcome_label.to_numpy(dtype=int)
    rows = []
    for (fold, seed, subject), indices in frame.groupby(["fold_id", "seed_id", "subject_id"]).indices.items():
        idx = np.asarray(indices, dtype=int)
        rows.append(
            {
                "fold_id": int(fold),
                "seed_id": int(seed),
                "subject_id": str(subject),
                "balanced_accuracy": float(balanced_accuracy_score(labels[idx], prediction[idx])),
                "accuracy": float(accuracy_score(labels[idx], prediction[idx])),
                "macro_f1": float(f1_score(labels[idx], prediction[idx], average="macro", zero_division=0)),
            }
        )
    return pd.DataFrame(rows).groupby("subject_id", as_index=False).agg(
        balanced_accuracy=("balanced_accuracy", "mean"),
        accuracy=("accuracy", "mean"),
        macro_f1=("macro_f1", "mean"),
    )


def _compare_value(
    checks: list[dict[str, Any]],
    name: str,
    actual: float,
    expected: float,
    tolerance: float = 1e-14,
) -> bool:
    difference = abs(float(actual) - float(expected))
    passed = bool(difference <= tolerance)
    checks.append(
        {
            "check": name,
            "passed": passed,
            "expected": float(expected),
            "actual": float(actual),
            "absolute_difference": difference,
        }
    )
    return passed


def reconstruct_v21(cache_root: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    ensure_directories()
    checks: list[dict[str, Any]] = []
    previous_reconstruction = json.loads(
        (V21_OUTPUTS / "protocol" / "V2_RECONSTRUCTION.json").read_text(encoding="utf-8")
    )
    previous_final = json.loads((V21_OUTPUTS / "FINAL_DECISION.json").read_text(encoding="utf-8"))
    passed = previous_reconstruction["status"] == "V2_RECONSTRUCTION_PASS"
    checks.append(
        {
            "check": "historical_V2_1_reconstruction_status",
            "passed": passed,
            "expected": "V2_RECONSTRUCTION_PASS",
            "actual": previous_reconstruction["status"],
            "absolute_difference": 0.0,
        }
    )
    lock_ok = previous_reconstruction["v2_policy_lock_hash"] == EXPECTED_V2_POLICY_LOCK
    checks.append(
        {
            "check": "V2_policy_lock_hash",
            "passed": lock_ok,
            "expected": EXPECTED_V2_POLICY_LOCK,
            "actual": previous_reconstruction["v2_policy_lock_hash"],
            "absolute_difference": 0.0,
        }
    )
    passed &= lock_ok

    leakage = json.loads((V21_OUTPUTS / "protocol" / "LEAKAGE_AUDIT.json").read_text(encoding="utf-8"))
    cache_hashes = {path.name: sha256_file(path) for path in sorted(cache_root.glob("*.parquet"))}
    for name, expected in leakage["source_cache_sha256"].items():
        actual = cache_hashes.get(name)
        item_ok = actual == expected
        checks.append(
            {
                "check": f"cache_sha256:{name}",
                "passed": item_ok,
                "expected": expected,
                "actual": actual,
                "absolute_difference": 0.0,
            }
        )
        passed &= item_ok

    target_rows = load_all_target_rows(cache_root)
    unique_trials = build_unique_trials(target_rows)
    historical_ensemble = pd.read_csv(V21_OUTPUTS / "results" / "ENSEMBLE_BASELINE_RESULTS.csv")
    historical_controls = pd.read_csv(V21_OUTPUTS / "results" / "CONSENSUS_CONTROL_RESULTS.csv")
    historical_deployment = pd.read_csv(V21_OUTPUTS / "results" / "DEPLOYMENT_LEVEL_RESULTS.csv")
    for pool, source_pool in (("exploration", "exploration"), ("holdout", "holdout")):
        frame = target_rows[target_rows.source_pool.eq(source_pool)].reset_index(drop=True)
        methods = build_ensemble_baselines(frame)
        baseline_subject = _target_run_subject_metrics(frame, methods["B0_TARGET_KEEP"].prediction)
        baseline_mean = float(baseline_subject.balanced_accuracy.mean())
        for method_id in (
            "B0_TARGET_KEEP",
            "B4_ALL_RUN_PROBABILITY_MEAN",
            "B6_ALL_RUN_LOGIT_MEAN",
            "B7_CONFIDENCE_WEIGHTED_KEEP_ENSEMBLE",
        ):
            subject = _target_run_subject_metrics(frame, methods[method_id].prediction)
            actual_ba = float(subject.balanced_accuracy.mean())
            actual_delta = actual_ba - baseline_mean
            expected = historical_ensemble[
                historical_ensemble.pool.eq(pool) & historical_ensemble.method_id.eq(method_id)
            ].iloc[0]
            passed &= _compare_value(checks, f"{pool}:{method_id}:mean_subject_BA", actual_ba, expected.mean_subject_BA)
            passed &= _compare_value(
                checks,
                f"{pool}:{method_id}:mean_subject_delta_BA",
                actual_delta,
                expected.mean_paired_delta_BA,
            )
        policies = v2_policies_for_frame(frame)
        for method_id in ("I003_CROSS_RUN_FULL", "I003_CROSS_RUN_PROTECTED_SAFE"):
            subject = _target_run_subject_metrics(frame, policies[method_id]["prediction"])
            actual_ba = float(subject.balanced_accuracy.mean())
            actual_delta = actual_ba - baseline_mean
            expected = historical_controls[
                historical_controls.pool.eq(pool) & historical_controls.method_id.eq(method_id)
            ].iloc[0]
            passed &= _compare_value(checks, f"{pool}:{method_id}:mean_subject_BA", actual_ba, expected.mean_subject_BA)
            passed &= _compare_value(
                checks,
                f"{pool}:{method_id}:mean_subject_delta_BA",
                actual_delta,
                expected.vs_target_mean_paired_delta_BA,
            )

        unique = unique_trials[unique_trials.source_pool.eq(source_pool)].reset_index(drop=True)
        labels = unique.outcome_label.to_numpy(dtype=int)
        prediction = unique.y_keep_ens.to_numpy(dtype=int)
        subject_rows = []
        for subject, indices in unique.groupby("subject_id").indices.items():
            idx = np.asarray(indices, dtype=int)
            subject_rows.append(balanced_accuracy_score(labels[idx], prediction[idx]))
        deployment_ba = float(np.mean(subject_rows))
        expected_deployment = historical_deployment[
            historical_deployment.pool.eq(pool)
            & historical_deployment.method_id.eq("B6_ALL_RUN_LOGIT_MEAN")
        ].iloc[0]
        passed &= _compare_value(
            checks,
            f"{pool}:B6:deployment_mean_subject_BA",
            deployment_ba,
            expected_deployment.mean_subject_BA,
        )

    identity_payload = [
        {
            "dataset": row.dataset,
            "subject_id": row.subject_id,
            "session_id": row.session_id,
            "manifest_index": int(row.manifest_index),
            "run_ids": list(row.run_ids),
        }
        for row in unique_trials.itertuples(index=False)
    ]
    prediction_payload = [
        {
            "trial_uid": row.trial_uid,
            "z_keep_ens": float(row.z_keep_ens),
            "p_keep_ens": float(row.p_keep_ens),
            "y_keep_ens": int(row.y_keep_ens),
        }
        for row in unique_trials.itertuples(index=False)
    ]
    payload = {
        "status": "V2_1_RECONSTRUCTION_PASS" if passed else "V2_1_RECONSTRUCTION_FAIL",
        "checks": checks,
        "numerical_tolerance": 1e-14,
        "dataset": DATASET_ID,
        "target_run_rows": int(len(target_rows)),
        "unique_trials": int(len(unique_trials)),
        "subjects": int(unique_trials.subject_id.nunique()),
        "sessions": sorted(unique_trials.session_id.unique().tolist()),
        "run_ids": sorted({item for values in unique_trials.run_ids for item in values}),
        "unique_trial_identity_sha256": canonical_hash(identity_payload),
        "B6_unique_prediction_sha256": canonical_hash(prediction_payload),
        "historical_v2_1_final_state": previous_final["primary_state"],
        "historical_v2_1_best_ensemble": previous_final["best_ensemble_selected_on_exploration"],
        "source_cache_sha256": cache_hashes,
        "OUTER_TEST_USED": False,
    }
    write_json(PROTOCOL / "V2_1_RECONSTRUCTION.json", payload)
    check_table = pd.DataFrame(checks)[["check", "passed", "absolute_difference"]]
    report = f"""# Exact V2.1 reconstruction

`{payload['status']}`

- Historical V2.1 state: `{payload['historical_v2_1_final_state']}`
- Frozen constructive reference: `{payload['historical_v2_1_best_ensemble']}`
- Target-run rows: `{payload['target_run_rows']}`
- Unique deployment trials: `{payload['unique_trials']}`
- Subjects: `{payload['subjects']}`
- Numerical tolerance: `1e-14`
- WBCIC outer accessed: `false`

B0, B4, B6, B7, FULL and protected-safe were rebuilt from the frozen cache.
B6 target-run and unique-trial subject metrics were compared to the historical
V2.1 tables. The unique B6 prediction hash is stored in the JSON artifact.

{markdown_table(check_table)}
"""
    (PROTOCOL / "V2_1_RECONSTRUCTION.md").write_text(report, encoding="utf-8")

    provenance = {
        "status": "PROVENANCE_AUDIT_PASS" if passed else "PROVENANCE_AUDIT_FAIL",
        "dataset": DATASET_ID,
        "source_cache_sha256": cache_hashes,
        "v2_policy_lock_hash": EXPECTED_V2_POLICY_LOCK,
        "v2_1_reconstruction_sha256": sha256_file(V21_OUTPUTS / "protocol" / "V2_RECONSTRUCTION.json"),
        "v2_1_final_decision_sha256": sha256_file(V21_OUTPUTS / "FINAL_DECISION.json"),
        "frozen_model_provenance": {
            "run_ids": payload["run_ids"],
            "available_run_counts": sorted(unique_trials.n_runs.unique().astype(int).tolist()),
            "identity_sha256": payload["unique_trial_identity_sha256"],
        },
        "identity_checks": {
            "one_subject_per_manifest": True,
            "one_session_per_manifest": True,
            "one_label_per_manifest": True,
            "no_duplicate_run_per_manifest": True,
            "unique_dataset_subject_session_manifest": True,
        },
        "outer_test_authorized": False,
        "OUTER_TEST_USED": False,
    }
    write_json(PROTOCOL / "PROVENANCE_AUDIT.json", provenance)
    if not passed:
        raise RuntimeError("V2.1/B6 reconstruction gate failed")
    return payload, target_rows, unique_trials
