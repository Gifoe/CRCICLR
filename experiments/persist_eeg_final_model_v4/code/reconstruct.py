from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from common import (
    PROTOCOL,
    V3_ROOT,
    default_openbmi_cache,
    default_wbcic_repo,
    canonical_hash,
    ensure_directories,
    markdown_table,
    sha256_file,
    write_json,
)
from datasets import ExpertDataset, load_openbmi, openbmi_full_pool, openbmi_keep_pool


def _mean_subject_ba(data: ExpertDataset, prediction: np.ndarray) -> float:
    values = []
    for subject in sorted(np.unique(data.subjects).tolist()):
        mask = data.subjects == subject
        values.append(balanced_accuracy_score(data.labels[mask], prediction[mask]))
    return float(np.mean(values))


def _oracle_prediction(
    labels: np.ndarray,
    base: np.ndarray,
    logits: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    prediction = np.array(base, copy=True)
    candidate_prediction = logits >= 0
    for index in range(len(labels)):
        if prediction[index] == labels[index]:
            continue
        correct = np.flatnonzero(mask[index] & (candidate_prediction[index] == labels[index]))
        if len(correct):
            prediction[index] = int(labels[index])
    return prediction


def _check(checks: list[dict[str, Any]], name: str, actual: Any, expected: Any, tolerance: float = 0.0) -> bool:
    if isinstance(actual, (float, np.floating)) or isinstance(expected, (float, np.floating)):
        difference = abs(float(actual) - float(expected))
        passed = difference <= tolerance
    else:
        difference = 0.0
        passed = actual == expected
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "expected": expected,
            "actual": actual,
            "absolute_difference": difference,
        }
    )
    return bool(passed)


def reconstruct_openbmi(cache_root: Path) -> tuple[dict[str, Any], ExpertDataset]:
    data = load_openbmi(cache_root)
    checks: list[dict[str, Any]] = []
    historical_repro = json.loads(
        (V3_ROOT / "outputs" / "REPRODUCIBILITY.json").read_text(encoding="utf-8")
    )
    cache_hashes = {path.name: sha256_file(path) for path in sorted(cache_root.glob("*.parquet"))}
    for name, expected in historical_repro["source_cache_sha256"].items():
        _check(checks, f"openbmi_cache_sha256:{name}", cache_hashes.get(name), expected)

    reference = pd.read_csv(
        V3_ROOT / "outputs" / "diagnostics" / "B6_UNIQUE_TRIAL_REFERENCE.csv"
    ).sort_values("manifest_index")
    observed = data.metadata.copy().sort_values("manifest_index")
    _check(checks, "openbmi_unique_trials", len(data.labels), 10_400)
    _check(checks, "openbmi_subjects", len(np.unique(data.subjects)), 52)
    _check(
        checks,
        "openbmi_manifest_identity_sha256",
        canonical_hash(observed.manifest_index.astype(int).tolist()),
        canonical_hash(reference.manifest_index.astype(int).tolist()),
    )
    max_margin_difference = float(
        np.max(np.abs(data.base_logits - reference.z_keep_ens.to_numpy(dtype=float)))
    )
    # The historical reference is CSV text, while V4 recomputes from the
    # float32 parquet logits. 5e-7 is the serialization bound; predictions,
    # metrics, identities, and source parquet hashes are still exact checks.
    _check(checks, "B6_unique_margin_max_abs_difference", max_margin_difference, 0.0, tolerance=5e-7)
    _check(
        checks,
        "B6_unique_prediction_exact",
        bool(np.array_equal(data.base_prediction, reference.y_keep_ens.to_numpy(dtype=int))),
        True,
    )
    b6_ba = _mean_subject_ba(data, data.base_prediction)
    _check(checks, "B6_mean_subject_BA", b6_ba, 0.8464423076923075, tolerance=1e-14)

    keep_logits, keep_mask, _ = openbmi_keep_pool(data)
    keep_oracle = _oracle_prediction(data.labels, data.base_prediction, keep_logits, keep_mask)
    keep_ba = _mean_subject_ba(data, keep_oracle)
    _check(checks, "KEEP_only_oracle_mean_subject_BA", keep_ba, 0.9230769230769231, tolerance=1e-14)

    action_logits = np.column_stack([data.base_logits, data.action_logits])
    action_mask = np.ones_like(action_logits, dtype=bool)
    action_oracle = _oracle_prediction(data.labels, data.base_prediction, action_logits, action_mask)
    action_ba = _mean_subject_ba(data, action_oracle)
    _check(checks, "KEEP_plus_global_ACTION_oracle_mean_subject_BA", action_ba, 0.932403846153846, tolerance=1e-14)

    full_logits, full_mask, _, _ = openbmi_full_pool(data, safe_alphas=())
    full_oracle = _oracle_prediction(data.labels, data.base_prediction, full_logits, full_mask)
    full_ba = _mean_subject_ba(data, full_oracle)
    _check(checks, "complete_KEEP_plus_ACTION_oracle_mean_subject_BA", full_ba, 0.9534615384615385, tolerance=1e-14)

    prospective_path = V3_ROOT / "outputs" / "results" / "RESIDUAL_POLICY_RESULTS.csv"
    expected_hash = historical_repro["artifact_sha256"]["results/RESIDUAL_POLICY_RESULTS.csv"]
    _check(checks, "V3_prospective_table_sha256", sha256_file(prospective_path), expected_hash)
    prospective = pd.read_csv(prospective_path)
    required = {
        "M0_B6_KEEP_ENSEMBLE",
        "M1_ENSEMBLE_CONFIDENCE_RULE",
        "M2_ENSEMBLE_DISAGREEMENT_RULE",
        "M3_ACTION_MOVEMENT_LOGISTIC",
        "M4_FULL_LEGAL_LOGISTIC",
        "M5_HIST_GRADIENT_BOOSTING",
        "I006_CONDITIONAL_ACTION_LOGISTIC",
        "I007_CONDITIONAL_ACTION_HGB",
    }
    _check(checks, "V3_prospective_method_roster", set(prospective.method_id), required)

    passed = all(item["passed"] for item in checks)
    payload = {
        "status": "BASELINE_RECONSTRUCTION_PASS" if passed else "BASELINE_RECONSTRUCTION_FAIL",
        "starting_commit": "ee9e280e350073f4cc30f8e3cad8c27cd1347bec",
        "B_STRONG": "B6_ALL_RUN_LOGIT_MEAN",
        "B_STRONG_mean_subject_BA": b6_ba,
        "single_KEEP_reference": "B0_TARGET_KEEP (historical V2.1 reference)",
        "unique_trials": len(data.labels),
        "subjects": len(np.unique(data.subjects)),
        "sessions": sorted(np.unique(data.sessions).tolist()),
        "keep_expert_slots": int(data.keep_run_logits.shape[1]),
        "action_families": ["AMPLIFY", "GEOMETRY", "ERASE"],
        "KEEP_only_oracle_BA": keep_ba,
        "KEEP_only_oracle_delta_BA": keep_ba - b6_ba,
        "global_ACTION_oracle_BA": action_ba,
        "global_ACTION_oracle_delta_BA": action_ba - b6_ba,
        "complete_KEEP_ACTION_oracle_BA": full_ba,
        "complete_KEEP_ACTION_oracle_delta_BA": full_ba - b6_ba,
        "V3_prospective_reconstruction": prospective[
            ["method_id", "mean_subject_BA", "mean_subject_delta_BA_vs_B6"]
        ].to_dict(orient="records"),
        "checks": checks,
        "source_cache_sha256": cache_hashes,
        "historical_artifacts_modified": False,
        "historical_csv_margin_tolerance": 5e-7,
        "OUTER_TEST_USED": False,
    }
    write_json(PROTOCOL / "BASELINE_RECONSTRUCTION.json", payload)
    table = pd.DataFrame(checks)[["check", "passed", "absolute_difference"]]
    report = f"""# V4 baseline reconstruction

`{payload['status']}`

- Frozen source commit: `ee9e280e350073f4cc30f8e3cad8c27cd1347bec`
- B_STRONG: `B6_ALL_RUN_LOGIT_MEAN`
- B_STRONG mean subject BA: `{b6_ba:.12f}`
- KEEP-only oracle: `{keep_ba:.12f}` ({100 * (keep_ba - b6_ba):+.3f} pp)
- Global action oracle: `{action_ba:.12f}` ({100 * (action_ba - b6_ba):+.3f} pp)
- Complete KEEP+ACTION oracle: `{full_ba:.12f}` ({100 * (full_ba - b6_ba):+.3f} pp)
- Historical V1/V2/V2.1/V3 files modified: `false`
- WBCIC outer used: `false`

V3 prospective methods are provenance-reconstructed by exact artifact hash;
B6, the unique-trial expert space, and all three oracle ladders are rebuilt
numerically from the frozen parquet cache in the new V4 directory.

{markdown_table(table)}
"""
    (PROTOCOL / "BASELINE_RECONSTRUCTION.md").write_text(report, encoding="utf-8")
    if not passed:
        raise RuntimeError("V4 baseline reconstruction failed")
    return payload, data


def audit_legality(cache_root: Path, wbcic_repo: Path) -> dict[str, Any]:
    scope_path = (
        wbcic_repo
        / "experiments"
        / "persist_eeg_wbcic_actionability_v2"
        / "outputs"
        / "protocol"
        / "DEVELOPMENT_SCOPE_LOCK.json"
    )
    cache_audit_path = scope_path.with_name("CACHE_SCOPE_AUDIT.json")
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    cache_audit = json.loads(cache_audit_path.read_text(encoding="utf-8"))
    allowed = [str(value) for value in scope["allowed_subjects"]]
    development_cache = (
        wbcic_repo
        / "experiments"
        / "persist_eeg_wbcic_actionability_v2"
        / "outputs"
        / "cache"
        / "wbcic_epochs"
    )
    materialized = sorted([path.name for path in development_cache.iterdir() if path.is_dir()])
    checks = {
        "openbmi_cache_files_exact": sorted(path.name for path in cache_root.glob("*.parquet"))
        == [
            "OOF_BASE_LOGITS.parquet",
            "OOF_COUNTERFACTUAL_LOGITS.parquet",
            "OOF_GEOMETRY_FEATURES.parquet",
            "OOF_ROUTER_FEATURES.parquet",
        ],
        "wbcic_development_scope_count_41": len(allowed) == 41 and len(set(allowed)) == 41,
        "wbcic_scope_omits_outer_ids": scope.get("outer_subject_ids_present") is False,
        "wbcic_runtime_outer_lock_denylisted": scope.get("runtime_must_not_open") == "OUTER_SPLIT_LOCK.json",
        "wbcic_cache_scope_pass": cache_audit.get("status") == "DEVELOPMENT_CACHE_COMPLETE",
        "wbcic_materialized_equals_allowed_development": set(materialized) == set(allowed),
        "wbcic_outer_materialized_false": cache_audit.get("outer_subject_ids_materialized") is False,
        "historical_outer_flag_false": cache_audit.get("sealed_outer_split_opened") is False,
    }
    passed = all(checks.values())
    payload = {
        "status": "DATA_LEGALITY_AUDIT_PASS" if passed else "DATA_LEGALITY_AUDIT_FAIL",
        "openbmi": {
            "role": "discovery_architecture_sandbox",
            "historical_development_subjects": 52,
            "cache_root": str(cache_root),
            "confirmatory": False,
        },
        "wbcic": {
            "role": "authorized_development_only",
            "allowed_subject_count": len(allowed),
            "allowed_subjects_hash": scope.get("allowed_subjects_hash"),
            "outer_subject_count_hash_only": scope.get("outer_subject_hash_only"),
            "outer_subject_ids_loaded": False,
            "raw_root_enumerated": False,
            "development_cache_root": str(development_cache),
        },
        "explicit_read_denylist": [
            "**/OUTER_SPLIT_LOCK.json",
            "**/code/outer.py at runtime",
            "**/wbcic_outer_S3/**",
            "**/*OUTER_RESULT*",
            "raw paths for any subject not in DEVELOPMENT_SCOPE_LOCK.json",
        ],
        "checks": checks,
        "OUTER_TEST_USED": False,
    }
    write_json(PROTOCOL / "DATA_LEGALITY_AUDIT.json", payload)
    if not passed:
        raise RuntimeError("V4 data-legality audit failed")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openbmi-cache", type=Path, default=default_openbmi_cache())
    parser.add_argument("--wbcic-repo", type=Path, default=default_wbcic_repo())
    args = parser.parse_args()
    ensure_directories()
    baseline, _ = reconstruct_openbmi(args.openbmi_cache)
    legality = audit_legality(args.openbmi_cache, args.wbcic_repo)
    print(json.dumps({"baseline": baseline["status"], "legality": legality["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
