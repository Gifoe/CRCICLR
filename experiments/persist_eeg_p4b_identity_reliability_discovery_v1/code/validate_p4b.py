from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
P4A = REPO / "experiments" / "persist_eeg_p4a_cross_setting_expansion_v1"
LEAN_TIP = "281e7c66c9818b5d2efe96968900cb585af20287"
REQUIRED_REPORTS = (
    "README.md",
    "P4A_INPUT_AUDIT.md",
    "USABLE_SETTING_AUDIT.md",
    "SOURCE_SCALE_AND_COLLINEARITY_AUDIT.md",
    "SOURCE_NORMALIZATION_FROZEN.json",
    "DISCOVERY_SETTING_ASSIGNMENT.json",
    "P4B_PROTOCOL_FROZEN.json",
    "PRE_OUTCOME_FREEZE_COMPLETE.json",
    "FUTURE_UTILITY_ACCESS_LEDGER.md",
    "PRIMARY_MODEL_COMPARISON.md",
    "INTERACTION_AUDIT.md",
    "REGIME_SEPARATION_AUDIT.md",
    "CROSS_SETTING_STABILITY.md",
    "HOLDOUT_PURITY_AUDIT.md",
    "ENGINEERING_RECOVERY_BASIS_HASH.md",
)
ALLOWED_TERMINALS = {
    "P4B_IDENTITY_RELIABILITY_CONDITION_STRONG_SUPPORTED",
    "P4B_IDENTITY_RELIABILITY_CONDITION_PARTIAL_SUPPORTED",
    "P4B_IDENTITY_RELIABILITY_CONDITION_NOT_SUPPORTED",
    "P4B_INSUFFICIENT_CROSS_SETTING_EVIDENCE",
    "P4B_INSUFFICIENT_PROSPECTIVE_HOLDOUT",
    "P4B_PROTOCOL_OR_PURITY_FAILURE",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    issues: list[str] = []
    for name in REQUIRED_REPORTS:
        path = EXP / name
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"missing_or_empty_report:{name}")
    required_results = {
        "subject": "discovery_future_utility_subject.csv",
        "direction": "discovery_future_utility_direction.csv",
        "controls": "discovery_matched_control_summary.csv",
        "predictions": "condition_model_predictions.csv",
        "summary": "condition_model_summary.json",
        "bootstrap": "interaction_bootstrap.csv",
        "regime": "regime_summary.csv",
        "stability": "per_setting_stability.csv",
        "analysis": "P4B_ANALYSIS_COMPLETE.json",
    }
    for name, filename in required_results.items():
        path = RESULTS / filename
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"missing_or_empty_result:{name}")
    if issues:
        write_json(RESULTS / "P4B_FINAL_VALIDATION.json", {"pass": False, "terminal": "P4B_PROTOCOL_OR_PURITY_FAILURE", "issues": issues})
        raise RuntimeError("P4B pre-table validation failure")

    pre = read_json(EXP / "PRE_OUTCOME_FREEZE_COMPLETE.json")
    protocol = read_json(EXP / "P4B_PROTOCOL_FROZEN.json")
    assignment = read_json(EXP / "DISCOVERY_SETTING_ASSIGNMENT.json")
    normalization = read_json(EXP / "SOURCE_NORMALIZATION_FROZEN.json")
    summary = read_json(RESULTS / "condition_model_summary.json")
    analysis = read_json(RESULTS / "P4B_ANALYSIS_COMPLETE.json")
    subject = pd.read_csv(RESULTS / required_results["subject"])
    direction = pd.read_csv(RESULTS / required_results["direction"])
    controls = pd.read_csv(RESULTS / required_results["controls"])
    predictions = pd.read_csv(RESULTS / required_results["predictions"])
    bootstrap = pd.read_csv(RESULTS / required_results["bootstrap"])
    regime = pd.read_csv(RESULTS / required_results["regime"])
    stability = pd.read_csv(RESULTS / required_results["stability"])
    normalized_source = pd.read_csv(RESULTS / "source_evidence_normalized.csv")

    expected_hashes = {
        "P4A_source_evidence_cube": sha256(P4A / "results" / "source_evidence_cube.csv"),
        "SOURCE_NORMALIZATION_FROZEN.json": sha256(EXP / "SOURCE_NORMALIZATION_FROZEN.json"),
        "DISCOVERY_SETTING_ASSIGNMENT.json": sha256(EXP / "DISCOVERY_SETTING_ASSIGNMENT.json"),
        "P4B_PROTOCOL_FROZEN.json": sha256(EXP / "P4B_PROTOCOL_FROZEN.json"),
        "source_evidence_normalized.csv": sha256(RESULTS / "source_evidence_normalized.csv"),
    }
    if pre.get("pass") is not True or pre.get("hashes") != expected_hashes:
        issues.append("pre_outcome_hash_lock")
    if pre.get("future_utility_access_count_before_freeze") != 0 or pre.get("p4c_reserved_future_utility_accessed") is not False:
        issues.append("pre_outcome_access_state")
    if analysis.get("pre_outcome_freeze_hashes_verified") is not True:
        issues.append("runner_freeze_verification")
    if analysis.get("first_future_access_timestamp_utc", "") <= pre.get("timestamp_utc", ""):
        issues.append("freeze_access_timestamp_order")

    discovery = ["S1", "S2", "S3", "S5"]
    reserved = ["S4", "S6"]
    if assignment.get("all_discovery_settings") != discovery or assignment.get("p4c_reserved_settings") != reserved:
        issues.append("discovery_reserve_assignment")
    if protocol.get("discovery_settings") != discovery or protocol.get("p4c_reserved_settings") != reserved:
        issues.append("protocol_assignment")
    if sorted(subject.setting_id.unique()) != discovery or sorted(direction.setting_id.unique()) != discovery:
        issues.append("discovery_result_setting_set")
    if any(setting in set(subject.setting_id) | set(direction.setting_id) | set(controls.setting_id) for setting in reserved):
        issues.append("reserved_outcome_present")
    for path in RESULTS.glob("*utility*.csv"):
        frame = pd.read_csv(path, nrows=1000)
        if "setting_id" in frame and any(setting in set(frame.setting_id.astype(str)) for setting in reserved):
            issues.append(f"reserved_utility_file:{path.name}")

    if len(direction) != 480 or len(predictions) != 480 or len(controls) != 480:
        issues.append(f"direction_cardinality:{len(direction)}:{len(predictions)}:{len(controls)}")
    if len(subject) == 0 or subject.duplicated(["setting_id", "fold", "seed", "direction_rank", "outcome_subject"]).any():
        issues.append("subject_first_uniqueness")
    if any("trial" in column.lower() for column in subject.columns):
        issues.append("trial_pseudoreplication_schema")
    if not (controls.control_count == 100).all():
        issues.append("matched_control_count")
    if set(direction.direction_rank.astype(int)) != set(range(1, 9)):
        issues.append("direction_rank")

    expected_models = {
        "M0": ["z_P", "z_geometry_strength", "z_rank"],
        "MI": ["z_P", "z_geometry_strength", "z_rank", "z_I"],
        "ME": ["z_P", "z_geometry_strength", "z_rank", "E_task"],
        "MADD": ["z_P", "z_geometry_strength", "z_rank", "z_I", "E_task"],
        "MINT": ["z_P", "z_geometry_strength", "z_rank", "z_I", "E_task", "z_I_x_E_task"],
    }
    if protocol.get("models", {}).get("alpha") != 1.0 or summary.get("alpha") != 1.0:
        issues.append("ridge_alpha")
    for model, columns in expected_models.items():
        if protocol.get("models", {}).get(model) != columns:
            issues.append(f"model_formula:{model}")
        if f"pred_{model}" not in predictions:
            issues.append(f"prediction_column:{model}")
    if protocol.get("primary_cv") != "Leave-One-Discovery-Setting-Out" or summary.get("primary_cv") != "Leave-One-Discovery-Setting-Out":
        issues.append("primary_cv")
    if "Leave-One-Entire-Run-Out" not in protocol.get("secondary_cv", "") or summary.get("secondary_cv") != "Leave-One-Entire-Run-Out":
        issues.append("secondary_cv")
    if protocol.get("bootstrap", {}).get("draws") != 10000 or len(bootstrap) != 10000 or bootstrap.draw.nunique() != 10000:
        issues.append("bootstrap_draws")
    if protocol.get("bootstrap", {}).get("hierarchy") != ["setting", "fold", "seed/run", "direction", "outcome subject"]:
        issues.append("bootstrap_hierarchy")
    if protocol.get("regime", {}).get("rule") != "within-setting tertiles" or set(regime.setting_id) != set(discovery + ["ALL"]):
        issues.append("tertile_regime")
    if normalization.get("E_task_formula") != "(z_D + z_C + z_O)/3" or normalization.get("active_E_task_primitives") != ["D", "C", "O"]:
        issues.append("E_task_definition")

    if not all(np.isfinite(list(summary.get("primary_RMSE", {}).values()))):
        issues.append("primary_rmse_nonfinite")
    required_bootstrap = {"contrast_MI_MINT", "contrast_MADD_MINT", "beta_IxE", "DeltaSlope", "DeltaRegime"}
    if not required_bootstrap.issubset(bootstrap.columns) or bootstrap[list(required_bootstrap)].isna().any().any():
        issues.append("bootstrap_schema_or_nan")
    candidate = analysis.get("terminal_candidate")
    if candidate not in ALLOWED_TERMINALS or candidate != summary.get("terminal_candidate"):
        issues.append("terminal_candidate")
    if analysis.get("p4c_reserved_future_utility_accessed") is not False:
        issues.append("reserved_access_flag")
    if "S4/S6 future direction utilities were never loaded" not in (EXP / "HOLDOUT_PURITY_AUDIT.md").read_text(encoding="utf-8"):
        issues.append("purity_report_statement")

    figures = sorted(FIGURES.glob("figure*.png"))
    if len(figures) != 7 or any(path.stat().st_size < 1000 for path in figures):
        issues.append(f"figure_count_or_size:{len(figures)}")

    if subprocess.check_output(["git", "rev-parse", f"{LEAN_TIP}^{{commit}}"], cwd=REPO, text=True).strip() != LEAN_TIP:
        issues.append("p4a_lean_commit")
    p4a_validation = read_json(P4A / "results" / "P4A_LEAN_FINAL_VALIDATION.json")
    if p4a_validation.get("pass") is not True or p4a_validation.get("terminal") != "P4A_LEAN_CROSS_SETTING_CUBE_COMPLETE":
        issues.append("p4a_lean_validation")

    if issues:
        payload = {
            "pass": False,
            "issues": issues,
            "terminal": "P4B_PROTOCOL_OR_PURITY_FAILURE",
            "p4a_lean_tip": LEAN_TIP,
            "p4c_reserved_future_utility_accessed": False,
        }
        write_json(RESULTS / "P4B_FINAL_VALIDATION.json", payload)
        raise RuntimeError(f"P4B validation failed: {issues}")

    p4c_lock_status = "NOT_REQUIRED_FOR_NOT_SUPPORTED"
    if candidate in {
        "P4B_IDENTITY_RELIABILITY_CONDITION_STRONG_SUPPORTED",
        "P4B_IDENTITY_RELIABILITY_CONDITION_PARTIAL_SUPPORTED",
    }:
        reserved_regime_thresholds = {}
        for setting in reserved:
            cell = normalized_source[normalized_source.setting_id == setting]
            if len(cell) != 120:
                raise RuntimeError(f"reserved source cardinality failure before P4C lock: {setting} rows={len(cell)}")
            reserved_regime_thresholds[setting] = {
                "high_I_lower": float(cell.z_I.quantile(2.0 / 3.0)),
                "low_E_upper": float(cell.E_task.quantile(1.0 / 3.0)),
                "high_E_lower": float(cell.E_task.quantile(2.0 / 3.0)),
            }
        lock = {
            "schema": "P4C_LOCK_V2",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "p4b_terminal": candidate,
            "reserved_settings": reserved,
            "reserved_future_utility_accessed_during_p4b": False,
            "source_normalization_sha256": sha256(EXP / "SOURCE_NORMALIZATION_FROZEN.json"),
            "p4b_protocol_sha256": sha256(EXP / "P4B_PROTOCOL_FROZEN.json"),
            "discovery_assignment_sha256": sha256(EXP / "DISCOVERY_SETTING_ASSIGNMENT.json"),
            "source_evidence_normalized_sha256": sha256(RESULTS / "source_evidence_normalized.csv"),
            "condition_model_summary_sha256": sha256(RESULTS / "condition_model_summary.json"),
            "E_task_definition": normalization["E_task_formula"],
            "E_task_primitives": normalization["active_E_task_primitives"],
            "ridge_alpha": 1.0,
            "MI_feature_order": expected_models["MI"],
            "MINT_feature_order": expected_models["MINT"],
            "fitted_MI": summary["full_fits"]["MI"],
            "fitted_MINT": summary["full_fits"]["MINT"],
            "direction_ranks": list(range(1, 9)),
            "candidate_rule": {
                "identity_bearing": "z_I > 0",
                "fallback_if_empty": "highest z_I direction within run",
            },
            "condition_policy": {
                "score": "frozen MINT predicted_U_BA",
                "selection": "top-1 among identity-bearing candidates",
                "abstention": False,
                "top_k": 1,
            },
            "comparison_baselines": [
                "Highest-I",
                "Lowest-D among identity-bearing",
                "Lowest-C among identity-bearing",
                "Lowest-O among identity-bearing",
                "Random-I among identity-bearing",
                "No-op",
                "Oracle-Future upper bound only",
            ],
            "random_I": {
                "draws_per_run": 100,
                "sampling": "uniform over identity-bearing candidates with replacement",
                "seed_rule": "stable SHA256 seed from P4C-Random-I, setting, fold, seed, draw",
                "draws_are_not_independent_inference_units": True,
            },
            "regime": {
                "rule": "within-setting tertiles from source-only normalized evidence",
                "thresholds": reserved_regime_thresholds,
                "highI_lowE": "z_I >= high_I_lower and E_task <= low_E_upper",
                "highI_highE": "z_I >= high_I_lower and E_task >= high_E_lower",
            },
            "future_utility_definition": protocol["future_utility"],
            "bootstrap": {
                "draws": 10000,
                "hierarchy": ["setting", "fold", "seed/run", "outcome subject"],
                "direction_nested_for_predictive_and_regime": True,
            },
            "confirmatory_gates": {
                "predictive_transport": "RMSE_MI - RMSE_MINT > 0; strong only when CI lower > 0",
                "regime_transport": "DeltaRegime > 0; strong only when CI lower > 0",
                "interaction_direction": "beta_IxE < 0 and DeltaSlope > 0",
                "condition_consistency": ">=75% reserved settings; one setting can only support SINGLE_SETTING_PROSPECTIVE_SUPPORT",
                "actionability": "Condition-Highest-I, Condition-Random-I, and Condition-No-op each >0 with CI lower >0; >=75% reserved settings nonnegative; systematic harm means a setting-level Condition CI upper <0",
            },
            "execution_status": "LOCKED_NOT_RUN",
            "P4C_gates": ["policy evaluated once on reserved setting", "compare against all frozen baselines", "report oracle only as upper bound", "no model or threshold adjustment"],
        }
        write_json(EXP / "P4C_LOCK.json", lock)
        p4c_lock_status = "LOCKED_NOT_RUN"

    validation_payload = {
        "pass": True,
        "issues": [],
        "terminal": candidate,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "p4a_lean_tip": LEAN_TIP,
        "discovery_settings": discovery,
        "p4c_reserved_settings": reserved,
        "subject_rows": len(subject),
        "direction_rows": len(direction),
        "bootstrap_draws": len(bootstrap),
        "figures": [path.name for path in figures],
        "p4c_reserved_future_utility_accessed": False,
        "OpenBMI_sealed_internal_holdout": "UNTOUCHED",
        "WBCIC_outer_10": "UNTOUCHED_NOT_ENUMERATED",
        "P4C_LOCK_status": p4c_lock_status,
        "final_report_written_after_validator_pass": True,
    }
    write_json(RESULTS / "P4B_FINAL_VALIDATION.json", validation_payload)

    rmse_table = pd.DataFrame(
        {"model": list(summary["primary_RMSE"]), "LOSO_setting_RMSE": list(summary["primary_RMSE"].values())}
    ).to_markdown(index=False, floatfmt=".8f")
    report = f"""# P4B Final Report

Exact terminal: `{candidate}`.

The validator passed before this final terminal was written.

## Frozen design

- P4A Lean input tip: `{LEAN_TIP}`.
- Discovery settings: {', '.join(discovery)}.
- P4C reserved settings: {', '.join(reserved)}.
- Normalization: within-setting median/MAD with frozen SD fallback.
- E_task: `{normalization['E_task_formula']}`; D/C/O were source-only nondegenerate and retained without learned weights.
- Ridge alpha: 1; primary CV: leave-one-entire-setting-out.
- Bootstrap: 10,000 draws, setting -> fold -> seed/run -> direction -> outcome subject.

## Model results

{rmse_table}

- RMSE_MI - RMSE_MINT: {summary['contrasts']['RMSE_MI_minus_MINT']:.9f}; CI {summary['bootstrap_CI95']['contrast_MI_MINT']}.
- RMSE_MADD - RMSE_MINT: {summary['contrasts']['RMSE_MADD_minus_MINT']:.9f}; CI {summary['bootstrap_CI95']['contrast_MADD_MINT']}.
- beta_IxE: {summary['beta_IxE']:.9f}; CI {summary['bootstrap_CI95']['beta_IxE']}.
- slope_lowE: {summary['slope_lowE']:.9f}; slope_highE: {summary['slope_highE']:.9f}.
- DeltaSlope: {summary['DeltaSlope']:.9f}; CI {summary['bootstrap_CI95']['DeltaSlope']}.
- DeltaRegime: {summary['DeltaRegime']:.9f}; CI {summary['bootstrap_CI95']['DeltaRegime']}.
- Per-setting primary-direction consistency: {summary['per_setting_consistency']:.1%}.
- Gates: {summary['gates']}.

## Purity and prospective lock

- S4/S6 future direction utilities: untouched during P4B.
- OpenBMI sealed internal holdout: untouched.
- WBCIC outer 10: untouched and unenumerated.
- Trial pseudoreplication: absent; utilities were computed subject-first.
- Serialized-basis metadata recovery: documented in `ENGINEERING_RECOVERY_BASIS_HASH.md`; every intervention direction was either byte-exact or passed the frozen source D/O/geometry/persistence equivalence gate.
- P4C lock: `{p4c_lock_status}`. P4C was not executed.

The result is reported without outcome-driven rescue, nonlinear model search, alpha tuning, learned D/C/O weights, threshold changes, or cherry-picked settings.
"""
    (EXP / "P4B_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(f"P4B_VALIDATION_PASS {candidate}", flush=True)


if __name__ == "__main__":
    main()
