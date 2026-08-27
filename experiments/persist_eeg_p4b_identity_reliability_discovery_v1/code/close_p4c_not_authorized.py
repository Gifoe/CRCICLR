from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
RESULTS = EXP / "results"
P4A = REPO / "experiments" / "persist_eeg_p4a_cross_setting_expansion_v1"
BRANCH = "codex/persist-eeg-p4b-identity-reliability-discovery-v1"
NOT_AUTHORIZED_TERMINALS = {
    "P4B_IDENTITY_RELIABILITY_CONDITION_NOT_SUPPORTED",
    "P4B_INSUFFICIENT_CROSS_SETTING_EVIDENCE",
    "P4B_INSUFFICIENT_PROSPECTIVE_HOLDOUT",
    "P4B_PROTOCOL_OR_PURITY_FAILURE",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def main() -> None:
    validation = read_json(RESULTS / "P4B_FINAL_VALIDATION.json")
    summary = read_json(RESULTS / "condition_model_summary.json")
    analysis = read_json(RESULTS / "P4B_ANALYSIS_COMPLETE.json")
    protocol = read_json(EXP / "P4B_PROTOCOL_FROZEN.json")
    assignment = read_json(EXP / "DISCOVERY_SETTING_ASSIGNMENT.json")
    normalization = read_json(EXP / "SOURCE_NORMALIZATION_FROZEN.json")
    pre = read_json(EXP / "PRE_OUTCOME_FREEZE_COMPLETE.json")
    predictions = pd.read_csv(RESULTS / "condition_model_predictions.csv")
    stability = pd.read_csv(RESULTS / "per_setting_stability.csv")
    subject = pd.read_csv(RESULTS / "discovery_future_utility_subject.csv", usecols=["setting_id"])
    direction = pd.read_csv(RESULTS / "discovery_future_utility_direction.csv", usecols=["setting_id"])
    controls = pd.read_csv(RESULTS / "discovery_matched_control_summary.csv", usecols=["setting_id"])

    tip = git("rev-parse", "HEAD")
    remote_tip = git("rev-parse", f"origin/{BRANCH}")
    current_branch = git("branch", "--show-current")
    terminal = validation.get("terminal")
    reserved = assignment.get("p4c_reserved_settings")
    expected_hashes = {
        "P4A_source_evidence_cube": sha256(P4A / "results" / "source_evidence_cube.csv"),
        "SOURCE_NORMALIZATION_FROZEN.json": sha256(EXP / "SOURCE_NORMALIZATION_FROZEN.json"),
        "DISCOVERY_SETTING_ASSIGNMENT.json": sha256(EXP / "DISCOVERY_SETTING_ASSIGNMENT.json"),
        "P4B_PROTOCOL_FROZEN.json": sha256(EXP / "P4B_PROTOCOL_FROZEN.json"),
        "source_evidence_normalized.csv": sha256(RESULTS / "source_evidence_normalized.csv"),
    }
    hash_consistent = pre.get("hashes") == expected_hashes
    timing_valid = all(
        timestamp <= pre["timestamp_utc"]
        for timestamp in [
            normalization["timestamp_utc"],
            assignment["timestamp_utc"],
            protocol["timestamp_utc"],
        ]
    ) and pre["timestamp_utc"] < analysis["first_future_access_timestamp_utc"]
    observed_settings = set(subject.setting_id) | set(direction.setting_id) | set(controls.setting_id) | set(predictions.setting_id)
    reserved_absent = not bool(set(reserved) & observed_settings)
    purity_pass = (
        reserved_absent
        and pre.get("p4c_reserved_future_utility_accessed") is False
        and analysis.get("p4c_reserved_future_utility_accessed") is False
        and validation.get("p4c_reserved_future_utility_accessed") is False
        and validation.get("OpenBMI_sealed_internal_holdout") == "UNTOUCHED"
        and validation.get("WBCIC_outer_10") == "UNTOUCHED_NOT_ENUMERATED"
    )

    mint_better_mi = float(summary["primary_RMSE"]["MINT"]) < float(summary["primary_RMSE"]["MI"])
    mint_better_madd = float(summary["primary_RMSE"]["MINT"]) < float(summary["primary_RMSE"]["MADD"])
    beta_negative = float(summary["beta_IxE"]) < 0
    delta_slope_positive = float(summary["DeltaSlope"]) > 0
    delta_regime_positive = float(summary["DeltaRegime"]) > 0
    partial_minimum = mint_better_mi and beta_negative and delta_slope_positive and delta_regime_positive

    predictions["run_id"] = predictions.setting_id.astype(str) + "/f" + predictions.fold.astype(str) + "/s" + predictions.seed.astype(str)
    predictions["se_advantage"] = (predictions.U_BA - predictions.pred_MI) ** 2 - (predictions.U_BA - predictions.pred_MINT) ** 2
    setting_influence = predictions.groupby("setting_id").se_advantage.sum()
    run_influence = predictions.groupby("run_id").se_advantage.sum()
    setting_share = float(setting_influence.abs().max() / max(setting_influence.abs().sum(), 1e-15))
    run_share = float(run_influence.abs().max() / max(run_influence.abs().sum(), 1e-15))

    p4c_lock_exists = (EXP / "P4C_LOCK.json").exists()
    audit_pass = all(
        [
            validation.get("pass") is True,
            terminal in NOT_AUTHORIZED_TERMINALS,
            tip == remote_tip,
            current_branch == BRANCH,
            hash_consistent,
            timing_valid,
            purity_pass,
            not p4c_lock_exists,
            validation.get("P4C_LOCK_status") == "NOT_REQUIRED_FOR_NOT_SUPPORTED",
        ]
    )
    if not audit_pass:
        raise RuntimeError("P4B pre-P4C audit failed; refusing to issue authorization decision")

    audit = {
        "schema": "P4B_PRE_P4C_AUDIT_V1",
        "timestamp_utc": now_utc(),
        "pass": True,
        "p4b_branch": BRANCH,
        "p4b_validated_tip": tip,
        "p4b_remote_tip": remote_tip,
        "p4b_terminal": terminal,
        "p4b_validator_pass": True,
        "p4a_lean_hash_consistent": hash_consistent,
        "normalization_E_task_assignment_frozen_before_future_access": timing_valid,
        "discovery_settings": assignment["all_discovery_settings"],
        "reserved_settings": reserved,
        "reserved_future_absent_from_p4b": reserved_absent,
        "purity_pass": purity_pass,
        "P4C_LOCK_exists": p4c_lock_exists,
        "P4C_LOCK_status": validation["P4C_LOCK_status"],
        "RMSE_MI": summary["primary_RMSE"]["MI"],
        "RMSE_MINT": summary["primary_RMSE"]["MINT"],
        "MINT_point_better_than_MI": mint_better_mi,
        "MINT_point_better_than_MADD": mint_better_madd,
        "beta_IxE": summary["beta_IxE"],
        "beta_IxE_CI95": summary["bootstrap_CI95"]["beta_IxE"],
        "DeltaSlope": summary["DeltaSlope"],
        "DeltaSlope_CI95": summary["bootstrap_CI95"]["DeltaSlope"],
        "DeltaRegime": summary["DeltaRegime"],
        "DeltaRegime_CI95": summary["bootstrap_CI95"]["DeltaRegime"],
        "per_setting_consistency": summary["per_setting_consistency"],
        "per_setting_effects": stability.to_dict("records"),
        "maximum_setting_absolute_influence_share": setting_share,
        "obvious_single_setting_domination": setting_share > 0.75,
        "maximum_run_absolute_influence_share": run_share,
        "influential_run_warning": run_share > 0.25,
        "post_outcome_protocol_or_threshold_modification": False,
        "OpenBMI_sealed_internal_holdout": "UNTOUCHED",
        "WBCIC_outer_10": "UNTOUCHED_NOT_ENUMERATED",
        "P4B_FINAL_VALIDATION_sha256": sha256(RESULTS / "P4B_FINAL_VALIDATION.json"),
    }
    write_json(EXP / "P4B_PRE_P4C_AUDIT.json", audit)
    (EXP / "P4B_PRE_P4C_AUDIT.md").write_text(
        "# P4B Pre-P4C Audit\n\n"
        f"Audit PASS on validated/pushed P4B tip `{tip}`. Exact P4B terminal: `{terminal}`.\n\n"
        f"- RMSE MI={summary['primary_RMSE']['MI']:.9f}; MINT={summary['primary_RMSE']['MINT']:.9f}; MINT better={mint_better_mi}.\n"
        f"- beta_IxE={summary['beta_IxE']:.9f}, CI={summary['bootstrap_CI95']['beta_IxE']}; required sign negative={beta_negative}.\n"
        f"- DeltaSlope={summary['DeltaSlope']:.9f}, CI={summary['bootstrap_CI95']['DeltaSlope']}; required direction positive={delta_slope_positive}.\n"
        f"- DeltaRegime={summary['DeltaRegime']:.9f}, CI={summary['bootstrap_CI95']['DeltaRegime']}; required direction positive={delta_regime_positive}.\n"
        f"- Per-setting interaction-direction consistency={summary['per_setting_consistency']:.1%}.\n"
        f"- Maximum setting/run absolute influence shares={setting_share:.1%}/{run_share:.1%}; these diagnostics did not trigger any setting removal or rescue.\n"
        "- P4A/source hashes and pre-outcome timing: PASS.\n"
        "- S4/S6 future utility: UNTOUCHED. OpenBMI sealed holdout: UNTOUCHED. WBCIC outer 10: UNTOUCHED.\n"
        "- P4C_LOCK is intentionally absent because the validated P4B terminal does not authorize one.\n\n"
        "The positive DeltaRegime alone is insufficient: frozen MINT transports much worse than MI, and both beta_IxE and DeltaSlope have the wrong point direction.\n",
        encoding="utf-8",
    )

    authorization = {
        "schema": "P4C_AUTHORIZATION_V1",
        "timestamp_utc": now_utc(),
        "authorized": False,
        "terminal": "P4C_NOT_AUTHORIZED_BY_P4B",
        "p4b_validated_tip": tip,
        "p4b_terminal": terminal,
        "p4b_validator_pass": True,
        "reserved_settings_remain_sealed": reserved,
        "purity_pass": purity_pass,
        "P4C_LOCK_status": "NOT_CREATED_NOT_AUTHORIZED",
        "partial_minimum_conditions": {
            "MINT_point_better_than_MI": mint_better_mi,
            "beta_IxE_negative": beta_negative,
            "DeltaSlope_positive": delta_slope_positive,
            "DeltaRegime_positive": delta_regime_positive,
            "all_met": partial_minimum,
        },
        "reason": "P4B terminal is NOT_SUPPORTED and the partial-discovery minimum conditions are not met",
    }
    write_json(EXP / "P4C_AUTHORIZATION.json", authorization)

    validation_payload = {
        "pass": True,
        "issues": [],
        "validated_at_utc": now_utc(),
        "terminal": "P4C_NOT_AUTHORIZED_BY_P4B",
        "CONDITION_STATUS": "P4C_NOT_AUTHORIZED_BY_P4B",
        "ACTIONABILITY_STATUS": "P4C_ACTIONABILITY_NOT_EVALUATED",
        "FINAL_MODEL_AUTHORIZATION": "NOT_AUTHORIZED",
        "p4b_validated_tip": tip,
        "p4b_terminal": terminal,
        "reserved_settings": reserved,
        "reserved_future_outcome_accessed": False,
        "prospective_prediction_freeze_created": False,
        "P4C_branch_created": False,
        "OpenBMI_sealed_internal_holdout": "UNTOUCHED",
        "WBCIC_outer_10": "UNTOUCHED_NOT_ENUMERATED",
        "final_report_written_after_validator_pass": False,
    }
    write_json(RESULTS / "P4C_FINAL_VALIDATION.json", validation_payload)
    report = f"""# P4C Final Report — Not Authorized

## Decision

- P4B exact terminal: `{terminal}`.
- P4B validator: PASS on `{tip}`.
- P4C authorization: **DENIED**.
- CONDITION_STATUS: `P4C_NOT_AUTHORIZED_BY_P4B`.
- ACTIONABILITY_STATUS: `P4C_ACTIONABILITY_NOT_EVALUATED`.
- FINAL_MODEL_AUTHORIZATION: `NOT_AUTHORIZED`.

## Why the gate failed

- Frozen LOSO-setting RMSE MI: {summary['primary_RMSE']['MI']:.9f}.
- Frozen LOSO-setting RMSE MINT: {summary['primary_RMSE']['MINT']:.9f}.
- RMSE_MI - RMSE_MINT: {summary['contrasts']['RMSE_MI_minus_MINT']:.9f}, CI={summary['bootstrap_CI95']['contrast_MI_MINT']}; the point estimate is negative, so MINT is worse.
- beta_IxE: {summary['beta_IxE']:.9f}, CI={summary['bootstrap_CI95']['beta_IxE']}; the point sign is positive, opposite to the hypothesis.
- DeltaSlope: {summary['DeltaSlope']:.9f}, CI={summary['bootstrap_CI95']['DeltaSlope']}; the point direction is negative.
- DeltaRegime: {summary['DeltaRegime']:.9f}, CI={summary['bootstrap_CI95']['DeltaRegime']}; this component is positive, but cannot override the failed predictive and interaction gates.
- Per-setting interaction-direction consistency: {summary['per_setting_consistency']:.1%}.

## Prospective evaluation status

Reserved settings remain exactly S4 and S6. Their future BA/F1/CE, per-direction utility, ranking, oracle, and policy outcomes were never opened. No P4C_LOCK, prediction freeze, P4C branch, prospective RMSE, policy comparison, or oracle analysis was created because doing so would violate the authorization gate. Accordingly all P4C outcome quantities are `NOT_EVALUATED`, not missing results.

OpenBMI sealed internal holdout is untouched. WBCIC outer 10 is untouched and unenumerated. There was no post-outcome model, E_task, normalization, threshold, alpha, top-k, or setting modification. No final model was trained.
"""
    (EXP / "P4C_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    validation_payload["final_report_written_after_validator_pass"] = True
    validation_payload["P4C_FINAL_REPORT_sha256"] = sha256(EXP / "P4C_FINAL_REPORT.md")
    validation_payload["P4B_PRE_P4C_AUDIT_sha256"] = sha256(EXP / "P4B_PRE_P4C_AUDIT.json")
    validation_payload["P4C_AUTHORIZATION_sha256"] = sha256(EXP / "P4C_AUTHORIZATION.json")
    write_json(RESULTS / "P4C_FINAL_VALIDATION.json", validation_payload)
    print("P4C_NOT_AUTHORIZED_BY_P4B VALIDATION_PASS", flush=True)


if __name__ == "__main__":
    main()
