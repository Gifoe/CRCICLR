from __future__ import annotations

import json

import pandas as pd

import common as c


def main() -> None:
    issues: list[str] = []
    protocol = c.protocol()
    freeze_path = c.EXP / "protocol" / "PRE_STAGE0_FREEZE.json"
    if not freeze_path.is_file():
        raise RuntimeError("missing PRE_STAGE0_FREEZE.json")
    freeze = c.read_json(freeze_path)
    for relative, expected in freeze.get("file_sha256", {}).items():
        path = c.EXP / relative
        if not path.is_file() or c.sha256(path) != expected:
            issues.append(f"post_freeze_hash_changed:{relative}")

    data_audit = c.read_json(c.EXP / "protocol" / "DATA_ACCESS_AUDIT.json")
    sealed = c.read_json(c.EXP / "protocol" / "SEALED_RESOURCE_AUDIT.json")
    if data_audit.get("pass") is not True or data_audit.get("all_four_settings_have_complete_erm_units") is not True:
        issues.append("data_access_audit")
    if data_audit.get("openbmi", {}).get("development_subject_count") != 40:
        issues.append("openbmi_development_count")
    if data_audit.get("wbcic", {}).get("development_subject_count") != 41:
        issues.append("wbcic_development_count")
    if data_audit.get("openbmi", {}).get("sealed_internal_holdout_membership_materialized") is not False:
        issues.append("openbmi_sealed_membership")
    if data_audit.get("wbcic", {}).get("sealed_outer_membership_materialized") is not False:
        issues.append("wbcic_outer_membership")
    if sealed.get("pass") is not True or sealed.get("outer_evaluation_authorized") is not False:
        issues.append("sealed_audit")

    unit_paths = sorted((c.RUNTIME / "stage0_units").glob("*/fold-*/*/UNIT_COMPLETE.json"))
    if len(unit_paths) != 40:
        issues.append(f"unit_count:{len(unit_paths)}")
    for path in unit_paths:
        payload = c.read_json(path)
        if payload.get("pass") is not True:
            issues.append(f"unit_failed:{path}")
        if payload.get("outcome_rows_loaded") != 0 or payload.get("future_session_rows_loaded") != 0:
            issues.append(f"unit_scope_violation:{path}")

    required = {
        "TRANSPORT_STABILITY.csv": {"matched_subject_same_class", "mismatched_subject_same_class", "wrong_class", "subject_permutation", "norm_matched_random"},
        "SUBJECT_FIDELITY.csv": {"no_transport", "scst", "norm_matched_random", "unconditional_subject_transport", "wrong_class", "subject_permutation", "same_class_mixup"},
        "CLASS_FIDELITY.csv": {"no_transport", "scst", "norm_matched_random", "unconditional_subject_transport", "same_class_mixup"},
        "MANIFOLD_VALIDITY.csv": {"no_transport", "scst", "norm_matched_random", "unconditional_subject_transport", "wrong_class", "subject_permutation", "same_class_mixup"},
    }
    for name, methods in required.items():
        path = c.RESULTS / name
        if not path.is_file():
            issues.append(f"missing:{name}")
            continue
        frame = pd.read_csv(path)
        column = "control" if name == "TRANSPORT_STABILITY.csv" else "method"
        if len(frame) == 0 or set(frame[column].astype(str).unique()) != methods:
            issues.append(f"controls:{name}")
        if set(frame.setting_id.unique()) != set(c.SETTINGS):
            issues.append(f"settings:{name}")
        if set(frame.layer.unique()) != set(protocol["candidate_layers"]):
            issues.append(f"layers:{name}")

    summary_path = c.RESULTS / "STAGE0_LAYER_SUMMARY.csv"
    result_path = c.RESULTS / "STAGE0_FINAL_RESULT.json"
    if not summary_path.is_file() or not result_path.is_file():
        issues.append("missing_stage0_summary")
        summary = pd.DataFrame()
        result = {}
    else:
        summary = pd.read_csv(summary_path)
        result = c.read_json(result_path)
        if len(summary) != 8:
            issues.append(f"summary_rows:{len(summary)}")
        if set(summary.setting_id.unique()) != set(c.SETTINGS):
            issues.append("summary_settings")
        if set(summary.layer.unique()) != set(protocol["candidate_layers"]):
            issues.append("summary_layers")
        if result.get("terminal") not in protocol["authorized_stage0_terminals"]:
            issues.append("terminal")
        if result.get("outer_or_future_performance_accessed") is not False:
            issues.append("future_access")
        supported = result.get("terminal") == "TRANSPORT_VALIDITY_SUPPORTED"
        if bool(result.get("stage1_authorized")) != supported:
            issues.append("stage1_gate")
        if supported and (len(result.get("selected_layers", {})) != 4 or not summary.groupby("setting_id").all_gates_pass.any().all()):
            issues.append("false_positive_supported_terminal")

    for name in ("FIGURE_1_TRANSPORT_CONCEPT_VALIDATION.png", "FIGURE_2_SUBJECT_VS_CLASS_FIDELITY.png"):
        if not (c.FIGURES / name).is_file():
            issues.append(f"missing_figure:{name}")

    validation = {
        "schema": "SCST_DR_STAGE0_VALIDATION_V1",
        "pass": len(issues) == 0,
        "issues": issues,
        "stage0_terminal": result.get("terminal"),
        "stage1_authorized": result.get("terminal") == "TRANSPORT_VALIDITY_SUPPORTED" if result else False,
        "OpenBMI_sealed_internal_holdout": "UNTOUCHED_UNENUMERATED",
        "WBCIC_outer_10": "UNTOUCHED_UNENUMERATED",
        "future_session_performance_accessed": False,
        "unit_count": len(unit_paths),
        "layer_summary_rows": len(summary),
    }
    c.write_json(c.RESULTS / "STAGE0_VALIDATION.json", validation)
    if issues:
        raise RuntimeError("Stage-0 validation failed: " + "; ".join(issues))
    print(f"SCST_DR_STAGE0_VALIDATION_PASS terminal={result['terminal']}", flush=True)


if __name__ == "__main__":
    main()
