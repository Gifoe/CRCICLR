from __future__ import annotations

import pandas as pd

import common as c


FINAL_TERMINAL = "FINAL_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED"
NOT_RUN = "NOT_RUN_BY_STAGE0_GATE"


def main() -> None:
    issues: list[str] = []
    repair_validation = c.read_json(c.RESULTS / "STAGE0_REPAIR2_VALIDATION.json")
    repair_result = c.read_json(c.RESULTS / "STAGE0_REPAIR2_FINAL_RESULT.json")
    if repair_validation.get("pass") is not True:
        issues.append("repair2_validation")
    if repair_validation.get("all_setting_pass_count") != 2:
        issues.append("repair2_setting_pass_count")
    if repair_result.get("terminal") != "TRANSPORT_VALIDITY_NOT_SUPPORTED":
        issues.append("repair2_terminal")
    if repair_result.get("stage1_authorized") is not False:
        issues.append("stage1_authorization")
    if repair_result.get("transport_validity_supported") is not False:
        issues.append("false_transport_support")
    if repair_result.get("outer_or_future_performance_accessed") is not False:
        issues.append("future_or_outer_access")

    report_json_path = c.EXP / "SCST_DR_FINAL_REPORT.json"
    report_md_path = c.EXP / "SCST_DR_FINAL_REPORT.md"
    if not report_json_path.is_file() or not report_md_path.is_file():
        issues.append("missing_final_report")
        report = {}
    else:
        report = c.read_json(report_json_path)
        if report.get("final_terminal") != FINAL_TERMINAL:
            issues.append("final_terminal")
        if report.get("stage0_repair2_terminal") != "TRANSPORT_VALIDITY_NOT_SUPPORTED":
            issues.append("report_repair2_terminal")
        if report.get("repair2_setting_pass_count") != 2:
            issues.append("report_repair2_setting_pass_count")
        if report.get("stage1_status") != NOT_RUN:
            issues.append("stage1_status")
        if report.get("outer_status") != "UNTOUCHED_UNENUMERATED_NOT_AUTHORIZED":
            issues.append("outer_status")
        if report.get("final_constructive_protocol_lock_created") is not False:
            issues.append("false_final_lock_claim")

    required_not_run = {
        "DEVELOPMENT_MAIN_RESULTS.csv",
        "PER_SUBJECT_RESULTS.csv",
        "MECHANISM_RESULTS.csv",
        "BASELINE_COMPARISON.csv",
    }
    for name in required_not_run:
        path = c.RESULTS / name
        if not path.is_file():
            issues.append(f"missing:{name}")
            continue
        frame = pd.read_csv(path)
        if len(frame) != 1 or set(frame.status.astype(str)) != {NOT_RUN}:
            issues.append(f"not_run_status:{name}")
        numeric_claim_columns = [
            column
            for column in frame.columns
            if column in {
                "balanced_accuracy",
                "macro_f1",
                "nll",
                "identity_evidence",
                "transport_decision_sensitivity",
                "scst_minus_baseline_ba",
            }
        ]
        if numeric_claim_columns and frame[numeric_claim_columns].notna().any().any():
            issues.append(f"fabricated_numeric_result:{name}")

    statistics_path = c.RESULTS / "STATISTICAL_TESTS.json"
    if not statistics_path.is_file() or c.read_json(statistics_path).get("status") != NOT_RUN:
        issues.append("statistical_tests_status")
    for name in ("README.md", "CLAIM_AUDIT.md", "REPRODUCIBILITY.md", "STAGE0_REPAIR2_REPORT.md"):
        if not (c.EXP / name).is_file():
            issues.append(f"missing:{name}")

    if (c.EXP / "protocol" / "FINAL_CONSTRUCTIVE_PROTOCOL_LOCK.json").exists():
        issues.append("forbidden_final_constructive_lock")
    if (c.EXP / "protocol" / "SCST_DEVELOPMENT_PROTOCOL_LOCK.json").exists():
        issues.append("forbidden_scst_development_lock")
    for name in ("OUTER_RESULTS.csv", "SCST_OUTER_RESULTS.csv", "SCST_OUTER_PER_SUBJECT.csv"):
        if (c.RESULTS / name).exists():
            issues.append(f"forbidden_outer_results:{name}")

    sealed = c.read_json(c.EXP / "protocol" / "SEALED_RESOURCE_AUDIT.json")
    if sealed.get("pass") is not True or sealed.get("outer_evaluation_authorized") is not False:
        issues.append("sealed_resource_audit")

    validation = {
        "schema": "SCST_DR_FINAL_CLOSURE_VALIDATION_V2",
        "pass": len(issues) == 0,
        "issues": issues,
        "final_terminal": report.get("final_terminal") if report else None,
        "stage0_repair2_terminal": repair_result.get("terminal"),
        "repair2_setting_pass_count": repair_validation.get("all_setting_pass_count"),
        "stage1_run": False,
        "outer_opened": False,
        "OpenBMI_sealed_internal_holdout": "UNTOUCHED_UNENUMERATED",
        "WBCIC_outer_10": "UNTOUCHED_UNENUMERATED",
    }
    c.write_json(c.RESULTS / "FINAL_CLOSURE_VALIDATION.json", validation)
    if issues:
        raise RuntimeError("Final closure validation failed: " + "; ".join(issues))
    print(f"SCST_DR_FINAL_CLOSURE_VALIDATION_PASS terminal={FINAL_TERMINAL}", flush=True)


if __name__ == "__main__":
    main()
