from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import common as c


REQUIRED_ROOT = (
    "README.md",
    "PROTOCOL.md",
    "FEATURE_DEFINITIONS.md",
    "ITERATION_LEDGER.md",
    "MECHANISM_AUDIT.md",
    "POLICY_AUDIT.md",
    "CLAIM_AUDIT.md",
    "REPRODUCIBILITY.md",
    "RELIABILITY_STAGE05_FINAL_REPORT.md",
    "RELIABILITY_STAGE05_FINAL_REPORT.json",
)

REQUIRED_RESULTS = (
    "PER_SUBJECT_FEATURES.csv",
    "PER_SUBJECT_RELIABILITY_OUTCOMES.csv",
    "UNIVARIATE_MECHANISMS.csv",
    "CROSS_VALIDATED_RELIABILITY.csv",
    "BACKBONE_MECHANISM_DECOMPOSITION.csv",
    "RELIABILITY_POLICY_RESULTS.csv",
    "PER_SUBJECT_POLICY.csv",
    "STATISTICAL_TESTS.json",
)

FIGURES = (
    "figure1_backbone_certificate_reliability.png",
    "figure2_stability_vs_signed_persistence.png",
    "figure3_identity_vs_mechanism_predictors.png",
    "figure4_policy_risk_coverage.png",
    "figure5_per_subject_policy_outcome.png",
)


def check(condition: bool, name: str, checks: list[dict]) -> None:
    checks.append({"check": name, "pass": bool(condition)})


def main() -> None:
    checks: list[dict] = []
    lock = c.verify_feature_lock(require_committed=True)
    execution = c.read_json(c.RUNTIME / "FEATURE_EXTRACTION_EXECUTION.json")
    final = c.read_json(c.EXP / "RELIABILITY_STAGE05_FINAL_REPORT.json")
    statistics = c.read_json(c.RESULTS / "STATISTICAL_TESTS.json")

    check(all((c.EXP / name).is_file() for name in REQUIRED_ROOT), "required_reports_present", checks)
    check(all((c.RESULTS / name).is_file() for name in REQUIRED_RESULTS), "required_results_present", checks)
    check(all((c.FIGURES / name).is_file() for name in FIGURES), "five_required_figures_present", checks)
    check((c.PROTOCOL / "DATA_ACCESS_LOCK.json").is_file(), "data_access_lock_present", checks)
    check((c.PROTOCOL / "RELIABILITY_FEATURE_PROTOCOL_LOCK.json").is_file(), "feature_protocol_lock_present", checks)
    check(lock["stage0_validated_tip"] == "46b8ecf2c39b0e32045cad9d78ca12327f0a3f0d", "stage0_tip_preserved", checks)
    check(lock["stage0_terminal_preserved"] == "TARGET_HISTORY_UTILITY_TRANSFER_PARTIAL", "stage0_terminal_preserved", checks)
    check(lock["stage0_authorization_preserved"] == "SCAA_DEVELOPMENT_NOT_AUTHORIZED", "stage0_non_authorization_preserved", checks)
    check(lock["outer_10"]["identifiers_present"] is False, "outer_identifiers_absent", checks)
    check(lock["OpenBMI"] == "NOT_ACCESSED", "openbmi_not_accessed", checks)
    check(execution.get("S3_signal_rows_read") == 0, "feature_extraction_read_no_S3_signal", checks)
    check(execution.get("signal_sessions_read") == [0, 1], "feature_sessions_exactly_S1_S2", checks)

    features = pd.read_csv(c.RESULTS / "PER_SUBJECT_FEATURES.csv", dtype={"subject_id": str})
    outcomes = pd.read_csv(c.RESULTS / "PER_SUBJECT_RELIABILITY_OUTCOMES.csv", dtype={"subject_id": str})
    cv = pd.read_csv(c.RESULTS / "CROSS_VALIDATED_RELIABILITY.csv")
    policy = pd.read_csv(c.RESULTS / "PER_SUBJECT_POLICY.csv", dtype={"subject_id": str})
    policy_results = pd.read_csv(c.RESULTS / "RELIABILITY_POLICY_RESULTS.csv")
    check(len(features) == 82 and features.subject_id.nunique() == 41, "paired_feature_rows_complete", checks)
    check(features.groupby("subject_id").backbone.nunique().eq(2).all(), "both_backbones_per_subject", checks)
    check(features.seed_count.eq(3).all(), "three_seed_aggregation", checks)
    check(features.feature_input_sessions.eq("S1_validation,S2").all(), "feature_input_label_S1_S2", checks)
    check(features.s3_feature_input_accessed.astype(str).str.lower().eq("false").all(), "feature_S3_flag_false", checks)
    check(features.identity_status.eq("UNAVAILABLE_NO_LEGAL_TARGET_LEVEL_FROZEN_SCORE").all(), "identity_not_fabricated", checks)
    check(len(outcomes) == 82 and outcomes.subject_id.nunique() == 41, "outcome_rows_complete", checks)
    check(set(outcomes.R_sign.unique()).issubset({0, 1}), "R_sign_binary", checks)
    check(set(outcomes.H.unique()).issubset({0, 1}), "H_binary", checks)
    check(len(cv) == 3 * 9, "M0_to_M8_all_outcomes", checks)
    identity = cv[cv.model == "M4"]
    check(identity.status.eq("UNAVAILABLE_NO_LEGAL_TARGET_LEVEL_IDENTITY").all(), "M4_unavailable_explicit", checks)
    check(identity.AUROC.isna().all(), "M4_has_no_fabricated_metric", checks)
    available = cv[cv.model != "M4"]
    check(available.AUROC.between(0, 1).all(), "available_AUROC_finite", checks)
    check(available.Brier.between(0, 1).all(), "available_Brier_finite", checks)
    check(policy.groupby("subject_id").backbone.nunique().eq(2).all(), "policy_subject_pairing", checks)
    check(set(policy_results.policy) == {"Anchor", "Always Adapt", "Simple S2 Gate", "Reliability-Gated S2"}, "policy_comparators_complete", checks)
    check(policy_results.coverage.between(0, 1).all(), "policy_coverage_valid", checks)
    check(policy_results.mean_S3_BA.between(0, 1).all(), "policy_BA_valid", checks)

    allowed_terminals = {
        "RELIABILITY_MECHANISM_SUPPORTED",
        "RELIABILITY_MECHANISM_PARTIAL",
        "RELIABILITY_MECHANISM_NOT_SUPPORTED",
    }
    check(final["terminal"] in allowed_terminals, "terminal_allowed", checks)
    check(statistics["terminal"] == final["terminal"], "terminal_consistent", checks)
    check(statistics["gates"] == final["gates"], "gates_consistent", checks)
    if final["terminal"] == "RELIABILITY_MECHANISM_SUPPORTED":
        check(all(final["gates"].values()), "supported_requires_all_gates", checks)
        check(final["authorization"] == "RELIABILITY_GATED_SCAA_DEVELOPMENT_AUTHORIZED", "supported_authorization", checks)
    else:
        check(final["authorization"] == "RELIABILITY_GATED_SCAA_DEVELOPMENT_NOT_AUTHORIZED", "non_supported_not_authorized", checks)
    check(final["outer_10_untouched_unenumerated"] is True, "final_outer_status", checks)
    check(final["OpenBMI_accessed"] is False, "final_openbmi_status", checks)
    check(final["feature_definitions_changed_after_S3_association"] is False, "no_post_outcome_feature_change", checks)

    passed = all(item["pass"] for item in checks)
    validation = {
        "schema": "PERSIST_EEG_SCAA_RELIABILITY_STAGE05_VALIDATION_V1",
        "pass": passed,
        "checks": checks,
        "terminal": final["terminal"],
        "authorization": final["authorization"],
        "outer_10": "UNTOUCHED_UNENUMERATED_UNPREPROCESSED_UNEVALUATED",
        "OpenBMI": "NOT_ACCESSED",
    }
    c.write_json(c.RESULTS / "VALIDATION.json", validation)
    if not passed:
        failed = [item["check"] for item in checks if not item["pass"]]
        raise RuntimeError(f"Stage-0.5 validation failed: {failed}")
    print("SCAA_RELIABILITY_STAGE05_VALIDATION_PASS")


if __name__ == "__main__":
    main()

