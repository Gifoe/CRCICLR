from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import common as c


REQUIRED = (
    "README.md",
    "PROTOCOL.md",
    "DATA_AUDIT.md",
    "ADAPTATION_COMPETENCE_AUDIT.md",
    "ADAPTATION_ITERATION_LEDGER.md",
    "CLAIM_AUDIT.md",
    "REPRODUCIBILITY.md",
    "SCAA_STAGE0_FINAL_REPORT.md",
    "SCAA_STAGE0_FINAL_REPORT.json",
    "protocol/DATA_ACCESS_LOCK.json",
    "protocol/ADAPTATION_RECIPE_SELECTION.json",
    "protocol/SCAA_STAGE0_PROTOCOL_LOCK.json",
    "results/ADAPTATION_COMPETENCE_GRID.csv",
    "results/PER_SUBJECT_UTILITY.csv",
    "results/PER_SUBJECT_SEED_UTILITY.csv",
    "results/BACKBONE_SUMMARY.csv",
    "results/UTILITY_TRANSFER_CORRELATION.csv",
    "results/SIGN_CONCORDANCE.csv",
    "results/HARM_AND_COVERAGE.csv",
    "results/POLICY_COMPARISON.csv",
    "results/SECONDARY_LCB_ANALYSIS.csv",
    "results/CONTROL_DIAGNOSTICS.csv",
    "results/STATISTICAL_TESTS.json",
    "figures/utility_transfer_scatter.png",
    "figures/utility_transfer_scatter.pdf",
    "figures/policy_comparison.png",
    "figures/policy_comparison.pdf",
    "figures/harm_coverage.png",
    "figures/harm_coverage.pdf",
    "figures/per_subject_transfer.png",
    "figures/per_subject_transfer.pdf",
)


def main() -> None:
    errors: list[str] = []
    for relative in REQUIRED:
        path = c.EXP / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing_or_empty:{relative}")

    try:
        lock = c.read_json(c.PROTOCOL / "SCAA_STAGE0_PROTOCOL_LOCK.json")
        data_lock = c.read_json(c.PROTOCOL / "DATA_ACCESS_LOCK.json")
        recipe = c.read_json(c.PROTOCOL / "ADAPTATION_RECIPE_SELECTION.json")
        final = c.read_json(c.EXP / "SCAA_STAGE0_FINAL_REPORT.json")
        tests = c.read_json(c.RESULTS / "STATISTICAL_TESTS.json")
    except Exception as exc:
        errors.append(f"json_read:{exc}")
        lock = data_lock = recipe = final = tests = {}

    if lock:
        for relative, expected in lock.get("code_hashes", {}).items():
            path = c.EXP / relative
            if not path.is_file() or c.sha256(path) != expected:
                errors.append(f"frozen_code_hash:{relative}")
        if lock.get("development_subject_count") != 41 or len(lock.get("development_subjects", [])) != 41:
            errors.append("protocol_subject_count")
        outer = lock.get("sealed_outer", {})
        if any(outer.get(key) is not False for key in ("identifiers_present", "enumerated", "accessed", "preprocessed", "evaluated")):
            errors.append("outer_lock_impure")
        if lock.get("primary_certificate") != "Delta_S2 > 0":
            errors.append("certificate_changed")
        if lock.get("aggregation", {}).get("statistical_unit") != "subject":
            errors.append("wrong_statistical_unit")
        if lock.get("statistics", {}).get("subject_bootstrap_resamples") != 10000:
            errors.append("wrong_bootstrap_count")

    if data_lock.get("pass") is not True or data_lock.get("S2_or_S3_adaptation_utility_inspected") is not False:
        errors.append("data_lock")
    if recipe.get("competence_gate_pass") is not True or recipe.get("adapter") != "classifier_head_only_supervised":
        errors.append("recipe")
    if recipe.get("selected_lr") != 0.001 or recipe.get("S2_or_S3_utility_accessed") is not False:
        errors.append("recipe_freeze")

    try:
        seed = pd.read_csv(c.RESULTS / "PER_SUBJECT_SEED_UTILITY.csv", dtype={"subject_id": str})
        subject = pd.read_csv(c.RESULTS / "PER_SUBJECT_UTILITY.csv", dtype={"subject_id": str})
        corr = pd.read_csv(c.RESULTS / "UTILITY_TRANSFER_CORRELATION.csv")
        sign = pd.read_csv(c.RESULTS / "SIGN_CONCORDANCE.csv")
        harm = pd.read_csv(c.RESULTS / "HARM_AND_COVERAGE.csv")
        policy = pd.read_csv(c.RESULTS / "POLICY_COMPARISON.csv")
        if len(seed) != 246 or seed.groupby(["backbone", "subject_id"]).seed.nunique().ne(3).any():
            errors.append("seed_grid")
        if seed.target_seen_by_anchor.astype(bool).any() or seed.S2_or_S3_used_for_training_or_selection.astype(bool).any():
            errors.append("target_or_outcome_leakage")
        expected_subjects = set(lock.get("development_subjects", []))
        if set(seed.subject_id) != expected_subjects:
            errors.append("seed_subject_set")
        if len(subject) != 123 or set(subject.scope) != {"EEGNet", "EEGConformer", "Pooled"}:
            errors.append("subject_aggregation")
        if subject.groupby("scope").size().to_dict() != {"EEGConformer": 41, "EEGNet": 41, "Pooled": 41}:
            errors.append("scope_counts")
        if len(corr) != 6 or set(corr.method) != {"pearson", "spearman"}:
            errors.append("correlation_table")
        if len(sign) != 3 or len(harm) != 3 or len(policy) != 3:
            errors.append("summary_tables")
        finite_columns = ["Delta_S2_BA", "Delta_S3_BA", "anchor_S3_BA", "adapted_S3_BA", "LCB_S2_90"]
        if not np.isfinite(subject[finite_columns].to_numpy(float)).all():
            errors.append("nonfinite_subject_values")
    except Exception as exc:
        errors.append(f"csv_validation:{exc}")

    terminals = {
        "TARGET_HISTORY_UTILITY_TRANSFER_SUPPORTED",
        "TARGET_HISTORY_UTILITY_TRANSFER_PARTIAL",
        "TARGET_HISTORY_UTILITY_TRANSFER_NOT_SUPPORTED",
    }
    terminal = final.get("terminal")
    authorization = final.get("authorization")
    if terminal not in terminals:
        errors.append("terminal")
    if (terminal == "TARGET_HISTORY_UTILITY_TRANSFER_SUPPORTED") != (authorization == "SCAA_DEVELOPMENT_AUTHORIZED"):
        errors.append("authorization_logic")
    if tests.get("terminal") != terminal or tests.get("authorization") != authorization:
        errors.append("decision_mismatch")
    if tests.get("statistical_unit") != "subject" or tests.get("bootstrap_resamples") != 10000:
        errors.append("statistics_metadata")
    if final.get("outer_10_untouched_unenumerated") is not True or final.get("target_never_seen_by_anchor") is not True:
        errors.append("purity_claim")

    report_text = (c.EXP / "SCAA_STAGE0_FINAL_REPORT.md").read_text(encoding="utf-8") if (c.EXP / "SCAA_STAGE0_FINAL_REPORT.md").is_file() else ""
    for number in range(1, 32):
        if f"{number}." not in report_text:
            errors.append(f"report_answer_{number}")
    if "SCAA improves generalization" in final.get("strongest_supported_claim", ""):
        errors.append("overclaim")

    tracked_runtime = subprocess.run(
        ["git", "ls-files", str((c.RUNTIME.relative_to(c.REPO)).as_posix())],
        cwd=c.REPO,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if tracked_runtime:
        errors.append("runtime_tracked")
    for extension in ("*.edf", "*.set", "*.mat", "*.npy", "*.npz", "*.pt", "*.pth", "*.ckpt"):
        if any(path.is_file() for path in c.EXP.glob(extension)):
            errors.append(f"raw_or_large_artifact:{extension}")

    validation = {
        "schema": "PERSIST_EEG_SCAA_STAGE0_VALIDATION_V1",
        "pass": not errors,
        "errors": errors,
        "branch": "codex/persist-eeg-scaa-stage0",
        "terminal": terminal,
        "authorization": authorization,
        "development_subjects": 41,
        "outer_10_untouched_unenumerated": True,
        "seed_level_rows": 246,
        "subject_level_rows": 123,
        "required_files": len(REQUIRED),
        "protocol_lock_sha256": c.sha256(c.PROTOCOL / "SCAA_STAGE0_PROTOCOL_LOCK.json") if (c.PROTOCOL / "SCAA_STAGE0_PROTOCOL_LOCK.json").is_file() else None,
    }
    c.write_json(c.RESULTS / "VALIDATION.json", validation)
    if errors:
        raise RuntimeError("SCAA Stage-0 validation failed: " + ", ".join(errors))
    print(f"SCAA_STAGE0_VALIDATION_PASS terminal={terminal}", flush=True)


if __name__ == "__main__":
    main()
