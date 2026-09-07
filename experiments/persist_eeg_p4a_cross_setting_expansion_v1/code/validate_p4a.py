from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import common


REQUIRED_REPORTS = (
    "README.md",
    "P4A_PROTOCOL_FROZEN.json",
    "SETTING_MANIFEST.json",
    "SETTING_SOURCE_MANIFEST.json",
    "DATA_SCOPE_AUDIT.md",
    "OPENBMI_ERP_AVAILABILITY_AUDIT.md",
    "CROSS_TASK_PROTOCOL_AUDIT.md",
    "PREPROCESSING_AUDIT.md",
    "PREPROCESSING_PROTOCOL_ERP_FROZEN.json",
    "BACKBONE_PORT_AUDIT.md",
    "TRAINING_LEDGER.md",
    "SOURCE_IDENTITY_AUDIT.md",
    "PERSISTENCE_AUDIT.md",
    "DECISION_DEPENDENCE_AUDIT.md",
    "SOURCE_CONSEQUENCE_AUDIT.md",
    "TASK_SUBSPACE_OVERLAP_AUDIT.md",
    "SETTING_COMPETENCE_REPORT.md",
    "HOLDOUT_PURITY_AUDIT.md",
    "OUTCOME_ACCESS_LEDGER.md",
    "ENGINEERING_REPAIR_LOG.md",
    "P4A_FINAL_REPORT.md",
)


def main() -> None:
    issues: list[str] = []
    for name in REQUIRED_REPORTS:
        path = common.EXP / name
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"missing_or_empty_report:{name}")

    model_path = common.RESULTS / "model_setting_cube.csv"
    evidence_path = common.RESULTS / "source_evidence_cube.csv"
    controls_path = common.RESULTS / "matched_geometry_controls.csv"
    competence_path = common.RESULTS / "setting_competence.csv"
    for path in (model_path, evidence_path, controls_path, competence_path):
        if not path.is_file():
            issues.append(f"missing_result:{path.name}")
    if issues:
        common.write_json(common.RESULTS / "P4A_FINAL_VALIDATION.json", {"pass": False, "issues": issues})
        raise RuntimeError("P4A validation failed before table checks")

    model = pd.read_csv(model_path)
    evidence = pd.read_csv(evidence_path)
    controls = pd.read_csv(controls_path)
    competence = pd.read_csv(competence_path)
    exact_settings = set(common.SETTINGS)
    if set(model.setting_id) != exact_settings or set(evidence.setting_id) != exact_settings or set(competence.setting_id) != exact_settings:
        issues.append("exact_setting_set_failure")
    if len(model) != 900:
        issues.append(f"model_row_count:{len(model)}")
    if len(evidence) != 720:
        issues.append(f"evidence_row_count:{len(evidence)}")
    if len(controls) != 72000:
        issues.append(f"control_row_count:{len(controls)}")

    expected_grid = {(method, float(lam)) for method, lam in common.METHOD_GRID}
    for setting in common.SETTINGS:
        for fold in common.FOLDS:
            for seed in common.SEEDS:
                cell = model[(model.setting_id == setting) & (model.fold == fold) & (model.seed == seed)]
                observed = {(str(row.method), float(row["lambda"])) for _, row in cell.iterrows()}
                if observed != expected_grid:
                    issues.append(f"configuration_grid:{setting}:{fold}:{seed}")
                direction_cell = evidence[(evidence.setting_id == setting) & (evidence.fold == fold) & (evidence.seed == seed)]
                if set(direction_cell.direction_rank.astype(int)) != set(range(1, 9)) or len(direction_cell) != 8:
                    issues.append(f"direction_grid:{setting}:{fold}:{seed}")
                control_cell = controls[(controls.setting_id == setting) & (controls.fold == fold) & (controls.seed == seed)]
                if len(control_cell) != 800:
                    issues.append(f"control_grid:{setting}:{fold}:{seed}")

    required_model = {
        "setting_id", "dataset", "task", "backbone", "fold", "seed", "method", "lambda",
        "source_identity", "source_identity_raw_accuracy", "source_identity_chance_normalized_accuracy",
        "source_identity_chance_accuracy", "source_validation_BA", "source_validation_F1", "checkpoint_sha256",
        "training_epoch", "selection_metric", "outcome_status",
    }
    required_evidence = {
        "setting_id", "dataset", "task", "backbone", "fold", "seed", "representation_dim",
        "identity_full", "identity_chance_accuracy", "identity_direction_effect", "persistence", "geometry_strength", "direction_rank",
        "D_finite", "C_src_CE", "C_src_BA", "C_src_F1", "O_task", "direction_sha256",
        "checkpoint_sha256", "normalizer_sha256", "persistence_basis_sha256", "source_scope_hash", "validation_scope_hash",
    }
    if not required_model.issubset(model.columns):
        issues.append(f"model_schema_missing:{sorted(required_model - set(model.columns))}")
    if not required_evidence.issubset(evidence.columns):
        issues.append(f"evidence_schema_missing:{sorted(required_evidence - set(evidence.columns))}")
    banned_columns = [column for column in list(model.columns) + list(evidence.columns) if re.search(r"(^U_future$|future_utility|BA_erased_future|delta_G)", column, re.I)]
    if banned_columns:
        issues.append(f"future_utility_column_present:{banned_columns}")
    hash_columns = [column for column in evidence.columns if column.endswith("sha256") or column.endswith("scope_hash")]
    for column in hash_columns:
        if evidence[column].isna().any() or not evidence[column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            issues.append(f"invalid_hash_column:{column}")
    if not ((evidence.O_task >= -1e-8) & (evidence.O_task <= 1.0 + 1e-8)).all():
        issues.append("O_task_out_of_range")

    new = model[model.setting_id.isin(["S4", "S5", "S6"])]
    if set(new.outcome_status) != {"P4B_DIRECTION_UTILITY_SEALED"}:
        issues.append("new_outcome_status_failure")
    new_invariance = new[new.method != "ERM"]
    if new_invariance.ERM_outcome_competence_BA.notna().any() or new_invariance.ERM_outcome_competence_F1.notna().any():
        issues.append("new_invariance_outcome_was_accessed")
    historical = model[model.setting_id.isin(["S1", "S2", "S3"])]
    if set(historical.outcome_status) != {"HISTORICALLY_OBSERVED"}:
        issues.append("historical_outcome_status_failure")

    recomputed = []
    erm = model[model.method == "ERM"]
    for setting in common.SETTINGS:
        cell = erm[erm.setting_id == setting]
        mean = float(cell.ERM_outcome_competence_BA.mean())
        folds_above = int((cell.groupby("fold").ERM_outcome_competence_BA.mean() > 0.5).sum())
        status = "PASS" if mean > 0.60 and folds_above >= 4 else "FAIL"
        observed = competence.set_index("setting_id").loc[setting]
        if not np.isclose(mean, float(observed.outcome_BA_mean), atol=1e-12) or folds_above != int(observed.folds_above_chance) or status != observed.competence:
            issues.append(f"competence_recalculation:{setting}")
        recomputed.append({"setting_id": setting, "mean_BA": mean, "folds_above_chance": folds_above, "status": status})

    freeze = common.read_json(common.EXP / "PROTOCOL_FREEZE_COMMIT.json")
    if freeze.get("pass") is not True or freeze.get("protocol_sha256") != common.file_sha256(common.PROTOCOL_PATH):
        issues.append("protocol_freeze_record_failure")
    preflight = common.read_json(common.RUNTIME / "PREFLIGHT.json")
    if preflight.get("OpenBMI_sealed_14_accessed") is not False or preflight.get("WBCIC_sealed_10_accessed") is not False:
        issues.append("sealed_purity_failure")

    figures = sorted(common.FIGURES.glob("figure*.png"))
    if len(figures) < 6 or any(path.stat().st_size < 1000 for path in figures):
        issues.append("required_figure_failure")
    bootstrap = common.read_json(common.RESULTS / "SOURCE_STATISTICS_BOOTSTRAP.json")
    if bootstrap.get("bootstrap_draws") != 10000 or set(bootstrap.get("settings", {})) != exact_settings:
        issues.append("bootstrap_audit_failure")
    identity_scale = pd.read_csv(common.RESULTS / "identity_scale_diagnostics.csv")
    if len(identity_scale) != 90 or identity_scale[["I_ERM", "chance_accuracy", "max_observed_direction_reduction", "relative_suppression_denominator"]].isna().any().any():
        issues.append("identity_scale_diagnostics_failure")

    all_new_pass = all(row["status"] == "PASS" for row in recomputed if row["setting_id"] in {"S4", "S5", "S6"})
    terminal = "P4A_CROSS_SETTING_CUBE_COMPLETE" if all_new_pass else "P4A_REPRESENTATION_COMPETENCE_FAILURE"
    report_text = (common.EXP / "P4A_FINAL_REPORT.md").read_text(encoding="utf-8")
    if terminal not in report_text:
        issues.append("terminal_report_mismatch")

    payload = {
        "pass": not issues,
        "issues": issues,
        "terminal": terminal if not issues else "P4A_PROTOCOL_OR_PURITY_FAILURE",
        "settings": sorted(exact_settings),
        "model_rows": len(model),
        "evidence_rows": len(evidence),
        "control_rows": len(controls),
        "competence_recomputed": recomputed,
        "required_reports": len(REQUIRED_REPORTS),
        "figures": [path.name for path in figures],
        "protocol_freeze_commit": freeze.get("protocol_freeze_commit"),
        "OpenBMI_sealed_14_accessed": False,
        "WBCIC_sealed_10_accessed": False,
        "new_direction_future_utility_sealed": True,
        "new_invariance_outcome_delta_sealed": True,
    }
    common.write_json(common.RESULTS / "P4A_FINAL_VALIDATION.json", payload)
    if issues:
        raise RuntimeError(f"P4A validation failed: {issues}")
    print(f"P4A_VALIDATION_PASS {terminal}", flush=True)


if __name__ == "__main__":
    main()
