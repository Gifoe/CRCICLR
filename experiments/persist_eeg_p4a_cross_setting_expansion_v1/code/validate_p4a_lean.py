from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import common


REQUIRED = (
    "P4A_GRID_PAUSE_SNAPSHOT.md",
    "P4A_GRID_PAUSE_SNAPSHOT.json",
    "P4A_PROTOCOL_AMENDMENT_LEAN_V1.md",
    "P4A_PROTOCOL_AMENDMENT_LEAN_V1.json",
    "SETTING_COMPETENCE_REPORT_LEAN.md",
    "SETTING_MANIFEST_LEAN.json",
    "P4A_LEAN_FINAL_REPORT.md",
)


def main() -> None:
    issues: list[str] = []
    for name in REQUIRED:
        path = common.EXP / name
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"missing_or_empty:{name}")

    result_paths = {
        "model": common.RESULTS / "erm_setting_cube.csv",
        "evidence": common.RESULTS / "source_evidence_cube.csv",
        "audit": common.RESULTS / "source_artifact_audit.csv",
        "competence": common.RESULTS / "setting_competence_lean.csv",
        "aggregate": common.RESULTS / "P4A_LEAN_AGGREGATION_COMPLETE.json",
    }
    for name, path in result_paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"missing_result:{name}")
    if issues:
        common.write_json(
            common.RESULTS / "P4A_LEAN_FINAL_VALIDATION.json",
            {"pass": False, "terminal": "P4A_LEAN_PROTOCOL_OR_PURITY_FAILURE", "issues": issues},
        )
        raise RuntimeError("P4A Lean pre-table validation failed")

    snapshot = common.read_json(common.EXP / "P4A_GRID_PAUSE_SNAPSHOT.json")
    amendment = common.read_json(common.EXP / "P4A_PROTOCOL_AMENDMENT_LEAN_V1.json")
    manifest = common.read_json(common.EXP / "SETTING_MANIFEST_LEAN.json")
    preflight = common.read_json(common.RUNTIME / "PREFLIGHT.json")
    model = pd.read_csv(result_paths["model"])
    evidence = pd.read_csv(result_paths["evidence"])
    audit = pd.read_csv(result_paths["audit"])
    competence = pd.read_csv(result_paths["competence"])

    if snapshot.get("label") != "OPTIONAL_PARTIAL_INVARIANCE_GRID":
        issues.append("partial_grid_label")
    progress = snapshot.get("progress", {})
    if progress.get("erm_completed") != 45 or progress.get("erm_expected") != 45:
        issues.append("snapshot_erm_count")
    if progress.get("grid_completed") != 205 or progress.get("grid_expected") != 405:
        issues.append("snapshot_grid_count")
    if progress.get("grid_counts_by_setting") != {"S4": 135, "S5": 70, "S6": 0}:
        issues.append("snapshot_setting_counts")
    if len(snapshot.get("completed_configuration_ids", [])) != 250:
        issues.append("snapshot_completed_id_count")
    if len(snapshot.get("incomplete_grid_configuration_ids", [])) != 200:
        issues.append("snapshot_incomplete_id_count")
    if snapshot.get("pause_boundary", {}).get("current_atomic_job_allowed_to_finish") is not True:
        issues.append("non_graceful_pause")
    if snapshot.get("scheduler", {}).get("State") != "Ready":
        issues.append("snapshot_scheduler_not_stopped")
    if not all(bool(row.get("exists")) for row in snapshot.get("checkpoints", [])):
        issues.append("snapshot_checkpoint_missing")

    if amendment.get("amendment_type") != "COMPUTATIONAL_SCOPE_AMENDMENT" or amendment.get("scientific_outcome_amendment") is not False:
        issues.append("amendment_type")
    if amendment.get("original_protocol", {}).get("required_full_invariance_grid") is not True:
        issues.append("original_protocol_not_acknowledged")
    if amendment.get("mandatory_erm_complete_before_amendment") is not True:
        issues.append("amendment_before_erm")
    if amendment.get("amendment_before_p4b_future_direction_utility_discovery") is not True:
        issues.append("amendment_after_future_access")
    if amendment.get("pause_decision_used_grid_scientific_outcomes") is not False:
        issues.append("outcome_driven_pause")
    if amendment.get("partial_grid_excluded_from_lean_primary_gate") is not True:
        issues.append("partial_grid_in_primary_gate")

    exact_settings = set(common.SETTINGS)
    if set(model.setting_id) != exact_settings or set(evidence.setting_id) != exact_settings or set(competence.setting_id) != exact_settings:
        issues.append("setting_set")
    if len(model) != 90 or len(evidence) != 720 or len(audit) != 90:
        issues.append(f"cardinality:model={len(model)}:evidence={len(evidence)}:audit={len(audit)}")
    for setting in common.SETTINGS:
        cell = model[model.setting_id == setting]
        if len(cell) != 15 or set(cell.method) != {"ERM"} or set(cell["lambda"].astype(float)) != {0.0}:
            issues.append(f"erm_grid:{setting}")
        direction = evidence[evidence.setting_id == setting]
        if len(direction) != 120:
            issues.append(f"direction_count:{setting}")
        for fold in common.FOLDS:
            for seed in common.SEEDS:
                run = direction[(direction.fold == fold) & (direction.seed == seed)]
                if len(run) != 8 or set(run.direction_rank.astype(int)) != set(range(1, 9)):
                    issues.append(f"direction_grid:{setting}:{fold}:{seed}")

    required_evidence = {
        "setting_id", "dataset", "task", "backbone", "fold", "seed", "direction_rank",
        "identity_full", "identity_direction_effect", "persistence", "geometry_strength",
        "D_finite", "C_src_CE", "C_src_BA", "C_src_F1", "O_task",
        "checkpoint_sha256", "direction_sha256", "persistence_basis_sha256",
        "source_scope_hash", "validation_scope_hash",
    }
    missing = required_evidence - set(evidence.columns)
    if missing:
        issues.append(f"evidence_schema:{sorted(missing)}")
    banned = [column for column in evidence.columns if re.search(r"U_future|future_(BA|CE|F1)_utility|BA_erased_future|delta_G", column, re.I)]
    if banned:
        issues.append(f"future_columns:{banned}")
    hash_columns = [column for column in evidence.columns if column.endswith("sha256") or column.endswith("scope_hash")]
    for column in hash_columns:
        if evidence[column].isna().any() or not evidence[column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            issues.append(f"hash_column:{column}")
    if not ((evidence.O_task >= -1e-8) & (evidence.O_task <= 1.0 + 1e-8)).all():
        issues.append("O_task_range")
    if not audit.artifact_complete.astype(bool).all() or not (audit.direction_rows == 8).all():
        issues.append("source_artifact_completeness")

    recomputed: list[dict[str, object]] = []
    for setting in common.SETTINGS:
        cell = model[model.setting_id == setting]
        mean_ba = float(cell.ERM_outcome_competence_BA.mean())
        folds_above = int((cell.groupby("fold").ERM_outcome_competence_BA.mean() > 0.5).sum())
        status = "COMPETENCE_PASS" if mean_ba > 0.60 and folds_above >= 4 else "COMPETENCE_FAIL"
        row = competence.set_index("setting_id").loc[setting]
        if not np.isclose(mean_ba, float(row.outcome_BA_mean), atol=1e-12) or folds_above != int(row.folds_above_chance) or status != row.competence:
            issues.append(f"competence_recalculation:{setting}")
        recomputed.append({"setting_id": setting, "mean_BA": mean_ba, "folds_above_chance": folds_above, "status": status})

    new_erm_paths = sorted(common.RUNS.glob("S[456]/fold-*/seed-*/candidates/erm__lambda-0.00.json"))
    new_source_complete = sorted(common.RUNS.glob("S[456]/fold-*/seed-*/source_freeze/erm__lambda-0.00/SOURCE_COMPLETE.json"))
    if len(new_erm_paths) != 45 or len(new_source_complete) != 45:
        issues.append(f"new_erm_files:{len(new_erm_paths)}:{len(new_source_complete)}")
    for path in new_source_complete:
        payload = common.read_json(path)
        if payload.get("direction_future_utility_accessed") is not False or payload.get("invariance_outcome_accessed") is not False:
            issues.append(f"new_source_purity:{path}")

    if preflight.get("OpenBMI_sealed_14_accessed") is not False or preflight.get("WBCIC_sealed_10_accessed") is not False:
        issues.append("sealed_holdout_purity")
    if manifest.get("partial_non_erm_grid_primary_gate") is not False or manifest.get("direction_future_utility_sealed") is not True:
        issues.append("lean_manifest_purity")

    task_live = json.loads(subprocess.check_output(
        [
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "$t=Get-ScheduledTask -TaskName 'PERSIST_EEG_P4A_PIPELINE';"
            "$g=Get-CimInstance Win32_Process|Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -match 'train.py.*--tier grid'};"
            "[pscustomobject]@{State=[string]$t.State;Arguments=[string]$t.Actions[0].Arguments;GridProcessCount=@($g).Count}|ConvertTo-Json -Compress",
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip())
    if "run_lean_pipeline.py" not in task_live["Arguments"]:
        issues.append("live_scheduler_action_not_lean")
    if int(task_live["GridProcessCount"]) != 0:
        issues.append(f"live_grid_process_count:{task_live['GridProcessCount']}")

    all_pass = all(row["status"] == "COMPETENCE_PASS" for row in recomputed)
    terminal = "P4A_LEAN_CROSS_SETTING_CUBE_COMPLETE" if all_pass else "P4A_LEAN_PARTIAL_SETTING_FAILURE"
    report = (common.EXP / "P4A_LEAN_FINAL_REPORT.md").read_text(encoding="utf-8")
    if terminal not in report:
        issues.append("terminal_report_mismatch")
    if issues:
        terminal = "P4A_LEAN_PROTOCOL_OR_PURITY_FAILURE"

    payload = {
        "pass": not issues,
        "issues": issues,
        "terminal": terminal,
        "settings": sorted(exact_settings),
        "erm_rows": len(model),
        "new_mandatory_erm_complete": len(new_erm_paths),
        "source_evidence_rows": len(evidence),
        "source_artifact_audit_rows": len(audit),
        "competence_recomputed": recomputed,
        "partial_grid_completed": progress.get("grid_completed"),
        "partial_grid_primary_gate": False,
        "direction_future_utility_sealed": True,
        "invariance_outcome_delta_sealed": True,
        "OpenBMI_sealed_14_accessed": False,
        "WBCIC_sealed_10_accessed_or_enumerated": False,
    }
    common.write_json(common.RESULTS / "P4A_LEAN_FINAL_VALIDATION.json", payload)
    if issues:
        raise RuntimeError(f"P4A Lean validation failed: {issues}")
    print(f"P4A_LEAN_VALIDATION_PASS {terminal}", flush=True)


if __name__ == "__main__":
    main()
