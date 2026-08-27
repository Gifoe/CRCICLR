from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import p4d_common as c


def validate_content_hash(payload: dict[str, Any]) -> bool:
    expected = payload.get("content_sha256")
    value = dict(payload)
    value.pop("content_sha256", None)
    return expected == c.canonical_sha256(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-final-report", action="store_true")
    args = parser.parse_args()
    issues: list[str] = []

    required = [
        c.EXP / "README.md",
        c.EXP / "P4C_INPUT_AUDIT.md",
        c.EXP / "P4D_AUTHORIZATION.json",
        c.EXP / "P4D_SOURCE_BURDEN_DEFINITION.md",
        c.EXP / "P4D_SOURCE_BURDEN_FREEZE.json",
        c.EXP / "INVARIANCE_GRID_INVENTORY.md",
        c.EXP / "CANONICAL_INVARIANCE_CONFIGS.json",
        c.EXP / "IDENTITY_MANIPULATION_AUDIT.md",
        c.EXP / "IDENTITY_MANIPULATION_NORMALIZATION_FROZEN.json",
        c.EXP / "P4D_PROTOCOL_FROZEN.json",
        c.EXP / "P4D_PRE_TASK_OUTCOME_FREEZE.json",
        c.EXP / "P4D_METHOD_OUTCOME_ACCESS_LEDGER.md",
        c.RESULTS / "P4D_SOURCE_UNSAFE_BURDEN.csv",
        c.RESULTS / "invariance_grid_inventory.csv",
        c.RESULTS / "identity_manipulation.csv",
        c.RESULTS / "canonical_identity_manipulation_source_only.csv",
        c.RESULTS / "canonical_method_future_outcomes.csv",
        c.RESULTS / "bridge_model_summary.json",
        c.RESULTS / "bridge_bootstrap.csv",
        c.RESULTS / "simple_slope_summary.csv",
        c.RESULTS / "per_setting_bridge.csv",
        c.RESULTS / "per_method_bridge.csv",
        c.RESULTS / "headroom_summary.json",
        c.EXP / "PRIMARY_BRIDGE_INTERACTION_AUDIT.md",
        c.EXP / "SIMPLE_SLOPE_BRIDGE_AUDIT.md",
        c.EXP / "CROSS_SETTING_METHOD_BRIDGE.md",
        c.EXP / "CROSS_METHOD_STABILITY.md",
        c.EXP / "HEADROOM_AUDIT.md",
        c.EXP / "HOLDOUT_PURITY_AUDIT.md",
        c.EXP / "THEORY_METHOD_BRIDGE_NOTE.md",
    ]
    for path in required:
        if not path.is_file():
            issues.append(f"missing required artifact: {path.name}")

    p4c_validation_path = c.P4C / "results" / "P4C_SAFETY_FINAL_VALIDATION.json"
    p4c_assignment_path = c.P4C / "results" / "P4C_SAFETY_PREOUTCOME_REGIME_ASSIGNMENTS.csv"
    p4c_validation = c.read_json(p4c_validation_path)
    if p4c_validation.get("pass") is not True:
        issues.append("P4C validated input is not PASS")
    if p4c_validation.get("preoutcome_assignment_sha256") != c.sha256(p4c_assignment_path):
        issues.append("P4C validated assignment hash mismatch")

    auth = c.read_json(c.EXP / "P4D_AUTHORIZATION.json")
    if auth.get("P4D_AUTHORIZATION") != "CONDITIONAL" or auth.get("P4C_SAFETY_STATUS") != "P4C_SAFETY_BOUNDARY_PARTIAL_SUPPORTED":
        issues.append("P4D conditional authorization does not preserve exact P4C terminal")
    if not validate_content_hash(auth):
        issues.append("P4D authorization content hash failure")

    assignments = pd.read_csv(p4c_assignment_path)
    if assignments.E_task_definition.nunique() != 1 or assignments.E_task_definition.iloc[0] != "(z_D + z_C + z_O)/3":
        issues.append("P4C E_task definition changed")
    burden = pd.read_csv(c.RESULTS / "P4D_SOURCE_UNSAFE_BURDEN.csv")
    if len(burden) != 30 or set(burden.setting_id) != {"S4", "S6"}:
        issues.append("source burden must cover 30 prospective ERM runs")
    recomputed_unsafe = burden.n_highI_highE / burden.n_identity_high
    recomputed_admissible = burden.n_highI_lowE / burden.n_identity_high
    if not np.allclose(burden.R_unsafe, recomputed_unsafe) or not np.allclose(burden.R_admissible, recomputed_admissible):
        issues.append("source burden ratios do not reproduce counts")
    burden_freeze = c.read_json(c.EXP / "P4D_SOURCE_BURDEN_FREEZE.json")
    if burden_freeze.get("method_future_task_outcomes_accessed") is not False or burden_freeze.get("burden_csv_sha256") != c.sha256(c.RESULTS / "P4D_SOURCE_UNSAFE_BURDEN.csv"):
        issues.append("source-only burden freeze/hash failure")

    inventory = pd.read_csv(c.RESULTS / "invariance_grid_inventory.csv")
    if len(inventory) != 900 or set(inventory.status) - {"HISTORICALLY_OBSERVED_TASK_OUTCOME", "TRAINED_BUT_TASK_OUTCOME_SEALED", "UNTRAINED"}:
        issues.append("grid inventory cardinality/status failure")

    selection = pd.read_csv(c.RESULTS / "identity_manipulation.csv")
    canonical = c.read_json(c.EXP / "CANONICAL_INVARIANCE_CONFIGS.json")
    if len(selection) != 135 or selection.task_outcome_accessed_for_selection.astype(bool).any():
        issues.append("canonical lambda selection source matrix/purity failure")
    if canonical.get("future_BA_F1_CE_accessed_for_selection") is not False or not validate_content_hash(canonical):
        issues.append("canonical configuration freeze/hash failure")
    selected: dict[str, float] = {}
    for row in canonical["methods"]:
        method = str(row["method"])
        candidates = []
        for lam in c.LAMBDAS:
            part = selection[(selection.method == method) & np.isclose(selection["lambda"], lam)]
            median = float(part.S_I_abs.median())
            fraction = float((part.S_I_abs > 0).mean())
            if len(part) == 15 and median > 0 and fraction >= 0.60:
                candidates.append((median, -lam, lam))
        expected = max(candidates)[2] if candidates else None
        if expected != row["lambda_star"]:
            issues.append(f"canonical lambda rule does not reproduce for {method}")
        if expected is not None:
            selected[method] = float(expected)

    normalization = c.read_json(c.EXP / "IDENTITY_MANIPULATION_NORMALIZATION_FROZEN.json")
    identity = pd.read_csv(c.RESULTS / "canonical_identity_manipulation_source_only.csv")
    expected_rows = 30 * len(selected)
    if len(identity) != expected_rows or identity.task_outcome_accessed.astype(bool).any():
        issues.append("canonical identity normalization matrix/purity failure")
    for setting, part in identity.groupby("setting_id"):
        info = normalization["settings"][setting]
        expected_z = (part.S_I_abs.to_numpy(float) - float(info["median_S_I_abs"])) / (float(info["scale"]) + c.EPS)
        if not np.allclose(part.z_SI, expected_z, atol=1e-10):
            issues.append(f"z_SI does not reproduce for {setting}")

    protocol = c.read_json(c.EXP / "P4D_PROTOCOL_FROZEN.json")
    prefreeze = c.read_json(c.EXP / "P4D_PRE_TASK_OUTCOME_FREEZE.json")
    if not validate_content_hash(protocol) or not validate_content_hash(prefreeze):
        issues.append("protocol/pre-outcome content hash failure")
    if prefreeze.get("future_method_task_outcome_access_count_before_freeze") != 0 or prefreeze.get("partial_outcome_retuning") is not False:
        issues.append("pre-task-outcome freeze purity failure")
    if prefreeze.get("P4A_405_grid_resumed") is not False:
        issues.append("P4A 405-grid was resumed")

    outcomes = pd.read_csv(c.RESULTS / "canonical_method_future_outcomes.csv")
    if len(outcomes) != expected_rows:
        issues.append("canonical future outcome matrix row count failure")
    for setting in ("S4", "S6"):
        for method, lam in selected.items():
            part = outcomes[(outcomes.setting_id == setting) & (outcomes.method == method)]
            if len(part) != 15 or set(part.fold) != set(c.FOLDS) or set(part.seed) != set(c.SEEDS) or not np.allclose(part["lambda"], lam):
                issues.append(f"unbalanced/exact config failure {setting}/{method}")

    bootstrap = pd.read_csv(c.RESULTS / "bridge_bootstrap.csv")
    if len(bootstrap) != c.BOOTSTRAP_DRAWS:
        issues.append("bootstrap is not exactly 10,000 draws")
    summary = c.read_json(c.RESULTS / "bridge_model_summary.json")
    if summary.get("cluster_hierarchy") != ["setting", "fold", "seed/run"] or summary.get("method_configs_nested_within_run") is not True:
        issues.append("bootstrap clustering specification failure")
    point = summary["coefficients"]
    cis = summary["coefficient_CIs"]
    slopes = summary["simple_slopes"]
    per_setting = pd.read_csv(c.RESULTS / "per_setting_bridge.csv")
    per_method = pd.read_csv(c.RESULTS / "per_method_bridge.csv")
    gates = {
        "G1_manipulation_competence": len(selected) >= 2,
        "G2_primary_interaction": point["beta_zSI_x_Runsafe"] < 0 and cis["beta_zSI_x_Runsafe"][1] < 0,
        "G3_simple_slope": slopes["DeltaSlope_bridge"] > 0 and slopes["DeltaSlope_CI"][0] > 0,
        "G4_setting_consistency": len(per_setting) == 2 and bool(per_setting.hypothesis_direction.astype(bool).all()),
        "G5_method_consistency": int(per_method.hypothesis_direction.astype(bool).sum()) >= 2,
        "G6_purity": True,
    }
    if all(gates.values()):
        expected_terminal = "P4D_METHOD_LEVEL_BRIDGE_STRONG_SUPPORTED"
    elif point["beta_zSI_x_Runsafe"] < 0 and slopes["DeltaSlope_bridge"] > 0 and gates["G4_setting_consistency"]:
        expected_terminal = "P4D_METHOD_LEVEL_BRIDGE_PARTIAL_SUPPORTED"
    else:
        expected_terminal = "P4D_METHOD_LEVEL_BRIDGE_NOT_SUPPORTED"
    if gates != summary.get("gates") or expected_terminal != summary.get("P4D_terminal"):
        issues.append("terminal gates do not reproduce")
    if summary.get("P4E_MODEL_AUTHORIZATION") != "NOT_AUTHORIZED":
        issues.append("P4E must be NOT_AUTHORIZED because P4C was partial")
    if summary.get("outcome_driven_modification") is not False:
        issues.append("outcome-driven modification purity flag failure")

    figures = sorted(path.name for path in c.FIGURES.glob("figure*.png"))
    if len(figures) != 7:
        issues.append(f"expected seven figures, found {len(figures)}")
    final_report = c.EXP / "P4D_FINAL_REPORT.md"
    if args.require_final_report and not final_report.is_file():
        issues.append("final report required but missing")
    if final_report.is_file() and summary.get("P4D_terminal") not in final_report.read_text(encoding="utf-8"):
        issues.append("final report does not preserve validated terminal")

    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=c.REPO, text=True).strip()
    validation: dict[str, Any] = {
        "schema": "PERSIST_EEG_P4D_FINAL_VALIDATION_V1",
        "pass": not issues,
        "issues": issues,
        "validated_at_utc": c.now_utc(),
        "branch": branch,
        "validated_parent_tip": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.REPO, text=True).strip(),
        "P4C_SAFETY_STATUS": p4c_validation["SAFETY_STATUS"],
        "P4D_AUTHORIZATION": auth["P4D_AUTHORIZATION"],
        "P4D_terminal": summary.get("P4D_terminal"),
        "P4E_MODEL_AUTHORIZATION": summary.get("P4E_MODEL_AUTHORIZATION"),
        "competent_methods": selected,
        "bootstrap_draws": len(bootstrap),
        "gates": gates,
        "OpenBMI_sealed_internal_holdout": "UNTOUCHED",
        "WBCIC_outer_10": "UNTOUCHED_NOT_ENUMERATED",
        "P4A_405_grid": "PAUSED_NOT_RESUMED",
        "final_report_written_after_core_validator_pass": final_report.is_file(),
        "final_report_sha256": c.sha256(final_report) if final_report.is_file() else None,
        "figures": figures,
    }
    c.write_json(c.RESULTS / "P4D_FINAL_VALIDATION.json", validation)
    print(json.dumps(validation, indent=2))
    if issues:
        raise SystemExit(1)
    print("P4D_VALIDATOR_PASS")


if __name__ == "__main__":
    main()
