from __future__ import annotations

import numpy as np
import pandas as pd

import common as c
import run_stage0_repair1 as repair


def main() -> None:
    issues: list[str] = []
    freeze_path = c.EXP / "protocol" / "PRE_STAGE0_REPAIR1_FREEZE.json"
    if not freeze_path.is_file():
        raise RuntimeError("missing PRE_STAGE0_REPAIR1_FREEZE.json")
    freeze = c.read_json(freeze_path)
    if freeze.get("pass") is not True or freeze.get("frozen_before_repair1_metrics") is not True:
        issues.append("repair1_freeze")
    for relative, expected in freeze.get("file_sha256", {}).items():
        path = c.EXP / relative
        if not path.is_file() or c.sha256(path) != expected:
            issues.append(f"post_freeze_hash_changed:{relative}")

    v0_validation = c.read_json(c.RESULTS / "STAGE0_VALIDATION.json")
    if v0_validation.get("pass") is not True:
        issues.append("v0_validation")
    if v0_validation.get("stage0_terminal") != "TRANSPORT_NOT_SUBJECT_FAITHFUL":
        issues.append("v0_terminal")
    if v0_validation.get("future_session_performance_accessed") is not False:
        issues.append("v0_future_access")

    unit_paths = sorted(
        (c.RUNTIME / "stage0_repair1_units").glob("*/fold-*/*/UNIT_COMPLETE.json")
    )
    if len(unit_paths) != 40:
        issues.append(f"repair1_unit_count:{len(unit_paths)}")
    seen_units: set[tuple[str, int, str]] = set()
    expected_methods = {repair.alpha_method(alpha) for alpha in repair.ALPHAS}
    for path in unit_paths:
        payload = c.read_json(path)
        setting = str(payload.get("setting_id"))
        fold = int(payload.get("fold", -1))
        layer = str(payload.get("layer"))
        seen_units.add((setting, fold, layer))
        if payload.get("pass") is not True or payload.get("schema") != "SCST_DR_STAGE0_REPAIR1_UNIT_V1":
            issues.append(f"unit_failed:{path}")
        if payload.get("outcome_rows_loaded") != 0 or payload.get("future_session_rows_loaded") != 0:
            issues.append(f"unit_scope_violation:{path}")
        if set(payload.get("methods", [])) != expected_methods:
            issues.append(f"unit_methods:{path}")
        if set(map(float, payload.get("alphas", []))) != set(repair.ALPHAS):
            issues.append(f"unit_alphas:{path}")

        v0_path = c.RUNTIME / "stage0_units" / setting / f"fold-{fold}" / layer / "UNIT_COMPLETE.json"
        if not v0_path.is_file():
            issues.append(f"missing_v0_unit:{path}")
        else:
            v0_payload = c.read_json(v0_path)
            if c.sha256(v0_path) != payload.get("v0_unit_complete_sha256"):
                issues.append(f"v0_unit_hash:{path}")
            for key in (
                "feature_scope_sha256",
                "scaling_center_sha256",
                "scaling_scale_sha256",
                "source_rows",
                "validation_rows",
                "bank_session",
                "evaluation_session",
            ):
                if payload.get(key) != v0_payload.get(key):
                    issues.append(f"v0_mismatch:{key}:{path}")

        for name in ("SUBJECT_FIDELITY.csv", "CLASS_FIDELITY.csv", "MANIFOLD_VALIDITY.csv"):
            output = path.parent / name
            if not output.is_file():
                issues.append(f"missing:{output}")
                continue
            if c.sha256(output) != payload.get("output_sha256", {}).get(name):
                issues.append(f"output_hash:{output}")
            frame = pd.read_csv(output)
            if len(frame) == 0 or set(frame.method.astype(str).unique()) != expected_methods:
                issues.append(f"methods:{output}")
            if set(np.round(frame.alpha.astype(float).unique(), 8)) != set(repair.ALPHAS):
                issues.append(f"alphas:{output}")
            if set(frame.setting_id.astype(str).unique()) != {setting}:
                issues.append(f"setting:{output}")
            if set(frame.layer.astype(str).unique()) != {layer}:
                issues.append(f"layer:{output}")

    expected_units = {
        (setting, fold, layer)
        for setting in c.SETTINGS
        for fold in range(5)
        for layer in repair.LAYERS
    }
    if seen_units != expected_units:
        issues.append("repair1_unit_coverage")

    summary_path = c.RESULTS / "STAGE0_REPAIR1_LAYER_SUMMARY.csv"
    result_path = c.RESULTS / "STAGE0_REPAIR1_FINAL_RESULT.json"
    if not summary_path.is_file() or not result_path.is_file():
        issues.append("missing_repair1_summary")
        summary = pd.DataFrame()
        result: dict = {}
    else:
        summary = pd.read_csv(summary_path)
        result = c.read_json(result_path)
        if len(summary) != 16:
            issues.append(f"summary_rows:{len(summary)}")
        if set(summary.setting_id.astype(str).unique()) != set(c.SETTINGS):
            issues.append("summary_settings")
        if set(summary.layer.astype(str).unique()) != set(repair.LAYERS):
            issues.append("summary_layers")
        if set(np.round(summary.alpha.astype(float).unique(), 8)) != set(repair.ALPHAS):
            issues.append("summary_alphas")
        terminal = result.get("terminal")
        if terminal not in c.protocol()["authorized_stage0_terminals"]:
            issues.append("terminal")
        if result.get("outer_or_future_performance_accessed") is not False:
            issues.append("future_access")

        eligible: list[float] = []
        for alpha in repair.ALPHAS:
            frame = summary[np.isclose(summary.alpha.astype(float), alpha)]
            if frame.groupby("setting_id").all_gates_pass.any().reindex(c.SETTINGS).fillna(False).all():
                eligible.append(alpha)
        supported = terminal == "TRANSPORT_VALIDITY_SUPPORTED"
        if supported != bool(eligible):
            issues.append("supported_terminal_consistency")
        if bool(result.get("stage1_authorized")) != supported:
            issues.append("stage1_gate")
        if sorted(map(float, result.get("eligible_alphas", []))) != sorted(eligible):
            issues.append("eligible_alphas")
        if supported:
            expected_alpha = 0.50 if 0.50 in eligible else 0.25
            if not np.isclose(float(result.get("selected_alpha")), expected_alpha):
                issues.append("selected_alpha")
            if len(result.get("selected_layers", {})) != len(c.SETTINGS):
                issues.append("selected_layers")
        else:
            if result.get("selected_alpha") is not None or result.get("selected_layers"):
                issues.append("false_selection")

    for name in (
        "FIGURE_STAGE0_REPAIR1_ALPHA_FIDELITY.png",
        "FIGURE_STAGE0_REPAIR1_MANIFOLD.png",
    ):
        if not (c.FIGURES / name).is_file():
            issues.append(f"missing_figure:{name}")
    if not (c.EXP / "STAGE0_REPAIR1_REPORT.md").is_file():
        issues.append("missing_repair1_report")

    validation = {
        "schema": "SCST_DR_STAGE0_REPAIR1_VALIDATION_V1",
        "pass": len(issues) == 0,
        "issues": issues,
        "stage0_terminal": result.get("terminal") if result else None,
        "stage1_authorized": result.get("terminal") == "TRANSPORT_VALIDITY_SUPPORTED" if result else False,
        "selected_alpha": result.get("selected_alpha") if result else None,
        "OpenBMI_sealed_internal_holdout": "UNTOUCHED_UNENUMERATED",
        "WBCIC_outer_10": "UNTOUCHED_UNENUMERATED",
        "future_session_performance_accessed": False,
        "unit_count": len(unit_paths),
        "layer_summary_rows": len(summary),
    }
    c.write_json(c.RESULTS / "STAGE0_REPAIR1_VALIDATION.json", validation)
    if issues:
        raise RuntimeError("Stage-0 Repair-1 validation failed: " + "; ".join(issues))
    print(
        f"SCST_DR_STAGE0_REPAIR1_VALIDATION_PASS terminal={result['terminal']} selected_alpha={result.get('selected_alpha')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
