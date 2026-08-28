from __future__ import annotations

import numpy as np
import pandas as pd

import common as c
import run_stage0_repair2 as repair


def main() -> None:
    issues: list[str] = []
    freeze_path = c.EXP / "protocol" / "PRE_STAGE0_REPAIR2_FREEZE.json"
    if not freeze_path.is_file():
        raise RuntimeError("missing PRE_STAGE0_REPAIR2_FREEZE.json")
    freeze = c.read_json(freeze_path)
    if freeze.get("pass") is not True or freeze.get("frozen_before_repair2_outcomes") is not True:
        issues.append("repair2_freeze")
    for relative, expected in freeze.get("file_sha256", {}).items():
        path = c.EXP / relative
        if not path.is_file() or c.sha256(path) != expected:
            issues.append(f"post_freeze_hash_changed:{relative}")

    sealed = c.read_json(c.EXP / "protocol" / "SEALED_RESOURCE_AUDIT.json")
    if sealed.get("pass") is not True or sealed.get("outer_evaluation_authorized") is not False:
        issues.append("sealed_resource_audit")
    repair1 = c.read_json(c.RESULTS / "STAGE0_REPAIR1_VALIDATION.json")
    if repair1.get("pass") is not True or repair1.get("stage0_terminal") != "TRANSPORT_OFF_MANIFOLD":
        issues.append("repair1_prerequisite")

    unit_paths = sorted((c.RUNTIME / "stage0_repair2_units").glob("*/fold-*/UNIT_COMPLETE.json"))
    if len(unit_paths) != 20:
        issues.append(f"unit_count:{len(unit_paths)}")
    seen: set[tuple[str, int]] = set()
    for complete_path in unit_paths:
        payload = c.read_json(complete_path)
        setting = str(payload.get("setting_id"))
        fold = int(payload.get("fold", -1))
        seen.add((setting, fold))
        if payload.get("pass") is not True or payload.get("schema") != "SCST_DR_STAGE0_REPAIR2_UNIT_V1":
            issues.append(f"unit_failed:{complete_path}")
        if payload.get("layer") != repair.LAYER:
            issues.append(f"unit_layer:{complete_path}")
        if payload.get("outcome_rows_loaded") != 0 or payload.get("future_session_rows_loaded") != 0:
            issues.append(f"unit_scope:{complete_path}")
        if not np.array_equal(np.asarray(payload.get("alpha_grid"), dtype=float), repair.ALPHA_GRID):
            issues.append(f"unit_grid:{complete_path}")
        if float(payload.get("alpha_max", -1)) != repair.ALPHA_MAX:
            issues.append(f"unit_alpha_max:{complete_path}")
        if set(payload.get("support_radius", {})) != {"0", "1"}:
            issues.append(f"support_radius_classes:{complete_path}")
        if any(not np.isfinite(float(value)) or float(value) <= 0 for value in payload.get("support_radius", {}).values()):
            issues.append(f"support_radius:{complete_path}")

        v0_path = c.RUNTIME / "stage0_units" / setting / f"fold-{fold}" / repair.LAYER / "UNIT_COMPLETE.json"
        if not v0_path.is_file():
            issues.append(f"missing_v0:{complete_path}")
        else:
            v0_payload = c.read_json(v0_path)
            if c.sha256(v0_path) != payload.get("v0_unit_complete_sha256"):
                issues.append(f"v0_hash:{complete_path}")
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
                    issues.append(f"v0_mismatch:{key}:{complete_path}")

        frames: dict[str, pd.DataFrame] = {}
        expected_methods = {
            "SUBJECT_FIDELITY.csv": set(repair.METHODS_SUBJECT_MANIFOLD),
            "CLASS_FIDELITY.csv": set(repair.METHODS_CLASS),
            "MANIFOLD_VALIDITY.csv": set(repair.METHODS_SUBJECT_MANIFOLD),
        }
        for name, methods in expected_methods.items():
            path = complete_path.parent / name
            if not path.is_file() or c.sha256(path) != payload.get("output_sha256", {}).get(name):
                issues.append(f"output_hash:{path}")
                continue
            frame = pd.read_csv(path)
            frames[name] = frame
            if len(frame) == 0 or set(frame.method.astype(str)) != methods:
                issues.append(f"methods:{path}")
            if set(frame.setting_id.astype(str)) != {setting} or set(frame.layer.astype(str)) != {repair.LAYER}:
                issues.append(f"scope:{path}")

        alpha_csv = complete_path.parent / "ALPHA_DISTRIBUTION.csv"
        npz_path = complete_path.parent / "ALPHA_VALUES.npz"
        for path in (alpha_csv, npz_path):
            if not path.is_file() or c.sha256(path) != payload.get("output_sha256", {}).get(path.name):
                issues.append(f"output_hash:{path}")
        if npz_path.is_file():
            values = np.load(npz_path, allow_pickle=False)
            for unit in ("centroid", "trial"):
                alpha = values[f"{unit}_alpha"].astype(float)
                if len(alpha) == 0 or not np.isfinite(alpha).all():
                    issues.append(f"alpha_values:{unit}:{npz_path}")
                    continue
                if alpha.min() < 0 or alpha.max() > repair.ALPHA_MAX:
                    issues.append(f"alpha_range:{unit}:{npz_path}")
                if not np.allclose(alpha * 64.0, np.round(alpha * 64.0), atol=1e-7, rtol=0):
                    issues.append(f"alpha_grid:{unit}:{npz_path}")
                norm = values[f"{unit}_norm_ratio"].astype(float)
                if len(norm) != len(alpha) or not np.isfinite(norm).all() or np.any(norm < 0):
                    issues.append(f"norm_ratio:{unit}:{npz_path}")

        key = ["setting_id", "fold", "layer", "source_subject", "target_subject", "class_label"]
        if "SUBJECT_FIDELITY.csv" in frames:
            frame = frames["SUBJECT_FIDELITY.csv"]
            a = frame[frame.method == "scst"][key + ["perturbation_norm"]].rename(columns={"perturbation_norm": "scst"})
            b = frame[frame.method == "norm_matched_random"][key + ["perturbation_norm"]].rename(columns={"perturbation_norm": "random"})
            merged = a.merge(b, on=key, validate="one_to_one")
            if not np.allclose(merged.scst, merged.random, atol=1e-9, rtol=1e-9):
                issues.append(f"centroid_random_norm_match:{complete_path}")
        if "CLASS_FIDELITY.csv" in frames:
            frame = frames["CLASS_FIDELITY.csv"]
            a = frame[frame.method == "scst"][key + ["mean_perturbation_norm"]].rename(columns={"mean_perturbation_norm": "scst"})
            b = frame[frame.method == "norm_matched_random"][key + ["mean_perturbation_norm"]].rename(columns={"mean_perturbation_norm": "random"})
            merged = a.merge(b, on=key, validate="one_to_one")
            if not np.allclose(merged.scst, merged.random, atol=1e-9, rtol=1e-9):
                issues.append(f"trial_random_norm_match:{complete_path}")

    expected_seen = {(setting, fold) for setting in c.SETTINGS for fold in range(5)}
    if seen != expected_seen:
        issues.append("unit_coverage")

    required_results = (
        "STAGE0_REPAIR2_LAYER_SUMMARY.csv",
        "STAGE0_REPAIR2_ALPHA_DISTRIBUTION.csv",
        "STAGE0_REPAIR2_SUBJECT_FIDELITY.csv",
        "STAGE0_REPAIR2_CLASS_FIDELITY.csv",
        "STAGE0_REPAIR2_MANIFOLD_VALIDITY.csv",
        "STAGE0_REPAIR2_STATISTICS.json",
        "STAGE0_REPAIR2_FINAL_RESULT.json",
    )
    for name in required_results:
        if not (c.RESULTS / name).is_file():
            issues.append(f"missing:{name}")

    summary_path = c.RESULTS / "STAGE0_REPAIR2_LAYER_SUMMARY.csv"
    result_path = c.RESULTS / "STAGE0_REPAIR2_FINAL_RESULT.json"
    if summary_path.is_file() and result_path.is_file():
        summary = pd.read_csv(summary_path)
        result = c.read_json(result_path)
        if len(summary) != 4 or set(summary.setting_id.astype(str)) != set(c.SETTINGS):
            issues.append("summary_coverage")
        if set(summary.layer.astype(str)) != {repair.LAYER}:
            issues.append("summary_layer")
        gate = c.protocol()["per_setting_layer_gates"]
        expected_manifold = (summary.manifold_knn_ratio_to_clean <= float(gate["manifold_knn_ratio_to_clean_max"])) & (
            summary.off_manifold_excess_over_random <= float(gate["manifold_off_rate_excess_over_random_max"])
        )
        if not np.array_equal(summary.gate_manifold.astype(bool).to_numpy(), expected_manifold.to_numpy()):
            issues.append("manifold_gate_recompute")
        supported = bool(summary.all_gates_pass.astype(bool).all())
        expected_terminal = "TRANSPORT_VALIDITY_SUPPORTED" if supported else "TRANSPORT_VALIDITY_NOT_SUPPORTED"
        if result.get("terminal") != expected_terminal:
            issues.append("terminal")
        if bool(result.get("stage1_authorized")) != supported:
            issues.append("stage1_gate")
        if result.get("outer_or_future_performance_accessed") is not False:
            issues.append("future_access")
    else:
        summary = pd.DataFrame()
        result = {}

    alpha_path = c.RESULTS / "STAGE0_REPAIR2_ALPHA_DISTRIBUTION.csv"
    if alpha_path.is_file():
        alpha = pd.read_csv(alpha_path)
        setting_rows = alpha[(alpha.scope == "setting") & (alpha.query_unit == "centroid")]
        if len(setting_rows) != 4 or set(setting_rows.setting_id.astype(str)) != set(c.SETTINGS):
            issues.append("alpha_setting_coverage")
        if (alpha.alpha_mean < 0).any() or (alpha.alpha_mean > repair.ALPHA_MAX).any():
            issues.append("alpha_summary_range")

    for name in ("FIGURE_STAGE0_REPAIR2_SUPPORT_AND_MANIFOLD.png", "FIGURE_STAGE0_REPAIR2_FIDELITY.png"):
        if not (c.FIGURES / name).is_file():
            issues.append(f"missing_figure:{name}")
    if not (c.EXP / "STAGE0_REPAIR2_REPORT.md").is_file():
        issues.append("missing_repair2_report")

    validation = {
        "schema": "SCST_DR_STAGE0_REPAIR2_VALIDATION_V1",
        "pass": len(issues) == 0,
        "issues": issues,
        "stage0_terminal": result.get("terminal") if result else None,
        "stage1_authorized": result.get("stage1_authorized") if result else False,
        "all_setting_pass_count": int(summary.all_gates_pass.sum()) if len(summary) else 0,
        "unit_count": len(unit_paths),
        "OpenBMI_sealed_internal_holdout": "UNTOUCHED_UNENUMERATED",
        "WBCIC_outer_10": "UNTOUCHED_UNENUMERATED",
        "future_session_performance_accessed": False,
    }
    c.write_json(c.RESULTS / "STAGE0_REPAIR2_VALIDATION.json", validation)
    if issues:
        raise RuntimeError("Stage-0 Repair-2 validation failed: " + "; ".join(issues))
    print(
        f"SCST_DR_STAGE0_REPAIR2_VALIDATION_PASS terminal={result['terminal']} all_setting_pass={validation['all_setting_pass_count']}/4",
        flush=True,
    )


if __name__ == "__main__":
    main()
