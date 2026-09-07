from __future__ import annotations

import json
import shutil
from typing import Any

import numpy as np
import pandas as pd

import p4d_common as c


def configs() -> list[tuple[str, float]]:
    payload = c.read_json(c.EXP / "CANONICAL_INVARIANCE_CONFIGS.json")
    return [
        (str(row["method"]), float(row["lambda_star"]))
        for row in payload["methods"]
        if row["status"] == "IDENTITY_MANIPULATION_COMPETENT"
    ]


def source_matrix(selected: list[tuple[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for setting in ("S4", "S6"):
        for fold in c.FOLDS:
            for seed in c.SEEDS:
                erm = c.read_json(c.require_file(c.source_complete(setting, fold, seed, "ERM", 0.0)))
                i_erm = float(erm["source_identity"]["identity_symmetric"])
                for method, lam in selected:
                    path = c.require_file(c.source_complete(setting, fold, seed, method, lam))
                    payload = c.read_json(path)
                    if payload.get("pass") is not True or payload.get("invariance_outcome_accessed") is not False or payload.get("direction_future_utility_accessed") is not False:
                        raise RuntimeError(f"prospective source artifact purity failure: {path}")
                    i_method = float(payload["source_identity"]["identity_symmetric"])
                    suppression = i_erm - i_method
                    rows.append(
                        {
                            "setting_id": setting,
                            "dataset": c.SETTINGS[setting]["dataset"],
                            "task": c.SETTINGS[setting]["task"],
                            "backbone": c.SETTINGS[setting]["backbone"],
                            "fold": fold,
                            "seed": seed,
                            "method": method,
                            "lambda": lam,
                            "I_ERM": i_erm,
                            "I_method": i_method,
                            "S_I_abs": suppression,
                            "S_I_rel": suppression / (abs(i_erm) + c.EPS),
                            "method_checkpoint_sha256": payload["checkpoint_sha256"],
                            "erm_checkpoint_sha256": erm["checkpoint_sha256"],
                            "task_outcome_accessed": False,
                        }
                    )
    frame = pd.DataFrame(rows)
    expected = 30 * len(selected)
    if len(frame) != expected or frame[["setting_id", "fold", "seed", "method"]].duplicated().any():
        raise RuntimeError(f"canonical matrix not balanced: {len(frame)} vs {expected}")
    return frame


def normalize(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = frame.copy()
    result["z_SI"] = np.nan
    setting_stats: dict[str, Any] = {}
    for setting, indices in result.groupby("setting_id").groups.items():
        values = result.loc[indices, "S_I_abs"].to_numpy(float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_scale = 1.4826 * mad
        fallback = False
        scale = robust_scale
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.std(values, ddof=1))
            fallback = True
        if not np.isfinite(scale) or scale <= 1e-12:
            raise RuntimeError(f"identity normalization degenerate for {setting}")
        result.loc[indices, "z_SI"] = (values - median) / (scale + c.EPS)
        setting_stats[str(setting)] = {
            "n": len(values),
            "median_S_I_abs": median,
            "MAD": mad,
            "scale": scale,
            "fallback_SD_used": fallback,
        }
    return result, setting_stats


def main() -> None:
    selected = configs()
    if not selected:
        raise RuntimeError("no competent canonical method")
    completion = c.read_json(c.RESULTS / "P4D_S6_CANONICAL_TRAINING_COMPLETE.json")
    if completion.get("pass") is not True or completion.get("task_outcomes_accessed") is not False:
        raise RuntimeError("S6 canonical source-only training is incomplete")
    inventory_path = c.RESULTS / "invariance_grid_inventory.csv"
    pretraining_inventory = c.RESULTS / "invariance_grid_inventory_pretraining.csv"
    if not pretraining_inventory.is_file():
        shutil.copy2(inventory_path, pretraining_inventory)
    from prepare_p4d import grid_inventory

    final_inventory = grid_inventory()
    inventory_summary = final_inventory.groupby(["setting_id", "status"]).size().rename("cells").reset_index()
    c.write_text(
        c.EXP / "INVARIANCE_GRID_INVENTORY.md",
        "# Invariance Grid Inventory\n\n"
        "This final source-only inventory distinguishes historical/observed outcomes, trained but task-outcome-sealed artifacts, and untrained cells. "
        "S4 retains its complete 135-cell non-ERM sealed grid. S5 remains partial and supplementary. S6 contains only the frozen manipulation-competent canonical configuration; all noncanonical cells remain untrained. The 405-grid was not resumed.\n\n"
        + c.markdown_table(inventory_summary),
    )
    frame, stats = normalize(source_matrix(selected))
    path = c.RESULTS / "canonical_identity_manipulation_source_only.csv"
    c.write_csv(path, frame)
    normalization: dict[str, Any] = {
        "schema": "PERSIST_EEG_P4D_IDENTITY_MANIPULATION_NORMALIZATION_FROZEN_V1",
        "timestamp_utc": c.now_utc(),
        "definition": "z_SI=(S_I_abs-setting_median)/(1.4826*setting_MAD+1e-12), fallback setting SD if MAD degenerate",
        "S_I_abs": "identity_symmetric_ERM - identity_symmetric_method",
        "settings": stats,
        "source_only": True,
        "task_outcomes_accessed": False,
        "matrix_rows": len(frame),
        "matrix_sha256": c.sha256(path),
    }
    normalization["content_sha256"] = c.canonical_sha256(normalization)
    c.write_json(c.EXP / "IDENTITY_MANIPULATION_NORMALIZATION_FROZEN.json", normalization)
    burden_path = c.RESULTS / "P4D_SOURCE_UNSAFE_BURDEN.csv"
    protocol_path = c.EXP / "P4D_PROTOCOL_FROZEN.json"
    freeze: dict[str, Any] = {
        "schema": "PERSIST_EEG_P4D_PRE_TASK_OUTCOME_FREEZE_V1",
        "pass": True,
        "timestamp_utc": c.now_utc(),
        "prospective_settings": ["S4", "S6"],
        "canonical_configs": [{"method": method, "lambda": lam} for method, lam in selected],
        "balanced_runs_per_setting_method": 15,
        "matrix_rows": len(frame),
        "future_method_task_outcome_access_count_before_freeze": 0,
        "partial_outcome_retuning": False,
        "P4A_405_grid_resumed": False,
        "OpenBMI_sealed_internal_holdout": "UNTOUCHED",
        "WBCIC_outer_10": "UNTOUCHED_NOT_ENUMERATED",
        "hashes": {
            "P4D_PROTOCOL_FROZEN.json": c.sha256(protocol_path),
            "CANONICAL_INVARIANCE_CONFIGS.json": c.sha256(c.EXP / "CANONICAL_INVARIANCE_CONFIGS.json"),
            "P4D_SOURCE_BURDEN_FREEZE.json": c.sha256(c.EXP / "P4D_SOURCE_BURDEN_FREEZE.json"),
            "P4D_SOURCE_UNSAFE_BURDEN.csv": c.sha256(burden_path),
            "IDENTITY_MANIPULATION_NORMALIZATION_FROZEN.json": c.sha256(c.EXP / "IDENTITY_MANIPULATION_NORMALIZATION_FROZEN.json"),
            "canonical_identity_manipulation_source_only.csv": c.sha256(path),
            "P4D_S6_CANONICAL_TRAINING_COMPLETE.json": c.sha256(c.RESULTS / "P4D_S6_CANONICAL_TRAINING_COMPLETE.json"),
            "invariance_grid_inventory_pretraining.csv": c.sha256(pretraining_inventory),
            "invariance_grid_inventory_final.csv": c.sha256(inventory_path),
        },
        "final_inventory_status_counts": final_inventory.status.value_counts().to_dict(),
    }
    freeze["content_sha256"] = c.canonical_sha256(freeze)
    c.write_json(c.EXP / "P4D_PRE_TASK_OUTCOME_FREEZE.json", freeze)
    print(json.dumps({"configs": freeze["canonical_configs"], "rows": len(frame), "normalization": stats}, indent=2))
    print("P4D_PRE_TASK_OUTCOME_FREEZE_COMPLETE")


if __name__ == "__main__":
    main()
