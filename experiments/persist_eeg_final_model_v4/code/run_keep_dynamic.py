from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import DIAGNOSTICS, LEADERBOARD, default_openbmi_cache, ensure_directories, write_csv, write_json
from datasets import load_openbmi
from models import deepsets, stacking
from run_keep_refine import _append_unique, _evaluate
from run_search import _write_iteration_logs
from training import run_nested_model_average_oof, run_nested_oof


def run(cache_root: Path) -> pd.DataFrame:
    ensure_directories()
    data = load_openbmi(cache_root)
    raw_logits = np.where(data.keep_run_mask, data.keep_run_logits, data.base_logits[:, None])
    session_values = sorted(np.unique(data.sessions).tolist())
    session_onehot = np.column_stack([(data.sessions == value).astype(float) for value in session_values])
    x = np.column_stack([raw_logits, data.keep_run_mask.astype(float), session_onehot])
    stable_configs = [
        {"l2": l2, "learn_scale": True, "session_specific": False}
        for l2 in (0.01, 0.1, 1.0)
    ]
    results = []
    print("[V4 dynamic] M1_MASKED_POOL_THR050", flush=True)
    results.append(
        run_nested_oof(
            data,
            "M1_MASKED_POOL_THR050",
            x,
            "stacking",
            stable_configs,
            stacking.build,
            thresholds=(0.5,),
        )
    )
    print("[V4 dynamic] M1_MASKED_POOL_SHRUNK_THR", flush=True)
    results.append(
        run_nested_oof(
            data,
            "M1_MASKED_POOL_SHRUNK_THR",
            x,
            "stacking",
            stable_configs,
            stacking.build,
            thresholds=(0.475, 0.5, 0.525),
        )
    )
    print("[V4 dynamic] M1_MASKED_POOL_CONFIG_AVG", flush=True)
    results.append(
        run_nested_model_average_oof(
            data,
            "M1_MASKED_POOL_CONFIG_AVG",
            x,
            "stacking",
            stable_configs,
            stacking.build,
            thresholds=(0.475, 0.5, 0.525),
        )
    )
    print("[V4 dynamic] M1_CONTEXTUAL_POSITIVE_POOL", flush=True)
    results.append(
        run_nested_oof(
            data,
            "M1_CONTEXTUAL_POSITIVE_POOL",
            x,
            "stacking",
            stacking.contextual_configurations(),
            stacking.build_contextual,
            thresholds=(0.475, 0.5, 0.525),
        )
    )
    print("[V4 dynamic] M1_DEEPSETS_KEEP", flush=True)
    results.append(
        run_nested_oof(
            data,
            "M1_DEEPSETS_KEEP",
            x,
            "deepsets",
            deepsets.configurations(),
            deepsets.build,
            thresholds=(0.475, 0.5, 0.525),
        )
    )

    leaderboard, subjects, folds, selections, predictions = _evaluate(data, results)
    write_csv(LEADERBOARD / "OPENBMI_KEEP_DYNAMIC_REFINEMENT.csv", leaderboard)
    write_csv(DIAGNOSTICS / "KEEP_DYNAMIC_SUBJECT_RESULTS.csv", subjects)
    write_csv(DIAGNOSTICS / "KEEP_DYNAMIC_FOLD_RESULTS.csv", folds)
    write_csv(DIAGNOSTICS / "KEEP_DYNAMIC_CALIBRATION_SELECTION.csv", selections)
    write_csv(DIAGNOSTICS / "KEEP_DYNAMIC_OOF_PREDICTIONS.csv", predictions)
    combined = _append_unique(LEADERBOARD / "OPENBMI_MODEL_LEADERBOARD.csv", leaderboard, "method_id")
    combined = combined.sort_values(["Delta_BA_vs_B_STRONG", "NLL"], ascending=[False, True]).reset_index(drop=True)
    write_csv(LEADERBOARD / "OPENBMI_MODEL_LEADERBOARD.csv", combined)
    _append_unique(DIAGNOSTICS / "SUBJECT_RESULTS.csv", subjects, "method_id")
    _append_unique(DIAGNOSTICS / "FOLD_RESULTS.csv", folds, "method_id")
    _append_unique(DIAGNOSTICS / "CALIBRATION_SELECTION.csv", selections, "method_id")
    _append_unique(DIAGNOSTICS / "OPENBMI_OOF_PREDICTIONS.csv", predictions, "method_id")

    feature_map = {method: "six frozen KEEP tokens + availability + session" for method in combined.method_id}
    feature_map.update(
        {
            "M2_KEEP_ACTION_LINEAR": "KEEP_ACTION",
            "M2_KEEP_ACTION_HGB": "KEEP_ACTION",
            "M3_KEEP_ACTION_PERSIST_LINEAR": "KEEP_ACTION_PERSIST",
            "M3_KEEP_ACTION_PERSIST_HGB": "KEEP_ACTION_PERSIST",
            "M2_BOUNDED_RESIDUAL": "KEEP_ACTION bounded offset",
            "M3_BOUNDED_RESIDUAL_PERSIST": "KEEP_ACTION_PERSIST bounded offset",
            "M0_B_STRONG_B6": "B6 only",
        }
    )
    _write_iteration_logs(combined, feature_map)
    best = combined.iloc[0]
    decision = {
        "status": "KEEP_DYNAMIC_REFINEMENT_COMPLETE",
        "best_method": best.method_id,
        "best_Delta_BA_vs_B_STRONG": float(best.Delta_BA_vs_B_STRONG),
        "best_CI95": [float(best.CI95_L), float(best.CI95_U)],
        "positive_fold_fraction": float(best.positive_fold_fraction),
        "positive_subject_fraction": float(best.positive_subject_fraction),
        "strong_candidate": bool(best.Delta_BA_vs_B_STRONG > 0 and best.CI95_L > 0),
        "OUTER_TEST_USED": False,
    }
    write_json(DIAGNOSTICS / "KEEP_DYNAMIC_DECISION.json", decision)
    failure_path = DIAGNOSTICS / "FAILURE_ANALYSIS.csv"
    failure = pd.read_csv(failure_path)
    new = pd.DataFrame(
        [
            {
                "stage": "keep_dynamic",
                "evidence": f"best={best.method_id}, Delta={best.Delta_BA_vs_B_STRONG:.6f}, CI_L={best.CI95_L:.6f}",
                "diagnosis": "dynamic KEEP aggregation evaluated with threshold shrinkage and set gating",
                "next_hypothesis": "transfer the winning family to WBCIC-dev before any further OpenBMI adaptation",
                "OUTER_TEST_USED": False,
            }
        ]
    )
    write_csv(failure_path, pd.concat([failure, new], ignore_index=True))
    print(leaderboard[["method_id", "mean_subject_BA", "Delta_BA_vs_B_STRONG", "CI95_L", "CI95_U", "positive_fold_fraction"]].to_string(index=False))
    return leaderboard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openbmi-cache", type=Path, default=default_openbmi_cache())
    args = parser.parse_args()
    run(args.openbmi_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
