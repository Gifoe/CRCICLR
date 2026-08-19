from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import DIAGNOSTICS, LEADERBOARD, default_openbmi_cache, ensure_directories, sigmoid, write_csv, write_json
from datasets import load_openbmi, openbmi_keep_pool
from evaluation import summarize_method
from features import build_openbmi_features
from models import linear, stacking
from run_search import _write_iteration_logs
from training import OOFResult, run_nested_oof, run_threshold_only_oof


def _linear_l2_configs() -> list[dict[str, object]]:
    return [{"C": c, "penalty": "l2"} for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)]


def _evaluate(data, results: list[OOFResult]):
    rows, subjects, folds, selections, predictions = [], [], [], [], []
    for result in results:
        row, subject, fold = summarize_method(
            data, result.method_id, result.prediction, result.probability, result.outer_fold
        )
        rows.append(row)
        subjects.append(subject)
        folds.append(fold)
        selections.append(result.selections)
        predictions.append(
            pd.DataFrame(
                {
                    "dataset": data.dataset_id,
                    "trial_uid": data.trial_uid,
                    "subject_id": data.subjects,
                    "session_id": data.sessions,
                    "method_id": result.method_id,
                    "outer_fold": result.outer_fold,
                    "label": data.labels,
                    "B_STRONG_prediction": data.base_prediction,
                    "prediction": result.prediction,
                    "probability": result.probability,
                    "OUTER_TEST_USED": False,
                }
            )
        )
    return (
        pd.DataFrame(rows),
        pd.concat(subjects, ignore_index=True),
        pd.concat(folds, ignore_index=True),
        pd.concat(selections, ignore_index=True),
        pd.concat(predictions, ignore_index=True),
    )


def _append_unique(path: Path, new: pd.DataFrame, key: str) -> pd.DataFrame:
    old = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    if not old.empty:
        old = old[~old[key].isin(set(new[key]))]
    merged = pd.concat([old, new], ignore_index=True)
    write_csv(path, merged)
    return merged


def run(cache_root: Path) -> pd.DataFrame:
    ensure_directories()
    data = load_openbmi(cache_root)
    bundle = build_openbmi_features(data)
    keep_logits, _, _ = openbmi_keep_pool(data)
    keep_probability_mean = sigmoid(keep_logits[:, 1])

    keep_names = bundle.names["KEEP"]
    summary_indices = [
        index
        for index, name in enumerate(keep_names)
        if "fold-" not in name and not name.startswith("keep_sorted_centered_")
    ]
    no_session_indices = [index for index, name in enumerate(keep_names) if not name.startswith("session_")]
    raw_logits = np.where(data.keep_run_mask, data.keep_run_logits, data.base_logits[:, None])
    session_values = sorted(np.unique(data.sessions).tolist())
    session_onehot = np.column_stack([(data.sessions == value).astype(float) for value in session_values])
    weighted_x = np.column_stack([raw_logits, data.keep_run_mask.astype(float), session_onehot])
    raw_x = np.column_stack([raw_logits, data.keep_run_mask.astype(float), session_onehot])

    results: list[OOFResult] = []
    print("[V4 refine] M1_B6_THRESHOLD_ONLY", flush=True)
    results.append(run_threshold_only_oof(data, "M1_B6_THRESHOLD_ONLY", data.base_probability))
    print("[V4 refine] M1_B4_THRESHOLD_ONLY", flush=True)
    results.append(run_threshold_only_oof(data, "M1_B4_THRESHOLD_ONLY", keep_probability_mean))
    print("[V4 refine] M1_KEEP_SUMMARY_LINEAR", flush=True)
    results.append(
        run_nested_oof(
            data,
            "M1_KEEP_SUMMARY_LINEAR",
            bundle.matrices["KEEP"][:, summary_indices],
            "linear",
            _linear_l2_configs(),
            linear.build,
        )
    )
    print("[V4 refine] M1_KEEP_RAW_LINEAR", flush=True)
    results.append(
        run_nested_oof(data, "M1_KEEP_RAW_LINEAR", raw_x, "linear", _linear_l2_configs(), linear.build)
    )
    print("[V4 refine] M1_KEEP_NO_SESSION_LINEAR", flush=True)
    results.append(
        run_nested_oof(
            data,
            "M1_KEEP_NO_SESSION_LINEAR",
            bundle.matrices["KEEP"][:, no_session_indices],
            "linear",
            _linear_l2_configs(),
            linear.build,
        )
    )
    print("[V4 refine] M1_MASKED_POSITIVE_POOL", flush=True)
    results.append(
        run_nested_oof(
            data,
            "M1_MASKED_POSITIVE_POOL",
            weighted_x,
            "stacking",
            stacking.configurations(),
            stacking.build,
        )
    )

    leaderboard, subjects, folds, selections, predictions = _evaluate(data, results)
    write_csv(LEADERBOARD / "OPENBMI_KEEP_REFINEMENT.csv", leaderboard)
    write_csv(DIAGNOSTICS / "KEEP_REFINEMENT_SUBJECT_RESULTS.csv", subjects)
    write_csv(DIAGNOSTICS / "KEEP_REFINEMENT_FOLD_RESULTS.csv", folds)
    write_csv(DIAGNOSTICS / "KEEP_REFINEMENT_CALIBRATION_SELECTION.csv", selections)
    write_csv(DIAGNOSTICS / "KEEP_REFINEMENT_OOF_PREDICTIONS.csv", predictions)

    combined = _append_unique(LEADERBOARD / "OPENBMI_MODEL_LEADERBOARD.csv", leaderboard, "method_id")
    combined = combined.sort_values(["Delta_BA_vs_B_STRONG", "NLL"], ascending=[False, True]).reset_index(drop=True)
    write_csv(LEADERBOARD / "OPENBMI_MODEL_LEADERBOARD.csv", combined)
    _append_unique(DIAGNOSTICS / "SUBJECT_RESULTS.csv", subjects, "method_id")
    _append_unique(DIAGNOSTICS / "FOLD_RESULTS.csv", folds, "method_id")
    _append_unique(DIAGNOSTICS / "CALIBRATION_SELECTION.csv", selections, "method_id")
    _append_unique(DIAGNOSTICS / "OPENBMI_OOF_PREDICTIONS.csv", predictions, "method_id")

    initial = pd.read_csv(LEADERBOARD / "OPENBMI_MODEL_LEADERBOARD.csv").set_index("method_id")
    failure = pd.DataFrame(
        [
            {
                "stage": "initial",
                "evidence": f"M1 linear Delta={initial.loc['M1_DYNAMIC_KEEP_LINEAR', 'Delta_BA_vs_B_STRONG']:.6f}, LCB={initial.loc['M1_DYNAMIC_KEEP_LINEAR', 'CI95_L']:.6f}",
                "diagnosis": "KEEP stacking has weak positive signal but is not robust",
                "next_hypothesis": "separate threshold calibration, raw run weighting, and summary effects",
            },
            {
                "stage": "initial",
                "evidence": f"M2 linear Delta={initial.loc['M2_KEEP_ACTION_LINEAR', 'Delta_BA_vs_B_STRONG']:.6f}; M2 HGB Delta={initial.loc['M2_KEEP_ACTION_HGB', 'Delta_BA_vs_B_STRONG']:.6f}",
                "diagnosis": "unrestricted action logits increase cross-subject error",
                "next_hypothesis": "do not add actions until a conservative correctness/eligibility model exists",
            },
            {
                "stage": "initial",
                "evidence": f"M3 linear minus M2 linear={initial.loc['M3_KEEP_ACTION_PERSIST_LINEAR', 'Delta_BA_vs_B_STRONG'] - initial.loc['M2_KEEP_ACTION_LINEAR', 'Delta_BA_vs_B_STRONG']:.6f}",
                "diagnosis": "flat PERSIST concatenation adds no initial performance value",
                "next_hypothesis": "test PERSIST as prior/constraint after a viable action selector exists",
            },
        ]
    )
    failure["OUTER_TEST_USED"] = False
    write_csv(DIAGNOSTICS / "FAILURE_ANALYSIS.csv", failure)
    feature_map = {
        "M1_DYNAMIC_KEEP_LINEAR": "KEEP full legal",
        "M1_DYNAMIC_KEEP_HGB": "KEEP full legal",
        "M2_KEEP_ACTION_LINEAR": "KEEP_ACTION",
        "M2_KEEP_ACTION_HGB": "KEEP_ACTION",
        "M3_KEEP_ACTION_PERSIST_LINEAR": "KEEP_ACTION_PERSIST",
        "M3_KEEP_ACTION_PERSIST_HGB": "KEEP_ACTION_PERSIST",
        "M2_BOUNDED_RESIDUAL": "KEEP_ACTION bounded offset",
        "M3_BOUNDED_RESIDUAL_PERSIST": "KEEP_ACTION_PERSIST bounded offset",
        "M1_B6_THRESHOLD_ONLY": "B6 probability only",
        "M1_B4_THRESHOLD_ONLY": "B4 probability mean only",
        "M1_KEEP_SUMMARY_LINEAR": "KEEP aggregate summaries",
        "M1_KEEP_RAW_LINEAR": "six run logits + availability + session",
        "M1_KEEP_NO_SESSION_LINEAR": "KEEP full legal without session",
        "M1_MASKED_POSITIVE_POOL": "positive availability-normalized run weights",
    }
    _write_iteration_logs(combined, feature_map)
    best = combined.iloc[0].to_dict()
    write_json(
        DIAGNOSTICS / "KEEP_REFINEMENT_DECISION.json",
        {
            "status": "KEEP_REFINEMENT_COMPLETE",
            "best_method": best["method_id"],
            "best_Delta_BA_vs_B_STRONG": best["Delta_BA_vs_B_STRONG"],
            "best_CI95": [best["CI95_L"], best["CI95_U"]],
            "strong_candidate": bool(best["Delta_BA_vs_B_STRONG"] > 0 and best["CI95_L"] > 0),
            "OUTER_TEST_USED": False,
        },
    )
    print(combined[["method_id", "mean_subject_BA", "Delta_BA_vs_B_STRONG", "CI95_L", "CI95_U"]].to_string(index=False))
    return combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openbmi-cache", type=Path, default=default_openbmi_cache())
    args = parser.parse_args()
    run(args.openbmi_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
