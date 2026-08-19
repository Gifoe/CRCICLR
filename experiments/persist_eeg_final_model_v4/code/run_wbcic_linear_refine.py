from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from common import DIAGNOSTICS, LEADERBOARD, OUTPUTS, default_wbcic_repo, ensure_directories, logit, sigmoid, write_csv, write_json
from datasets import load_wbcic_development
from evaluation import summarize_method
from models import linear, residual
from run_wbcic_keep_search import _raw_features
from training import run_nested_model_average_oof, run_nested_oof


def _probability_features(data) -> np.ndarray:
    logits = data.keep_run_logits[:, :5]
    probability = sigmoid(logits)
    base = data.base_probability
    vote = (logits >= 0).mean(axis=1)
    return np.column_stack(
        [
            probability,
            probability - base[:, None],
            probability.std(axis=1),
            probability.max(axis=1) - probability.min(axis=1),
            vote,
            np.abs(base - 0.5),
        ]
    )


def _evaluate(data, results) -> pd.DataFrame:
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
    leaderboard = pd.DataFrame(rows).sort_values(
        ["Delta_BA_vs_B_STRONG", "NLL"], ascending=[False, True]
    ).reset_index(drop=True)
    write_csv(LEADERBOARD / "WBCIC_DEV_LINEAR_REFINEMENT.csv", leaderboard)
    write_csv(DIAGNOSTICS / "WBCIC_DEV_LINEAR_REFINEMENT_SUBJECT_RESULTS.csv", pd.concat(subjects, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_DEV_LINEAR_REFINEMENT_FOLD_RESULTS.csv", pd.concat(folds, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_DEV_LINEAR_REFINEMENT_SELECTIONS.csv", pd.concat(selections, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_DEV_LINEAR_REFINEMENT_OOF_PREDICTIONS.csv", pd.concat(predictions, ignore_index=True))
    return leaderboard


def run(expert_table: Path, wbcic_repo: Path) -> pd.DataFrame:
    ensure_directories()
    raw_data = load_wbcic_development(expert_table, wbcic_repo)
    logits = raw_data.keep_run_logits[:, :5]
    data = replace(raw_data, base_logits=logit(sigmoid(logits).mean(axis=1)))
    raw_x = _raw_features(data)
    probability_x = _probability_features(data)
    linear_configs = [
        {"C": c, "penalty": "l2"} for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
    ]
    residual_configs = [
        {"l2": l2, "clip": clip}
        for l2 in (1.0, 10.0, 100.0)
        for clip in (0.25, 0.5, 1.0)
    ]
    jobs = [
        ("W1_RAW_LINEAR_THR050", lambda: run_nested_oof(data, "W1_RAW_LINEAR_THR050", raw_x, "linear", linear_configs, linear.build, thresholds=(0.5,))),
        ("W1_RAW_LINEAR_SHRUNK_THR", lambda: run_nested_oof(data, "W1_RAW_LINEAR_SHRUNK_THR", raw_x, "linear", linear_configs, linear.build, thresholds=(0.475, 0.5, 0.525))),
        ("W1_RAW_LINEAR_CONFIG_AVG", lambda: run_nested_model_average_oof(data, "W1_RAW_LINEAR_CONFIG_AVG", raw_x, "linear", linear_configs, linear.build, thresholds=(0.475, 0.5, 0.525))),
        ("W1_PROB_LINEAR_THR050", lambda: run_nested_oof(data, "W1_PROB_LINEAR_THR050", probability_x, "linear", linear_configs, linear.build, thresholds=(0.5,))),
        ("W1_PROB_LINEAR_SHRUNK_THR", lambda: run_nested_oof(data, "W1_PROB_LINEAR_SHRUNK_THR", probability_x, "linear", linear_configs, linear.build, thresholds=(0.475, 0.5, 0.525))),
        ("W1_PROB_LINEAR_CONFIG_AVG", lambda: run_nested_model_average_oof(data, "W1_PROB_LINEAR_CONFIG_AVG", probability_x, "linear", linear_configs, linear.build, thresholds=(0.475, 0.5, 0.525))),
        ("W1_KEEP_ANCHORED_RESIDUAL_THR050", lambda: run_nested_oof(data, "W1_KEEP_ANCHORED_RESIDUAL_THR050", probability_x, "residual", residual_configs, residual.build, thresholds=(0.5,))),
        ("W1_KEEP_ANCHORED_RESIDUAL_SHRUNK_THR", lambda: run_nested_oof(data, "W1_KEEP_ANCHORED_RESIDUAL_SHRUNK_THR", probability_x, "residual", residual_configs, residual.build, thresholds=(0.475, 0.5, 0.525))),
    ]
    results = []
    for method_id, execute in jobs:
        print(f"[WBCIC linear refine] {method_id}", flush=True)
        results.append(execute())
    leaderboard = _evaluate(data, results)
    best = leaderboard.iloc[0]
    write_json(
        DIAGNOSTICS / "WBCIC_DEV_LINEAR_REFINEMENT_DECISION.json",
        {
            "status": "WBCIC_DEVELOPMENT_LINEAR_REFINEMENT_COMPLETE",
            "best_method": str(best.method_id),
            "best_Delta_BA_vs_B_STRONG": float(best.Delta_BA_vs_B_STRONG),
            "best_CI95": [float(best.CI95_L), float(best.CI95_U)],
            "positive_fold_fraction": float(best.positive_fold_fraction),
            "strong_candidate": bool(best.Delta_BA_vs_B_STRONG > 0 and best.CI95_L > 0),
            "outer_subject_ids_loaded": False,
            "OUTER_TEST_USED": False,
        },
    )
    print(leaderboard[["method_id", "mean_subject_BA", "Delta_BA_vs_B_STRONG", "CI95_L", "CI95_U", "positive_fold_fraction", "worst_subject_delta"]].to_string(index=False))
    return leaderboard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-table", type=Path, default=OUTPUTS / "cache" / "WBCIC_DEV_KEEP_EXPERTS.parquet")
    parser.add_argument("--wbcic-repo", type=Path, default=default_wbcic_repo())
    args = parser.parse_args()
    run(args.expert_table, args.wbcic_repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
