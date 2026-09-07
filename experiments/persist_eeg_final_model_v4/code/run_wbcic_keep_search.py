from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from common import DIAGNOSTICS, LEADERBOARD, OUTPUTS, default_wbcic_repo, ensure_directories, logit, sigmoid, write_csv, write_json
from datasets import load_wbcic_development
from evaluation import summarize_method
from models import deepsets, linear, stacking, trees
from training import OOFResult, run_nested_model_average_oof, run_nested_oof, run_threshold_only_oof


def _fixed(data, method_id: str, margin: np.ndarray) -> OOFResult:
    margin = np.asarray(margin, dtype=float)
    return OOFResult(
        method_id=method_id,
        prediction=(margin >= 0).astype(int),
        probability=sigmoid(margin),
        outer_fold=data.metadata.outer_fold.to_numpy(dtype=int),
        selections=pd.DataFrame(),
    )


def _raw_features(data) -> np.ndarray:
    filled = np.where(data.keep_run_mask, data.keep_run_logits, data.base_logits[:, None])
    return np.column_stack(
        [filled, data.keep_run_mask.astype(float), np.ones(len(data.labels)), np.zeros(len(data.labels))]
    )


def _summary_features(data) -> np.ndarray:
    keep = data.keep_run_logits
    mask = data.keep_run_mask
    filled = np.where(mask, keep, np.nan)
    probability = sigmoid(keep)
    count = mask.sum(axis=1).astype(float)
    vote = np.nanmean(np.where(mask, (keep >= 0).astype(float), np.nan), axis=1)
    vote_clipped = np.clip(vote, 1e-7, 1 - 1e-7)
    vote_entropy = -(vote_clipped * np.log(vote_clipped) + (1 - vote_clipped) * np.log1p(-vote_clipped))
    base_probability = data.base_probability
    base_entropy = -(
        base_probability * np.log(np.clip(base_probability, 1e-7, 1.0))
        + (1 - base_probability) * np.log(np.clip(1 - base_probability, 1e-7, 1.0))
    )
    columns = [
        data.base_logits,
        base_probability,
        np.abs(data.base_logits),
        base_entropy,
        np.nanmean(filled, axis=1),
        np.nanstd(filled, axis=1),
        np.nanmin(filled, axis=1),
        np.nanmax(filled, axis=1),
        np.nanmax(filled, axis=1) - np.nanmin(filled, axis=1),
        np.nanmedian(filled, axis=1),
        np.nanmean(probability, axis=1),
        vote,
        vote_entropy,
        np.sum(mask & ((keep >= 0) != (data.base_logits[:, None] >= 0)), axis=1) / count,
        count,
    ]
    for position in range(6):
        columns.extend(
            [
                np.where(mask[:, position], keep[:, position], data.base_logits),
                mask[:, position].astype(float),
            ]
        )
    return np.column_stack(columns)


def _evaluate(data, results: list[OOFResult]) -> pd.DataFrame:
    rows, subjects, folds, selections, predictions = [], [], [], [], []
    for result in results:
        row, subject, fold = summarize_method(
            data, result.method_id, result.prediction, result.probability, result.outer_fold
        )
        rows.append(row)
        subjects.append(subject)
        folds.append(fold)
        if not result.selections.empty:
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
    write_csv(LEADERBOARD / "WBCIC_DEV_KEEP_SEARCH.csv", leaderboard)
    write_csv(DIAGNOSTICS / "WBCIC_DEV_KEEP_SEARCH_SUBJECT_RESULTS.csv", pd.concat(subjects, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_DEV_KEEP_SEARCH_FOLD_RESULTS.csv", pd.concat(folds, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_DEV_KEEP_SEARCH_OOF_PREDICTIONS.csv", pd.concat(predictions, ignore_index=True))
    if selections:
        write_csv(DIAGNOSTICS / "WBCIC_DEV_KEEP_SEARCH_SELECTIONS.csv", pd.concat(selections, ignore_index=True))
    return leaderboard


def run(expert_table: Path, wbcic_repo: Path) -> pd.DataFrame:
    ensure_directories()
    raw_data = load_wbcic_development(expert_table, wbcic_repo)
    logits = raw_data.keep_run_logits[:, :5]
    strong_probability = sigmoid(logits).mean(axis=1)
    data = replace(raw_data, base_logits=logit(strong_probability))
    raw_x = _raw_features(data)
    summary_x = _summary_features(data)
    stable_configs = [
        {"l2": l2, "learn_scale": True, "session_specific": False}
        for l2 in (0.01, 0.1, 1.0)
    ]
    contextual_configs = [
        {"l2": l2, "context_mode": "prediction"} for l2 in (0.01, 0.1, 1.0)
    ]
    linear_configs = [
        {"C": c, "penalty": "l2"} for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
    ]
    results: list[OOFResult] = [
        _fixed(data, "W0_B_STRONG_PROBABILITY_MEAN", data.base_logits),
        _fixed(data, "W0_ALL_LOGIT_MEAN", logits.mean(axis=1)),
    ]
    jobs = [
        ("W1_BSTRONG_THRESHOLD_ONLY", lambda: run_threshold_only_oof(data, "W1_BSTRONG_THRESHOLD_ONLY", data.base_probability)),
        ("W1_RAW_LINEAR", lambda: run_nested_oof(data, "W1_RAW_LINEAR", raw_x, "linear", linear_configs, linear.build)),
        ("W1_SUMMARY_LINEAR", lambda: run_nested_oof(data, "W1_SUMMARY_LINEAR", summary_x, "linear", linear_configs, linear.build)),
        ("W1_SUMMARY_HGB", lambda: run_nested_oof(data, "W1_SUMMARY_HGB", summary_x, "tree", trees.configurations(), trees.build)),
        ("W1_POSITIVE_PROB_POOL_THR050", lambda: run_nested_oof(data, "W1_POSITIVE_PROB_POOL_THR050", raw_x, "stacking_probability", stable_configs, stacking.build_probability, thresholds=(0.5,))),
        ("W1_POSITIVE_PROB_POOL_SHRUNK_THR", lambda: run_nested_oof(data, "W1_POSITIVE_PROB_POOL_SHRUNK_THR", raw_x, "stacking_probability", stable_configs, stacking.build_probability, thresholds=(0.475, 0.5, 0.525))),
        ("W1_POSITIVE_PROB_POOL_CONFIG_AVG", lambda: run_nested_model_average_oof(data, "W1_POSITIVE_PROB_POOL_CONFIG_AVG", raw_x, "stacking_probability", stable_configs, stacking.build_probability, thresholds=(0.475, 0.5, 0.525))),
        ("W1_CONTEXTUAL_POSITIVE_LOGIT_POOL", lambda: run_nested_oof(data, "W1_CONTEXTUAL_POSITIVE_LOGIT_POOL", raw_x, "stacking", contextual_configs, stacking.build_contextual, thresholds=(0.475, 0.5, 0.525))),
        ("W1_DEEPSETS_KEEP", lambda: run_nested_oof(data, "W1_DEEPSETS_KEEP", raw_x, "deepsets", deepsets.configurations(), deepsets.build, thresholds=(0.475, 0.5, 0.525))),
    ]
    for method_id, execute in jobs:
        print(f"[WBCIC KEEP search] {method_id}", flush=True)
        results.append(execute())
    leaderboard = _evaluate(data, results)
    best = leaderboard.iloc[0]
    write_json(
        DIAGNOSTICS / "WBCIC_DEV_KEEP_SEARCH_DECISION.json",
        {
            "status": "WBCIC_DEVELOPMENT_KEEP_SEARCH_COMPLETE",
            "B_STRONG": "W0_B_STRONG_PROBABILITY_MEAN",
            "best_method": str(best.method_id),
            "best_Delta_BA_vs_B_STRONG": float(best.Delta_BA_vs_B_STRONG),
            "best_CI95": [float(best.CI95_L), float(best.CI95_U)],
            "strong_candidate": bool(best.Delta_BA_vs_B_STRONG > 0 and best.CI95_L > 0),
            "outer_subject_ids_loaded": False,
            "OUTER_TEST_USED": False,
        },
    )
    print(leaderboard[["method_id", "mean_subject_BA", "Delta_BA_vs_B_STRONG", "CI95_L", "CI95_U", "positive_fold_fraction"]].to_string(index=False))
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
