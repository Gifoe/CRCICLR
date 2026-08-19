from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from common import DIAGNOSTICS, LEADERBOARD, OUTPUTS, default_wbcic_repo, ensure_directories, logit, sigmoid, write_csv, write_json
from datasets import load_wbcic_development
from evaluation import paired_subject_bootstrap, summarize_method
from models import stacking
from training import OOFResult, run_nested_model_average_oof, run_nested_oof


def _fixed(data, method_id: str, logit: np.ndarray) -> OOFResult:
    probability = sigmoid(logit)
    return OOFResult(
        method_id=method_id,
        prediction=(logit >= 0).astype(int),
        probability=probability,
        outer_fold=data.metadata.outer_fold.to_numpy(dtype=int),
        selections=pd.DataFrame(),
    )


def run(expert_table: Path, wbcic_repo: Path) -> pd.DataFrame:
    ensure_directories()
    raw_data = load_wbcic_development(expert_table, wbcic_repo)
    logits = raw_data.keep_run_logits
    # The legal static audit established that the probability mean of all
    # five competent experts is WBCIC B_STRONG. Nested configuration and
    # threshold selection must optimize against that same reference, not the
    # weaker EEGNet_STABLE member or the weaker raw-logit mean.
    stable_prediction = (logits[:, 0] >= 0).astype(int)
    strong_probability = np.mean(sigmoid(logits[:, :5]), axis=1)
    strong_logits = logit(strong_probability)
    data = replace(raw_data, base_logits=strong_logits)
    results = [
        _fixed(data, "W0_EEGNET_STABLE", logits[:, 0]),
        _fixed(data, "W0_EEGNET_STD", logits[:, 1]),
        _fixed(data, "W0_STATIC_EEGNET_PAIR", np.mean(logits[:, [0, 1]], axis=1)),
        _fixed(data, "W0_STATIC_STABLE_DEEP", np.mean(logits[:, [0, 2]], axis=1)),
        _fixed(data, "W0_STATIC_STABLE_STD_DEEP", np.mean(logits[:, [0, 1, 2]], axis=1)),
        _fixed(data, "W0_STATIC_ALL_LOGIT_MEAN", np.mean(logits[:, :5], axis=1)),
        _fixed(data, "W0_STATIC_ALL_PROBABILITY_MEAN", strong_logits),
    ]
    filled = np.where(data.keep_run_mask, data.keep_run_logits, data.base_logits[:, None])
    session_onehot = np.column_stack([np.ones(len(data.labels)), np.zeros(len(data.labels))])
    x = np.column_stack([filled, data.keep_run_mask.astype(float), session_onehot])
    stable_configs = [
        {"l2": l2, "learn_scale": True, "session_specific": False}
        for l2 in (0.01, 0.1, 1.0)
    ]
    print("[WBCIC transfer] W1_MASKED_POOL_SHRUNK_THR", flush=True)
    results.append(
        run_nested_oof(
            data,
            "W1_MASKED_POOL_SHRUNK_THR",
            x,
            "stacking",
            stable_configs,
            stacking.build,
            thresholds=(0.475, 0.5, 0.525),
        )
    )
    print("[WBCIC transfer] W1_MASKED_POOL_CONFIG_AVG", flush=True)
    results.append(
        run_nested_model_average_oof(
            data,
            "W1_MASKED_POOL_CONFIG_AVG",
            x,
            "stacking",
            stable_configs,
            stacking.build,
            thresholds=(0.475, 0.5, 0.525),
        )
    )

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
                    "EEGNet_STABLE_prediction": stable_prediction,
                    "WBCIC_B_STRONG_prediction": data.base_prediction,
                    "prediction": result.prediction,
                    "probability": result.probability,
                    "OUTER_TEST_USED": False,
                }
            )
        )
    leaderboard = pd.DataFrame(rows)
    subject_table = pd.concat(subjects, ignore_index=True)
    fixed_ids = [value.method_id for value in results if value.method_id.startswith("W0_")]
    static_best_id = str(
        leaderboard[leaderboard.method_id.isin(fixed_ids)].sort_values("mean_subject_BA", ascending=False).iloc[0].method_id
    )
    if static_best_id != "W0_STATIC_ALL_PROBABILITY_MEAN":
        raise RuntimeError(
            "Frozen WBCIC B_STRONG changed; rerun the static-baseline audit before dynamic selection"
        )
    static_subject = subject_table[subject_table.method_id.eq(static_best_id)].set_index("subject_id")
    versus_rows = []
    for method_id, group in subject_table.groupby("method_id", sort=False):
        aligned = group.set_index("subject_id").loc[static_subject.index]
        delta = aligned.balanced_accuracy.to_numpy(dtype=float) - static_subject.balanced_accuracy.to_numpy(dtype=float)
        lower, upper = paired_subject_bootstrap(delta, f"{method_id}_VS_{static_best_id}")
        versus_rows.append(
            {
                "method_id": method_id,
                "WBCIC_B_STRONG": static_best_id,
                "Delta_BA_vs_WBCIC_B_STRONG": float(delta.mean()),
                "CI95_L_vs_WBCIC_B_STRONG": lower,
                "CI95_U_vs_WBCIC_B_STRONG": upper,
                "positive_subject_fraction_vs_WBCIC_B_STRONG": float(np.mean(delta > 0)),
                "nonnegative_subject_fraction_vs_WBCIC_B_STRONG": float(np.mean(delta >= 0)),
                "worst_subject_delta_vs_WBCIC_B_STRONG": float(delta.min()),
            }
        )
    leaderboard = leaderboard.merge(pd.DataFrame(versus_rows), on="method_id", validate="one_to_one")
    leaderboard = leaderboard.sort_values("mean_subject_BA", ascending=False).reset_index(drop=True)
    write_csv(LEADERBOARD / "WBCIC_DEV_MODEL_LEADERBOARD.csv", leaderboard)
    write_csv(DIAGNOSTICS / "WBCIC_DEV_SUBJECT_RESULTS.csv", subject_table)
    write_csv(DIAGNOSTICS / "WBCIC_DEV_FOLD_RESULTS.csv", pd.concat(folds, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_DEV_OOF_PREDICTIONS.csv", pd.concat(predictions, ignore_index=True))
    if selections:
        write_csv(DIAGNOSTICS / "WBCIC_DEV_CALIBRATION_SELECTION.csv", pd.concat(selections, ignore_index=True))

    openbmi = pd.read_csv(LEADERBOARD / "OPENBMI_MODEL_LEADERBOARD.csv")
    open_best = openbmi.iloc[0]
    w_transfer = leaderboard[leaderboard.method_id.eq("W1_MASKED_POOL_SHRUNK_THR")].iloc[0]
    cross = pd.DataFrame(
        [
            {
                "candidate": "STATIC_STRONG_REFERENCE",
                "OpenBMI_method": "M0_B_STRONG_B6",
                "OpenBMI_Delta_BA": 0.0,
                "WBCIC_method": static_best_id,
                "WBCIC_Delta_BA_vs_static": 0.0,
                "worst_benchmark_gain": 0.0,
            },
            {
                "candidate": "TRANSFERRED_MASKED_POSITIVE_POOL",
                "OpenBMI_method": open_best.method_id,
                "OpenBMI_Delta_BA": float(open_best.Delta_BA_vs_B_STRONG),
                "WBCIC_method": "W1_MASKED_POOL_SHRUNK_THR",
                "WBCIC_Delta_BA_vs_static": float(w_transfer.Delta_BA_vs_WBCIC_B_STRONG),
                "worst_benchmark_gain": min(
                    float(open_best.Delta_BA_vs_B_STRONG), float(w_transfer.Delta_BA_vs_WBCIC_B_STRONG)
                ),
            },
        ]
    )
    cross["mean_normalized_gain"] = cross[["OpenBMI_Delta_BA", "WBCIC_Delta_BA_vs_static"]].mean(axis=1)
    cross["OUTER_TEST_USED"] = False
    write_csv(LEADERBOARD / "CROSS_BENCHMARK_LEADERBOARD.csv", cross)
    decision = {
        "status": "WBCIC_DEVELOPMENT_TRANSFER_COMPLETE",
        "WBCIC_B_STRONG": static_best_id,
        "WBCIC_B_STRONG_BA": float(
            leaderboard[leaderboard.method_id.eq(static_best_id)].iloc[0].mean_subject_BA
        ),
        "transfer_method": "W1_MASKED_POOL_SHRUNK_THR",
        "transfer_BA": float(w_transfer.mean_subject_BA),
        "transfer_delta_vs_WBCIC_B_STRONG": float(w_transfer.Delta_BA_vs_WBCIC_B_STRONG),
        "transfer_CI95_vs_WBCIC_B_STRONG": [
            float(w_transfer.CI95_L_vs_WBCIC_B_STRONG),
            float(w_transfer.CI95_U_vs_WBCIC_B_STRONG),
        ],
        "outer_subject_ids_loaded": False,
        "OUTER_TEST_USED": False,
    }
    write_json(OUTPUTS / "diagnostics" / "WBCIC_TRANSFER_DECISION.json", decision)
    print(leaderboard[["method_id", "mean_subject_BA", "Delta_BA_vs_WBCIC_B_STRONG", "CI95_L_vs_WBCIC_B_STRONG", "CI95_U_vs_WBCIC_B_STRONG"]].to_string(index=False))
    return leaderboard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expert-table", type=Path, default=OUTPUTS / "cache" / "WBCIC_DEV_KEEP_EXPERTS.parquet"
    )
    parser.add_argument("--wbcic-repo", type=Path, default=default_wbcic_repo())
    args = parser.parse_args()
    run(args.expert_table, args.wbcic_repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
