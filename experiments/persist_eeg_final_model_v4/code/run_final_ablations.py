from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import DIAGNOSTICS, OUTPUTS, default_openbmi_cache, ensure_directories, write_csv, write_json
from datasets import load_openbmi
from evaluation import paired_subject_bootstrap, summarize_method
from features import build_openbmi_features, select_features
from models import linear, stacking
from training import baseline_result, run_nested_oof


def _generic_capacity_features(values: np.ndarray, extra_columns: int) -> np.ndarray:
    """Add deterministic non-structural transforms to match PERSIST input width."""
    source = np.asarray(values, dtype=float)
    generated = []
    for index in range(extra_columns):
        left = source[:, index % source.shape[1]]
        right = source[:, (index * 7 + 3) % source.shape[1]]
        mode = index % 3
        if mode == 0:
            value = np.sign(left) * np.log1p(np.abs(left))
        elif mode == 1:
            value = np.clip(left, -10.0, 10.0) * np.clip(right, -10.0, 10.0)
        else:
            value = np.square(np.clip(left, -10.0, 10.0))
        generated.append(value)
    return np.column_stack([source, *generated]) if generated else source.copy()


def run(cache_root: Path) -> pd.DataFrame:
    ensure_directories()
    data = load_openbmi(cache_root)
    bundle = build_openbmi_features(data)
    raw_logits = np.where(data.keep_run_mask, data.keep_run_logits, data.base_logits[:, None])
    session_values = sorted(np.unique(data.sessions).tolist())
    session_onehot = np.column_stack([(data.sessions == value).astype(float) for value in session_values])
    positive_pool_x = np.column_stack([raw_logits, data.keep_run_mask.astype(float), session_onehot])
    stable_configs = [
        {"l2": l2, "learn_scale": True, "session_specific": False}
        for l2 in (0.01, 0.1, 1.0)
    ]
    jobs = [
        (
            "A1_DYNAMIC_KEEP_FINAL",
            positive_pool_x,
            "stacking",
            stable_configs,
            stacking.build,
            (0.475, 0.5, 0.525),
        ),
        ("A2_KEEP_ACTION_NO_PERSIST", bundle.matrices["KEEP_ACTION"], "linear", linear.configurations(), linear.build, None),
        ("A3_KEEP_ACTION_PERSIST", bundle.matrices["KEEP_ACTION_PERSIST"], "linear", linear.configurations(), linear.build, None),
    ]
    exclusions = {
        "A4_WITHOUT_PROTECTED": bundle.categories["protected"],
        "A5_WITHOUT_DECISION_DEPENDENCE": bundle.categories["decision_dependence"],
        "A6_WITHOUT_PERSISTENCE": bundle.categories["persistence"],
        "A7_WITHOUT_ACTION_MOVEMENT": bundle.categories["action_movement"],
        "A8_WITHOUT_ENSEMBLE_DISAGREEMENT": bundle.categories["ensemble_disagreement"],
    }
    for method_id, excluded in exclusions.items():
        values, _ = select_features(bundle, "KEEP_ACTION_PERSIST", excluded)
        jobs.append((method_id, values, "linear", linear.configurations(), linear.build, None))
    generic = _generic_capacity_features(
        bundle.matrices["KEEP_ACTION"], len(bundle.names["KEEP_ACTION_PERSIST"]) - len(bundle.names["KEEP_ACTION"])
    )
    if generic.shape[1] != bundle.matrices["KEEP_ACTION_PERSIST"].shape[1]:
        raise RuntimeError("A9 generic capacity control width mismatch")
    jobs.append(("A9_CAPACITY_MATCHED_GENERIC", generic, "linear", linear.configurations(), linear.build, None))

    results = [baseline_result(data)]
    results[0].method_id = "A0_STATIC_B_STRONG"
    for method_id, values, family, configurations, builder, thresholds in jobs:
        print(f"[V4 final ablation] {method_id}", flush=True)
        kwargs = {} if thresholds is None else {"thresholds": thresholds}
        results.append(
            run_nested_oof(data, method_id, values, family, configurations, builder, **kwargs)
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
    leaderboard = pd.DataFrame(rows).sort_values("method_id").reset_index(drop=True)
    subject_table = pd.concat(subjects, ignore_index=True)
    write_csv(OUTPUTS / "ablations" / "FINAL_MODEL_ABLATIONS.csv", leaderboard)
    write_csv(DIAGNOSTICS / "FINAL_ABLATION_SUBJECT_RESULTS.csv", subject_table)
    write_csv(DIAGNOSTICS / "FINAL_ABLATION_FOLD_RESULTS.csv", pd.concat(folds, ignore_index=True))
    write_csv(DIAGNOSTICS / "FINAL_ABLATION_SELECTIONS.csv", pd.concat(selections, ignore_index=True))
    write_csv(DIAGNOSTICS / "FINAL_ABLATION_OOF_PREDICTIONS.csv", pd.concat(predictions, ignore_index=True))

    comparisons = [
        ("dynamic_KEEP_value", "A1_DYNAMIC_KEEP_FINAL", "A0_STATIC_B_STRONG"),
        ("ACTION_increment", "A2_KEEP_ACTION_NO_PERSIST", "A1_DYNAMIC_KEEP_FINAL"),
        ("PERSIST_increment", "A3_KEEP_ACTION_PERSIST", "A2_KEEP_ACTION_NO_PERSIST"),
        ("protected_increment", "A3_KEEP_ACTION_PERSIST", "A4_WITHOUT_PROTECTED"),
        ("decision_dependence_increment", "A3_KEEP_ACTION_PERSIST", "A5_WITHOUT_DECISION_DEPENDENCE"),
        ("persistence_increment", "A3_KEEP_ACTION_PERSIST", "A6_WITHOUT_PERSISTENCE"),
        ("action_movement_increment", "A3_KEEP_ACTION_PERSIST", "A7_WITHOUT_ACTION_MOVEMENT"),
        ("disagreement_increment", "A3_KEEP_ACTION_PERSIST", "A8_WITHOUT_ENSEMBLE_DISAGREEMENT"),
        ("PERSIST_vs_capacity_control", "A3_KEEP_ACTION_PERSIST", "A9_CAPACITY_MATCHED_GENERIC"),
    ]
    incremental_rows = []
    for name, numerator, denominator in comparisons:
        high = subject_table[subject_table.method_id.eq(numerator)].set_index("subject_id")
        low = subject_table[subject_table.method_id.eq(denominator)].set_index("subject_id")
        delta = high.loc[low.index, "balanced_accuracy"].to_numpy(dtype=float) - low.balanced_accuracy.to_numpy(dtype=float)
        lower, upper = paired_subject_bootstrap(delta, f"V4_ABLATION_{name}")
        high_row = leaderboard[leaderboard.method_id.eq(numerator)].iloc[0]
        low_row = leaderboard[leaderboard.method_id.eq(denominator)].iloc[0]
        incremental_rows.append(
            {
                "comparison": name,
                "numerator": numerator,
                "denominator": denominator,
                "Delta_BA": float(delta.mean()),
                "CI95_L": lower,
                "CI95_U": upper,
                "Delta_worst_subject": float(high_row.worst_subject_delta - low_row.worst_subject_delta),
                "Delta_harm_rate": float(high_row.harm_rate - low_row.harm_rate),
                "Delta_NLL": float(high_row.NLL - low_row.NLL),
                "OUTER_TEST_USED": False,
            }
        )
    incremental = pd.DataFrame(incremental_rows)
    write_csv(OUTPUTS / "ablations" / "PERSIST_INCREMENTAL_VALUE.csv", incremental)
    persist_row = incremental[incremental.comparison.eq("PERSIST_increment")].iloc[0]
    write_json(
        OUTPUTS / "ablations" / "FINAL_ABLATION_DECISION.json",
        {
            "status": "FINAL_ABLATIONS_COMPLETE",
            "selected_final_method": "A1_DYNAMIC_KEEP_FINAL",
            "PERSIST_increment_Delta_BA": float(persist_row.Delta_BA),
            "PERSIST_increment_CI95": [float(persist_row.CI95_L), float(persist_row.CI95_U)],
            "PERSIST_raw_gain_supported": bool(persist_row.Delta_BA > 0 and persist_row.CI95_L > 0),
            "OUTER_TEST_USED": False,
        },
    )
    print(leaderboard[["method_id", "mean_subject_BA", "Delta_BA_vs_B_STRONG", "CI95_L", "CI95_U", "harm_rate", "worst_subject_delta"]].to_string(index=False))
    print(incremental.to_string(index=False))
    return leaderboard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openbmi-cache", type=Path, default=default_openbmi_cache())
    args = parser.parse_args()
    run(args.openbmi_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
