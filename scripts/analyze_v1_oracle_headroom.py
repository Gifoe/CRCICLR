#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hsc_tta.v2.access_guard import OldFinalAccessGuard


ROOT = Path("/root/autodl-tmp/hsc_tta_eeg")
V1 = ROOT / "outputs" / "full_experiment"
OUT = ROOT / "outputs" / "v2_joint_certified" / "diagnostics"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    guard = OldFinalAccessGuard(ROOT)
    counter = guard.read_parquet(V1 / "ALL_COUNTERFACTUAL_ACTION_OUTCOMES.parquet", purpose="v1_oracle_diagnostic")
    decisions = guard.read_parquet(V1 / "ALL_SUBJECT_DECISIONS.parquet", purpose="v1_oracle_diagnostic")
    predictions = pd.read_parquet(V1 / "ALL_CRITICAL_INDEX_PREDICTIONS.parquet")
    residuals = pd.read_parquet(V1 / "ALL_CALIBRATION_RESIDUALS.parquet")
    actions = ["no_tta", "t3a", "entropy_adapter"]
    cost = {name: index for index, name in enumerate(actions)}
    subject_rows, summary_rows, safe_rows, regret_rows = [], [], [], []
    for keys, group in counter.groupby(["dataset", "seed", "alpha"], sort=True):
        dataset, seed, alpha = keys
        no = group[group.action == "no_tta"].set_index("subject_id")
        local = group.copy()
        local["gain_vs_no_tta"] = local["subject_id"].map(no["argmax_error"]) - local["argmax_error"]
        local["action_cost"] = local.action.map(cost)
        current_decisions = decisions[(decisions.dataset == dataset) & (decisions.seed == seed) & np.isclose(decisions.alpha, alpha)]
        selected = current_decisions.set_index("subject_id")
        for subject, sg in local.groupby("subject_id", sort=True):
            oracle = sg.sort_values(["argmax_error", "action_cost", "action"]).iloc[0]
            genuinely_safe = sg[(sg.true_future_risk <= alpha) & sg.nontrivial_candidate.astype(bool) & (sg["lambda"] < 1)]
            safe = genuinely_safe.sort_values(["argmax_error", "action_cost", "action"]).iloc[0] if len(genuinely_safe) else sg[sg.action == "no_tta"].iloc[0]
            certified = sg[sg.nontrivial_candidate.astype(bool)]
            certified_best = certified.sort_values(["argmax_error", "action_cost", "action"]).iloc[0] if len(certified) else sg[sg.action == "no_tta"].iloc[0]
            chosen = selected.loc[subject]
            chosen_outcome = sg[sg.action == chosen.selected_action].iloc[0]
            row = {"dataset": dataset, "seed": int(seed), "alpha": float(alpha), "subject_id": subject,
                   "no_tta_error": float(no.loc[subject, "argmax_error"]),
                   "oracle_action": oracle.action, "oracle_error": float(oracle.argmax_error),
                   "oracle_gain": float(oracle.gain_vs_no_tta),
                   "any_tta_beneficial": bool((sg[sg.action != "no_tta"].gain_vs_no_tta > 0).any()),
                   "best_tta_gain": float(sg[sg.action != "no_tta"].gain_vs_no_tta.max()),
                   "worst_tta_gain": float(sg[sg.action != "no_tta"].gain_vs_no_tta.min()),
                   "safe_oracle_action": safe.action, "safe_oracle_gain": float(safe.gain_vs_no_tta),
                   "safe_oracle_csr": bool(len(genuinely_safe)), "safe_oracle_set_size": float(safe.future_average_set_size),
                   "safe_and_beneficial": bool(len(genuinely_safe) and safe.gain_vs_no_tta > 0),
                   "certified_oracle_action": certified_best.action,
                   "certified_oracle_gain": float(certified_best.gain_vs_no_tta),
                   "hsc_action": chosen.selected_action, "hsc_gain": float(chosen_outcome.gain_vs_no_tta),
                   "hsc_safe_oracle_regret": float(safe.gain_vs_no_tta - chosen_outcome.gain_vs_no_tta)}
            subject_rows.append(row)
        frame = pd.DataFrame([r for r in subject_rows if (r["dataset"], r["seed"], r["alpha"]) == keys])
        action_rates = frame.oracle_action.value_counts(normalize=True).to_dict()
        summary_rows.append({"dataset": dataset, "seed": seed, "alpha": alpha,
            "oracle_argmax_error": frame.oracle_error.mean(), "oracle_gain_vs_no_tta": frame.oracle_gain.mean(),
            "at_least_one_tta_beneficial_rate": frame.any_tta_beneficial.mean(),
            **{f"oracle_{a}_rate": action_rates.get(a, 0.0) for a in actions},
            "tta_improvement_mean": frame.best_tta_gain.mean(), "tta_harm_mean": frame.worst_tta_gain.mean()})
        safe_rates = frame.safe_oracle_action.value_counts(normalize=True).to_dict()
        safe_rows.append({"dataset": dataset, "seed": seed, "alpha": alpha,
            "safe_oracle_gain_vs_no_tta": frame.safe_oracle_gain.mean(),
            "safe_and_beneficial_subject_rate": frame.safe_and_beneficial.mean(),
            "safe_oracle_csr": frame.safe_oracle_csr.mean(), "safe_oracle_set_size": frame.safe_oracle_set_size.mean(),
            **{f"safe_oracle_{a}_rate": safe_rates.get(a, 0.0) for a in actions}})
        regret_rows.append({"dataset": dataset, "seed": seed, "alpha": alpha,
            "hsc_gain": frame.hsc_gain.mean(), "safe_oracle_gain": frame.safe_oracle_gain.mean(),
            "regret": frame.hsc_safe_oracle_regret.mean(), "certified_oracle_gain": frame.certified_oracle_gain.mean()})
    subjects = pd.DataFrame(subject_rows)
    subjects.to_parquet(OUT / "V1_ORACLE_HEADROOM_BY_SUBJECT.parquet", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "V1_ORACLE_HEADROOM_SUMMARY.csv", index=False)
    pd.DataFrame(safe_rows).to_csv(OUT / "V1_SAFE_ORACLE_SUMMARY.csv", index=False)
    pd.DataFrame(regret_rows).to_csv(OUT / "V1_POLICY_REGRET.csv", index=False)
    pred_residual = predictions["critical_index"] - predictions["predicted_critical_index"]
    shift = ["# V1 residual shift audit", "",
        f"Predictor rows: {len(predictions)}; calibration subject-score rows: {len(residuals)}.",
        f"Prediction residual MAE: {pred_residual.abs().mean():.4f}.",
        f"Critical-index underestimation frequency: {(pred_residual > 0).mean():.4f}.",
        f"Calibration residual mean / 90th percentile: {residuals.residual.mean():.4f} / {residuals.residual.quantile(.9):.4f}.",
        "Final-test critical-index residual shift cannot be reconstructed from the v1 aggregate file because it stores only each action's selected lambda outcome, not the complete final-test lambda curve. It is not fabricated.",
        "Repeated-subject dependence across seeds is retained and must not be treated as independent episodes."]
    (OUT / "V1_RESIDUAL_SHIFT_REPORT.md").write_text("\n\n".join(shift) + "\n", encoding="utf-8")
    aggregate = subjects.groupby("dataset").agg(
        beneficial_headroom=("any_tta_beneficial", "mean"), safe_beneficial=("safe_and_beneficial", "mean"),
        oracle_gain=("oracle_gain", "mean"), safe_gain=("safe_oracle_gain", "mean"),
        certified_gain=("certified_oracle_gain", "mean"), hsc_gain=("hsc_gain", "mean"),
        regret=("hsc_safe_oracle_regret", "mean"), no_tta_error=("no_tta_error", "mean"),
    ).reset_index()
    diagnosis = "# V1 failure diagnosis\n\n" + aggregate.to_markdown(index=False) + "\n\n"
    diagnosis += (
        "Interpretation rules: oracle minus safe-oracle gain estimates loss to the risk constraint; safe-oracle minus "
        "certified-oracle gain estimates certificate conservatism; certified-oracle minus HSC gain estimates selector weakness. "
        "Low No-TTA quality, especially EEGMMIDB, is a source-model qualification failure rather than a selector result.\n"
    )
    (OUT / "V1_FAILURE_DIAGNOSIS.md").write_text(diagnosis, encoding="utf-8")
    tainted = {dataset: sorted(values.subject_id.unique().tolist()) for dataset, values in decisions.groupby("dataset")}
    provenance = ROOT / "outputs" / "v2_joint_certified" / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    (provenance / "OLD_FINAL_TEST_TAINTED.json").write_text(json.dumps({
        "status": "tainted_already_observed", "v1_method_commit": "28fe62593a30833acd6b925317d6645ed6c15a04",
        "subjects": tainted, "reason": "v1 final-test outcomes were inspected before v2 development",
        "allowed_pre_freeze_stage": "v1_oracle_diagnostic_only",
        "forbidden_stages": ["source_head_selection", "action_selection", "predictor_training", "scale_estimation", "calibration", "selector_tuning"],
        "post_freeze_use": "exploratory replication only; never confirmatory"
    }, indent=2), encoding="utf-8")
    print(aggregate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
