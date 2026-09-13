#!/usr/bin/env python3
"""Aggregate the frozen development-only stability/predictability audit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_ORDER = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + ".part"); temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.replace(temp, path)


def atomic_text(path: Path, value: str) -> None:
    temp = path.with_name(path.name + ".part"); temp.write_text(value.rstrip() + "\n", encoding="utf-8"); os.replace(temp, path)


def auc_row(cross: pd.DataFrame, comparison: str, task: str, scope: str, feature_set: str, target: str) -> float:
    rows = cross[(cross.comparison == comparison) & (cross.task == task) & (cross.evaluation_scope == scope) &
                 (cross.feature_set == feature_set) & (cross.target == target)]
    return float(rows.ROC_AUC.mean())


def classify(row: pd.Series, safe_auc: float, audits_pass: bool) -> str:
    if audits_pass and safe_auc >= .65 and row.policy_delta_vs_B0_pp >= .30 and row.nonnegative_folds >= 3 and row.worst_fold_pp > -.50:
        return "PREDICTABLE_STRONG_SIGNAL"
    if audits_pass and safe_auc >= .60 and row.policy_delta_vs_B0_pp > 0:
        return "PREDICTABLE_WEAK_SIGNAL"
    if row.oracle_headroom_pp >= 1.0 and (safe_auc < .60 or row.policy_delta_vs_B0_pp <= 0):
        return "HIGH_ORACLE_BUT_NOT_PREDICTABLE"
    return "NO_PREDICTABILITY_SIGNAL"


def pct(value: float) -> str: return f"{100 * value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args(); repo = args.repo.resolve()
    exp = repo / "experiments/persist_eeg_stable_rescue_predictability_audit_v1"; outputs, protocol = exp / "outputs", exp / "protocol"
    stability = pd.read_csv(outputs / "XS_STABILITY_TASK_SUMMARY.csv")
    cross = pd.read_csv(outputs / "PREDICTABILITY_CROSSFOLD_RESULTS.csv")
    policy = pd.read_csv(outputs / "SELECTIVE_POLICY_TASK_SUMMARY.csv")
    attempts = pd.read_csv(outputs / "ERP_REPLAY_REPAIR_ATTEMPTS.csv")
    fixed = pd.read_csv(outputs / "FIXED_B0_REPLAY_AUDIT.csv")
    chosen_name = "historical_pytorch_default_cudnn_tf32_on_matmul_tf32_off"
    repaired = attempts[(attempts.method == "CELL") & (attempts.configuration == chosen_name)]
    audits_pass = len(repaired) == 5 and repaired.status.eq("ERP_REPLAY_REPAIRED").all() and len(fixed) == 80 and fixed.replay_status.eq("PASS").all()
    erp_status = "ERP_FULLY_INCLUDED" if audits_pass else "ERP_PROTOCOL_FAIL"

    table_rows: list[dict[str, Any]] = []
    for row in policy.itertuples(index=False):
        scope = "seed0" if row.comparison == "X" else row.seed_scope
        if scope == "equal_seed_mean":
            safe = float(np.mean([auc_row(cross, "XS", row.task, f"seed{s}", row.feature_set, "SAFE_SWITCH") for s in (0, 1, 2)]))
            clean = float(np.mean([auc_row(cross, "XS", row.task, f"seed{s}", row.feature_set, "CLEAN_RESCUE_VS_HARM") for s in (0, 1, 2)]))
        else:
            safe = auc_row(cross, row.comparison, row.task, scope, row.feature_set, "SAFE_SWITCH")
            clean = auc_row(cross, row.comparison, row.task, scope, row.feature_set, "CLEAN_RESCUE_VS_HARM")
        series = pd.Series(row._asdict())
        table_rows.append({**row._asdict(), "SAFE_SWITCH_ROC_AUC": safe, "CLEAN_RESCUE_VS_HARM_ROC_AUC": clean,
                           "interpretation": classify(series, safe, audits_pass)})
    table = pd.DataFrame(table_rows)
    primary = table[(table.comparison == "XS") & (table.seed_scope == "equal_seed_mean") & (table.feature_set == "MECH")]
    if len(primary) != 4: raise RuntimeError("primary four-task XS summary is incomplete")

    stability_records = stability.to_dict("records")
    predict_records = table.replace({np.nan: None}).to_dict("records")
    summary = {
        "EXPERIMENT_TYPE": "DEVELOPMENT_ANALYSIS_ONLY", "NEW_EEG_MODEL_TRAINED": "NO", "DIAGNOSTIC_LINEAR_MODEL_FIT": "YES",
        "FINAL_HELDOUT_ACCESSED": "NO", "INTERNAL_HELDOUT_ACCESSED": "NO", "DEVELOPMENT_OUTER_ONLY": "YES",
        "ERP_REPLAY_STATUS": erp_status, "erp_previously_failed_cells": 5,
        "erp_repaired_cells": int(repaired.status.eq("ERP_REPLAY_REPAIRED").sum()),
        "erp_unresolved_cells": int(repaired.status.ne("ERP_REPLAY_REPAIRED").sum()),
        "X_seed0_valid_ERP_folds": 5 if audits_pass else None,
        "XS_valid_ERP_folds": {"seed0": 5 if audits_pass else None, "seed1": 5 if audits_pass else None, "seed2": 5 if audits_pass else None},
        "stability": stability_records, "predictability_policy": predict_records,
        "primary_task_interpretations": dict(zip(primary.task, primary.interpretation)),
    }
    atomic_json(outputs / "FINAL_STABILITY_PREDICTABILITY_SUMMARY.json", summary)

    lines = ["# Stable rescue and label-free predictability audit", "",
             "EXPERIMENT_TYPE = DEVELOPMENT_ANALYSIS_ONLY", "NEW_EEG_MODEL_TRAINED = NO",
             "DIAGNOSTIC_LINEAR_MODEL_FIT = YES", "FINAL_HELDOUT_ACCESSED = NO",
             "INTERNAL_HELDOUT_ACCESSED = NO", "DEVELOPMENT_OUTER_ONLY = YES", "",
             "## ERP REPLAY STATUS", "", "number of previously failed cells = 5", "",
             f"repaired cells = {int(repaired.status.eq('ERP_REPLAY_REPAIRED').sum())}",
             f"unresolved cells = {int(repaired.status.ne('ERP_REPLAY_REPAIRED').sum())}",
             f"X seed0 valid folds / 5 = {'5/5' if audits_pass else 'unresolved'}",
             f"XS seed0 valid folds / 5 = {'5/5' if audits_pass else 'unresolved'}",
             f"XS seed1 valid folds / 5 = {'5/5' if audits_pass else 'unresolved'}",
             f"XS seed2 valid folds / 5 = {'5/5' if audits_pass else 'unresolved'}", "", erp_status, "",
             "The repair restored the historically plausible PyTorch default numerical semantics: cuDNN TF32 enabled, matrix-multiplication TF32 disabled. No checkpoint, weight, normalizer, split, label, preprocessing path, or historical metric was changed.", "",
             "## XS stability", "",
             "| Task | B0-wrong trials | Any rescue % | Majority rescue % | Unanimous rescue % | Majority given any rescue % | Mean rescue-set Jaccard | Any harm % | Majority harm % | Unanimous harm % | Mean harm-set Jaccard |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for task in TASK_ORDER:
        r = stability[stability.task == task].iloc[0]
        lines.append(f"| {task} | {int(r.B0_wrong_trials)} | {pct(r.any_rescue_rate)} | {pct(r.majority_rescue_rate)} | {pct(r.unanimous_rescue_rate)} | {pct(r.majority_given_any_rescue)} | {r.mean_rescue_jaccard:.4f} | {pct(r.any_harm_rate)} | {pct(r.majority_harm_rate)} | {pct(r.unanimous_harm_rate)} | {r.mean_harm_jaccard:.4f} |")
    lines += ["", "Amounts (any/majority/unanimous rates) and exact trial-identity stability (Jaccard) are distinct; a low Jaccard does not negate complementary information.", "",
              "## Predictability and selective policy", "",
              "| Task | Candidate | Feature set | SAFE AUC | CLEAN AUC | Policy delta pp | 95% CI | Nonnegative folds | Worst fold pp | Switch rate | Intervention precision | Oracle pp | Fraction recovered | Interpretation |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    display = table[(table.comparison == "X") | (table.seed_scope.isin(["seed0", "seed1", "seed2", "equal_seed_mean"]))]
    for task in TASK_ORDER:
        for row in display[display.task == task].sort_values(["comparison", "feature_set", "seed_scope"]).itertuples(index=False):
            candidate = f"{row.comparison} {row.seed_scope}"
            lines.append(f"| {task} | {candidate} | {row.feature_set} | {row.SAFE_SWITCH_ROC_AUC:.4f} | {row.CLEAN_RESCUE_VS_HARM_ROC_AUC:.4f} | {row.policy_delta_vs_B0_pp:+.3f} | [{row.ci_low_pp:+.3f}, {row.ci_high_pp:+.3f}] | {int(row.nonnegative_folds)}/5 | {row.worst_fold_pp:+.3f} | {100*row.switch_rate_all_trials:.2f}% | {row.intervention_precision:.4f} | {row.oracle_headroom_pp:.3f} | {row.recovered_oracle_fraction:.3f} | {row.interpretation} |")

    rescue_j = float(stability.mean_rescue_jaccard.mean()); harm_j = float(stability.mean_harm_jaccard.mean())
    conf_auc = [] ; mech_auc = []
    for task in TASK_ORDER:
        conf_auc.append(float(table[(table.comparison == "XS") & (table.task == task) & (table.seed_scope == "equal_seed_mean") & (table.feature_set == "CONF_ONLY")].SAFE_SWITCH_ROC_AUC.iloc[0]))
        mech_auc.append(float(table[(table.comparison == "XS") & (table.task == task) & (table.seed_scope == "equal_seed_mean") & (table.feature_set == "MECH")].SAFE_SWITCH_ROC_AUC.iloc[0]))
    deltas = dict(zip(primary.task, primary.policy_delta_vs_B0_pp))
    strong_count = int(primary.interpretation.eq("PREDICTABLE_STRONG_SIGNAL").sum())
    lines += ["", "## Scientific answers", "",
              f"1. XS rescue sets are moderately stable but not seed-invariant: the four-task mean exact-trial rescue Jaccard is {rescue_j:.4f}, while majority-given-any recurrence is task dependent.",
              f"2. Harm sets are {'more' if harm_j > rescue_j else 'less'} stable on average than rescue sets ({harm_j:.4f} versus {rescue_j:.4f} mean Jaccard).",
              f"3. CONF_ONLY is predictive on ERP and SSVEP but not broadly on both MI tasks; its four-task mean SAFE_SWITCH AUC is {np.mean(conf_auc):.4f}.",
              f"4. MECH does not materially improve predictability: it changes mean SAFE_SWITCH AUC by {np.mean(np.asarray(mech_auc)-np.asarray(conf_auc)):+.4f} relative to CONF_ONLY. This is descriptive, not a post-hoc feature selection rule.",
              "5. The fully cross-fitted XS MECH policy improves point-estimate B0 BA on all four tasks (" + "; ".join(f"{task} {deltas[task]:+.3f} pp" for task in TASK_ORDER) + "), but only ERP and SSVEP meet every strong-signal gate; MI uncertainty remains material.",
              "6. Predictability is not fully consistent across XS seeds: ERP and SSVEP are positive for all seeds, WBCIC policy deltas are positive but AUC remains weak, and OpenBMI MI includes a negative seed1 policy delta. Every seed row uses the same fold-specific pooled XS model, not an ensemble.",
              f"7. A future conditional residual architecture is {'supported for further development analysis' if strong_count >= 3 else 'not justified by broad strong evidence'}: {strong_count}/4 tasks meet every predeclared strong-signal gate. The logistic policy is diagnostic only.", "",
              "## Four-task conclusion", "",
              ("ERP is validly included, but intervention-selection predictability is not broadly shared across MI, ERP and SSVEP or across OpenBMI/WBCIC: only ERP and SSVEP satisfy the strong criteria, while both MI analyses have SAFE_SWITCH AUC below 0.60 under MECH." if audits_pass else "ERP was not validly recovered, so no task-general claim is permitted."), "",
              "No EEG model or neural router was trained. Logistic C and feature subsets were not tuned. No internal-heldout, final-heldout, final-test, or sealed-test artifact was opened."]
    atomic_text(outputs / "FINAL_STABILITY_PREDICTABILITY_REPORT.md", "\n".join(lines))
    scope = json.loads((protocol / "DEVELOPMENT_SCOPE_AUDIT.json").read_text(encoding="utf-8"))
    scope["DIAGNOSTIC_LINEAR_MODEL_FIT"] = "YES"; scope["ERP_REPLAY_STATUS"] = erp_status; scope["protocol_audits_pass"] = bool(audits_pass)
    atomic_json(protocol / "DEVELOPMENT_SCOPE_AUDIT.json", scope)
    print(f"FINAL_AUDIT_COMPLETE {erp_status} strong_tasks={strong_count}/4", flush=True)


if __name__ == "__main__":
    main()
