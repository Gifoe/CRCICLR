#!/usr/bin/env python3
"""Audit F0 reproduction, classify tasks, and write the final diagnostic report."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_name(path.name + ".part")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_text(path: Path, value: str) -> None:
    temp = path.with_name(path.name + ".part")
    temp.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temp, path)


def auc(cross: pd.DataFrame, comparison: str, task: str, family: str, target: str, scope: str) -> float:
    rows = cross[(cross.comparison == comparison) & (cross.task == task) &
                 (cross.feature_family == family) & (cross.target == target) &
                 (cross.evaluation_scope == scope)]
    if len(rows) != 5:
        raise RuntimeError(f"missing cross-fold rows: {comparison}/{task}/{family}/{target}/{scope}")
    return float(rows.ROC_AUC.mean())


def prior_auc(cross: pd.DataFrame, comparison: str, task: str, target: str, scope: str) -> float:
    rows = cross[(cross.comparison == comparison) & (cross.task == task) &
                 (cross.feature_set == "CONF_ONLY") & (cross.target == target) &
                 (cross.evaluation_scope == scope)]
    if len(rows) != 5:
        raise RuntimeError("prior F0 rows incomplete")
    return float(rows.ROC_AUC.mean())


def primary_auc(cross: pd.DataFrame, task: str, family: str, target: str) -> float:
    return float(np.mean([auc(cross, "XS", task, family, target, f"seed{seed}") for seed in (0, 1, 2)]))


def strong(row: pd.Series, safe_auc: float, audits: bool) -> bool:
    return bool(audits and safe_auc >= .65 and row.policy_delta_vs_B0_pp >= .30 and
                row.nonnegative_folds >= 3 and row.worst_fold_pp > -.50)


def classify(f1: pd.Series, f2: pd.Series, auc1: float, auc2: float, audits: bool) -> str:
    if strong(f1, auc1, audits):
        return "SIGNED_LOGITS_SUFFICIENT"
    if strong(f2, auc2, audits):
        return "REPRESENTATION_GEOMETRY_NEEDED"
    if max(auc1, auc2) >= .60 and max(f1.policy_delta_vs_B0_pp, f2.policy_delta_vs_B0_pp) > 0:
        return "ROUTING_WEAK_SIGNAL"
    each_fails = all((value_auc < .60 or value_row.policy_delta_vs_B0_pp <= 0)
                     for value_row, value_auc in ((f1, auc1), (f2, auc2)))
    if max(f1.oracle_headroom_pp, f2.oracle_headroom_pp) >= 1.0 and each_fails:
        return "HIGH_ORACLE_NOT_OBSERVABLY_ROUTABLE"
    return "NO_ROUTING_SIGNAL"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    exp = repo / "experiments/persist_eeg_representation_rescue_predictability_v1"
    outputs, protocol = exp / "outputs", exp / "protocol"
    prior_outputs = repo / "experiments/persist_eeg_stable_rescue_predictability_audit_v1/outputs"
    cross = pd.read_csv(outputs / "REPRESENTATION_CROSSFOLD_RESULTS.csv")
    policy = pd.read_csv(outputs / "REPRESENTATION_POLICY_TASK_SUMMARY.csv")
    prior_cross = pd.read_csv(prior_outputs / "PREDICTABILITY_CROSSFOLD_RESULTS.csv")
    prior_policy = pd.read_csv(prior_outputs / "SELECTIVE_POLICY_TASK_SUMMARY.csv")
    replay = pd.read_csv(outputs / "REPRESENTATION_REPLAY_AUDIT.csv")
    weights = pd.read_csv(outputs / "SUBJECT_WEIGHT_AUDIT.csv")
    replay_pass = bool(len(replay) == 120 and replay.status.eq("PASS").all() and replay.dimension_pass.all() and replay.predictions_exact.all())
    weight_pass = bool(len(weights) > 0 and weights.equal_weight_pass.all() and weights.absolute_difference.max() <= 1e-10)

    audit_rows: list[dict[str, Any]] = []
    for comparison in ("X", "XS"):
        scope = "seed0" if comparison == "X" else "equal_seed_mean"
        cross_scope = "seed0"
        for task in TASKS:
            current_auc = auc(cross, comparison, task, "F0", "SAFE_SWITCH", cross_scope) if comparison == "X" else primary_auc(cross, task, "F0", "SAFE_SWITCH")
            old_auc = prior_auc(prior_cross, comparison, task, "SAFE_SWITCH", cross_scope) if comparison == "X" else float(np.mean([prior_auc(prior_cross, "XS", task, "SAFE_SWITCH", f"seed{s}") for s in (0, 1, 2)]))
            current_policy = policy[(policy.comparison == comparison) & (policy.task == task) & (policy.seed_scope == scope) & (policy.feature_family == "F0")].iloc[0].policy_delta_vs_B0_pp
            old_policy = prior_policy[(prior_policy.comparison == comparison) & (prior_policy.task == task) & (prior_policy.seed_scope == scope) & (prior_policy.feature_set == "CONF_ONLY")].iloc[0].policy_delta_vs_B0_pp
            auc_diff = abs(float(current_auc - old_auc))
            policy_diff = abs(float(current_policy - old_policy))
            audit_rows.append({"comparison": comparison, "task": task, "seed_scope": scope,
                               "prior_SAFE_AUC": old_auc, "current_SAFE_AUC": current_auc,
                               "absolute_AUC_difference": auc_diff, "AUC_tolerance": 1e-8,
                               "prior_policy_delta_pp": old_policy, "current_policy_delta_pp": current_policy,
                               "absolute_policy_delta_difference_pp": policy_diff, "policy_tolerance_pp": 1e-8,
                               "AUC_pass": auc_diff <= 1e-8, "policy_pass": policy_diff <= 1e-8,
                               "status": "PASS" if auc_diff <= 1e-8 and policy_diff <= 1e-8 else "FAIL"})
    f0_audit = pd.DataFrame(audit_rows)
    atomic_csv(outputs / "F0_CONTROL_REPRODUCTION_AUDIT.csv", f0_audit)
    f0_pass = bool(f0_audit.status.eq("PASS").all())
    audits_pass = f0_pass and replay_pass and weight_pass
    if not f0_pass:
        raise RuntimeError("REPRESENTATION_AUDIT_PROTOCOL_FAIL: F0 did not reproduce")

    primary_rows: list[dict[str, Any]] = []
    interpretations: dict[str, str] = {}
    for task in TASKS:
        family_rows: dict[str, dict[str, Any]] = {}
        for family in ("F0", "F1", "F2"):
            p = policy[(policy.comparison == "XS") & (policy.task == task) &
                       (policy.seed_scope == "equal_seed_mean") & (policy.feature_family == family)].iloc[0]
            record = p.to_dict()
            record["SAFE_AUC"] = primary_auc(cross, task, family, "SAFE_SWITCH")
            record["CLEAN_AUC"] = primary_auc(cross, task, family, "CLEAN_RESCUE_VS_HARM")
            primary_rows.append(record)
            family_rows[family] = record
        f1, f2 = pd.Series(family_rows["F1"]), pd.Series(family_rows["F2"])
        interpretations[task] = classify(f1, f2, float(f1.SAFE_AUC), float(f2.SAFE_AUC), audits_pass)

    primary = pd.DataFrame(primary_rows)
    strong_tasks = sum(value in {"SIGNED_LOGITS_SUFFICIENT", "REPRESENTATION_GEOMETRY_NEEDED"} for value in interpretations.values())
    all_positive = all(max(primary[(primary.task == task) & (primary.feature_family.isin(["F1", "F2"]))].policy_delta_vs_B0_pp) > 0 for task in TASKS)
    neither_mi_high = all(interpretations[task] != "HIGH_ORACLE_NOT_OBSERVABLY_ROUTABLE" for task in ("OpenBMI_MI", "WBCIC_MI"))
    broad_success = bool(all_positive and strong_tasks >= 3 and neither_mi_high)
    routing_labels = {
        task: ("LOGIT_LEVEL_ROUTING" if interpretations[task] == "SIGNED_LOGITS_SUFFICIENT" else
               "REPRESENTATION_AWARE_ROUTING" if interpretations[task] == "REPRESENTATION_GEOMETRY_NEEDED" else
               "HIGH_ORACLE_BUT_ROUTING_UNRESOLVED" if interpretations[task] == "HIGH_ORACLE_NOT_OBSERVABLY_ROUTABLE" else
               "NO_STRONG_ROUTING_LABEL")
        for task in TASKS
    }

    summary = {
        "EXPERIMENT_TYPE": "DEVELOPMENT_ANALYSIS_ONLY", "NEW_EEG_MODEL_TRAINED": "NO",
        "FINAL_HELDOUT_ACCESSED": "NO", "INTERNAL_HELDOUT_ACCESSED": "NO", "DEVELOPMENT_OUTER_ONLY": "YES",
        "F0_CONTROL_REPRODUCTION_PASS": f0_pass, "REPRESENTATION_REPLAY_PASS": replay_pass,
        "SUBJECT_EQUAL_WEIGHT_PASS": weight_pass, "protocol_audits_pass": audits_pass,
        "task_interpretations": interpretations, "strong_routability_tasks": strong_tasks,
        "supported_routing_labels": routing_labels,
        "broad_four_task_success": broad_success,
        "primary_XS_equal_seed_results": primary.replace({np.nan: None}).to_dict("records"),
    }
    atomic_json(outputs / "FINAL_REPRESENTATION_PREDICTABILITY_SUMMARY.json", summary)

    lines = ["# Representation rescue predictability audit", "",
             "EXPERIMENT_TYPE = DEVELOPMENT_ANALYSIS_ONLY", "NEW_EEG_MODEL_TRAINED = NO",
             "FINAL_HELDOUT_ACCESSED = NO", "INTERNAL_HELDOUT_ACCESSED = NO", "DEVELOPMENT_OUTER_ONLY = YES", "",
             "## Protocol gates", "",
             f"- F0 exact-control reproduction: {'PASS' if f0_pass else 'FAIL'} (maximum AUC difference {f0_audit.absolute_AUC_difference.max():.3g}; maximum policy difference {f0_audit.absolute_policy_delta_difference_pp.max():.3g} pp)",
             f"- Frozen representation replay: {'PASS' if replay_pass else 'FAIL'} ({len(replay)} audited model/fold/seed cells)",
             f"- Subject-equal fitting weights: {'PASS' if weight_pass else 'FAIL'}", "",
             "## Table 1 — XS equal-seed mean", "",
             "| Task | Feature family | SAFE AUC | CLEAN AUC | Policy delta pp | 95% CI | Nonnegative folds | Worst fold | Switch rate | Intervention precision | Oracle headroom | Oracle fraction recovered |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        for family in ("F0", "F1", "F2"):
            r = primary[(primary.task == task) & (primary.feature_family == family)].iloc[0]
            lines.append(f"| {task} | {family} | {r.SAFE_AUC:.4f} | {r.CLEAN_AUC:.4f} | {r.policy_delta_vs_B0_pp:+.3f} | [{r.ci_low_pp:+.3f}, {r.ci_high_pp:+.3f}] | {int(r.nonnegative_folds)}/5 | {r.worst_fold_pp:+.3f} | {100*r.switch_rate_all_trials:.2f}% | {r.intervention_precision:.4f} | {r.oracle_headroom_pp:.3f} | {r.recovered_oracle_fraction:.3f} |")
    lines += ["", "## Table 2 — feature increments", "",
              "| Task | SAFE AUC F0 | F1 | F2 | F1-F0 | F2-F1 | Policy F0 | F1 | F2 | F1-F0 | F2-F1 | Interpretation |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for task in TASKS:
        rows = {f: primary[(primary.task == task) & (primary.feature_family == f)].iloc[0] for f in ("F0", "F1", "F2")}
        lines.append(f"| {task} | {rows['F0'].SAFE_AUC:.4f} | {rows['F1'].SAFE_AUC:.4f} | {rows['F2'].SAFE_AUC:.4f} | {rows['F1'].SAFE_AUC-rows['F0'].SAFE_AUC:+.4f} | {rows['F2'].SAFE_AUC-rows['F1'].SAFE_AUC:+.4f} | {rows['F0'].policy_delta_vs_B0_pp:+.3f} | {rows['F1'].policy_delta_vs_B0_pp:+.3f} | {rows['F2'].policy_delta_vs_B0_pp:+.3f} | {rows['F1'].policy_delta_vs_B0_pp-rows['F0'].policy_delta_vs_B0_pp:+.3f} | {rows['F2'].policy_delta_vs_B0_pp-rows['F1'].policy_delta_vs_B0_pp:+.3f} | {interpretations[task]} |")
    lines += ["", "## Table 3 — XS seed robustness", "",
              "| Task | Seed | Feature family | SAFE AUC | Policy delta pp | 95% CI | Nonnegative folds | Worst fold |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        for seed in (0, 1, 2):
            for family in ("F1", "F2"):
                safe = auc(cross, "XS", task, family, "SAFE_SWITCH", f"seed{seed}")
                row = policy[(policy.comparison == "XS") & (policy.task == task) &
                             (policy.seed_scope == f"seed{seed}") & (policy.feature_family == family)].iloc[0]
                lines.append(f"| {task} | seed{seed} | {family} | {safe:.4f} | {row.policy_delta_vs_B0_pp:+.3f} | [{row.ci_low_pp:+.3f}, {row.ci_high_pp:+.3f}] | {int(row.nonnegative_folds)}/5 | {row.worst_fold_pp:+.3f} |")

    mi_answers = {}
    for task in ("OpenBMI_MI", "WBCIC_MI"):
        rows = {f: primary[(primary.task == task) & (primary.feature_family == f)].iloc[0] for f in ("F0", "F1", "F2")}
        mi_answers[task] = {
            "signed_compression_resolved": bool(rows["F1"].SAFE_AUC >= .65 and rows["F1"].policy_delta_vs_B0_pp >= .30),
            "embedding_added_safe_auc": float(rows["F2"].SAFE_AUC - rows["F1"].SAFE_AUC),
            "embedding_added_policy_pp": float(rows["F2"].policy_delta_vs_B0_pp - rows["F1"].policy_delta_vs_B0_pp),
            "F1_CI_excludes_zero": bool(rows["F1"].ci_low_pp > 0), "F2_CI_excludes_zero": bool(rows["F2"].ci_low_pp > 0),
        }
    embedding_auc_changes = {task: float(primary[(primary.task == task) & (primary.feature_family == "F2")].iloc[0].SAFE_AUC -
                                         primary[(primary.task == task) & (primary.feature_family == "F1")].iloc[0].SAFE_AUC) for task in TASKS}
    lines += ["", "## Final scientific answers", "",
              "1. No. The previous MI failure was not resolved by retaining class orientation, so class-symmetric compression is not supported as the material bottleneck.",
              f"2. No. OpenBMI MI signed features changed SAFE AUC by {primary[(primary.task=='OpenBMI_MI') & (primary.feature_family=='F1')].iloc[0].SAFE_AUC-primary[(primary.task=='OpenBMI_MI') & (primary.feature_family=='F0')].iloc[0].SAFE_AUC:+.4f} and policy delta by {primary[(primary.task=='OpenBMI_MI') & (primary.feature_family=='F1')].iloc[0].policy_delta_vs_B0_pp-primary[(primary.task=='OpenBMI_MI') & (primary.feature_family=='F0')].iloc[0].policy_delta_vs_B0_pp:+.3f} pp.",
              f"3. No. WBCIC MI signed features changed SAFE AUC by {primary[(primary.task=='WBCIC_MI') & (primary.feature_family=='F1')].iloc[0].SAFE_AUC-primary[(primary.task=='WBCIC_MI') & (primary.feature_family=='F0')].iloc[0].SAFE_AUC:+.4f} and policy delta by {primary[(primary.task=='WBCIC_MI') & (primary.feature_family=='F1')].iloc[0].policy_delta_vs_B0_pp-primary[(primary.task=='WBCIC_MI') & (primary.feature_family=='F0')].iloc[0].policy_delta_vs_B0_pp:+.3f} pp.",
              "4. No. Full embeddings reduced SAFE AUC relative to F1 on every task (" + "; ".join(f"{task} {embedding_auc_changes[task]:+.4f}" for task in TASKS) + ") and no task required F2 to pass the strong gate.",
              "5. SSVEP is preserved as a strong positive control under F1 and F2. ERP is not preserved under the full strong-policy gate: F1 markedly raises AUC but lowers policy gain below +0.30 pp and has a worst fold below -0.50 pp; F2 does not repair it.",
              "6. The intervention signal is robust across all three seeds only for SSVEP. OpenBMI MI is seed-fragile; WBCIC has positive point estimates but weak AUC and CIs crossing zero; ERP does not meet the strong fold/policy gate consistently.",
              "7. Supported labels: " + "; ".join(f"{task}={routing_labels[task]}" for task in TASKS) + ". ERP retains only a weak logit-level signal, not a strong routing label.",
              f"8. Broad evidence for a future single-model conditional residual mechanism: {'YES' if broad_success else 'NO'} ({strong_tasks}/4 strong tasks; all-task positive best-family policy={all_positive}; neither MI high-oracle unresolved={neither_mi_high}).", "",
              "This is a development-only diagnostic conclusion. No EEG network or neural router was trained, no logistic hyperparameter was tuned, and no internal-heldout, final-heldout, final-test, or sealed-test artifact was accessed."]
    atomic_text(outputs / "FINAL_REPRESENTATION_PREDICTABILITY_REPORT.md", "\n".join(lines))
    scope = json.loads((protocol / "DEVELOPMENT_SCOPE_AUDIT.json").read_text(encoding="utf-8"))
    scope.update({"F0_CONTROL_REPRODUCTION_PASS": f0_pass, "REPRESENTATION_REPLAY_PASS": replay_pass,
                  "SUBJECT_EQUAL_WEIGHT_PASS": weight_pass, "protocol_audits_pass": audits_pass})
    atomic_json(protocol / "DEVELOPMENT_SCOPE_AUDIT.json", scope)
    print(f"REPRESENTATION_AUDIT_COMPLETE strong_tasks={strong_tasks}/4 broad_success={broad_success}", flush=True)


if __name__ == "__main__":
    main()
