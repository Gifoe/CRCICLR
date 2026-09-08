"""Subject-unit aggregation and immutable terminal decision for the task test."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from task_datasets import EXP, OUTPUTS, PROTOCOL, TASKS, write_json

BOOTSTRAPS = 10_000
MI_DATASET = EXP.parent / "persist_eeg_final_heldout_confirmation_v1" / "outputs" / "DATASET_RESULTS.csv"


def bootstrap(delta_pp: np.ndarray) -> dict[str, float | int]:
    rng = np.random.default_rng(0)
    draws = rng.choice(delta_pp, size=(BOOTSTRAPS, len(delta_pp)), replace=True).mean(1)
    return {"resamples": BOOTSTRAPS, "seed": 0, "n_subjects": int(len(delta_pp)),
            "mean_delta_pp": float(delta_pp.mean()), "median_delta_pp": float(np.median(delta_pp)),
            "ci_low_pp": float(np.quantile(draws, .025)), "ci_high_pp": float(np.quantile(draws, .975))}


def subject_average(frame: pd.DataFrame, task: str, heldout: bool) -> pd.DataFrame:
    part = frame[frame.task == task]
    expected_subjects = 14 if heldout else 40
    expected_reps = 15 if heldout else 3
    grouped = part.groupby(["subject_id", "method"], as_index=False)[["BA", "macro_F1", "accuracy", "AUROC", "AUPRC"]].mean()
    if grouped.subject_id.nunique() != expected_subjects or not (grouped.groupby("subject_id").size() == 3).all():
        raise RuntimeError(f"subject/method cardinality invalid for {task}/{heldout}")
    wide = grouped.pivot(index="subject_id", columns="method", values=["BA", "macro_F1", "accuracy", "AUROC", "AUPRC"])
    wide.columns = [f"{method}_{metric}" for metric, method in wide.columns]
    out = wide.reset_index(); out.insert(0, "task", task); out["replicates_per_subject"] = expected_reps
    out["LOGIT50_minus_EEGNet_pp"] = (out["LOGIT50_BA"] - out["EEGNet_BA"]) * 100
    out["LOGIT50_minus_LiteBN_pp"] = (out["LOGIT50_BA"] - out["LiteBN_BA"]) * 100
    return out.sort_values("subject_id", key=lambda x: x.astype(int)).reset_index(drop=True)


def replicate_stability(held: pd.DataFrame) -> pd.DataFrame:
    avg = held.groupby(["task", "fold", "seed", "method"], as_index=False).BA.mean()
    w = avg.pivot(index=["task", "fold", "seed"], columns="method", values="BA").reset_index()
    if len(w) != 30 or set(("EEGNet", "LiteBN", "LOGIT50")) - set(w.columns):
        raise RuntimeError("heldout replicate grid invalid")
    w["LOGIT50_minus_EEGNet_pp"] = (w.LOGIT50 - w.EEGNet) * 100
    w["fold_mean_delta_pp"] = w.groupby(["task", "fold"])["LOGIT50_minus_EEGNet_pp"].transform("mean")
    w["seed_mean_delta_pp"] = w.groupby(["task", "seed"])["LOGIT50_minus_EEGNet_pp"].transform("mean")
    return w.sort_values(["task", "fold", "seed"]).reset_index(drop=True)


def task_rows(subjects: pd.DataFrame, stability: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    rows = []; statuses: dict[str, str] = {}
    for task in ("ERP", "SSVEP"):
        p = subjects[subjects.task == task]; d = p.LOGIT50_minus_EEGNet_pp.to_numpy(float); bs = bootstrap(d)
        strong = bs["mean_delta_pp"] >= 1.0 and bs["median_delta_pp"] > 0 and bs["ci_low_pp"] > 0 and int((d > 0).sum()) >= 8
        status = "STRONG_TASK_GENERALIZATION" if strong else ("POSITIVE_TASK_GENERALIZATION" if bs["mean_delta_pp"] > 0 and bs["median_delta_pp"] >= 0 else "NO_TASK_GAIN")
        statuses[task] = status; st = stability[stability.task == task]
        rows.append({"task": task, "n_classes": TASKS[task]["classes"], "n_subjects": int(len(p)),
                     "EEGNet_BA": float(p.EEGNet_BA.mean()), "LiteBN_BA": float(p.LiteBN_BA.mean()), "LOGIT50_BA": float(p.LOGIT50_BA.mean()),
                     "LOGIT50_minus_EEGNet_pp": bs["mean_delta_pp"], "LOGIT50_minus_LiteBN_pp": float(p.LOGIT50_minus_LiteBN_pp.mean()),
                     "median_subject_delta_pp": bs["median_delta_pp"], "bootstrap_ci_low_pp": bs["ci_low_pp"], "bootstrap_ci_high_pp": bs["ci_high_pp"],
                     "positive_subjects": int((d > 0).sum()), "harm_le_minus_1pp": int((d <= -1).sum()), "harm_le_minus_3pp": int((d <= -3).sum()), "harm_le_minus_5pp": int((d <= -5).sum()),
                     "positive_replicates_of_15": int((st.LOGIT50_minus_EEGNet_pp > 0).sum()), "task_confirmation": status})
    return pd.DataFrame(rows), statuses


def harm_profile(subjects: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in ("ERP", "SSVEP"):
        p = subjects[subjects.task == task].sort_values("LOGIT50_minus_EEGNet_pp"); d = p.LOGIT50_minus_EEGNet_pp
        out[task] = {"relative_to": "EEGNet", "n_subjects": int(len(p)), "positive": int((d > 0).sum()),
                     "gain_ge_plus_1pp": int((d >= 1).sum()), "gain_ge_plus_3pp": int((d >= 3).sum()), "abs_delta_lt_0_5pp": int((d.abs() < .5).sum()),
                     "harm_le_minus_1pp": int((d <= -1).sum()), "harm_le_minus_3pp": int((d <= -3).sum()), "harm_le_minus_5pp": int((d <= -5).sum()),
                     "largest_improvement": {"subject_id": str(p.iloc[-1].subject_id), "delta_pp": float(p.iloc[-1].LOGIT50_minus_EEGNet_pp)},
                     "largest_harm": {"subject_id": str(p.iloc[0].subject_id), "delta_pp": float(p.iloc[0].LOGIT50_minus_EEGNet_pp)}}
    return out


def search_rows(subjects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in ("ERP", "SSVEP"):
        p = subjects[subjects.task == task]; d = p.LOGIT50_minus_EEGNet_pp.to_numpy(float); bs = bootstrap(d)
        rows.append({"task": task, "n_subjects": int(len(p)), "EEGNet_BA": float(p.EEGNet_BA.mean()), "LiteBN_BA": float(p.LiteBN_BA.mean()), "LOGIT50_BA": float(p.LOGIT50_BA.mean()), **bs, "positive_subjects": int((d > 0).sum())})
    return pd.DataFrame(rows)


def terminal(rows: pd.DataFrame) -> str:
    d = rows.set_index("task").LOGIT50_minus_EEGNet_pp
    if d.ERP > 0 and d.SSVEP > 0 and set(rows.task_confirmation) == {"STRONG_TASK_GENERALIZATION"}:
        return "OPENBMI_TASK_GENERALITY_STRONG"
    if d.ERP > 0 and d.SSVEP > 0: return "OPENBMI_TASK_GENERALITY_POSITIVE"
    if (d.ERP > 0) != (d.SSVEP > 0): return "OPENBMI_TASK_GENERALITY_PARTIAL"
    return "OPENBMI_TASK_GENERALITY_NOT_SUPPORTED"


def table_markdown(table: pd.DataFrame) -> str:
    lines = ["| Task | Classes | EEGNet BA | LiteBN BA | LOGIT50 BA | Δ vs EEGNet | Median Δ | 95% CI | Positive subjects |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in table.itertuples():
        lines.append(f"| {r.Task} | {int(r.Classes)} | {r.EEGNet_BA:.4f} | {r.LiteBN_BA:.4f} | {r.LOGIT50_BA:.4f} | {r.LOGIT50_minus_EEGNet_pp:+.3f} pp | {r.Median_delta_pp:+.3f} pp | [{r.bootstrap_ci_low_pp:+.3f}, {r.bootstrap_ci_high_pp:+.3f}] pp | {r.Positive_subjects} |")
    return "\n".join(lines) + "\n"


def decision_markdown(table: pd.DataFrame, status: dict[str, str], harm: dict[str, Any], comp: dict[str, Any], value: str) -> str:
    t = table.set_index("Task"); positive = [task for task in ("ERP", "SSVEP") if t.loc[task, "LOGIT50_minus_EEGNet_pp"] > 0]
    h = "SUPPORTED" if len(positive) == 2 else ("PARTIALLY SUPPORTED" if len(positive) == 1 else "NOT SUPPORTED")
    specific = "NO" if len(positive) == 2 else ("PARTIALLY" if len(positive) == 1 else "YES")
    comp_count = sum(comp[task].get("complementarity_fraction", 0) > 0 for task in ("ERP", "SSVEP"))
    comp_status = "YES" if comp_count == 2 else ("PARTIAL" if comp_count == 1 else "NO")
    main = table_markdown(table)
    details = []
    for task in ("ERP", "SSVEP"):
        r=t.loc[task]; hp=harm[task]; cp=comp[task]
        details += [f"## {task}", "", f"Mean gain: {r.LOGIT50_minus_EEGNet_pp:+.3f} pp; median: {r.Median_delta_pp:+.3f} pp; 95% CI [{r.bootstrap_ci_low_pp:+.3f}, {r.bootstrap_ci_high_pp:+.3f}] pp; positive subjects: {int(r.Positive_subjects)}/14.", f"Harmful tail (≤−1/−3/−5 pp): {hp['harm_le_minus_1pp']}/{hp['harm_le_minus_3pp']}/{hp['harm_le_minus_5pp']}; complementarity fraction: {cp.get('complementarity_fraction', float('nan')):.4f}.", f"Classification: `{status[task]}`.", ""]
    return "\n".join(["# PERSIST-EEG OpenBMI Task-Generality Test", "", "The final fusion rule was frozen from the MI experiments before ERP and SSVEP outcomes were evaluated.", "", "## 1. Protocol", "", "- Same OpenBMI 40 SEARCH / 14 held-out membership; cache-session 1 to cache-session 2.", "- EEGNet and LiteBN trained from scratch over 5 folds × 3 seeds.", "- Fixed 50/50 logit fusion; no task-specific fusion tuning.", "", "## 2. Main held-out table", "", main, "## 3. Prospective hypothesis", "", "H_TASK: LOGIT50 > EEGNet on both ERP and SSVEP held-out mean BA.", "", f"Answer: **{h}**.", "", *details, "## 6. Is the frozen fusion benefit MI-specific?", "", f"**{specific}** under the locked ERP/SSVEP task test.", "", "## 7. Does carrier complementarity appear in all three OpenBMI tasks?", "", f"**{comp_status}**.", "", "## 8. Overall terminal", "", f"`{value}`", "", "## 9. Scientific interpretation", "", "The primary unit is the held-out subject after averaging 15 fixed carrier-pair replicates. ERP and SSVEP were run under the single protocol lock after both task grids completed. The fixed predictor has no learned fusion parameter and no target-session adaptation. The terminal follows the preregistered subject-level criteria without exclusions or post-outcome changes. These results assess constructive task generality and carrier complementarity, not universality of every historical PERSIST mechanism.", ""])


def main() -> None:
    search = pd.read_csv(OUTPUTS / "SEARCH_REPLICATE_RESULTS.csv")
    held = pd.read_csv(OUTPUTS / "HELDOUT_REPLICATE_RESULTS.csv")
    if len(search) != 720 or len(held) != 1260: raise RuntimeError("required evaluation outputs incomplete")
    ss = pd.concat([subject_average(search, t, False) for t in ("ERP", "SSVEP")], ignore_index=True)
    hs = pd.concat([subject_average(held, t, True) for t in ("ERP", "SSVEP")], ignore_index=True)
    dev = search_rows(ss); stability = replicate_stability(held); task_results, status = task_rows(hs, stability); harm = harm_profile(hs)
    comp = json.loads((OUTPUTS / "CARRIER_COMPLEMENTARITY.json").read_text(encoding="utf-8"))
    mi = pd.read_csv(MI_DATASET).query("dataset == 'OpenBMI'").iloc[0]
    rows = [{"Task": "MI", "Classes": 2, "EEGNet_BA": float(mi.EEGNet_BA), "LiteBN_BA": float(mi.LiteBN_BA), "LOGIT50_BA": float(mi.FROZEN_LOGIT50_BA), "LOGIT50_minus_EEGNet_pp": float(mi.LOGIT50_minus_EEGNet_pp), "Median_delta_pp": float(mi.median_delta_pp), "bootstrap_ci_low_pp": float(mi.bootstrap_ci_low_pp), "bootstrap_ci_high_pp": float(mi.bootstrap_ci_high_pp), "Positive_subjects": f"{int(mi.positive_subjects)}/14"}]
    for r in task_results.itertuples(): rows.append({"Task": r.task, "Classes": r.n_classes, "EEGNet_BA": r.EEGNet_BA, "LiteBN_BA": r.LiteBN_BA, "LOGIT50_BA": r.LOGIT50_BA, "LOGIT50_minus_EEGNet_pp": r.LOGIT50_minus_EEGNet_pp, "Median_delta_pp": r.median_subject_delta_pp, "bootstrap_ci_low_pp": r.bootstrap_ci_low_pp, "bootstrap_ci_high_pp": r.bootstrap_ci_high_pp, "Positive_subjects": f"{r.positive_subjects}/14"})
    table = pd.DataFrame(rows); terminal_value = terminal(task_results)
    held_map=task_results.set_index("task"); dev_map=dev.set_index("task")
    dvh = pd.DataFrame([{"task": t, "search_EEGNet_BA": dev_map.loc[t].EEGNet_BA, "search_LiteBN_BA": dev_map.loc[t].LiteBN_BA, "search_LOGIT50_BA": dev_map.loc[t].LOGIT50_BA, "search_delta_vs_EEGNet_pp": dev_map.loc[t].mean_delta_pp, "heldout_EEGNet_BA": held_map.loc[t].EEGNet_BA, "heldout_LiteBN_BA": held_map.loc[t].LiteBN_BA, "heldout_LOGIT50_BA": held_map.loc[t].LOGIT50_BA, "heldout_delta_vs_EEGNet_pp": held_map.loc[t].LOGIT50_minus_EEGNet_pp} for t in ("ERP", "SSVEP")])
    ss.to_csv(OUTPUTS / "SEARCH_SUBJECT_RESULTS.csv", index=False); dev.to_csv(OUTPUTS / "SEARCH_DATASET_RESULTS.csv", index=False); hs.to_csv(OUTPUTS / "HELDOUT_SUBJECT_RESULTS.csv", index=False); task_results.to_csv(OUTPUTS / "HELDOUT_TASK_RESULTS.csv", index=False); stability.to_csv(OUTPUTS / "REPLICATE_STABILITY.csv", index=False); dvh.to_csv(OUTPUTS / "DEVELOPMENT_VS_HOLDOUT_TASKS.csv", index=False); table.to_csv(OUTPUTS / "OPENBMI_TASK_GENERALITY_TABLE.csv", index=False); (OUTPUTS / "OPENBMI_TASK_GENERALITY_TABLE.md").write_text(table_markdown(table), encoding="utf-8"); write_json(OUTPUTS / "SUBJECT_HARM_PROFILE.json", harm); (OUTPUTS / "FINAL_TASK_GENERALITY_DECISION.md").write_text(decision_markdown(table, status, harm, comp, terminal_value), encoding="utf-8"); write_json(OUTPUTS / "FINAL_TERMINAL.json", {"terminal": terminal_value, "task_confirmation": status, "subject_is_primary_unit": True, "replicates_per_heldout_subject": 15, "rows": task_results.to_dict("records")}); print(terminal_value)


if __name__ == "__main__": main()