#!/usr/bin/env python3
"""Compute exact-trial XS rescue and harm stability across seeds."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["task", "fold", "subject_id", "session", "trial_id"]
PAIRS = ((0, 1), (0, 2), (1, 2))


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_name(path.name + ".part"); frame.to_csv(temp, index=False); os.replace(temp, path)


def jaccard(left: pd.Series, right: pd.Series) -> float:
    union = (left | right).sum()
    return float((left & right).sum() / union) if union else np.nan


def summarize(group: pd.DataFrame) -> dict[str, float | int]:
    wrong, correct = group[~group.B0_FIXED_correct], group[group.B0_FIXED_correct]
    result: dict[str, float | int] = {"B0_wrong_trials": len(wrong), "B0_correct_trials": len(correct)}
    for count in range(4):
        result[f"rescued_exactly_{count}_of_3"] = int(wrong.rescue_count.eq(count).sum())
        result[f"harmed_exactly_{count}_of_3"] = int(correct.harm_count.eq(count).sum())
    any_rescue = int(wrong.rescue_count.ge(1).sum()); any_harm = int(correct.harm_count.ge(1).sum())
    result.update({
        "any_rescue_rate": float(wrong.rescue_count.ge(1).mean()), "majority_rescue_rate": float(wrong.rescue_count.ge(2).mean()),
        "unanimous_rescue_rate": float(wrong.rescue_count.eq(3).mean()),
        "majority_given_any_rescue": float(wrong.rescue_count.ge(2).sum() / any_rescue) if any_rescue else np.nan,
        "unanimous_given_any_rescue": float(wrong.rescue_count.eq(3).sum() / any_rescue) if any_rescue else np.nan,
        "any_harm_rate": float(correct.harm_count.ge(1).mean()), "majority_harm_rate": float(correct.harm_count.ge(2).mean()),
        "unanimous_harm_rate": float(correct.harm_count.eq(3).mean()),
        "majority_given_any_harm": float(correct.harm_count.ge(2).sum() / any_harm) if any_harm else np.nan,
        "unanimous_given_any_harm": float(correct.harm_count.eq(3).sum() / any_harm) if any_harm else np.nan,
    })
    rescue_js, harm_js = [], []
    for left, right in PAIRS:
        rj = jaccard(wrong[f"rescue_{left}"], wrong[f"rescue_{right}"]); hj = jaccard(correct[f"harm_{left}"], correct[f"harm_{right}"])
        result[f"rescue_jaccard_seed{left}_seed{right}"] = rj; result[f"harm_jaccard_seed{left}_seed{right}"] = hj
        rescue_js.append(rj); harm_js.append(hj)
    result["mean_rescue_jaccard"] = float(np.nanmean(rescue_js)); result["mean_harm_jaccard"] = float(np.nanmean(harm_js))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args(); outputs = args.repo.resolve() / "experiments/persist_eeg_stable_rescue_predictability_audit_v1/outputs"
    rows = pd.read_csv(args.runtime.resolve() / "FIXED_B0_XS_ROWS.csv.gz", dtype={"subject_id": str})
    base = rows[rows.seed == 0][KEY + ["true_label", "B0_prediction", "B0_correct", "B0_logits", "B0_checkpoint_sha256", "normalizer_sha256"]].copy()
    base = base.rename(columns={"B0_prediction": "B0_FIXED_prediction", "B0_correct": "B0_FIXED_correct", "B0_logits": "B0_FIXED_logits"}).set_index(KEY)
    for seed in (0, 1, 2):
        candidate = rows[rows.seed == seed].set_index(KEY)
        if not base.index.equals(candidate.index): raise RuntimeError(f"fixed-row key mismatch seed{seed}")
        base[f"XS_seed{seed}_prediction"] = candidate.candidate_prediction.astype(int)
        base[f"XS_seed{seed}_correct"] = candidate.candidate_correct.astype(bool)
        base[f"XS_seed{seed}_checkpoint_sha256"] = candidate.candidate_checkpoint_sha256
        base[f"rescue_{seed}"] = ~base.B0_FIXED_correct & base[f"XS_seed{seed}_correct"]
        base[f"harm_{seed}"] = base.B0_FIXED_correct & ~base[f"XS_seed{seed}_correct"]
    trial = base.reset_index(); trial["rescue_count"] = trial[[f"rescue_{s}" for s in (0,1,2)]].sum(axis=1)
    trial["harm_count"] = trial[[f"harm_{s}" for s in (0,1,2)]].sum(axis=1)
    trial["ANY_RESCUE"] = trial.rescue_count.ge(1); trial["MAJORITY_RESCUE"] = trial.rescue_count.ge(2); trial["UNANIMOUS_RESCUE"] = trial.rescue_count.eq(3)
    trial["ANY_HARM"] = trial.harm_count.ge(1); trial["MAJORITY_HARM"] = trial.harm_count.ge(2); trial["UNANIMOUS_HARM"] = trial.harm_count.eq(3)
    atomic_csv(outputs / "XS_STABILITY_TRIAL_RESULTS.csv", trial)
    subject_rows = []
    for (task, fold, subject), group in trial.groupby(["task", "fold", "subject_id"], sort=True):
        subject_rows.append({"task": task, "fold": int(fold), "subject_id": str(subject), **summarize(group)})
    subjects = pd.DataFrame(subject_rows); atomic_csv(outputs / "XS_STABILITY_SUBJECT_SUMMARY.csv", subjects)
    task_rows = []
    for task, group in trial.groupby("task", sort=True):
        raw = summarize(group); ss = subjects[subjects.task == task]
        raw["subject_equal_mean_rescue_jaccard"] = float(ss.mean_rescue_jaccard.mean())
        raw["subject_equal_mean_harm_jaccard"] = float(ss.mean_harm_jaccard.mean())
        raw["task"] = task; raw["subjects"] = int(ss.subject_id.nunique()); task_rows.append(raw)
    atomic_csv(outputs / "XS_STABILITY_TASK_SUMMARY.csv", pd.DataFrame(task_rows))
    print(f"XS_STABILITY_COMPLETE trials={len(trial)} subjects={len(subjects)}", flush=True)


if __name__ == "__main__":
    main()
