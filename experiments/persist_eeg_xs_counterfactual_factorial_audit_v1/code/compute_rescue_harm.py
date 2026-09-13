#!/usr/bin/env python3
"""Compute subject-equal Rescue/Harm/NET summaries and paired bootstraps."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

TASK_NAMES = {"OMI": "OpenBMI_MI", "OERP": "OpenBMI_ERP", "OSSVEP": "OpenBMI_SSVEP", "WMI": "WBCIC_MI"}


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_name(path.name + ".part")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def subject_row(group: pd.DataFrame) -> dict[str, Any]:
    labels = group.true_label.to_numpy(int)
    classes = np.unique(labels)
    b0 = group.B0_prediction.to_numpy(int)
    candidate = group.counterfactual_prediction.to_numpy(int)
    b0_correct = b0 == labels
    candidate_correct = candidate == labels
    rescue = np.mean([np.mean((~b0_correct & candidate_correct)[labels == cls]) for cls in classes])
    harm = np.mean([np.mean((b0_correct & ~candidate_correct)[labels == cls]) for cls in classes])
    b0_ba = balanced_accuracy_score(labels, b0)
    candidate_ba = balanced_accuracy_score(labels, candidate)
    identity_difference = abs((candidate_ba - b0_ba) - (rescue - harm))
    if identity_difference > 1e-12:
        raise RuntimeError(f"BA/Rescue/Harm identity failed: {identity_difference}")
    return {
        "B0_BA": float(b0_ba), "counterfactual_BA": float(candidate_ba),
        "delta_BA_pp": float(100 * (candidate_ba - b0_ba)),
        "RESCUE_PP": float(100 * rescue), "HARM_PP": float(100 * harm),
        "NET_PP": float(100 * (rescue - harm)), "identity_absolute_difference": float(identity_difference),
        "trials": len(group), "CC_count": int(group.error_state.eq("CC").sum()),
        "RESCUE_count": int(group.error_state.eq("RESCUE").sum()),
        "HARM_count": int(group.error_state.eq("HARM").sum()), "WW_count": int(group.error_state.eq("WW").sum()),
    }


def bootstrap(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    draws = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    outputs = repo / "experiments/persist_eeg_xs_counterfactual_factorial_audit_v1/outputs"
    trials = pd.read_csv(outputs / "COUNTERFACTUAL_TRIAL_RESULTS.csv", dtype={"subject_id": str})
    trials["task"] = trials.task.map(TASK_NAMES)
    trials.loc[trials.task == "WBCIC_MI", "subject_id"] = "sub-" + trials.loc[trials.task == "WBCIC_MI", "subject_id"].astype(str)
    if trials.task.isna().any():
        raise RuntimeError("unknown compact task code")
    subject_rows = []
    for keys, group in trials.groupby(["task", "seed", "fold", "subject_id", "state_id", "C", "S", "M"], sort=True):
        subject_rows.append({**dict(zip(["task", "seed", "fold", "subject_id", "state_id", "C", "S", "M"], keys)), **subject_row(group)})
    subjects = pd.DataFrame(subject_rows)
    atomic_csv(outputs / "COUNTERFACTUAL_SUBJECT_RESULTS.csv", subjects)

    summary = subjects.groupby(["task", "seed", "state_id", "C", "S", "M"], as_index=False).agg(
        B0_BA=("B0_BA", "mean"), counterfactual_BA=("counterfactual_BA", "mean"),
        delta_BA_pp=("delta_BA_pp", "mean"), RESCUE_PP=("RESCUE_PP", "mean"),
        HARM_PP=("HARM_PP", "mean"), NET_PP=("NET_PP", "mean"),
        identity_absolute_difference=("identity_absolute_difference", "max"),
        subjects=("subject_id", "nunique"), trials=("trials", "sum"),
        CC_count=("CC_count", "sum"), RESCUE_count=("RESCUE_count", "sum"),
        HARM_count=("HARM_count", "sum"), WW_count=("WW_count", "sum"))
    full = summary[summary.state_id == "C1S1M1"][["task", "seed", "RESCUE_PP", "HARM_PP", "NET_PP"]].rename(
        columns={"RESCUE_PP": "FULL_RESCUE_PP", "HARM_PP": "FULL_HARM_PP", "NET_PP": "FULL_NET_PP"})
    summary = summary.merge(full, on=["task", "seed"], validate="many_to_one")
    summary["delta_rescue_vs_full_pp"] = summary.RESCUE_PP - summary.FULL_RESCUE_PP
    summary["delta_harm_vs_full_pp"] = summary.HARM_PP - summary.FULL_HARM_PP
    summary["delta_net_vs_full_pp"] = summary.NET_PP - summary.FULL_NET_PP
    summary["rescue_retention"] = summary.RESCUE_PP / summary.FULL_RESCUE_PP.replace(0, np.nan)
    summary["harm_reduction"] = 1 - summary.HARM_PP / summary.FULL_HARM_PP.replace(0, np.nan)
    atomic_csv(outputs / "COUNTERFACTUAL_TASK_SEED_SUMMARY.csv", summary)

    subject_full = subjects[subjects.state_id == "C1S1M1"][["task", "seed", "subject_id", "NET_PP"]].rename(columns={"NET_PP": "FULL_NET_PP"})
    paired = subjects.merge(subject_full, on=["task", "seed", "subject_id"], validate="many_to_one")
    paired["delta_net_vs_full_pp"] = paired.NET_PP - paired.FULL_NET_PP
    boot_rows = []
    for (task, seed, state), group in paired.groupby(["task", "seed", "state_id"], sort=True):
        net, net_low, net_high = bootstrap(group.NET_PP.to_numpy(float))
        delta, delta_low, delta_high = bootstrap(group.delta_net_vs_full_pp.to_numpy(float))
        boot_rows.append({"task": task, "seed": int(seed), "state_id": state,
                          "bootstrap_unit": "paired_real_subject", "bootstrap_resamples": 10000,
                          "NET_PP": net, "NET_ci_low_pp": net_low, "NET_ci_high_pp": net_high,
                          "delta_net_vs_full_pp": delta, "delta_ci_low_pp": delta_low, "delta_ci_high_pp": delta_high})
    atomic_csv(outputs / "COUNTERFACTUAL_SUBJECT_BOOTSTRAP.csv", pd.DataFrame(boot_rows))
    print(f"RESCUE_HARM_COMPLETE subjects={len(subjects)} task_seed_states={len(summary)} bootstraps={len(boot_rows)}", flush=True)


if __name__ == "__main__":
    main()
