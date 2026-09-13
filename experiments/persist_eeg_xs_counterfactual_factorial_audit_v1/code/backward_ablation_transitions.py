#!/usr/bin/env python3
"""Compute trial transitions from Full XS to each one-module backward ablation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["task", "seed", "fold", "subject_id", "session", "trial_id"]
ABLATIONS = {"C": "C0S1M1", "S": "C1S0M1", "M": "C1S1M0"}
TASK_NAMES = {"OMI": "OpenBMI_MI", "OERP": "OpenBMI_ERP", "OSSVEP": "OpenBMI_SSVEP", "WMI": "WBCIC_MI"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    outputs = args.repo.resolve() / "experiments/persist_eeg_xs_counterfactual_factorial_audit_v1/outputs"
    trials = pd.read_csv(outputs / "COUNTERFACTUAL_TRIAL_RESULTS.csv", dtype={"subject_id": str})
    trials["task"] = trials.task.map(TASK_NAMES)
    trials.loc[trials.task == "WBCIC_MI", "subject_id"] = "sub-" + trials.loc[trials.task == "WBCIC_MI", "subject_id"].astype(str)
    if trials.task.isna().any():
        raise RuntimeError("unknown compact task code")
    summary = pd.read_csv(outputs / "COUNTERFACTUAL_TASK_SEED_SUMMARY.csv")
    full = trials[trials.state_id == "C1S1M1"][KEY + ["error_state", "counterfactual_correct"]].rename(
        columns={"error_state": "full_error_state", "counterfactual_correct": "full_correct"})
    rows = []
    for module, state_id in ABLATIONS.items():
        reduced = trials[trials.state_id == state_id][KEY + ["error_state", "counterfactual_correct"]].rename(
            columns={"error_state": "reduced_error_state", "counterfactual_correct": "reduced_correct"})
        merged = full.merge(reduced, on=KEY, validate="one_to_one")
        for (task, seed), group in merged.groupby(["task", "seed"], sort=True):
            full_rescue = group.full_error_state.eq("RESCUE")
            full_harm = group.full_error_state.eq("HARM")
            rescue_lost = full_rescue & ~group.reduced_correct
            harm_reversed = full_harm & group.reduced_correct
            rescue_loss_rate = float(rescue_lost.sum() / full_rescue.sum()) if full_rescue.sum() else np.nan
            harm_reversal_rate = float(harm_reversed.sum() / full_harm.sum()) if full_harm.sum() else np.nan
            reduced_metrics = summary[(summary.task == task) & (summary.seed == seed) & (summary.state_id == state_id)].iloc[0]
            rows.append({"task": task, "seed": int(seed), "module_removed": module, "reduced_state": state_id,
                         "FULL_rescue_preserved": int((full_rescue & group.reduced_correct).sum()),
                         "FULL_rescue_lost": int(rescue_lost.sum()),
                         "FULL_harm_preserved": int((full_harm & ~group.reduced_correct).sum()),
                         "FULL_harm_reversed": int(harm_reversed.sum()),
                         "new_rescue_introduced": int((group.reduced_error_state.eq("RESCUE") & ~full_rescue).sum()),
                         "new_harm_introduced": int((group.reduced_error_state.eq("HARM") & ~full_harm).sum()),
                         "harm_reversal_rate": harm_reversal_rate, "rescue_loss_rate": rescue_loss_rate,
                         "selectivity_index_pp": float(100 * (harm_reversal_rate - rescue_loss_rate)),
                         "delta_rescue_vs_full_pp": float(reduced_metrics.delta_rescue_vs_full_pp),
                         "delta_harm_vs_full_pp": float(reduced_metrics.delta_harm_vs_full_pp),
                         "delta_net_vs_full_pp": float(reduced_metrics.delta_net_vs_full_pp)})
    pd.DataFrame(rows).to_csv(outputs / "FULL_BACKWARD_ABLATION_TRANSITIONS.csv", index=False)
    print(f"BACKWARD_ABLATION_TRANSITIONS_COMPLETE rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
