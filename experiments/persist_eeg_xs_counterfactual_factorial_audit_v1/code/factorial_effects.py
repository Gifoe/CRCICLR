#!/usr/bin/env python3
"""Compute standard balanced 2-level factorial effects for Rescue/Harm/NET."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EFFECTS = {"C": ("C",), "S": ("S",), "M": ("M",), "CxS": ("C", "S"),
           "CxM": ("C", "M"), "SxM": ("S", "M"), "CxSxM": ("C", "S", "M")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    outputs = args.repo.resolve() / "experiments/persist_eeg_xs_counterfactual_factorial_audit_v1/outputs"
    data = pd.read_csv(outputs / "COUNTERFACTUAL_TASK_SEED_SUMMARY.csv")
    rows = []
    for (task, seed), group in data.groupby(["task", "seed"], sort=True):
        if len(group) != 8:
            raise RuntimeError(f"incomplete factorial cube {task}/seed{seed}")
        for effect, factors in EFFECTS.items():
            sign = np.prod([2 * group[factor].to_numpy(float) - 1 for factor in factors], axis=0)
            row = {"task": task, "seed": int(seed), "effect": effect,
                   "sign_convention": "mean(product_sign=+1)-mean(product_sign=-1); ON is +1"}
            for metric in ("RESCUE_PP", "HARM_PP", "NET_PP"):
                row[f"{metric}_effect"] = float(np.sum(sign * group[metric].to_numpy(float)) / 4.0)
            rows.append(row)
    pd.DataFrame(rows).to_csv(outputs / "FACTORIAL_EFFECTS.csv", index=False)
    print(f"FACTORIAL_EFFECTS_COMPLETE rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
