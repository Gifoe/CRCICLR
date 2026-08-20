from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import OUTPUTS


def run() -> None:
    root = OUTPUTS
    print("ORACLE")
    print((root / "diagnostics" / "NEW_COMPONENT_ORACLE_SUMMARY.json").read_text(encoding="utf-8"))
    utility = pd.read_csv(root / "diagnostics" / "UTILITY_PREDICTABILITY.csv")
    print("UTILITY BY MODE")
    print(utility.groupby(["benchmark", "mode"])[["utility_R2", "utility_pearson", "utility_spearman", "utility_sign_accuracy"]].mean().to_string())
    future = pd.read_csv(root / "diagnostics" / "FUTURE_UTILITY_DIAGNOSTICS.csv")
    component = future.groupby(["benchmark", "component_id"]).agg(
        CE=("future_ce_gain", "mean"),
        BA=("future_ba_gain", "mean"),
        positive=("future_ce_gain", lambda value: (value > 0).mean()),
    ).sort_values(["benchmark", "CE"], ascending=[True, False])
    print("COMPONENT MEAN UTILITY")
    print(component.groupby(level=0).head(10).to_string())
    selection = pd.read_csv(root / "diagnostics" / "OUTCOME_ACTION_SELECTIONS.csv")
    action = selection.groupby(["benchmark", "mode"]).agg(
        adapt=("action", lambda value: (value == "ADAPT").mean()),
        delta=("Delta_BA", "mean"),
        harm=("Delta_BA", lambda value: (value < 0).mean()),
        worst=("Delta_BA", "min"),
    )
    print("ACTIONS")
    print(action.to_string())
    outcome = pd.read_csv(root / "diagnostics" / "OUTCOME_PROSPECTIVE_UTILITY.csv")
    rows = []
    for (benchmark, mode), group in outcome.groupby(["benchmark", "controller_mode"]):
        rows.append({
            "benchmark": benchmark,
            "mode": mode,
            "utility_pearson": float(np.corrcoef(group.future_ce_gain, group.predicted_utility)[0, 1]),
            "utility_sign_accuracy": float(np.mean((group.future_ce_gain > 0) == (group.predicted_utility > 0))),
        })
    print("OUTCOME PREDICTABILITY")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    run()
