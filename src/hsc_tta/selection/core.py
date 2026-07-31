from __future__ import annotations

import pandas as pd


ACTION_COST = {"no_tta": 0, "t3a": 1, "entropy_adapter": 2}


def select_safe_action(surface: pd.DataFrame, alpha: float) -> dict[str, object]:
    required = {"action", "lambda", "certified_upper_bound", "average_set_size", "singleton_rate"}
    if not required.issubset(surface.columns):
        raise ValueError(f"missing columns: {sorted(required - set(surface.columns))}")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    table = surface.copy()
    unknown = set(table.action) - set(ACTION_COST)
    if unknown:
        raise ValueError(f"unknown actions: {sorted(unknown)}")
    table["feasible"] = table.certified_upper_bound <= alpha
    feasible = table[table.feasible].copy()
    if feasible.empty:
        return {"status": "uncertified", "certified": False, "nontrivial_certified": False, "selected_action": None, "selected_lambda": None, "selection_reason": "no candidate has certified_upper_bound <= alpha", "candidates": table}
    feasible["_cost"] = feasible.action.map(ACTION_COST)
    chosen = feasible.sort_values(["average_set_size", "singleton_rate", "_cost", "lambda", "action"], ascending=[True, False, True, False, True], kind="mergesort").iloc[0]
    return {"status": "certified", "certified": True, "nontrivial_certified": bool(chosen.average_set_size < surface.get("n_classes", pd.Series([float("inf")])).iloc[0]), "selected_action": str(chosen.action), "selected_lambda": float(chosen["lambda"]), "selection_reason": "lexicographic: set_size, singleton_rate, adaptation_cost, descending_lambda, action", "selected_row": chosen.to_dict(), "candidates": table}

