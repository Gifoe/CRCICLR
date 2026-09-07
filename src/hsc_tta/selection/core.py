from __future__ import annotations

import numpy as np
import pandas as pd


ACTION_COST = {"no_tta": 0, "t3a": 1, "entropy_adapter": 2}
IDENTITY_COLUMNS = ("dataset", "seed", "episode_id", "subject_id", "alpha")
FORBIDDEN_EXACT = {
    "argmax_error",
    "macro_f1",
    "balanced_accuracy",
    "harmful_adaptation",
    "selected_error",
    "no_tta_error",
    "true_future_risk",
}


def _reject_future_columns(columns: list[str]) -> None:
    forbidden = sorted(
        name for name in columns if name.startswith("future_") or name in FORBIDDEN_EXACT
    )
    if forbidden:
        raise ValueError(f"selector cannot access future outcome columns: {forbidden}")


def select_safe_action(candidates: pd.DataFrame, alpha: float | None = None) -> dict[str, object]:
    """Select from certified critical-index candidates using U_s-only utility."""
    _reject_future_columns(list(candidates.columns))
    required = {
        *IDENTITY_COLUMNS,
        "action",
        "predicted_critical_index",
        "q_alpha",
        "certified_critical_index",
        "selected_lambda",
        "nontrivial_candidate",
        "context_average_set_size",
        "context_singleton_rate",
        "adaptation_cost",
        "n_classes",
        "n_nontrivial_lambdas",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if candidates.empty:
        raise ValueError("candidate table must not be empty")
    table = candidates.copy()
    if table[list(IDENTITY_COLUMNS)].drop_duplicates().shape[0] != 1:
        raise ValueError("selector accepts exactly one dataset/seed/episode/subject/alpha group")
    row_alpha = float(table["alpha"].iloc[0])
    if alpha is not None and not np.isclose(float(alpha), row_alpha):
        raise ValueError("alpha argument does not match candidate rows")
    if not 0 < row_alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    unknown = set(table["action"]) - set(ACTION_COST)
    if unknown:
        raise ValueError(f"unknown actions: {sorted(unknown)}")
    if table["action"].duplicated().any():
        raise ValueError("one candidate row per action is required")
    n_classes = table["n_classes"].to_numpy(float)
    if np.any(~np.isfinite(n_classes)) or np.any(n_classes < 2) or len(np.unique(n_classes)) != 1:
        raise ValueError("n_classes must be present, finite, >=2, and constant within a subject")
    for column in (
        "predicted_critical_index",
        "q_alpha",
        "certified_critical_index",
        "selected_lambda",
        "context_average_set_size",
        "context_singleton_rate",
        "adaptation_cost",
    ):
        if np.any(~np.isfinite(table[column].to_numpy(float))):
            raise ValueError(f"{column} must be finite")
    if np.any((table["context_average_set_size"] < 1) | (table["context_average_set_size"] > n_classes)):
        raise ValueError("context_average_set_size must be in [1, n_classes]")
    if np.any((table["context_singleton_rate"] < 0) | (table["context_singleton_rate"] > 1)):
        raise ValueError("context_singleton_rate must be in [0, 1]")
    sentinel_values = table["n_nontrivial_lambdas"].to_numpy(float)
    if (
        np.any(~np.isfinite(sentinel_values))
        or len(np.unique(sentinel_values)) != 1
        or int(sentinel_values[0]) < 1
    ):
        raise ValueError("n_nontrivial_lambdas must be a positive constant within a subject")
    sentinel_index = int(sentinel_values[0])
    if np.any((table["certified_critical_index"] < 0) | (table["certified_critical_index"] > sentinel_index)):
        raise ValueError("certified_critical_index must be in [0, L]")
    computed_nontrivial = (
        (table["certified_critical_index"] < sentinel_index)
        & (table["context_average_set_size"] < table["n_classes"])
        & (table["selected_lambda"] < 1.0)
    )
    if not np.array_equal(table["nontrivial_candidate"].astype(bool).to_numpy(), computed_nontrivial.to_numpy()):
        raise ValueError("nontrivial_candidate is inconsistent with index, lambda, or n_classes")
    table["_fixed_action_cost"] = table["action"].map(ACTION_COST)
    if not np.array_equal(table["adaptation_cost"].to_numpy(float), table["_fixed_action_cost"].to_numpy(float)):
        raise ValueError("adaptation_cost must follow no_tta < t3a < entropy_adapter")
    feasible = table[computed_nontrivial].copy()
    sort_columns = [
        "context_average_set_size",
        "context_singleton_rate",
        "adaptation_cost",
        "selected_lambda",
        "action",
    ]
    ascending = [True, False, True, False, True]
    if feasible.empty:
        fallback = table.sort_values(
            ["adaptation_cost", "action"], ascending=[True, True], kind="mergesort"
        ).iloc[0]
        return {
            "status": "uncertified",
            "certified": False,
            "nontrivial_certified": False,
            "selected_action": str(fallback["action"]),
            "selected_lambda": 1.0,
            "certified_critical_index": sentinel_index,
            "selection_reason": "no nontrivial candidate; full-set fallback",
            "selected_row": fallback.to_dict(),
            "candidates": table.drop(columns="_fixed_action_cost"),
        }
    chosen = feasible.sort_values(sort_columns, ascending=ascending, kind="mergesort").iloc[0]
    return {
        "status": "certified",
        "certified": True,
        "nontrivial_certified": True,
        "selected_action": str(chosen["action"]),
        "selected_lambda": float(chosen["selected_lambda"]),
        "certified_critical_index": int(chosen["certified_critical_index"]),
        "selection_reason": (
            "lexicographic: context_set_size, context_singleton_rate, "
            "adaptation_cost, conservative_lambda, action"
        ),
        "selected_row": chosen.to_dict(),
        "candidates": table.drop(columns="_fixed_action_cost"),
    }
