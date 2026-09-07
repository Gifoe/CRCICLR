from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from v3_common import sigmoid


ACTIONS = ("amplify", "geometry", "erase")


def build_action_candidates(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial_index, trial in trials.iterrows():
        keep = np.asarray(trial.keep_margins, dtype=float)
        n_runs = len(keep)
        rows.append(
            {
                "trial_index": int(trial_index),
                "trial_uid": trial.trial_uid,
                "candidate_id": "KEEP_ENSEMBLE",
                "action_family": "keep",
                "action_scope": "REFERENCE",
                "run_id": None,
                "fold_id": np.nan,
                "seed_id": np.nan,
                "margin": float(keep.mean()),
                "probability": float(sigmoid(np.asarray([keep.mean()]))[0]),
                "prediction": int(keep.mean() >= 0),
                "protected_safe": True,
                "uses_erase": False,
                "n_runs": n_runs,
                "OUTER_TEST_USED": False,
            }
        )
        for action in ACTIONS:
            action_margins = np.asarray(trial[f"{action}_margins"], dtype=float)
            global_margin = float(action_margins.mean())
            rows.append(
                {
                    "trial_index": int(trial_index),
                    "trial_uid": trial.trial_uid,
                    "candidate_id": f"ALL_{action.upper()}",
                    "action_family": action.upper(),
                    "action_scope": "GLOBAL",
                    "run_id": None,
                    "fold_id": np.nan,
                    "seed_id": np.nan,
                    "margin": global_margin,
                    "probability": float(sigmoid(np.asarray([global_margin]))[0]),
                    "prediction": int(global_margin >= 0),
                    "protected_safe": action != "erase",
                    "uses_erase": action == "erase",
                    "n_runs": n_runs,
                    "OUTER_TEST_USED": False,
                }
            )
            for position, run_id in enumerate(trial.run_ids):
                replacement_margin = float((keep.sum() - keep[position] + action_margins[position]) / n_runs)
                rows.append(
                    {
                        "trial_index": int(trial_index),
                        "trial_uid": trial.trial_uid,
                        "candidate_id": f"{run_id}->{action.upper()}",
                        "action_family": action.upper(),
                        "action_scope": "SINGLE_REPLACEMENT",
                        "run_id": run_id,
                        "fold_id": int(trial.fold_ids[position]),
                        "seed_id": int(trial.seed_ids[position]),
                        "margin": replacement_margin,
                        "probability": float(sigmoid(np.asarray([replacement_margin]))[0]),
                        "prediction": int(replacement_margin >= 0),
                        "protected_safe": action != "erase",
                        "uses_erase": action == "erase",
                        "n_runs": n_runs,
                        "OUTER_TEST_USED": False,
                    }
                )
    candidates = pd.DataFrame(rows)
    expected = int(sum(1 + 3 + 3 * int(n) for n in trials.n_runs))
    if len(candidates) != expected:
        raise RuntimeError(f"Action candidate row mismatch: {len(candidates)} != {expected}")
    if candidates.duplicated(["trial_index", "candidate_id"]).any():
        raise RuntimeError("Duplicate action candidate within a trial")
    return candidates


def action_menu_mask(candidates: pd.DataFrame, menu_id: str) -> np.ndarray:
    keep = candidates.candidate_id.eq("KEEP_ENSEMBLE").to_numpy()
    scope = candidates.action_scope.to_numpy(dtype=str)
    safe = candidates.protected_safe.to_numpy(dtype=bool)
    if menu_id == "PROTECTED_SAFE_GLOBAL":
        return keep | ((scope == "GLOBAL") & safe)
    if menu_id == "FULL_GLOBAL":
        return keep | (scope == "GLOBAL")
    if menu_id == "PROTECTED_SAFE_SINGLE_REPLACEMENT":
        return keep | ((scope == "SINGLE_REPLACEMENT") & safe)
    if menu_id == "FULL_SINGLE_REPLACEMENT":
        return keep | (scope == "SINGLE_REPLACEMENT")
    raise ValueError(menu_id)


def summarize_action_candidates(trials: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    enriched = candidates.merge(
        trials[["outcome_label", "y_keep_ens", "source_pool"]],
        left_on="trial_index",
        right_index=True,
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for (scope, candidate_id, family, run_id, fold, seed), group in enriched.groupby(
        ["action_scope", "candidate_id", "action_family", "run_id", "fold_id", "seed_id"],
        dropna=False,
        sort=True,
    ):
        differs = group.prediction.to_numpy(dtype=int) != group.y_keep_ens.to_numpy(dtype=int)
        labels = group.outcome_label.to_numpy(dtype=int)
        rescue = differs & (group.y_keep_ens.to_numpy(dtype=int) != labels) & (group.prediction.to_numpy(dtype=int) == labels)
        harm = differs & (group.y_keep_ens.to_numpy(dtype=int) == labels) & (group.prediction.to_numpy(dtype=int) != labels)
        rows.append(
            {
                "action_scope": scope,
                "candidate_id": candidate_id,
                "action_family": family,
                "run_id": run_id,
                "fold_id": fold,
                "seed_id": seed,
                "eligible_trials": int(len(group)),
                "prediction_differs_from_B6": int(differs.sum()),
                "rescue_count": int(rescue.sum()),
                "harm_count": int(harm.sum()),
                "net_correctness": int(rescue.sum() - harm.sum()),
                "rescue_precision": float(rescue.sum() / differs.sum()) if differs.any() else np.nan,
                "OUTER_TEST_USED": False,
            }
        )
    return pd.DataFrame(rows)
