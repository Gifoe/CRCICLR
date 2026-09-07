from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common import sigmoid
from datasets import ACTION_NAMES, OPENBMI_RUNS, ExpertDataset


def _entropy(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    return -(value * np.log(value) + (1 - value) * np.log(1 - value))


@dataclass
class FeatureBundle:
    matrices: dict[str, np.ndarray]
    names: dict[str, list[str]]
    categories: dict[str, list[str]]


def _column_stack(columns: list[np.ndarray]) -> np.ndarray:
    return np.column_stack([np.asarray(column, dtype=float) for column in columns])


def build_openbmi_features(data: ExpertDataset) -> FeatureBundle:
    keep = data.keep_run_logits
    mask = data.keep_run_mask
    count = mask.sum(axis=1).astype(float)
    base = data.base_logits
    base_probability = sigmoid(base)
    keep_probability = sigmoid(keep)
    votes = np.where(mask, (keep >= 0).astype(float), np.nan)
    vote_fraction = np.nanmean(votes, axis=1)
    sorted_deviation = np.sort(np.where(mask, keep - base[:, None], np.inf), axis=1)
    sorted_deviation[~np.isfinite(sorted_deviation)] = 0.0

    keep_columns = [
        base,
        base_probability,
        np.abs(base),
        _entropy(base_probability),
        np.nanstd(keep, axis=1),
        np.nanmin(keep, axis=1),
        np.nanmax(keep, axis=1),
        np.nanmax(keep, axis=1) - np.nanmin(keep, axis=1),
        np.nanmedian(keep, axis=1),
        vote_fraction,
        _entropy(vote_fraction),
        np.sum(mask & ((keep >= 0) != (base[:, None] >= 0)), axis=1) / count,
        count,
    ]
    keep_names = [
        "b6_margin",
        "b6_probability",
        "b6_abs_margin",
        "b6_entropy",
        "keep_margin_std",
        "keep_margin_min",
        "keep_margin_max",
        "keep_margin_range",
        "keep_margin_median",
        "keep_vote_fraction_class1",
        "keep_vote_entropy",
        "keep_disagreement_fraction",
        "keep_expert_count",
    ]
    for position, run_id in enumerate(OPENBMI_RUNS):
        keep_columns.extend(
            [
                np.where(mask[:, position], keep[:, position], base),
                mask[:, position].astype(float),
                sorted_deviation[:, position],
            ]
        )
        keep_names.extend(
            [
                f"keep_margin_{run_id}",
                f"keep_present_{run_id}",
                f"keep_sorted_centered_{position}",
            ]
        )
    session_values = sorted(np.unique(data.sessions).tolist())
    for session in session_values:
        keep_columns.append((data.sessions == session).astype(float))
        keep_names.append(f"session_{session}")
    keep_matrix = _column_stack(keep_columns)

    action_columns: list[np.ndarray] = []
    action_names: list[str] = []
    for action_index, action in enumerate(ACTION_NAMES):
        values = data.action_run_logits[:, action_index, :]
        aggregate = data.action_logits[:, action_index]
        probability = sigmoid(aggregate)
        movement = values - keep
        action_votes = np.where(mask, (values >= 0).astype(float), np.nan)
        action_vote_fraction = np.nanmean(action_votes, axis=1)
        action_columns.extend(
            [
                aggregate,
                probability,
                np.abs(aggregate),
                _entropy(probability),
                aggregate - base,
                sigmoid(aggregate) - base_probability,
                np.nanstd(values, axis=1),
                np.nanstd(movement, axis=1),
                np.nanmin(movement, axis=1),
                np.nanmax(movement, axis=1),
                action_vote_fraction,
                _entropy(action_vote_fraction),
                ((aggregate >= 0) != (base >= 0)).astype(float),
            ]
        )
        prefix = action.lower()
        action_names.extend(
            [
                f"{prefix}_margin",
                f"{prefix}_probability",
                f"{prefix}_abs_margin",
                f"{prefix}_entropy",
                f"{prefix}_margin_movement",
                f"{prefix}_probability_movement",
                f"{prefix}_run_margin_std",
                f"{prefix}_movement_std",
                f"{prefix}_movement_min",
                f"{prefix}_movement_max",
                f"{prefix}_vote_fraction_class1",
                f"{prefix}_vote_entropy",
                f"{prefix}_boundary_cross",
            ]
        )
    action_matrix = _column_stack(action_columns)
    full_no_persist = np.column_stack([keep_matrix, action_matrix])
    full_no_persist_names = keep_names + action_names
    full_persist = np.column_stack([full_no_persist, data.persist_context])
    full_persist_names = full_no_persist_names + list(data.persist_names)

    movement_names = [name for name in action_names if "movement" in name]
    protected_names = [name for name in data.persist_names if "protected_" in name]
    dependence_names = [
        name
        for name in data.persist_names
        if any(token in name for token in ("agreement", "margin_change", "entropy_change", "kl_", "js_", "probability_shift"))
    ]
    persistence_names = [name for name in data.persist_names if name not in dependence_names]
    disagreement_names = [name for name in keep_names if any(token in name for token in ("std", "range", "vote", "disagreement", "centered"))]
    return FeatureBundle(
        matrices={
            "KEEP": keep_matrix,
            "KEEP_ACTION": full_no_persist,
            "KEEP_ACTION_PERSIST": full_persist,
        },
        names={
            "KEEP": keep_names,
            "KEEP_ACTION": full_no_persist_names,
            "KEEP_ACTION_PERSIST": full_persist_names,
        },
        categories={
            "protected": protected_names,
            "decision_dependence": dependence_names,
            "persistence": persistence_names,
            "action_movement": movement_names,
            "ensemble_disagreement": disagreement_names,
            "persist_all": list(data.persist_names),
        },
    )


def select_features(bundle: FeatureBundle, base_set: str, exclude: list[str]) -> tuple[np.ndarray, list[str]]:
    names = bundle.names[base_set]
    excluded = set(exclude)
    keep_indices = [index for index, name in enumerate(names) if name not in excluded]
    return bundle.matrices[base_set][:, keep_indices], [names[index] for index in keep_indices]
