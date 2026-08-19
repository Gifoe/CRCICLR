from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v3_common import DATASET_ID, V21_CODE, sigmoid


if str(V21_CODE) not in sys.path:
    sys.path.append(str(V21_CODE))
from reconstruct_v2 import load_pool  # noqa: E402


RUN_COLUMNS = ("fold_id", "seed_id")
ACTIONS = ("amplify", "geometry", "erase")


def load_all_target_rows(cache_root: Path) -> pd.DataFrame:
    exploration = load_pool(cache_root, "EXPLORATION_POOL")
    exploration["source_pool"] = "exploration"
    holdout = load_pool(cache_root, "DEVELOPMENT_HOLDOUT")
    holdout["source_pool"] = "holdout"
    if set(exploration.manifest_index) & set(holdout.manifest_index):
        raise RuntimeError("Exploration and holdout manifest identities overlap")
    frame = pd.concat([exploration, holdout], ignore_index=True)
    if len(frame) != 40_800 or frame.subject_id.nunique() != 52:
        raise RuntimeError(f"Unexpected target-row coverage: rows={len(frame)} subjects={frame.subject_id.nunique()}")
    if frame.outer_test_used.astype(bool).any():
        raise RuntimeError("Outer-test dependency in target rows")
    return frame


def build_unique_trials(target_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manifest, group in target_rows.groupby("manifest_index", sort=True):
        group = group.sort_values(list(RUN_COLUMNS)).reset_index(drop=True)
        for column in ("subject_id", "session_id", "outcome_label", "source_pool"):
            if group[column].nunique() != 1:
                raise RuntimeError(f"Manifest {manifest} maps to multiple {column} values")
        if group.duplicated(list(RUN_COLUMNS)).any():
            raise RuntimeError(f"Manifest {manifest} duplicates a frozen run")
        n_runs = len(group)
        if n_runs not in (2, 4, 6):
            raise RuntimeError(f"Manifest {manifest} has unexpected run count {n_runs}")
        run_ids = [f"fold-{int(fold)}_seed-{int(seed)}" for fold, seed in group[list(RUN_COLUMNS)].itertuples(index=False)]
        keep = group.margin_noop.to_numpy(dtype=float)
        record: dict[str, Any] = {
            "dataset": DATASET_ID,
            "trial_uid": f"{DATASET_ID}:{int(manifest)}",
            "manifest_index": int(manifest),
            "subject_id": str(group.subject_id.iloc[0]),
            "session_id": str(group.session_id.iloc[0]),
            "outcome_label": int(group.outcome_label.iloc[0]),
            "source_pool": str(group.source_pool.iloc[0]),
            "run_ids": tuple(run_ids),
            "fold_ids": tuple(group.fold_id.astype(int)),
            "seed_ids": tuple(group.seed_id.astype(int)),
            "router_fold_ids": tuple(group.router_fold_id.astype(int)),
            "n_runs": n_runs,
            "keep_margins": keep,
            "keep_probabilities": sigmoid(keep),
            "keep_predictions": (keep >= 0).astype(int),
            "z_keep_ens": float(keep.mean()),
            "p_keep_ens": float(sigmoid(np.asarray([keep.mean()]))[0]),
            "y_keep_ens": int(keep.mean() >= 0),
            "outer_test_used": False,
        }
        for action in ACTIONS:
            margins = group[f"margin_{action}"].to_numpy(dtype=float)
            record[f"{action}_margins"] = margins
            record[f"{action}_probabilities"] = sigmoid(margins)
            record[f"{action}_predictions"] = (margins >= 0).astype(int)
        rows.append(record)
    trials = pd.DataFrame(rows)
    if len(trials) != 10_400 or trials.subject_id.nunique() != 52:
        raise RuntimeError(f"Unexpected unique-trial coverage: trials={len(trials)} subjects={trials.subject_id.nunique()}")
    if trials.duplicated(["dataset", "subject_id", "session_id", "manifest_index"]).any():
        raise RuntimeError("Unique deployment identity is not unique")
    return trials.reset_index(drop=True)


def serialize_unique_trial_reference(trials: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "trial_uid",
        "manifest_index",
        "subject_id",
        "session_id",
        "source_pool",
        "outcome_label",
        "n_runs",
        "z_keep_ens",
        "p_keep_ens",
        "y_keep_ens",
        "outer_test_used",
    ]
    return trials[columns].copy()
