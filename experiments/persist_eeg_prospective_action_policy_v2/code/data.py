from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import (
    HOLDOUT_THRESHOLD,
    PROTOCOL,
    SPLIT_SALT,
    canonical_hash,
    markdown_table,
    require_false_outer,
    sha256_file,
    stable_unit,
    write_json,
)


ID_COLUMNS = ["fold", "seed", "router_fold", "manifest_index", "subject", "session"]
FILES = {
    "features": "OOF_ROUTER_FEATURES.parquet",
    "base": "OOF_BASE_LOGITS.parquet",
    "counterfactual": "OOF_COUNTERFACTUAL_LOGITS.parquet",
    "geometry": "OOF_GEOMETRY_FEATURES.parquet",
}
ACTIONS = ("erase", "amplify", "geometry")


@dataclass
class PolicyData:
    frame: pd.DataFrame
    single_run_features: list[str]
    cross_run_features: list[str]
    split: dict[str, Any]


def _paths(cache_root: Path) -> dict[str, Path]:
    paths = {key: cache_root / value for key, value in FILES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing router cache: " + ", ".join(missing))
    return paths


def _subject_key(subject: object) -> str:
    text = str(subject)
    return str(int(float(text))) if text.replace(".", "", 1).isdigit() else text


def _cv_folds(subjects: list[str]) -> dict[str, int]:
    ordered = sorted(subjects, key=lambda subject: stable_unit(SPLIT_SALT, "cv", subject))
    return {subject: index % 5 for index, subject in enumerate(ordered)}


def create_split_protocol(cache_root: Path) -> dict[str, Any]:
    paths = _paths(cache_root)
    # Identity-only scan. No label or intervention outcome is materialized.
    identity = pd.read_parquet(paths["features"], columns=ID_COLUMNS)
    if len(identity) != 40800:
        raise RuntimeError(f"Expected 40800 router rows, found {len(identity)}")
    subjects = sorted({_subject_key(value) for value in identity.subject.unique()}, key=int)
    if len(subjects) != 52:
        raise RuntimeError(f"Expected 52 TRAIN-only subjects, found {len(subjects)}")
    assignments: list[dict[str, Any]] = []
    exploration: list[str] = []
    holdout: list[str] = []
    for subject in subjects:
        unit = stable_unit(SPLIT_SALT, subject)
        pool = "DEVELOPMENT_HOLDOUT" if unit < HOLDOUT_THRESHOLD else "EXPLORATION_POOL"
        (holdout if pool == "DEVELOPMENT_HOLDOUT" else exploration).append(subject)
        assignments.append(
            {
                "subject_id": subject,
                "sha256": canonical_hash({"salt": SPLIT_SALT, "subject_id": subject}),
                "hash_unit_interval": unit,
                "pool": pool,
            }
        )
    if len(exploration) != 40 or len(holdout) != 12:
        raise RuntimeError(f"Frozen hash split changed: exploration={len(exploration)}, holdout={len(holdout)}")
    folds = _cv_folds(exploration)
    for row in assignments:
        row["exploration_cv_fold"] = folds.get(row["subject_id"])
    assignment_hash = canonical_hash(assignments)
    payload = {
        "status": "AUTONOMOUS_RESEARCH_SPLIT_FROZEN",
        "algorithm": "SHA256(salt:canonical_subject_id), first 64 bits mapped to [0,1)",
        "salt": SPLIT_SALT,
        "development_holdout_rule": f"hash_unit_interval < {HOLDOUT_THRESHOLD}",
        "assignment_hash": assignment_hash,
        "counts": {
            "all_train_subjects": len(subjects),
            "exploration_pool": len(exploration),
            "development_holdout": len(holdout),
        },
        "exploration_fraction": len(exploration) / len(subjects),
        "development_holdout_fraction": len(holdout) / len(subjects),
        "assignments": assignments,
        "source_identity_rows": len(identity),
        "source_sha256": {str(path): sha256_file(path) for path in paths.values()},
        "labels_scanned_for_split": False,
        "intervention_outcomes_scanned_for_split": False,
        "DEVELOPMENT_HOLDOUT_OPENED": False,
        "OUTER_TEST_USED": False,
    }
    write_json(PROTOCOL / "AUTONOMOUS_RESEARCH_SPLIT.json", payload)
    table = pd.DataFrame(assignments)[["subject_id", "pool", "exploration_cv_fold", "hash_unit_interval"]]
    table["hash_unit_interval"] = table.hash_unit_interval.map(lambda value: f"{value:.12f}")
    md = f"""# Autonomous research split

`OUTER_TEST_USED = false`

The split was computed from subject identifiers only. Labels, correctness,
intervention effects, and WBCIC outer data were not scanned to choose it.

- Salt: `{SPLIT_SALT}`
- Rule: `{payload['development_holdout_rule']}`
- Exploration: `{len(exploration)}/52` subjects
- Sealed development holdout: `{len(holdout)}/52` subjects
- Assignment hash: `{assignment_hash}`

The five exploration CV folds are deterministic rank-balanced folds and are
used for train/calibration/validation separation. The holdout has no CV fold.

{markdown_table(table)}
"""
    (PROTOCOL / "AUTONOMOUS_RESEARCH_SPLIT.md").write_text(md, encoding="utf-8")
    return payload


def read_split() -> dict[str, Any]:
    path = PROTOCOL / "AUTONOMOUS_RESEARCH_SPLIT.json"
    if not path.exists():
        raise FileNotFoundError("Create the autonomous split before loading a pool")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _filtered_read(path: Path, subjects: list[str], columns: list[str] | None = None) -> pd.DataFrame:
    # The historical cache stores canonical subject identifiers as
    # large_string. Keeping the predicate values as strings is required for
    # Arrow pushdown and, critically, prevents materializing the excluded pool.
    frame = pd.read_parquet(path, columns=columns, filters=[("subject", "in", subjects)])
    observed = {_subject_key(value) for value in frame.subject.unique()}
    if not observed.issubset(set(subjects)):
        raise RuntimeError("Parquet predicate admitted a subject outside the requested pool")
    return frame.sort_values(ID_COLUMNS).reset_index(drop=True)


def _entropy_binary(p1: np.ndarray) -> np.ndarray:
    clipped = np.clip(p1, 1e-9, 1 - 1e-9)
    return -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))


def _leave_one_out(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group = frame.groupby("manifest_index")[column]
    count = group.transform("size").to_numpy(dtype=float)
    values = frame[column].to_numpy(dtype=float)
    total = group.transform("sum").to_numpy(dtype=float)
    square_total = frame.assign(_square=values**2).groupby("manifest_index")._square.transform("sum").to_numpy()
    if np.any(count < 2):
        raise RuntimeError("Cross-run consensus requires at least two frozen runs per manifest sample")
    other_n = count - 1
    mean = (total - values) / other_n
    variance = np.maximum((square_total - values**2) / other_n - mean**2, 0.0)
    return mean, np.sqrt(variance), other_n


def load_pool(cache_root: Path, pool: str) -> PolicyData:
    split = read_split()
    if pool not in ("EXPLORATION_POOL", "DEVELOPMENT_HOLDOUT"):
        raise ValueError(pool)
    subjects = [row["subject_id"] for row in split["assignments"] if row["pool"] == pool]
    paths = _paths(cache_root)
    frames = {name: _filtered_read(path, subjects) for name, path in paths.items()}
    reference = frames["features"][ID_COLUMNS]
    for name, frame in frames.items():
        require_false_outer(frame, name)
        if not reference.equals(frame[ID_COLUMNS]):
            raise RuntimeError(f"Router cache identity mismatch: {name}")
        if not np.array_equal(frames["features"].label.to_numpy(), frame.label.to_numpy()):
            raise RuntimeError(f"Router label mismatch: {name}")

    raw = frames["features"].copy()
    for name in ("base", "counterfactual", "geometry"):
        extra = [column for column in frames[name] if column not in ID_COLUMNS and column != "label"]
        raw = pd.concat([raw, frames[name][extra]], axis=1)
    data = pd.DataFrame(
        {
            "fold_id": raw.fold.astype(int),
            "seed_id": raw.seed.astype(int),
            "router_fold_id": raw.router_fold.astype(int),
            "manifest_index": raw.manifest_index.astype(int),
            "subject_id": raw.subject.map(_subject_key),
            "session_id": raw.session.astype(str),
            "outcome_label": raw.label.astype(int),
        }
    )
    logit_pairs = {
        "noop": ("keep_logit_0", "keep_logit_1"),
        "erase": ("erase_logit_0", "erase_logit_1"),
        "amplify": ("amplify_logit_0", "amplify_logit_1"),
        "geometry": ("geometry_logit_0", "geometry_logit_1"),
    }
    for action, (left, right) in logit_pairs.items():
        margin = raw[right].to_numpy(dtype=float) - raw[left].to_numpy(dtype=float)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(margin, -50, 50)))
        data[f"margin_{action}"] = margin
        data[f"p1_{action}"] = p1
        data[f"pred_{action}"] = (margin >= 0).astype(np.int8)
        data[f"confidence_{action}"] = np.maximum(p1, 1 - p1)
        data[f"entropy_{action}"] = _entropy_binary(p1)
    base_correct = data.pred_noop.to_numpy() == data.outcome_label.to_numpy()
    data["target_baseline_error"] = (~base_correct).astype(np.int8)
    for action in ACTIONS:
        action_correct = data[f"pred_{action}"].to_numpy() == data.outcome_label.to_numpy()
        data[f"effect_{action}"] = action_correct.astype(np.int8) - base_correct.astype(np.int8)
        data[f"flip_{action}"] = (data[f"pred_{action}"] != data.pred_noop).astype(np.int8)
        data[f"delta_margin_{action}"] = data[f"margin_{action}"] - data.margin_noop
        data[f"delta_p1_{action}"] = data[f"p1_{action}"] - data.p1_noop
        data[f"confidence_change_{action}"] = data[f"confidence_{action}"] - data.confidence_noop
    data["flip_count"] = data[[f"flip_{action}" for action in ACTIONS]].sum(axis=1)
    data["action_vote_fraction_class1"] = data[[f"pred_{action}" for action in ACTIONS]].mean(axis=1)
    data["action_margin_mean"] = data[[f"margin_{action}" for action in ACTIONS]].mean(axis=1)
    data["action_margin_std"] = data[[f"margin_{action}" for action in ACTIONS]].std(axis=1, ddof=0)
    data["action_confidence_mean"] = data[[f"confidence_{action}" for action in ACTIONS]].mean(axis=1)

    excluded = set(ID_COLUMNS + ["label"])
    original_features: list[str] = []
    for column in frames["features"].columns:
        if column in excluded:
            continue
        name = f"original_{column}"
        data[name] = frames["features"][column].to_numpy(dtype=float)
        original_features.append(name)

    cross_features: list[str] = []
    for action in ("noop", *ACTIONS):
        for kind in ("margin", "p1", "pred"):
            column = f"{kind}_{action}"
            mean, std, other_n = _leave_one_out(data, column)
            mean_name = f"other_run_mean_{column}"
            std_name = f"other_run_std_{column}"
            data[mean_name] = mean
            data[std_name] = std
            cross_features.extend((mean_name, std_name))
        data["other_run_count"] = other_n
    data["other_run_base_majority"] = (data.other_run_mean_pred_noop >= 0.5).astype(np.int8)
    data["other_run_base_disagrees"] = (data.other_run_base_majority != data.pred_noop).astype(np.int8)
    data["other_run_base_vote_strength"] = np.abs(data.other_run_mean_pred_noop - 0.5) * 2
    data["target_vs_other_base_margin"] = data.margin_noop - data.other_run_mean_margin_noop
    cross_features.extend(
        [
            "other_run_count",
            "other_run_base_majority",
            "other_run_base_disagrees",
            "other_run_base_vote_strength",
            "target_vs_other_base_margin",
        ]
    )
    repeats = data.groupby("manifest_index").manifest_index.transform("size")
    data["unit_weight"] = 1.0 / repeats
    fold_map = {
        row["subject_id"]: row["exploration_cv_fold"]
        for row in split["assignments"]
        if row["pool"] == "EXPLORATION_POOL"
    }
    data["exploration_cv_fold"] = data.subject_id.map(fold_map)
    data["pool"] = pool
    data["outer_test_used"] = False

    derived_single = [
        column
        for column in data.columns
        if column.startswith(("margin_", "p1_", "confidence_", "entropy_", "flip_", "delta_", "action_"))
    ]
    single_features = sorted(set(original_features + derived_single))
    # Explicitly forbid outcome, identity, and any effect from the model matrix.
    forbidden_tokens = ("effect_", "outcome", "target_", "subject", "manifest", "fold_id", "seed_id", "session")
    single_features = [
        column for column in single_features if not any(token in column for token in forbidden_tokens)
    ]
    cross_run_features = sorted(set(single_features + cross_features))
    return PolicyData(data.reset_index(drop=True), single_features, cross_run_features, split)
