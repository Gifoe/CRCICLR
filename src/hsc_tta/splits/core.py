from __future__ import annotations

import numpy as np
import re
from sklearn.model_selection import GroupKFold

TARGETS = {
    "hmc": [("task_head_train", 70), ("meta_risk_train", 35), ("conformal_calibration", 20), ("final_test", 26)],
    "eegmmidb": [("task_head_train", 45), ("meta_risk_train", 30), ("conformal_calibration", 15), ("final_test", 19)],
}


def _cap_stratum(subject_id: str) -> str:
    stem = subject_id.split(":", 1)[-1]
    match = re.match(r"[A-Za-z]+", stem)
    return match.group(0).lower() if match else "unknown"


def _stratified_cap_calibration(ids: list[str], n_cal: int, rng: np.random.Generator) -> list[str]:
    groups: dict[str, list[str]] = {}
    for subject_id in ids:
        groups.setdefault(_cap_stratum(subject_id), []).append(subject_id)
    shuffled = {
        name: [values[i] for i in rng.permutation(len(values))]
        for name, values in sorted(groups.items())
    }
    ideal = {name: n_cal * len(values) / len(ids) for name, values in shuffled.items()}
    ensure_each = n_cal >= len(shuffled)
    allocation = {
        name: min(len(values), max(1 if ensure_each else 0, int(np.floor(ideal[name]))))
        for name, values in shuffled.items()
    }
    while sum(allocation.values()) < n_cal:
        candidates = [name for name, values in shuffled.items() if allocation[name] < len(values)]
        chosen = max(candidates, key=lambda name: (ideal[name] - allocation[name], name))
        allocation[chosen] += 1
    while sum(allocation.values()) > n_cal:
        floor = 1 if ensure_each else 0
        candidates = [name for name in shuffled if allocation[name] > floor]
        chosen = max(candidates, key=lambda name: (allocation[name] - ideal[name], name))
        allocation[chosen] -= 1
    return sorted(subject_id for name, values in shuffled.items() for subject_id in values[: allocation[name]])


def make_subject_split(subject_ids: list[str], dataset: str, seed: int) -> dict[str, list[str]]:
    ids = sorted(set(subject_ids))
    if len(ids) != len(subject_ids):
        raise ValueError("subject_ids must be unique")
    rng = np.random.default_rng(seed)
    ordered = [ids[i] for i in rng.permutation(len(ids))]
    if dataset == "cap":
        n_cal = 25 if len(ids) >= 45 else max(20, round(len(ids) * 0.25))
        if len(ids) - n_cal < 1 or n_cal > len(ids):
            raise ValueError("insufficient CAP subjects for formal split")
        calibration = _stratified_cap_calibration(ids, n_cal, rng)
        result = {"target_site_calibration": calibration, "external_final_test": sorted(set(ids) - set(calibration))}
    else:
        if dataset not in TARGETS:
            raise ValueError("unknown dataset")
        targets = TARGETS[dataset]
        required_min = 35 if dataset == "hmc" else 27
        if len(ids) < required_min:
            raise ValueError("insufficient subjects for calibration/test minima")
        target_total = sum(n for _, n in targets)
        counts = [n for _, n in targets] if len(ids) == target_total else [int(np.floor(len(ids) * n / target_total)) for _, n in targets]
        counts[-1] += len(ids) - sum(counts)
        min_cal, min_test = (15, 20) if dataset == "hmc" else (12, 15)
        counts[-2] = max(counts[-2], min_cal)
        counts[-1] = max(counts[-1], min_test)
        overflow = sum(counts) - len(ids)
        for idx in [0, 1]:
            take = min(overflow, max(0, counts[idx] - 1)); counts[idx] -= take; overflow -= take
        if overflow:
            raise ValueError("insufficient subjects after enforcing calibration/test minima")
        result, cursor = {}, 0
        for (role, _), count in zip(targets, counts):
            result[role] = sorted(ordered[cursor:cursor + count]); cursor += count
    validate_subject_split(result, ids)
    return result


def validate_subject_split(split: dict[str, list[str]], all_subjects: list[str] | None = None) -> None:
    roles = [set(v) for v in split.values()]
    for i, left in enumerate(roles):
        for right in roles[i + 1:]:
            if left & right:
                raise ValueError("subject leakage across split roles")
    union = set().union(*roles) if roles else set()
    if all_subjects is not None and union != set(all_subjects):
        raise ValueError("split does not cover subjects exactly")


def make_internal_subject_split(
    roles: dict[str, list[str]], dataset: str, seed: int
) -> dict[str, object]:
    """Freeze task-head fit/val and meta-risk five-fold subject assignments."""
    if dataset == "cap":
        if set(roles) != {"target_site_calibration", "external_final_test"}:
            raise ValueError("CAP must contain only calibration and external final-test roles")
        validate_subject_split(roles)
        return {
            "dataset": "cap",
            "seed": int(seed),
            "task_head": "inherit_hmc",
            "meta_predictor": "inherit_hmc",
            "task_head_fit": [],
            "task_head_val": [],
            "meta_risk_folds": {},
        }
    counts = {"hmc": (60, 10, 35), "eegmmidb": (38, 7, 30)}
    if dataset not in counts:
        raise ValueError("unknown dataset")
    n_fit, n_val, n_meta = counts[dataset]
    task_subjects = sorted(roles.get("task_head_train", []))
    meta_subjects = sorted(roles.get("meta_risk_train", []))
    if len(task_subjects) != n_fit + n_val or len(meta_subjects) != n_meta:
        raise ValueError("large split role counts do not match the frozen protocol")
    rng = np.random.default_rng(seed)
    task_order = [task_subjects[i] for i in rng.permutation(len(task_subjects))]
    meta_order = [meta_subjects[i] for i in rng.permutation(len(meta_subjects))]
    fit = sorted(task_order[:n_fit])
    val = sorted(task_order[n_fit:])
    folds: dict[str, int] = {}
    groups = np.asarray(meta_order)
    dummy = np.zeros((len(groups), 1))
    for fold, (_, valid) in enumerate(GroupKFold(n_splits=5).split(dummy, groups=groups)):
        for index in valid:
            folds[str(groups[index])] = int(fold)
    if set(fit) & set(val) or set(fit) | set(val) != set(task_subjects):
        raise AssertionError("internal task-head split is not subject-disjoint and exhaustive")
    if set(folds) != set(meta_subjects) or set(folds.values()) != set(range(5)):
        raise AssertionError("meta-risk folds are incomplete")
    return {
        "dataset": dataset,
        "seed": int(seed),
        "task_head_fit": fit,
        "task_head_val": val,
        "meta_risk_folds": dict(sorted(folds.items())),
    }
