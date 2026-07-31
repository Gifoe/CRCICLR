from __future__ import annotations

import numpy as np

TARGETS = {
    "hmc": [("task_head_train", 70), ("meta_risk_train", 35), ("conformal_calibration", 20), ("final_test", 26)],
    "eegmmidb": [("task_head_train", 45), ("meta_risk_train", 30), ("conformal_calibration", 15), ("final_test", 19)],
}


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
        result = {"target_site_calibration": sorted(ordered[:n_cal]), "external_final_test": sorted(ordered[n_cal:])}
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

