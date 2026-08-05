from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable

import pandas as pd

MASTER_SALT = "contextual-risk-master-split-v1-20260805"
FOLD_SALT = "contextual-risk-screening-folds-v1"


def _u64(dataset: str, subject_id: str, salt: str) -> int:
    digest = hashlib.sha256(f"{dataset}|{subject_id}|{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def build_master_cohorts(
    dataset: str, subjects: Iterable[str], salt: str = MASTER_SALT
) -> pd.DataFrame:
    unique = sorted(set(map(str, subjects)))
    ranked = sorted(unique, key=lambda subject: (_u64(dataset, subject, salt), subject))
    n = len(ranked)
    n_dev, n_cal = math.floor(0.60 * n), math.floor(0.20 * n)
    if min(n_dev, n_cal, n - n_dev - n_cal) < 15 or n_dev < 30:
        raise ValueError(
            f"{dataset} cannot use 60/20/20: n={n}, counts={(n_dev,n_cal,n-n_dev-n_cal)}"
        )
    rows = []
    for rank, subject in enumerate(ranked):
        if rank < n_dev:
            cohort = "method_development"
        elif rank < n_dev + n_cal:
            cohort = "formal_calibration"
        else:
            cohort = "internal_final_evaluation"
        rows.append(
            {
                "dataset": dataset,
                "subject_id": subject,
                "master_cohort": cohort,
                "master_rank": rank,
                "master_hash_u64": _u64(dataset, subject, salt),
                "master_salt": salt,
            }
        )
    return pd.DataFrame(rows)


def screening_fold(dataset: str, subject_id: str, salt: str = FOLD_SALT) -> int:
    """Deterministic hash fold shared by every source-head seed."""
    return _u64(dataset, subject_id, salt) % 5


def attach_screening_folds(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["screening_fold"] = [
        screening_fold(str(ds), str(subject))
        if cohort == "method_development"
        else -1
        for ds, subject, cohort in zip(
            result.dataset, result.subject_id, result.master_cohort, strict=True
        )
    ]
    return result
