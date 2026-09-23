"""Frozen, split-isolated recipient/donor selection for mechanism evaluation.

Labels only define within-subject/session evaluation strata and donor classes.
No mapping, direction, checkpoint, or hyperparameter is fitted here.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


CAP = 8


@dataclass(frozen=True)
class TrialView:
    x: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    split: str


@dataclass(frozen=True)
class PairPlan:
    recipient: np.ndarray
    donor: np.ndarray
    donor_label: np.ndarray
    coverage: tuple[dict[str, object], ...]


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, ("pc-mechanism-pairs-v1", *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def trial_view(data: dict[str, object], split: str) -> TrialView:
    """Build one legal data view using only the named upstream development keys.

    In particular, this interface has no final-heldout key or fallback path.
    TRAIN includes source and future TRAIN partitions; OUTER_DEVELOPMENT
    includes source and future outer-development partitions. It never mixes
    TRAIN and outer data, including when a source/future session ID is equal.
    """
    fields = {
        "TRAIN": (("train_x", "train_y", "train_subjects"),
                  ("future_train_x", "future_train_y", "future_train_subjects")),
        "OUTER_DEVELOPMENT": (("outer_source_x", "outer_source_y", "outer_source_subjects"),
                              ("outer_future_x", "outer_future_y", "outer_future_subjects")),
    }
    if split not in fields:
        raise ValueError("split must be TRAIN or OUTER_DEVELOPMENT")
    if "source_session" not in data or "future_session" not in data:
        raise KeyError("source/future session ID missing")
    chunks = []
    for names, session in zip(fields[split], (data["source_session"], data["future_session"])):
        if any(name not in data for name in names):
            raise KeyError(f"missing legal {split} field: {names}")
        x, y, subjects = (np.asarray(data[name]) for name in names)
        if not (len(x) == len(y) == len(subjects)):
            raise ValueError(f"unaligned {split} rows in {names}")
        chunks.append((x, y, subjects.astype(str), np.full(len(y), int(session), np.int64)))
    return TrialView(
        np.concatenate([chunk[0] for chunk in chunks]),
        np.concatenate([chunk[1] for chunk in chunks]).astype(np.int64),
        np.concatenate([chunk[2] for chunk in chunks]),
        np.concatenate([chunk[3] for chunk in chunks]), split,
    )


def _ordered(values: np.ndarray) -> list[object]:
    # Numeric biological-subject IDs retain numeric ordering. Other IDs use
    # lexical ordering, without relying on their original input order.
    return sorted(set(values.tolist()), key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))


def select_pairs(
    activations: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    *,
    model: str,
    task: str,
    fold: int,
    stage: str,
    split: str,
    cap: int = CAP,
) -> PairPlan:
    """Cap recipients at eight per stratum; match other-class donors by norm.

    Donors may come from all rows in their eligible class, including rows not
    sampled as recipients. Equal-distance donors use the lowest input row
    index. TRAIN and outer must be passed in separate calls and distinct split
    keys; the function never combines them.
    """
    if split not in {"TRAIN", "OUTER_DEVELOPMENT"}:
        raise ValueError("split must be TRAIN or OUTER_DEVELOPMENT")
    if cap != CAP:
        raise ValueError(f"recipient cap is frozen at {CAP}")
    a = np.asarray(activations)
    y = np.asarray(labels)
    s = np.asarray(subjects).astype(str)
    se = np.asarray(sessions)
    if a.ndim != 2 or not (len(a) == len(y) == len(s) == len(se)) or not len(a):
        raise ValueError("nonempty, aligned flattened activations and metadata required")
    if not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
        raise ValueError("activations must be finite numeric values")
    norms = np.linalg.norm(a.astype(np.float64, copy=False), axis=1)
    return select_pairs_from_norms(norms, y, s, se, model=model, task=task,
                                   fold=fold, stage=stage, split=split, cap=cap)


def select_pairs_from_norms(
    norms: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    *, model: str, task: str, fold: int, stage: str, split: str,
    cap: int = CAP,
) -> PairPlan:
    """Equivalent plan from a streaming pass of activation norms.

    This permits all legal donors to be searched without retaining every
    high-dimensional activation in physical memory.
    """
    if split not in {"TRAIN", "OUTER_DEVELOPMENT"}:
        raise ValueError("split must be TRAIN or OUTER_DEVELOPMENT")
    if cap != CAP:
        raise ValueError(f"recipient cap is frozen at {CAP}")
    norms = np.asarray(norms, np.float64)
    y = np.asarray(labels)
    s = np.asarray(subjects).astype(str)
    se = np.asarray(sessions)
    if norms.ndim != 1 or not (len(norms) == len(y) == len(s) == len(se)) or not len(norms):
        raise ValueError("nonempty aligned activation norms and metadata required")
    if not np.isfinite(norms).all() or np.any(norms < 0):
        raise ValueError("activation norms must be finite and nonnegative")
    if not np.issubdtype(y.dtype, np.integer) or not np.issubdtype(se.dtype, np.integer):
        raise ValueError("labels and sessions must be integer arrays")
    if any(not item for item in s):
        raise ValueError("empty biological-subject ID")
    if not model or not task or not stage or not isinstance(fold, int) or fold not in range(5):
        raise ValueError("invalid frozen cell identity")

    recipients: list[int] = []
    donors: list[int] = []
    donor_labels: list[int] = []
    coverage: list[dict[str, object]] = []
    for subject in _ordered(s):
        subject_mask = s == subject
        for session in _ordered(se[subject_mask]):
            group = subject_mask & (se == session)
            group_labels = _ordered(y[group])
            for label in group_labels:
                eligible = np.flatnonzero(group & (y == label))
                rng = np.random.default_rng(stable_seed(model, task, fold, stage, split, subject, session, label))
                chosen = np.sort(rng.choice(eligible, size=min(cap, len(eligible)), replace=False))
                other_labels = [other for other in group_labels if other != label]
                n_pairs = 0
                for recipient in chosen:
                    for other in other_labels:
                        candidates = np.flatnonzero(group & (y == other))
                        # np.flatnonzero returns increasing original row indices,
                        # and np.argmin selects the first tied candidate.
                        donor = int(candidates[np.argmin(np.abs(norms[candidates] - norms[recipient]))])
                        recipients.append(int(recipient))
                        donors.append(donor)
                        donor_labels.append(int(other))
                        n_pairs += 1
                coverage.append({
                    "subject": str(subject), "session": int(session), "class": int(label),
                    "eligible_recipients": int(len(eligible)), "selected_recipients": int(len(chosen)),
                    "other_classes": [int(other) for other in other_labels],
                    "pair_count": n_pairs, "missing_eligible_donor_class": not bool(other_labels),
                })
    return PairPlan(np.asarray(recipients, dtype=np.int64), np.asarray(donors, dtype=np.int64),
                    np.asarray(donor_labels, dtype=np.int64), tuple(coverage))
