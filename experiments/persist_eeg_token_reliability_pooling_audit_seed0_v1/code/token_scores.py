"""Source-only channel-by-time token reliability and its falsification controls.

Inputs have shape (trial, channel*time, embedding). No OUTER arrays are accepted.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


def seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def directions(z: np.ndarray, y: np.ndarray, subjects: np.ndarray, sessions: np.ndarray,
               ordered_subjects: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return d[subject,session,token,dimension] and class counts.

    Missing class/session pairs fail closed; the reliability estimand requires
    matched subject and both binary classes in both sessions.
    """
    if z.ndim != 3 or len(z) != len(y) or len(y) != len(subjects) or len(y) != len(sessions):
        raise ValueError("token/label/subject/session shape mismatch")
    if set(np.unique(y).tolist()) != {0, 1} or set(np.unique(sessions).tolist()) != {1, 2}:
        raise ValueError("expected binary MI labels and sessions 1,2")
    out = np.empty((len(ordered_subjects), 2, z.shape[1], z.shape[2]), dtype=np.float32)
    counts = np.zeros((len(ordered_subjects), 2, 2), dtype=np.int32)
    for i, subject in enumerate(ordered_subjects):
        for j, session in enumerate((1, 2)):
            means = []
            for label in (0, 1):
                mask = (subjects == subject) & (sessions == session) & (y == label)
                counts[i, j, label] = int(mask.sum())
                if not mask.any():
                    raise ValueError(f"missing subject/session/class: {subject}/{session}/{label}")
                means.append(z[mask].mean(axis=0, dtype=np.float64))
            out[i, j] = means[1] - means[0]
    return out, counts


def cosine_reliability(d: np.ndarray) -> np.ndarray:
    if d.ndim != 4 or d.shape[1] != 2:
        raise ValueError("directions must be [subject,2,token,d_model]")
    a, b = d[:, 0].astype(np.float64), d[:, 1].astype(np.float64)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    valid = (na > 1e-12) & (nb > 1e-12)
    if not valid.all():
        raise ValueError(f"{int((~valid).sum())} zero class-direction vectors")
    return np.mean(np.sum(a * b, axis=-1) / (na * nb), axis=0)


def magnitude_reliability(d: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(d.astype(np.float64), axis=-1)
    if np.any(norms <= 1e-12):
        raise ValueError("zero class-direction vector")
    return -np.mean(np.abs(np.log(norms[:, 0] / norms[:, 1])), axis=0)


def spread(r: np.ndarray) -> dict[str, float]:
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 1 or not np.isfinite(r).all():
        raise ValueError("invalid reliability vector")
    n = len(r)
    order = np.argsort(r, kind="stable")
    q = max(1, int(np.ceil(.25 * n)))
    dec = max(1, int(np.ceil(.10 * n)))
    return {"mean": float(r.mean()), "std": float(r.std()),
            "iqr": float(np.quantile(r, .75) - np.quantile(r, .25)),
            "min": float(r.min()), "max": float(r.max()),
            "top_decile_mean": float(r[order[-dec:]].mean()),
            "bottom_decile_mean": float(r[order[:dec]].mean()),
            "top_quartile_mean": float(r[order[-q:]].mean()),
            "bottom_quartile_mean": float(r[order[:q]].mean()),
            "top_bottom_quartile_gap": float(r[order[-q:]].mean() - r[order[:q]].mean())}


def pairing_null(d: np.ndarray, repeats: int, *, fold: int) -> list[dict[str, float]]:
    if repeats < 500:
        raise ValueError("pairing null requires at least 500 draws")
    rng = np.random.default_rng(seed("pairing-null", fold))
    out = []
    for repeat in range(repeats):
        permutation = rng.permutation(len(d))
        # A derangement is not required by the predeclared permutation null.
        r = cosine_reliability(np.stack((d[:, 0], d[permutation, 1]), axis=1))
        out.append({"fold": fold, "null": "SESSION_SUBJECT_PAIRING", "repeat": repeat,
                    **spread(r)})
    return out


def label_null(z: np.ndarray, y: np.ndarray, subjects: np.ndarray, sessions: np.ndarray,
               ordered_subjects: list[str], repeats: int, *, fold: int) -> list[dict[str, float]]:
    if repeats < 500:
        raise ValueError("label null requires at least 500 draws")
    rng = np.random.default_rng(seed("session2-label-null", fold))
    real_d, _ = directions(z, y, subjects, sessions, ordered_subjects)
    first = real_d[:, 0].astype(np.float32)
    out = []
    # Block the BLAS work to avoid allocating all repetitions' token tensors.
    # Each row of weights preserves the subject's original Session-2 class
    # counts exactly; only the mapping from label to trial is permuted.
    for start in range(0, repeats, 20):
        stop = min(start + 20, repeats)
        second = np.empty((stop - start, len(ordered_subjects), z.shape[1], z.shape[2]), np.float32)
        for i, subject in enumerate(ordered_subjects):
            idx = np.flatnonzero((subjects == subject) & (sessions == 2))
            original = y[idx]
            n1, n0 = int((original == 1).sum()), int((original == 0).sum())
            if min(n1, n0) < 1:
                raise ValueError("label null subject lacks one of the MI classes")
            weights = np.empty((stop - start, len(idx)), dtype=np.float32)
            for row in range(stop - start):
                shuffled = rng.permutation(original)
                weights[row] = np.where(shuffled == 1, 1 / n1, -1 / n0)
            second[:, i] = (weights @ z[idx].reshape(len(idx), -1)).reshape(stop-start, z.shape[1], z.shape[2])
        a = first[None].astype(np.float64)
        b = second.astype(np.float64)
        norm_a = np.linalg.norm(a, axis=-1)
        norm_b = np.linalg.norm(b, axis=-1)
        if np.any(norm_a <= 1e-12) or np.any(norm_b <= 1e-12):
            raise ValueError("label null generated a zero class-direction vector")
        reliability = (np.sum(a * b, axis=-1) / (norm_a * norm_b)).mean(axis=1)
        for row, repeat in enumerate(range(start, stop)):
            out.append({"fold": fold, "null": "SESSION2_WITHIN_SUBJECT_LABEL", "repeat": repeat,
                        **spread(reliability[row])})
    return out


def _top_indices(r: np.ndarray, fraction: float) -> set[int]:
    count = max(1, int(np.ceil(len(r) * fraction)))
    return set(np.argsort(-r, kind="stable")[:count].tolist())


def reproducibility(d: np.ndarray, repeats: int, *, fold: int) -> list[dict[str, float]]:
    if repeats < 200 or len(d) < 4:
        raise ValueError("split-half requires 200 draws and at least four TRAIN subjects")
    rng = np.random.default_rng(seed("split-half", fold))
    out = []
    for repeat in range(repeats):
        perm = rng.permutation(len(d))
        a, b = cosine_reliability(d[perm[:len(d)//2]]), cosine_reliability(d[perm[len(d)//2:]])
        q1, q2 = _top_indices(a, .25), _top_indices(b, .25)
        t1, t2 = _top_indices(a, .10), _top_indices(b, .10)
        out.append({"fold": fold, "repeat": repeat,
                    "spearman": float(spearmanr(a, b).statistic),
                    "pearson": float(pearsonr(a, b).statistic),
                    "top_quartile_jaccard": len(q1 & q2) / len(q1 | q2),
                    "top_decile_jaccard": len(t1 & t2) / len(t1 | t2)})
    return out


def decomposition(r: np.ndarray, channels: int, patches: int) -> dict[str, float]:
    matrix = np.asarray(r, dtype=np.float64).reshape(channels, patches)
    grand = matrix.mean()
    channel = matrix.mean(axis=1, keepdims=True) - grand
    time = matrix.mean(axis=0, keepdims=True) - grand
    interaction = matrix - grand - channel - time
    total = np.sum((matrix - grand) ** 2)
    if total <= 0:
        return {"channel_fraction": 0., "time_fraction": 0., "interaction_fraction": 0.}
    return {"channel_fraction": float(patches * np.sum(channel ** 2) / total),
            "time_fraction": float(channels * np.sum(time ** 2) / total),
            "interaction_fraction": float(np.sum(interaction ** 2) / total)}
