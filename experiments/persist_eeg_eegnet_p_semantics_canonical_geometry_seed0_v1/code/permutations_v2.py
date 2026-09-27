"""Gram-accelerated stage-wide trial-label permutation control.

Version-forward extension of the embedding-only V1 control. Session-identity
permutation rows are unchanged; the trial-label null is added at the three
earlier native EEGNet stages without repeatedly multiplying high-D trial data.
"""
from __future__ import annotations

import numpy as np

import data_geometry as dg
import decoders
import permutations
import semantics


def label_null_gram(train_relation: dict, held_x: np.ndarray, held: dg.Population,
                    sessions: tuple[int, int], draws: int, seed_parts: tuple) -> tuple[float, float, float, float]:
    owners = sorted(set(held.subject))
    train_owners = sorted({key[0] for key in train_relation})
    refs = {t: np.mean([train_relation[(s, t)] for s in train_owners], axis=0).astype(np.float64)
            for t in sessions}
    rng = np.random.default_rng(dg.seed("within-subject-session-label-null-v2", *seed_parts))
    per_cell = []
    for subject in owners:
        for session in sessions:
            ix = np.flatnonzero((held.subject == subject) & (held.session == session))
            y = held.y[ix]
            n0, n1 = int((y == 0).sum()), int((y == 1).sum())
            if min(n0, n1) == 0:
                raise RuntimeError("label permutation cell missing a class")
            x = np.asarray(held_x[ix], dtype=np.float64)
            ref = refs[session]
            ref_norm = max(float(np.linalg.norm(ref)), 1e-10)
            trial_ref = (x @ ref) / ref_norm
            gram = x @ x.T
            weights = np.where(y == 1, 1 / n1, -1 / n0).astype(np.float64)
            assignments = np.vstack([weights] + [rng.permutation(weights) for _ in range(draws)])
            numerator = assignments @ trial_ref
            squared = np.einsum("bi,ij,bj->b", assignments, gram, assignments, optimize=True)
            denominator = np.sqrt(np.maximum(squared, 1e-20))
            per_cell.append(numerator / denominator)
    statistics = np.mean(np.stack(per_cell), axis=0)
    observed, null = float(statistics[0]), statistics[1:]
    return (observed, float(null.mean()), float(np.quantile(null, .95)),
            float((1 + (null >= observed).sum()) / (draws + 1)))


def control_rows(fold: int, stage: str, geometry: dict, populations: dict,
                 sessions: tuple[int, int], draws: int = 200) -> list[dict]:
    if stage == "embedding_64d":
        raise RuntimeError("embedding V1 already has the label control; do not duplicate")
    rows = permutations.control_rows(fold, stage, geometry, populations, sessions, draws)
    train, train_raw = populations["TRAIN_GEOMETRY"]
    held, held_raw = populations["OUTER_DEVELOPMENT"]
    for family in ("FULL", "P", "C", "RANDOM"):
        for draw in (range(20) if family == "RANDOM" else (-1,)):
            basis = (dg.random_subspace(len(geometry["mu"]), geometry["q"].shape[1], fold, stage, draw)
                     if family == "RANDOM" else geometry["q"])
            train_x = decoders.coordinates(train_raw, geometry["mu"], basis, family)
            held_x = decoders.coordinates(held_raw, geometry["mu"], basis, family)
            grid = semantics.centroid_grid(train_x, train)
            _, relation = semantics.baseline_relation(grid, sessions)
            observed, null_mean, null95, p = label_null_gram(
                relation, held_x, held, sessions, draws, (fold, stage, family, draw))
            rows.append({"fold": fold, "stage": stage, "population": "OUTER_DEVELOPMENT",
                         "family": family, "random_draw": draw if family == "RANDOM" else "",
                         "control": "WITHIN_SUBJECT_SESSION_TRIAL_LABEL_PERMUTATION",
                         "quantity": "cross_subject_task_relation", "metric": "cosine_to_TRUE_TRAIN_reference",
                         "observed": observed, "null_mean": null_mean, "null_p95": null95,
                         "one_sided_permutation_p": p, "permutations": draws,
                         "target_labels_used_for_fit": False,
                         "label_permutation_scope": "OUTER_DEVELOPMENT evaluation trials; TRAIN reference frozen"})
    return rows
