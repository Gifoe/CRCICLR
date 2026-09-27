"""Biological-session matching and within-cell label permutation controls."""
from __future__ import annotations

import numpy as np

import data_geometry as dg
import decoders
import semantics


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return semantics.sim(a, b)["cosine"]


def _paired_null(vectors: dict, sessions: tuple[int, int], permutations: int,
                 seed_parts: tuple) -> tuple[float, float, float, float]:
    owners = sorted({key[0] for key in vectors})
    first = np.stack([vectors[(s, sessions[0])] for s in owners])
    second = np.stack([vectors[(s, sessions[1])] for s in owners])
    first = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-10)
    second = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-10)
    matrix = first @ second.T
    observed = float(np.diag(matrix).mean())
    rng = np.random.default_rng(dg.seed("session-pair-null", *seed_parts))
    null = [float(matrix[np.arange(len(owners)), rng.permutation(len(owners))].mean())
            for _ in range(permutations)]
    return (observed, float(np.mean(null)), float(np.quantile(null, .95)),
            float((1 + sum(value >= observed for value in null)) / (permutations + 1)))


def _label_null(train_relation: dict, held_x: np.ndarray, held: dg.Population,
                sessions: tuple[int, int], permutations: int, seed_parts: tuple):
    owners = sorted(set(held.subject))
    train_owners = sorted({key[0] for key in train_relation})
    refs = {session: np.mean([train_relation[(s, session)] for s in train_owners], axis=0)
            for session in sessions}
    true_grid = semantics.centroid_grid(held_x, held)
    true_d = {key: true_grid[(key[0], key[1], 1)] - true_grid[(key[0], key[1], 0)]
              for key in [(s, t) for s in owners for t in sessions]}
    observed = float(np.mean([_cosine(true_d[(s, t)], refs[t]) for s in owners for t in sessions]))
    rng = np.random.default_rng(dg.seed("within-subject-session-label-null", *seed_parts))
    cells = {(s, t): np.flatnonzero((held.subject == s) & (held.session == t)) for s in owners for t in sessions}
    null = []
    for _ in range(permutations):
        scored = []
        for (subject, session), ix in cells.items():
            shuffled = rng.permutation(held.y[ix])
            if set(shuffled) != {0, 1}:
                raise RuntimeError("permuted cell lost a class")
            d = held_x[ix][shuffled == 1].mean(axis=0) - held_x[ix][shuffled == 0].mean(axis=0)
            scored.append(_cosine(d, refs[session]))
        null.append(float(np.mean(scored)))
    return (observed, float(np.mean(null)), float(np.quantile(null, .95)),
            float((1 + sum(value >= observed for value in null)) / (permutations + 1)))


def control_rows(fold: int, stage: str, geometry: dict, populations: dict,
                 sessions: tuple[int, int], permutations: int = 200) -> list[dict]:
    rows = []
    for family in ("FULL", "P", "C", "RANDOM"):
        for draw in (range(20) if family == "RANDOM" else (-1,)):
            basis = (dg.random_subspace(len(geometry["mu"]), geometry["q"].shape[1], fold, stage, draw)
                     if family == "RANDOM" else geometry["q"])
            role_views = {}
            for role, (pop, raw) in populations.items():
                x = decoders.coordinates(raw, geometry["mu"], basis, family)
                grid = semantics.centroid_grid(x, pop)
                baseline, relation = semantics.baseline_relation(grid, sessions)
                role_views[role] = (x, baseline, relation)
                energy = float(np.square(x.astype(np.float64)).sum(axis=1).mean())
                variance = float(np.square(x.astype(np.float64) - x.mean(axis=0)).sum(axis=1).mean())
                for quantity, vectors in (("baseline", baseline), ("task_relation", relation)):
                    observed, null_mean, null95, p = _paired_null(
                        vectors, sessions, permutations, (fold, stage, role, family, draw, quantity))
                    rows.append({"fold": fold, "stage": stage, "population": role,
                                 "family": family, "random_draw": draw if family == "RANDOM" else "",
                                 "control": "SESSION2_SUBJECT_ID_PERMUTATION",
                                 "quantity": quantity, "metric": "within_biological_subject_cosine",
                                 "observed": observed, "null_mean": null_mean, "null_p95": null95,
                                 "one_sided_permutation_p": p, "permutations": permutations,
                                 "trial_mean_squared_norm": energy, "trial_total_variance": variance,
                                 "representation_rank": basis.shape[1] if family in ("P", "RANDOM") else "ambient_or_complement"})
            if stage == "embedding_64d":
                held, _ = populations["OUTER_DEVELOPMENT"]
                observed, null_mean, null95, p = _label_null(
                    role_views["TRAIN_GEOMETRY"][2], role_views["OUTER_DEVELOPMENT"][0], held,
                    sessions, permutations, (fold, stage, family, draw))
                rows.append({"fold": fold, "stage": stage, "population": "OUTER_DEVELOPMENT",
                             "family": family, "random_draw": draw if family == "RANDOM" else "",
                             "control": "WITHIN_SUBJECT_SESSION_TRIAL_LABEL_PERMUTATION",
                             "quantity": "cross_subject_task_relation", "metric": "cosine_to_TRUE_TRAIN_reference",
                             "observed": observed, "null_mean": null_mean, "null_p95": null95,
                             "one_sided_permutation_p": p, "permutations": permutations,
                             "target_labels_used_for_fit": False,
                             "label_permutation_scope": "OUTER_DEVELOPMENT evaluation trials; TRAIN reference frozen"})
    return rows
