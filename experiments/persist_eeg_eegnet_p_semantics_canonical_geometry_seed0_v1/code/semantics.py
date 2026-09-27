"""Centroid-based P/C semantic decomposition (frozen representations only)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

import data_geometry as dg


def centroid_grid(x: np.ndarray, pop: dg.Population) -> dict[tuple[str, int, int], np.ndarray]:
    result = {}
    for subject in sorted(set(pop.subject.astype(str))):
        for session in sorted(set(pop.session.astype(int))):
            for label in (0, 1):
                mask = (pop.subject == subject) & (pop.session == session) & (pop.y == label)
                if not mask.any():
                    raise RuntimeError(f"missing subject/session/class cell {pop.role} {subject} {session} {label}")
                result[(subject, session, label)] = x[mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    return result


def decomposed_centroids(raw: dict, mu: np.ndarray, q: np.ndarray, family: str) -> dict:
    return {key: dg.representation(value[None, :], mu, q, family)[0] for key, value in raw.items()}


def baseline_relation(centroids: dict, sessions: tuple[int, int]) -> tuple[dict, dict]:
    subjects = sorted({key[0] for key in centroids})
    baseline, relation = {}, {}
    for subject in subjects:
        for session in sessions:
            a, b = centroids[(subject, session, 0)], centroids[(subject, session, 1)]
            baseline[(subject, session)] = (a + b) / 2
            relation[(subject, session)] = b - a
    return baseline, relation


def sim(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a, b = a.astype(np.float64), b.astype(np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if min(na, nb) < 1e-10:
        return {"cosine": float("nan"), "pearson": float("nan"),
                "normalized_euclidean": float("nan"), "sign_agreement": float("nan"),
                "optimal_scalar_residual": float("nan")}
    cosine = float(np.dot(a, b) / (na * nb))
    ac, bc = a - a.mean(), b - b.mean()
    denom = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    pearson = float(np.dot(ac, bc) / denom) if denom >= 1e-10 else float("nan")
    scale = float(np.dot(a, b) / (nb * nb))
    return {"cosine": cosine, "pearson": pearson,
            "normalized_euclidean": float(np.linalg.norm(a / na - b / nb)),
            "sign_agreement": float(np.mean(np.sign(a) == np.sign(b))),
            "optimal_scalar_residual": float(np.linalg.norm(a - scale * b) / na)}


def subject_bootstrap(values: list[float], *parts: object) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=np.float64)
    if not len(v) or not np.isfinite(v).all():
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(dg.seed("subject-bootstrap", *parts))
    sample = rng.integers(0, len(v), size=(20_000, len(v)))
    means = v[sample].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def persistence_rows(vectors: dict, *, role: str, fold: int, stage: str, family: str,
                     draw: int, kind: str, sessions: tuple[int, int]) -> list[dict]:
    subjects = sorted({key[0] for key in vectors})
    scored = {subject: sim(vectors[(subject, sessions[0])], vectors[(subject, sessions[1])]) for subject in subjects}
    rows = []
    for metric in ("cosine", "pearson", "normalized_euclidean"):
        mean, lo, hi = subject_bootstrap([scored[s][metric] for s in subjects], fold, stage, role, family, draw, kind, metric)
        rows.append({"fold": fold, "seed": 0, "stage": stage, "population": role,
                     "family": family, "random_draw": draw if family == "RANDOM" else "",
                     "quantity": kind, "metric": metric, "subject_equal_mean": mean,
                     "bootstrap_CI_low": lo, "bootstrap_CI_high": hi,
                     "bootstrap_draws": 20_000, "subjects": len(subjects)})
    return rows


def cross_subject_rows(train_relation: dict, target_relation: dict, *, role: str, fold: int,
                       stage: str, family: str, draw: int, sessions: tuple[int, int]) -> list[dict]:
    train_subjects = sorted({key[0] for key in train_relation})
    held_subjects = sorted({key[0] for key in target_relation})
    rows = []
    for target_session in sessions:
        for ref_session in sessions:
            per_subject = defaultdict(list)
            for subject in held_subjects:
                references = [s for s in train_subjects if role != "TRAIN_GEOMETRY" or s != subject]
                if not references:
                    raise RuntimeError("LOSO task relation has no reference subjects")
                ref = np.mean([train_relation[(s, ref_session)] for s in references], axis=0)
                values = sim(target_relation[(subject, target_session)], ref)
                for metric in ("cosine", "pearson", "sign_agreement", "optimal_scalar_residual"):
                    per_subject[metric].append(values[metric])
            for metric, values in per_subject.items():
                mean, lo, hi = subject_bootstrap(values, fold, stage, role, family, draw,
                                                 target_session, ref_session, metric)
                rows.append({"fold": fold, "seed": 0, "stage": stage, "population": role,
                             "family": family, "random_draw": draw if family == "RANDOM" else "",
                             "target_session": target_session, "reference_session": ref_session,
                             "reference_population": "TRAIN_GEOMETRY_LOSO" if role == "TRAIN_GEOMETRY" else "TRAIN_GEOMETRY_ONLY",
                             "metric": metric, "subject_equal_mean": mean,
                             "bootstrap_CI_low": lo, "bootstrap_CI_high": hi,
                             "bootstrap_draws": 20_000, "subjects": len(held_subjects)})
    return rows


def variance_rows(centroids: dict, *, role: str, fold: int, stage: str, family: str,
                  draw: int, sessions: tuple[int, int]) -> list[dict]:
    subjects = sorted({key[0] for key in centroids})
    cube = np.stack([[ [centroids[(s, t, y)] for y in (0, 1)] for t in sessions] for s in subjects]).astype(np.float64)
    grand = cube.mean(axis=(0, 1, 2), keepdims=True)
    subject = cube.mean(axis=(1, 2), keepdims=True) - grand
    session = cube.mean(axis=(0, 2), keepdims=True) - grand
    label = cube.mean(axis=(0, 1), keepdims=True) - grand
    subject_class = cube.mean(axis=1, keepdims=True) - grand - subject - label
    session_class = cube.mean(axis=0, keepdims=True) - grand - session - label
    residual = cube - grand - subject - session - label - subject_class - session_class
    total = float(np.square(cube - grand).sum())
    effects = {"subject": subject, "session": session, "class": label,
               "subject_x_class": subject_class, "session_x_class": session_class,
               "residual_including_subject_x_session": residual}
    rows = []
    for name, value in effects.items():
        count = int(np.prod(cube.shape[:-1]) / np.prod(value.shape[:-1]))
        ss = float(np.square(value).sum()) * count
        rows.append({"fold": fold, "stage": stage, "population": role, "family": family,
                     "random_draw": draw if family == "RANDOM" else "", "effect": name,
                     "sum_squares": ss, "fraction": ss / total if total > 1e-12 else float("nan"),
                     "total_sum_squares": total, "subjects": len(subjects),
                     "decomposition": "balanced three-way centroid ANOVA; subject-session and triple interaction in residual"})
    if total > 1e-12 and abs(sum(r["sum_squares"] for r in rows) - total) / total > 1e-5:
        raise RuntimeError("variance decomposition does not close")
    return rows


def summarize_centroids(fold: int, stage: str, geometry: dict, populations: dict[str, tuple[dg.Population, np.ndarray]],
                        sessions: tuple[int, int], random_draws: int = 20) -> dict[str, list[dict]]:
    result = {"BASELINE_PERSISTENCE": [], "TASK_RELATION_WITHIN_SUBJECT": [],
              "TASK_RELATION_CROSS_SUBJECT": [], "REPRESENTATION_VARIANCE_DECOMPOSITION": [],
              "ENERGY_COVARIANCE": []}
    raw = {role: centroid_grid(x, pop) for role, (pop, x) in populations.items()}
    mu, q = geometry["mu"], geometry["q"]
    for family in ("FULL", "P", "C", "RANDOM"):
        draws = range(random_draws) if family == "RANDOM" else (-1,)
        for draw in draws:
            basis = dg.random_subspace(len(mu), q.shape[1], fold, stage, draw) if family == "RANDOM" else q
            views = {role: decomposed_centroids(grid, mu, basis, family) for role, grid in raw.items()}
            br = {role: baseline_relation(grid, sessions) for role, grid in views.items()}
            train_relation = br["TRAIN_GEOMETRY"][1]
            for role, (baseline, relation) in br.items():
                common = dict(role=role, fold=fold, stage=stage, family=family, draw=draw, sessions=sessions)
                result["BASELINE_PERSISTENCE"].extend(persistence_rows(baseline, kind="class_common_baseline", **common))
                result["TASK_RELATION_WITHIN_SUBJECT"].extend(persistence_rows(relation, kind="class_relation", **common))
                result["TASK_RELATION_CROSS_SUBJECT"].extend(cross_subject_rows(train_relation, relation, **common))
                result["REPRESENTATION_VARIANCE_DECOMPOSITION"].extend(variance_rows(views[role], **common))
                values = np.stack(list(views[role].values())).astype(np.float64)
                result["ENERGY_COVARIANCE"].append({"fold": fold, "stage": stage, "population": role,
                    "family": family, "random_draw": draw if family == "RANDOM" else "",
                    "mean_centroid_squared_norm": float(np.square(values).sum(axis=1).mean()),
                    "centroid_total_variance": float(np.square(values - values.mean(axis=0)).sum(axis=1).mean()),
                    "rank": int(basis.shape[1])})
    return result
