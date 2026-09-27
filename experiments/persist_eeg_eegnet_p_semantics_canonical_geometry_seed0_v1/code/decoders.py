"""TRAIN-only frozen linear decoders and descriptive subject/task probes."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

import data_geometry as dg
import semantics as sm


def coordinates(x: np.ndarray, mu: np.ndarray, q: np.ndarray, family: str) -> np.ndarray:
    centered = np.asarray(x - mu, dtype=np.float32)
    if family == "FULL":
        return centered
    z = centered @ q
    if family in ("P", "RANDOM"):
        # Orthonormal coordinates preserve all information in the projected P
        # subspace and avoid a redundant high-dimensional parameterization.
        return z.astype(np.float32)
    if family == "C":
        return (centered - z @ q.T).astype(np.float32)
    raise KeyError(family)


def centroid_training(pop: dg.Population, x: np.ndarray, sessions: tuple[int, int]):
    grid = sm.centroid_grid(x, pop)
    keys = sorted(grid)
    matrix = np.stack([grid[key] for key in keys]).astype(np.float32)
    labels = np.asarray([key[2] for key in keys], dtype=np.int64)
    subjects = np.asarray([key[0] for key in keys], dtype=str)
    session = np.asarray([key[1] for key in keys], dtype=np.int64)
    if set(session) != set(sessions):
        raise RuntimeError("TRAIN centroid session mismatch")
    return matrix, labels, subjects, session


def fit_linear(x: np.ndarray, y: np.ndarray, *, standardizer: StandardScaler | None = None):
    scale = standardizer or StandardScaler()
    if standardizer is None:
        scale.fit(x)
    z = scale.transform(x)
    solver = "liblinear" if len(np.unique(y)) == 2 else "lbfgs"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = LogisticRegression(C=1.0, penalty="l2", solver=solver,
                                   dual=(solver == "liblinear" and z.shape[1] > z.shape[0]),
                                   max_iter=2000, random_state=0)
        model.fit(z, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError("TRAIN-only linear decoder failed to converge")
    return scale, model


def predict_batched(scale: StandardScaler, model: LogisticRegression, x: np.ndarray,
                    mu: np.ndarray, q: np.ndarray, family: str, batch: int = 64) -> np.ndarray:
    parts = []
    for i in range(0, len(x), batch):
        z = coordinates(x[i:i + batch], mu, q, family)
        parts.append(model.predict_proba(scale.transform(z)).astype(np.float32))
    return np.concatenate(parts)


def metrics(y: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> dict:
    pred = classes[probabilities.argmax(axis=1)]
    return {"BA": float(balanced_accuracy_score(y, pred)),
            "macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "NLL": float(log_loss(y, probabilities, labels=classes)),
            "rows": len(y)}


def decoder_rows(fold: int, stage: str, geometry: dict, populations: dict[str, tuple[dg.Population, np.ndarray]],
                 sessions: tuple[int, int], family: str, draw: int):
    mu, q = geometry["mu"], geometry["q"]
    if family == "RANDOM":
        q = dg.random_subspace(len(mu), q.shape[1], fold, stage, draw)
    train_pop, train_raw = populations["TRAIN_GEOMETRY"]
    cent, labels, subjects, session = centroid_training(train_pop, train_raw, sessions)
    source = coordinates(cent, mu, q, family)
    common = {"fold": fold, "seed": 0, "stage": stage, "family": family,
              "random_draw": draw if family == "RANDOM" else "",
              "fit_unit": "TRAIN subject-session-class centroid; subject-equal", "C": 1.0,
              "target_decoder_refit": False}
    transfer_rows, cross_rows, boundary_rows = [], [], []
    fitted = {}
    for train_session in sessions:
        fit = session == train_session
        scaler, decoder = fit_linear(source[fit], labels[fit])
        fitted[train_session] = (scaler, decoder)
        target_session = next(s for s in sessions if s != train_session)
        for role, (pop, raw) in populations.items():
            ix = pop.session == target_session
            if not ix.any():
                raise RuntimeError(f"missing session transfer target {role} {target_session}")
            probabilities = predict_batched(scaler, decoder, raw[ix], mu, q, family)
            transfer_rows.append({**common, "fit_population": "TRAIN_GEOMETRY",
                                  "fit_session": train_session, "eval_population": role,
                                  "eval_session": target_session,
                                  **metrics(pop.y[ix], probabilities, decoder.classes_)})
    scaler_all, decoder_all = fit_linear(source, labels)
    for role in ("CHECKPOINT_VALIDATION", "OUTER_DEVELOPMENT"):
        pop, raw = populations[role]
        probabilities = predict_batched(scaler_all, decoder_all, raw, mu, q, family)
        cross_rows.append({**common, "fit_population": "TRAIN_GEOMETRY_BOTH_SESSIONS",
                           "eval_population": role, "eval_session": "both",
                           **metrics(pop.y, probabilities, decoder_all.classes_)})
        for target_session in sessions:
            ix = pop.session == target_session
            probabilities = predict_batched(scaler_all, decoder_all, raw[ix], mu, q, family)
            cross_rows.append({**common, "fit_population": "TRAIN_GEOMETRY_BOTH_SESSIONS",
                               "eval_population": role, "eval_session": target_session,
                               **metrics(pop.y[ix], probabilities, decoder_all.classes_)})
    # Compare normal vectors in the same original feature coordinates despite
    # separate fit-only scalers for the two frozen session decoders.
    s1, m1 = fitted[sessions[0]]
    s2, m2 = fitted[sessions[1]]
    w1, w2 = m1.coef_[0] / s1.scale_, m2.coef_[0] / s2.scale_
    b1 = float(m1.intercept_[0] - w1 @ s1.mean_)
    b2 = float(m2.intercept_[0] - w2 @ s2.mean_)
    norm1, norm2 = float(np.linalg.norm(w1)), float(np.linalg.norm(w2))
    angle = float(np.degrees(np.arccos(np.clip(float(w1 @ w2) / max(norm1 * norm2, 1e-12), -1, 1))))
    agreements = []
    for _, (pop, raw) in populations.items():
        p1 = predict_batched(s1, m1, raw, mu, q, family)
        p2 = predict_batched(s2, m2, raw, mu, q, family)
        agreements.append(float(np.mean(m1.classes_[p1.argmax(1)] == m2.classes_[p2.argmax(1)])))
    boundary_rows.append({**common, "session1": sessions[0], "session2": sessions[1],
                          "normal_cosine": float(w1 @ w2 / max(norm1 * norm2, 1e-12)),
                          "normal_angle_degrees": angle, "raw_bias_shift": b2 - b1,
                          "prediction_agreement_population_equal": float(np.mean(agreements))})
    return transfer_rows, cross_rows, boundary_rows


def probe_rows(fold: int, stage: str, geometry: dict, populations: dict[str, tuple[dg.Population, np.ndarray]],
               sessions: tuple[int, int], family: str, draw: int):
    mu, q = geometry["mu"], geometry["q"]
    if family == "RANDOM":
        q = dg.random_subspace(len(mu), q.shape[1], fold, stage, draw)
    train_pop, raw = populations["TRAIN_GEOMETRY"]
    cent, labels, subjects, session = centroid_training(train_pop, raw, sessions)
    x = coordinates(cent, mu, q, family)
    common = {"fold": fold, "stage": stage, "family": family,
              "random_draw": draw if family == "RANDOM" else "",
              "fit_population": "TRAIN_GEOMETRY", "fit_unit": "subject-session-class centroid"}
    rows = []
    for fit_session in sessions:
        fit, eva = session == fit_session, session != fit_session
        scale = StandardScaler().fit(x[fit])
        model = NearestCentroid().fit(scale.transform(x[fit]), subjects[fit])
        prediction = model.predict(scale.transform(x[eva]))
        chance = 1 / len(np.unique(subjects))
        ba = float(balanced_accuracy_score(subjects[eva], prediction))
        rows.append({**common, "probe": "subject_ID", "eval_population": "TRAIN_GEOMETRY",
                     "fit_session": fit_session, "eval_session": next(t for t in sessions if t != fit_session),
                     "grouping": "entire subject-session held out; no trial mixing",
                     "balanced_accuracy": ba, "chance": chance,
                     "normalized_above_chance": (ba - chance) / (1 - chance)})
    for fit_session in sessions:
        fit = session == fit_session
        scale, model = fit_linear(x[fit], labels[fit])
        for role, (pop, target) in populations.items():
            target_session = next(t for t in sessions if t != fit_session)
            ix = pop.session == target_session
            probability = predict_batched(scale, model, target[ix], mu, q, family)
            score = metrics(pop.y[ix], probability, model.classes_)
            rows.append({**common, "probe": "task_MI", "eval_population": role,
                         "fit_session": fit_session, "eval_session": target_session,
                         "grouping": "TRAIN subjects only for fit; held subject roles never fit",
                         "balanced_accuracy": score["BA"], "chance": .5,
                         "normalized_above_chance": 2 * score["BA"] - 1,
                         "macro_F1": score["macro_F1"], "NLL": score["NLL"]})
    # TRAIN-only held-subject task probe, grouped by biological subject.
    splitter = GroupKFold(n_splits=5)
    scores = []
    for fit, eva in splitter.split(x, labels, subjects):
        scale, model = fit_linear(x[fit], labels[fit])
        probability = model.predict_proba(scale.transform(x[eva]))
        scores.append(metrics(labels[eva], probability, model.classes_)["BA"])
    rows.append({**common, "probe": "task_MI_grouped_subject_CV",
                 "eval_population": "TRAIN_GEOMETRY_HELD_SUBJECT_CV", "fit_session": "both",
                 "eval_session": "both", "grouping": "5-fold GroupKFold by biological subject",
                 "balanced_accuracy": float(np.mean(scores)), "chance": .5,
                 "normalized_above_chance": 2 * float(np.mean(scores)) - 1})
    return rows
