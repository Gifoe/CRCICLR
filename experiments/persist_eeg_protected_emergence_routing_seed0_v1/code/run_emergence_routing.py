"""Frozen trial-level Protected emergence and selective-routing audit.

This program reuses the *fixed* final Protected coordinates and frozen native
heads from the two preceding audits.  It is deliberately fail-closed: it never
trains a backbone or head, never reselects Protected, and never loads the final
heldout cohort.  All tuning is nested inside each fold's TRAIN subjects.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


EXP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
PATHWAY_EXP = ROOT / "experiments" / "persist_eeg_protected_pathway_mechanism_seed0_v1"
OUT, PROTOCOL = EXP / "outputs", EXP / "protocol"
RUNTIME = Path(os.environ.get("ROUTING_RUNTIME", str(ROOT.parent / "protected_emergence_routing_runtime")))
MODELS, TASKS, FOLDS, SEED = ("EEGNet", "EEGConformer", "FBCNet"), ("OpenBMI_MI", "OpenBMI_SSVEP"), tuple(range(5)), 0
CAP, RANDOM_DRAWS, SUBJECT_FOLDS, BOOTSTRAPS = 16, 100, 5, 2000
RIDGE_ALPHAS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
SUPPRESS_ALPHAS, GATE_CS, GATE_TAUS = (0.0, 0.25, 0.5, 0.75), (0.01, 0.1, 1.0, 10.0), (0.25, 0.5, 0.75)
RANDOM_ROUTING_WORKERS = max(1, min(4, int(os.environ.get("ROUTING_RANDOM_WORKERS", "1"))))
EPS = 1e-12


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PW = load("routing_upstream_pathway", PATHWAY_EXP / "code" / "run_protected_pathway.py")
UP = PW.UP


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**32 - 1)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if np.isfinite(value) else None
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: clean(row.get(k, "")) for k in fields} for row in rows])
    os.replace(tmp, path)


def target_path(model: str, task: str, fold: int) -> Path:
    return RUNTIME / "cells" / model.lower() / task.lower() / f"fold{fold}_seed0.json"


def natural(values: np.ndarray | list[str]) -> list[str]:
    return UP.natural(values)


def centered(z: np.ndarray) -> np.ndarray:
    return z - z.mean(axis=1, keepdims=True)


def softmax(z: np.ndarray) -> np.ndarray:
    x = z - z.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=1, keepdims=True), EPS)


def cross_entropy(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(softmax(z)[np.arange(len(y)), y], EPS, 1.0))


def true_margin(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    own = z[np.arange(len(y)), y]
    other = z.copy()
    other[np.arange(len(y)), y] = -np.inf
    return own - other.max(axis=1)


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    labels = np.unique(y)
    return float(np.mean([np.mean(pred[y == label] == label) for label in labels])) if len(labels) else float("nan")


def per_subject_metrics(y: np.ndarray, z: np.ndarray, subjects: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    pred = z.argmax(axis=1)
    for subject in natural(subjects):
        ix = np.flatnonzero(subjects.astype(str) == subject)
        out[subject] = {
            "BA": balanced_accuracy(y[ix], pred[ix]),
            "MacroF1": float(f1_score(y[ix], pred[ix], average="macro", zero_division=0)),
            "CE": float(np.mean(cross_entropy(z[ix], y[ix]))),
            "n_trials": int(len(ix)),
        }
    return out


def cap_indices(subjects: np.ndarray, labels: np.ndarray, session: int, model: str, task: str, fold: int, purpose: str) -> np.ndarray:
    chosen: list[int] = []
    for subject in natural(subjects):
        for label in sorted(np.unique(labels).astype(int)):
            idx = np.flatnonzero((subjects.astype(str) == subject) & (labels == label))
            if len(idx) > CAP:
                rng = np.random.default_rng(stable_seed("emergence-routing-cap-v1", model, task, fold, purpose, subject, session, label, CAP))
                idx = np.sort(rng.choice(idx, CAP, replace=False))
            chosen.extend(map(int, idx))
    return np.asarray(sorted(chosen), dtype=np.int64)


def trial_train(data: dict[str, Any], model: str, task: str, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = cap_indices(data["train_subjects"], data["train_y"], int(data["source_session"]), model, task, fold, "source")
    b = cap_indices(data["future_train_subjects"], data["future_train_y"], int(data["future_session"]), model, task, fold, "future")
    return (
        np.concatenate([data["train_x"][a], data["future_train_x"][b]]),
        np.concatenate([data["train_y"][a], data["future_train_y"][b]]).astype(np.int64),
        np.concatenate([data["train_subjects"][a], data["future_train_subjects"][b]]).astype(str),
        np.concatenate([np.full(len(a), data["source_session"]), np.full(len(b), data["future_session"])]).astype(np.int64),
    )


def all_train(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([data["train_x"], data["future_train_x"]]),
        np.concatenate([data["train_y"], data["future_train_y"]]).astype(np.int64),
        np.concatenate([data["train_subjects"], data["future_train_subjects"]]).astype(str),
        np.concatenate([np.full(len(data["train_y"]), data["source_session"]), np.full(len(data["future_train_y"]), data["future_session"])]).astype(np.int64),
    )


def forward_stages(runner: Any, x: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    acts: dict[str, list[np.ndarray]] = defaultdict(list)
    hs: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), 32):
            values, h, z = runner.all(runner.tensor(x[start:start + 32]))
            for name, value in values.items():
                acts[name].append(value.reshape(len(h), -1).float().cpu().numpy())
            hs.append(h.float().cpu().numpy())
            zs.append(z.float().cpu().numpy())
    return {name: np.concatenate(parts).astype(np.float32) for name, parts in acts.items()}, np.concatenate(hs).astype(np.float32), np.concatenate(zs).astype(np.float32)


def exact_target(h: np.ndarray, spec: dict[str, Any], dims: np.ndarray) -> tuple[np.ndarray, float]:
    transform = (spec["basis"] / spec["scale"][None, :]) @ spec["directions"][:, dims]
    q_exact = (h - spec["mean"]) @ transform
    q_stored = PW.canonical(h, spec)[:, dims]
    err = float(np.max(np.abs(q_exact - q_stored)))
    if err >= 1e-5:
        raise RuntimeError(f"exact final Protected transform sanity failure: {err}")
    return q_exact.astype(np.float32), err


def standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(x.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    return (x - mean) / std, mean, std


def kernel_predictions(xfit: np.ndarray, yfit: np.ndarray, xtest: np.ndarray) -> np.ndarray:
    """All fixed ridge alphas in a memory-safe dual solve; output A,N,T,K."""
    xf, mean, std = standardise(xfit)
    xt = (xtest - mean) / std
    n, width = len(xf), xfit.shape[1]
    if n < 2:
        raise RuntimeError("insufficient TRAIN trials for dual ridge")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        a = torch.from_numpy(np.ascontiguousarray(xf)).to(dev)
        b = torch.from_numpy(np.ascontiguousarray(yfit.reshape(n, -1))).to(dev)
        t = torch.from_numpy(np.ascontiguousarray(xt)).to(dev)
        kernel = (a @ a.T) / max(width, 1)
        eig, vec = torch.linalg.eigh(kernel)
        vt_y = vec.T @ b
        kt = (t @ a.T) / max(width, 1)
        floor = torch.clamp(eig[-1].abs() * 1e-8, min=1e-8)
        parts = []
        for alpha in RIDGE_ALPHAS:
            den = torch.clamp(eig, min=floor) if alpha == 0 else eig + alpha
            parts.append((kt @ (vec @ (vt_y / den[:, None]))).cpu().numpy())
    del a, b, t, kernel, eig, vec, vt_y, kt
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return np.stack(parts, axis=0).reshape(len(RIDGE_ALPHAS), len(xtest), *yfit.shape[1:]).astype(np.float32)


def r2_matrix(y: np.ndarray, pred: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Subject-equal R2 for predictions A,N,T,K, returned A,T."""
    scores: list[np.ndarray] = []
    for subject in natural(subjects):
        ix = np.flatnonzero(subjects.astype(str) == subject)
        truth = y[ix]
        variance = np.mean((truth - truth.mean(axis=0, keepdims=True)) ** 2, axis=(0, 2))
        mse = np.mean((pred[:, ix] - truth[None]) ** 2, axis=(1, 3))
        scores.append(1.0 - mse / np.maximum(variance[None], EPS))
    return np.nanmean(np.stack(scores, axis=0), axis=0)


def one_r2(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    mse = float(np.mean((y - pred) ** 2))
    variance = float(np.mean((y - y.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - mse / max(variance, EPS)
    corr = []
    for i in range(y.shape[1]):
        if np.std(y[:, i]) > EPS and np.std(pred[:, i]) > EPS:
            corr.append(float(np.corrcoef(y[:, i], pred[:, i])[0, 1]))
    return float(r2), float(mse / max(variance, EPS)), float(np.mean(corr)) if corr else float("nan")


def subject_groups(subjects: np.ndarray) -> list[np.ndarray]:
    ids = natural(subjects)
    return [np.asarray([s for i, s in enumerate(ids) if i % SUBJECT_FOLDS == k]) for k in range(SUBJECT_FOLDS)]


def choose_ridge_alpha(a: np.ndarray, targets: np.ndarray, subjects: np.ndarray, sessions: np.ndarray, source_session: int, allowed_subjects: np.ndarray) -> np.ndarray:
    """Nested subject-CV; alpha is separately selected for P and each random control."""
    records: list[np.ndarray] = []
    allowed = np.isin(subjects, allowed_subjects)
    for held in subject_groups(allowed_subjects):
        fit = (sessions == source_session) & allowed & ~np.isin(subjects, held)
        test = (sessions == source_session) & allowed & np.isin(subjects, held)
        if fit.sum() < 2 or not test.any():
            continue
        pred = kernel_predictions(a[fit], targets[fit], a[test])
        records.append(r2_matrix(targets[test], pred, subjects[test]))
    if not records:
        raise RuntimeError("nested subject-CV produced no ridge validation split")
    score = np.nanmean(np.stack(records), axis=0)
    return np.asarray([int(np.nanargmax(score[:, j])) for j in range(score.shape[1])], dtype=int)


def mapping_rows(a: np.ndarray, targets: np.ndarray, subjects: np.ndarray, sessions: np.ndarray, source: int, future: int) -> list[dict[str, Any]]:
    """Four held-subject session directions with nested alpha selection."""
    rows: list[dict[str, Any]] = []
    directions = (("same_s1", source, source), ("same_s2", future, future), ("s1_to_s2", source, future), ("s2_to_s1", future, source))
    all_subjects = np.asarray(natural(subjects))
    for label, fit_session, test_session in directions:
        for held in subject_groups(all_subjects):
            allowed = all_subjects[~np.isin(all_subjects, held)]
            alpha_index = choose_ridge_alpha(a, targets, subjects, sessions, fit_session, allowed)
            fit = (sessions == fit_session) & ~np.isin(subjects, held)
            test = (sessions == test_session) & np.isin(subjects, held)
            if fit.sum() < 2 or not test.any():
                continue
            pred_all = kernel_predictions(a[fit], targets[fit], a[test])
            pred = np.stack([pred_all[alpha_index[j], :, j] for j in range(targets.shape[1])], axis=1)
            for subject in natural(subjects[test]):
                ix = np.flatnonzero(subjects[test].astype(str) == subject)
                protected = one_r2(targets[test][ix, 0], pred[ix, 0])
                random = [one_r2(targets[test][ix, j], pred[ix, j]) for j in range(1, targets.shape[1])]
                rows.append({
                    "subject_id": subject, "direction": label,
                    "R2_P": protected[0], "normalized_MSE_P": protected[1], "coordinate_correlation_P": protected[2],
                    "random_R2_mean": float(np.mean([r[0] for r in random])),
                    "random_normalized_MSE_mean": float(np.mean([r[1] for r in random])),
                    "random_coordinate_correlation_mean": float(np.nanmean([r[2] for r in random])),
                    "emergence_excess": float(protected[0] - np.mean([r[0] for r in random])),
                    "chosen_alpha_P": float(RIDGE_ALPHAS[alpha_index[0]]),
                    "chosen_alpha_random_mean": float(np.mean([RIDGE_ALPHAS[i] for i in alpha_index[1:]])),
                })
    return rows


def collapse_mapping(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subject_id"])].append(row)
    out = []
    for subject, values in grouped.items():
        same = [r for r in values if r["direction"].startswith("same")]
        cross = [r for r in values if "to" in r["direction"]]
        if not same or not cross:
            continue
        avg = lambda rs, key: float(np.nanmean([r[key] for r in rs]))
        out.append({
            "layer_name": stage, "subject_id": subject,
            "same_session_R2_P": avg(same, "R2_P"), "cross_session_R2_P": avg(cross, "R2_P"),
            "random_R2_mean": avg(cross, "random_R2_mean"), "emergence_excess": avg(cross, "emergence_excess"),
            "normalized_MSE": avg(cross, "normalized_MSE_P"), "coordinate_correlation": avg(cross, "coordinate_correlation_P"),
            "chosen_alpha_P": avg(cross, "chosen_alpha_P"), "chosen_alpha_random_mean": avg(cross, "chosen_alpha_random_mean"),
        })
    return out


def decompose(h: np.ndarray, z: np.ndarray, spec: dict[str, Any], dims: np.ndarray, head: torch.nn.Module) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(head, torch.nn.Linear):
        raise RuntimeError("native classifier head is not affine")
    q = PW.canonical(h, spec)
    raw = UP.raw_base(spec)
    w = head.weight.detach().float().cpu().numpy()
    bias = head.bias.detach().float().cpu().numpy() if head.bias is not None else 0.0
    z0 = (spec["mean"] @ w.T + bias)[None, :]
    zp = (q[:, dims] @ raw[dims]) @ w.T
    zc = z - z0 - zp
    if not np.allclose(z, z0 + zp + zc, rtol=2e-5, atol=3e-5):
        raise RuntimeError("exact P/complement decomposition failed")
    return q.astype(np.float32), z0.astype(np.float32), zp.astype(np.float32), zc.astype(np.float32)


def top_margin(z: np.ndarray) -> np.ndarray:
    part = np.partition(z, -2, axis=1)
    return part[:, -1] - part[:, -2]


def class_centroids(q: np.ndarray, y: np.ndarray, classes: int) -> np.ndarray:
    rows = []
    for label in range(classes):
        ix = np.flatnonzero(y == label)
        if not len(ix):
            raise RuntimeError("TRAIN class is absent while building inference-only geometry feature")
        rows.append(q[ix].mean(axis=0))
    return np.asarray(rows, dtype=np.float32)


def gate_features(zp: np.ndarray, zc: np.ndarray, z: np.ndarray, q: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    classes = z.shape[1]
    pp, pc, pf = zp.argmax(1), zc.argmax(1), z.argmax(1)
    onehot = lambda values: np.eye(classes, dtype=np.float32)[values]
    cp, cc = centered(zp), centered(zc)
    cosine = np.sum(cp * cc, axis=1) / np.maximum(np.linalg.norm(cp, axis=1) * np.linalg.norm(cc, axis=1), EPS)
    distances = np.sqrt(np.sum((q[:, None] - centroids[None]) ** 2, axis=2))
    near = np.partition(distances, 1, axis=1)[:, :2]
    return np.concatenate([
        onehot(pp), onehot(pc), onehot(pf),
        (pp == pc)[:, None], (pp == pf)[:, None],
        top_margin(zp)[:, None], top_margin(zc)[:, None], top_margin(z)[:, None],
        (-np.sum(softmax(zp) * np.log(np.clip(softmax(zp), EPS, 1)), axis=1))[:, None],
        (-np.sum(softmax(zc) * np.log(np.clip(softmax(zc), EPS, 1)), axis=1))[:, None],
        (-np.sum(softmax(z) * np.log(np.clip(softmax(z), EPS, 1)), axis=1))[:, None],
        cosine[:, None], np.linalg.norm(cp, axis=1)[:, None], np.linalg.norm(cc, axis=1)[:, None],
        (np.linalg.norm(zp, axis=1) / np.maximum(np.linalg.norm(zc, axis=1), EPS))[:, None],
        near[:, :1], near[:, 1:2], (near[:, 1:2] - near[:, :1]),
    ], axis=1).astype(np.float32)


def fit_logistic(x: np.ndarray, target: np.ndarray, c: float, seed: int) -> tuple[Any | None, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(x.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    if len(np.unique(target)) < 2:
        return None, mean, std
    model = LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=1000, random_state=seed)
    model.fit((x - mean) / std, target)
    return model, mean, std


def gate_probability(model: Any | None, mean: np.ndarray, std: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.zeros(len(x), dtype=np.float32) if model is None else model.predict_proba((x - mean) / std)[:, 1].astype(np.float32)


def mean_subject_ba(y: np.ndarray, z: np.ndarray, subjects: np.ndarray) -> float:
    return float(np.mean([balanced_accuracy(y[subjects.astype(str) == s], z[subjects.astype(str) == s].argmax(1)) for s in natural(subjects)]))


def choose_alpha(z0: np.ndarray, zp: np.ndarray, zc: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> float:
    """Nested subject-CV selection of one global suppression strength, TRAIN only."""
    base = np.broadcast_to(z0, zp.shape)
    candidates: list[list[float]] = [[] for _ in SUPPRESS_ALPHAS]
    for held in subject_groups(subjects):
        ix = np.isin(subjects, held)
        for i, alpha in enumerate(SUPPRESS_ALPHAS):
            candidates[i].append(mean_subject_ba(y[ix], base[ix] + zp[ix] + alpha * zc[ix], subjects[ix]))
    scores = np.asarray([np.mean(v) for v in candidates])
    return float(SUPPRESS_ALPHAS[int(np.argmax(scores))])


def choose_gate(z0: np.ndarray, zp: np.ndarray, zc: np.ndarray, q: np.ndarray, y: np.ndarray, subjects: np.ndarray, classes: int, alpha: float, seed_parts: tuple[object, ...]) -> tuple[float, float, Any | None, np.ndarray, np.ndarray, np.ndarray, str]:
    z = z0 + zp + zc
    zsup = z0 + zp + alpha * zc
    target = (cross_entropy(zsup, y) < cross_entropy(z, y)).astype(int)
    if len(np.unique(target)) < 2:
        cent = class_centroids(q, y, classes)
        features = gate_features(zp, zc, z, q, cent)
        _, mean, std = fit_logistic(features, target, GATE_CS[0], stable_seed(*seed_parts, "degenerate"))
        return GATE_CS[0], 1.0, None, mean, std, cent, "GATE_DEGENERATE_NO_TRAIN_VARIATION"
    probs = {c: np.zeros(len(y), dtype=np.float32) for c in GATE_CS}
    for split, held in enumerate(subject_groups(subjects)):
        fit = ~np.isin(subjects, held)
        val = np.isin(subjects, held)
        cent = class_centroids(q[fit], y[fit], classes)
        xf = gate_features(zp[fit], zc[fit], z[fit], q[fit], cent)
        xv = gate_features(zp[val], zc[val], z[val], q[val], cent)
        for c in GATE_CS:
            model, mean, std = fit_logistic(xf, target[fit], c, stable_seed(*seed_parts, split, c))
            probs[c][val] = gate_probability(model, mean, std, xv)
    best: tuple[float, float, float] | None = None
    for c in GATE_CS:
        for tau in GATE_TAUS:
            routed = np.where((probs[c] > tau)[:, None], zsup, z)
            score = mean_subject_ba(y, routed, subjects)
            candidate = (score, -c, -tau)
            if best is None or candidate > best:
                best = candidate
                choice = (c, tau)
    c, tau = choice
    cent = class_centroids(q, y, classes)
    features = gate_features(zp, zc, z, q, cent)
    model, mean, std = fit_logistic(features, target, c, stable_seed(*seed_parts, "refit", c, tau))
    return c, tau, model, mean, std, cent, "OK"


def failure_modes(zp: np.ndarray, zc: np.ndarray, z: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    comp = z.copy(); comp[np.arange(len(y)), y] = -np.inf; cstar = comp.argmax(1)
    mp = zp[np.arange(len(y)), y] - zp[np.arange(len(y)), cstar]
    mc = zc[np.arange(len(y)), y] - zc[np.arange(len(y)), cstar]
    correct = z.argmax(1) == y
    failure1 = ~correct & (mp > 0) & (mc < 0)
    failure3 = correct & (mp < 0) & (mc > 0)
    rows = []
    for subject in natural(subjects):
        ix = np.flatnonzero(subjects.astype(str) == subject)
        p_ok, c_ok = mp[ix] > 0, mc[ix] > 0
        rows.append({
            "subject_id": subject, "n_trials": int(len(ix)),
            "P_correct_C_correct": float(np.mean(p_ok & c_ok)), "P_correct_C_oppose": float(np.mean(p_ok & ~c_ok)),
            "P_wrong_C_correct": float(np.mean(~p_ok & c_ok)), "P_wrong_C_wrong": float(np.mean(~p_ok & ~c_ok)),
            "complement_override_error_rate": float(np.mean(failure1[ix])),
            "P_wrong_native_error_rate": float(np.mean((~correct[ix]) & (mp[ix] < 0))),
            "complement_rescue_rate": float(np.mean(failure3[ix])),
        })
    return rows, failure1, failure3


def route_one(qtrain: np.ndarray, ztrain: np.ndarray, ytrain: np.ndarray, strain: np.ndarray, qouter: np.ndarray, zouter: np.ndarray, youter: np.ndarray, souter: np.ndarray, spec: dict[str, Any], dims: np.ndarray, weight: np.ndarray, bias: np.ndarray | float, classes: int, seed_parts: tuple[object, ...], detailed: bool) -> dict[str, Any]:
    raw = UP.raw_base(spec)
    z0 = (spec["mean"] @ weight.T + bias)[None, :].astype(np.float32)
    zptrain = (qtrain[:, dims] @ raw[dims]) @ weight.T; zpouter = (qouter[:, dims] @ raw[dims]) @ weight.T
    zctrain = ztrain - z0 - zptrain; zcouter = zouter - z0 - zpouter
    if not np.allclose(zouter, z0 + zpouter + zcouter, rtol=2e-5, atol=3e-5):
        raise RuntimeError("routing decomposition mismatch")
    alpha = choose_alpha(z0, zptrain, zctrain, ytrain, strain)
    c, tau, gate, mean, std, centroids, gate_status = choose_gate(z0, zptrain, zctrain, qtrain[:, dims], ytrain, strain, classes, alpha, seed_parts)
    native = zouter; ponly = z0 + zpouter; conly = z0 + zcouter
    fixed = {f"alpha_{alpha:g}": z0 + zpouter + alpha * zcouter for alpha in SUPPRESS_ALPHAS}
    features = gate_features(zpouter, zcouter, native, qouter[:, dims], centroids)
    probability = gate_probability(gate, mean, std, features)
    suppressed = z0 + zpouter + alpha * zcouter
    routed = np.where((probability > tau)[:, None], suppressed, native)
    candidates = [fixed[f"alpha_{a:g}"] for a in SUPPRESS_ALPHAS] + [native]
    margins = np.stack([true_margin(value, youter) for value in candidates], axis=1)
    losses = np.stack([cross_entropy(value, youter) for value in candidates], axis=1)
    oracle_margin = np.stack(candidates, axis=1)[np.arange(len(youter)), margins.argmax(axis=1)]
    oracle_ce = np.stack(candidates, axis=1)[np.arange(len(youter)), losses.argmin(axis=1)]
    native_correct = native.argmax(1) == youter
    recoverable = (~native_correct) & np.any(np.stack([value.argmax(1) == youter for value in fixed.values()], axis=1), axis=1)
    harm = native_correct & np.any(np.stack([value.argmax(1) != youter for value in fixed.values()], axis=1), axis=1)
    rows = []
    for subject in natural(souter):
        ix = np.flatnonzero(souter.astype(str) == subject)
        metric = lambda value: per_subject_metrics(youter[ix], value[ix], np.asarray([subject] * len(ix)))[subject]
        n = metric(native); p = metric(ponly); co = metric(conly); r = metric(routed); om = metric(oracle_margin); oc = metric(oracle_ce)
        row = {"subject_id": subject, "native_BA": n["BA"], "native_MacroF1": n["MacroF1"], "native_CE": n["CE"], "P_only_BA": p["BA"], "complement_only_BA": co["BA"], "oracle_BA": om["BA"], "oracle_CE": oc["CE"], "routing_BA": r["BA"], "routing_MacroF1": r["MacroF1"], "routing_CE": r["CE"], "delta_BA": r["BA"]-n["BA"], "delta_MacroF1": r["MacroF1"]-n["MacroF1"], "delta_CE": r["CE"]-n["CE"], "oracle_BA_gain": om["BA"]-n["BA"], "oracle_CE_gain": n["CE"]-oc["CE"], "recoverable_error_rate": float(np.mean(recoverable[ix])), "suppression_harm_rate": float(np.mean(harm[ix])), "routing_fraction": float(np.mean(probability[ix] > tau)), "alpha_s": alpha, "gate_C": c, "gate_threshold": tau, "gate_status": gate_status, "worst_session_BA": r["BA"]}
        for name, value in fixed.items():
            fm = metric(value); row[f"fixed_{name}_BA"] = fm["BA"]; row[f"fixed_{name}_MacroF1"] = fm["MacroF1"]; row[f"fixed_{name}_CE"] = fm["CE"]
        rows.append(row)
    result = {"config": {"alpha_s": alpha, "gate_C": c, "gate_threshold": tau, "gate_status": gate_status}, "subject_rows": rows}
    if detailed:
        failure, failure1, failure3 = failure_modes(zpouter, zcouter, native, youter, souter)
        transitions = []
        route_correct = routed.argmax(1) == youter
        for subject in natural(souter):
            ix = np.flatnonzero(souter.astype(str) == subject)
            corrected = ~native_correct[ix] & route_correct[ix]
            introduced = native_correct[ix] & ~route_correct[ix]
            transitions.append({"subject_id": subject, "corrected_errors": int(corrected.sum()), "introduced_errors": int(introduced.sum()), "unchanged_correct": int((native_correct[ix] & route_correct[ix]).sum()), "unchanged_wrong": int((~native_correct[ix] & ~route_correct[ix]).sum()), "targeted_correction_rate": float(np.mean(corrected & failure1[ix])), "collateral_damage_rate": float(np.mean(introduced & failure3[ix])), "net_corrected_trials": int(corrected.sum()-introduced.sum())})
        result.update({"failure_rows": failure, "transition_rows": transitions})
    return result


def prior_pathway_effects(model: str, task: str, fold: int) -> tuple[float, float]:
    old = PW.path(model, task, fold)
    if not old.is_file():
        raise FileNotFoundError(f"previous pathway cell missing: {old}")
    value = json.loads(old.read_text(encoding="utf-8"))
    if value.get("status") != "COMPLETE":
        return float("nan"), float("nan")
    causal = value.get("causal", [])
    interaction = value.get("interaction", [])
    c = [r.get("delta_qP", np.nan)-r.get("random_delta_qP", np.nan) for r in causal]
    i = [r.get("interaction_excess", np.nan) for r in interaction]
    return float(np.nanmean(c)) if c else float("nan"), float(np.nanmean(i)) if i else float("nan")


def cell(model: str, task: str, fold: int) -> None:
    target = target_path(model, task, fold)
    if target.is_file():
        print("CELL_CACHED", model, task, fold, flush=True)
        return
    base = {"model": model, "task": task, "fold": fold, "seed": SEED, "backbone_training": False, "head_refit": False, "final_heldout_accessed": False}
    try:
        if not (PROTOCOL / "PROVENANCE.json").is_file():
            raise RuntimeError("protocol lock is required before a cell may run")
        record, stored, ckpt, pathway = PW.previous(model, task, fold)
        data = UP.outer_data(task, fold)
        if data["normalizer"]["mean_std_sha256"] != record["normalizer"]["mean_std_sha256"]:
            raise RuntimeError("TRAIN normalizer hash mismatch")
        if pathway.get("status") == "EMPTY_PROTECTED":
            write_json(target, {**base, "status": "EMPTY_PROTECTED", "previous_cell_sha256": sha(PW.path(model, task, fold))})
            return
        if pathway.get("status") != "COMPLETE":
            raise RuntimeError("preceding pathway cell is not complete")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net, head = UP.helper(model).build_model({"Model": model, "Task": task, "fold": fold, "seed": 0, "channels": int(record.get("channels") or 62), "samples": int(record.get("samples") or 1000), "classes": int(record["classes"]), "checkpoint_path": str(ckpt), "recipe_name": record.get("recipe", {}).get("name"), "trainable_parameters": int(record.get("trainable_parameters", record.get("parameters", 0)))}, device)
        runner = PW.Stages(net, head, model, device)
        basis_x, basis_y, basis_s, basis_sessions = UP.capped_train(data, task, model, fold)
        PW.stage_check(runner, basis_x)
        basis_h, _, _ = UP.hook_representations(net, head, basis_x, model, device)
        spec = UP.helper(model).spectrum(basis_h, basis_y, basis_s, basis_sessions, task, model, fold)
        dims = np.asarray(stored.get("protected_blocks", []), dtype=int)
        basis_sha = UP.array_sha(spec["mean"], spec["basis"], spec["scale"], spec["directions"])
        if basis_sha != pathway.get("basis_sha256") or int(spec["rank"]) != int(stored.get("rank", -1)) or np.any(dims < 0) or np.any(dims >= spec["rank"]):
            raise RuntimeError("fixed canonical basis or Protected coordinates cannot be reproduced")
        x, y, subjects, sessions = trial_train(data, model, task, fold)
        acts, h, z = forward_stages(runner, x)
        href, zdirect, zhead = UP.hook_representations(net, head, x, model, device)
        if not np.allclose(h, href, rtol=1e-5, atol=1e-6) or not np.allclose(z, zhead, rtol=1e-5, atol=1e-6) or not np.allclose(zdirect, zhead, rtol=1e-5, atol=1e-6):
            raise RuntimeError("manual trial forward does not exactly reproduce frozen native head")
        q_p, exact_error = exact_target(h, spec, dims)
        random_dims = UP.random_dims(spec["rank"], len(dims), model, task, fold, "native-equal-rank-random")
        targets = np.stack([q_p] + [PW.canonical(h, spec)[:, d].astype(np.float32) for d in random_dims], axis=1)
        emergence_subject: list[dict[str, Any]] = []
        protocol_rows: list[dict[str, Any]] = []
        final_sanity = None
        for stage in runner.names:
            mapped = collapse_mapping(mapping_rows(acts[stage], targets, subjects, sessions, int(data["source_session"]), int(data["future_session"])), stage)
            if not mapped:
                raise RuntimeError(f"no trial-level mapping records at stage {stage}")
            for row in mapped:
                emergence_subject.append(row)
            final_r2 = float(np.mean([r["cross_session_R2_P"] for r in mapped])) if stage == "classifier_input" else float("nan")
            sanity = bool(exact_error < 1e-5 and final_r2 > 0.98) if stage == "classifier_input" else True
            protocol_rows.append({"layer_name": stage, "activation_dimension": int(acts[stage].shape[1]), "trial_count": int(len(x)), "chosen_ridge_alpha_P_mean": float(np.mean([r["chosen_alpha_P"] for r in mapped])), "final_layer_exact_max_abs_error": exact_error if stage == "classifier_input" else float("nan"), "final_layer_learned_cross_session_R2": final_r2, "sanity_pass": sanity})
            if stage == "classifier_input": final_sanity = sanity
        if not final_sanity:
            raise RuntimeError("FAIL_PROTOCOL: final classifier-input learned mapping sanity is below 0.98")
        gx, gy, gs, _ = all_train(data)
        _, gh, gz = forward_stages(runner, gx)
        oh, oz_direct, oz = UP.hook_representations(net, head, data["outer_future_x"], model, device)
        if not np.allclose(oz_direct, oz, rtol=1e-5, atol=1e-6):
            raise RuntimeError("outer native-head replay mismatch")
        qg = PW.canonical(gh, spec).astype(np.float32); qo = PW.canonical(oh, spec).astype(np.float32)
        weight = head.weight.detach().float().cpu().numpy().copy(); bias = head.bias.detach().float().cpu().numpy().copy() if head.bias is not None else 0.0
        protected = route_one(qg, gz, gy, gs, qo, oz, data["outer_future_y"], data["outer_future_subjects"].astype(str), spec, dims, weight, bias, data["classes"], (model, task, fold, "Protected"), True)
        def one_random(item: tuple[int, np.ndarray]) -> dict[str, Any]:
            draw, rd = item
            control = route_one(qg, gz, gy, gs, qo, oz, data["outer_future_y"], data["outer_future_subjects"].astype(str), spec, rd, weight, bias, data["classes"], (model, task, fold, "random", draw), False)
            gains = [r["delta_BA"] for r in control["subject_rows"]]
            return {"draw": draw, "random_dims": rd.tolist(), "gain_BA": float(np.mean(gains)), "gain_MacroF1": float(np.mean([r["delta_MacroF1"] for r in control["subject_rows"]])), "gain_CE": float(np.mean([r["delta_CE"] for r in control["subject_rows"]])), **control["config"]}
        if RANDOM_ROUTING_WORKERS == 1:
            random_rows = [one_random(item) for item in enumerate(random_dims)]
        else:
            # Each draw is seed-isolated and has no GPU/model mutation; map preserves draw order.
            with ThreadPoolExecutor(max_workers=RANDOM_ROUTING_WORKERS, thread_name_prefix="routing-random") as pool:
                random_rows = list(pool.map(one_random, enumerate(random_dims)))
        p_gain = float(np.mean([r["delta_BA"] for r in protected["subject_rows"]]))
        values = np.asarray([r["gain_BA"] for r in random_rows])
        for row in protected["subject_rows"]:
            row["random_gain_BA_mean"] = float(values.mean()); row["random_gain_BA_low"] = float(np.quantile(values, .025)); row["random_gain_BA_high"] = float(np.quantile(values, .975)); row["Protected_gain_percentile"] = float(np.mean(values <= p_gain)); row["random_empirical_p"] = float((1 + np.sum(values >= p_gain)) / (len(values) + 1))
        causal, interaction = prior_pathway_effects(model, task, fold)
        prov = {"checkpoint_sha256": sha(ckpt), "normalizer_sha256": data["normalizer"]["mean_std_sha256"], "basis_sha256": basis_sha, "protected_assignment_sha256": hashlib.sha256(json.dumps(stored.get("protected_assignment", []), sort_keys=True).encode()).hexdigest(), "split_sha256": data["split_sha256"], "previous_native_cell_sha256": sha(UP.target_path(model, task, fold)), "previous_pathway_cell_sha256": sha(PW.path(model, task, fold)), "protected_dims": dims.tolist(), "random_subsets_sha256": hashlib.sha256(np.concatenate(random_dims).tobytes()).hexdigest()}
        write_json(target, {**base, "status": "COMPLETE", **prov, "stage_names": list(runner.names), "protocol_audit": protocol_rows, "emergence_subject": emergence_subject, "failure_rows": protected["failure_rows"], "oracle_routing_rows": protected["subject_rows"], "random_routing_rows": random_rows, "transition_rows": protected["transition_rows"], "previous_causal_P_path_excess": causal, "previous_nonlinear_interaction_excess": interaction})
        print("CELL_COMPLETE", model, task, fold, flush=True)
        del net
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as exc:
        write_json(target, {**base, "status": "FAIL_CLOSED", "reason": f"{type(exc).__name__}: {exc}"})
        print("CELL_FAIL_CLOSED", model, task, fold, str(exc), flush=True)


def bootstrap_subject(rows: list[dict[str, Any]], key: str, seed: int) -> tuple[float, float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(key, np.nan)
        if value is not None and np.isfinite(value): grouped[str(row["subject_id"])].append(float(value))
    values = np.asarray([np.mean(v) for v in grouped.values()], dtype=float)
    if not len(values): return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.mean(values[rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))], axis=1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def aggregate() -> None:
    cells = []
    for model in MODELS:
        for task in TASKS:
            for fold in FOLDS:
                path = target_path(model, task, fold)
                if not path.is_file(): raise RuntimeError(f"missing {model}/{task}/f{fold}")
                cells.append(json.loads(path.read_text(encoding="utf-8")))
    protocol, emergence, failure, routing, randoms, transitions = [], [], [], [], [], []
    status = []
    for cell_value in cells:
        base = {k: cell_value.get(k) for k in ("model", "task", "fold", "seed", "status", "checkpoint_sha256", "basis_sha256", "protected_assignment_sha256", "split_sha256")}
        status.append(base)
        if cell_value.get("status") != "COMPLETE": continue
        protocol.extend([{**base, **row} for row in cell_value.get("protocol_audit", [])])
        emergence.extend([{**base, **row} for row in cell_value.get("emergence_subject", [])])
        failure.extend([{**base, **row} for row in cell_value.get("failure_rows", [])])
        routing.extend([{**base, **row} for row in cell_value.get("oracle_routing_rows", [])])
        randoms.extend([{**base, **row} for row in cell_value.get("random_routing_rows", [])])
        transitions.extend([{**base, **row} for row in cell_value.get("transition_rows", [])])
    write_csv(OUT / "CELL_STATUS.csv", status)
    write_csv(OUT / "EMERGENCE_PROTOCOL_AUDIT.csv", protocol)
    write_csv(OUT / "FAILURE_MODE_SUMMARY.csv", failure)
    write_csv(OUT / "ORACLE_HEADROOM.csv", routing)
    write_csv(OUT / "SELECTIVE_ROUTING_RESULTS.csv", routing)
    write_csv(OUT / "RANDOM_ROUTING_CONTROLS.csv", randoms)
    write_csv(OUT / "ROUTING_ERROR_TRANSITIONS.csv", transitions)
    layer_rows = []
    onset: dict[tuple[str, str], str] = {}
    for model in MODELS:
        for task in TASKS:
            candidates = [r for r in emergence if r["model"] == model and r["task"] == task]
            order = next((c.get("stage_names", []) for c in cells if c.get("model") == model and c.get("task") == task and c.get("status") == "COMPLETE"), [])
            first = None
            for stage in order:
                rows = [r for r in candidates if r["layer_name"] == stage]
                if not rows: continue
                r2, r2lo, r2hi = bootstrap_subject(rows, "cross_session_R2_P", stable_seed(model, task, stage, "r2"))
                ex, exlo, exhi = bootstrap_subject(rows, "emergence_excess", stable_seed(model, task, stage, "ex"))
                same, _, _ = bootstrap_subject(rows, "same_session_R2_P", stable_seed(model, task, stage, "same"))
                fold_signs = []
                for fold in FOLDS:
                    rr = [x for x in rows if int(x["fold"]) == fold]
                    if rr: fold_signs.append(float(np.mean([x["cross_session_R2_P"] for x in rr])) > 0 and float(np.mean([x["emergence_excess"] for x in rr])) > 0)
                qualifies = bool(r2 > 0 and r2lo > 0 and ex > 0 and exlo > 0 and sum(fold_signs) >= 4)
                if qualifies and first is None: first = stage
                layer_rows.append({"model": model, "task": task, "layer_name": stage, "cross_session_R2_P": r2, "cross_session_R2_P_CI_low": r2lo, "cross_session_R2_P_CI_high": r2hi, "same_session_R2_P": same, "random_R2_mean": float(np.mean([r["random_R2_mean"] for r in rows])), "emergence_excess": ex, "emergence_excess_CI_low": exlo, "emergence_excess_CI_high": exhi, "positive_folds": int(sum(fold_signs)), "onset_flag": False})
            onset[(model, task)] = first or "NO_RELIABLE_EMERGENCE_ONSET"
    for row in layer_rows:
        row["onset_flag"] = row["layer_name"] == onset[(row["model"], row["task"])]
    write_csv(OUT / "LAYERWISE_EMERGENCE_FIXED.csv", layer_rows)
    summary = []
    for model in MODELS:
        for task in TASKS:
            rows = [r for r in routing if r["model"] == model and r["task"] == task]
            if not rows:
                summary.append({"model": model, "task": task, "status": "INCOMPLETE_OR_EMPTY_PROTECTED", "emergence_onset": onset.get((model, task), "NO_RELIABLE_EMERGENCE_ONSET")})
                continue
            dba, dba_lo, dba_hi = bootstrap_subject(rows, "delta_BA", stable_seed(model, task, "dba"))
            df1, df1_lo, df1_hi = bootstrap_subject(rows, "delta_MacroF1", stable_seed(model, task, "df1"))
            dce, dce_lo, dce_hi = bootstrap_subject(rows, "delta_CE", stable_seed(model, task, "dce"))
            oracle, _, _ = bootstrap_subject(rows, "oracle_BA_gain", stable_seed(model, task, "oracle"))
            random = [r["gain_BA"] for r in randoms if r["model"] == model and r["task"] == task]
            summary.append({"model": model, "task": task, "status": "COMPLETE", "emergence_onset": onset[(model, task)], "native_BA": float(np.mean([r["native_BA"] for r in rows])), "oracle_BA_gain": oracle, "routing_delta_BA": dba, "routing_delta_BA_CI_low": dba_lo, "routing_delta_BA_CI_high": dba_hi, "routing_delta_MacroF1": df1, "routing_delta_MacroF1_CI_low": df1_lo, "routing_delta_MacroF1_CI_high": df1_hi, "routing_delta_CE": dce, "routing_delta_CE_CI_low": dce_lo, "routing_delta_CE_CI_high": dce_hi, "random_routing_gain_mean": float(np.mean(random)), "random_routing_gain_CI_low": float(np.quantile(random, .025)), "random_routing_gain_CI_high": float(np.quantile(random, .975)), "mean_P_gain_percentile": float(np.mean([r["Protected_gain_percentile"] for r in rows])), "mean_empirical_p": float(np.mean([r["random_empirical_p"] for r in rows]))})
    write_csv(OUT / "MODEL_TASK_SUMMARY.csv", summary)
    report = ["# Protected emergence and selective-routing audit", "", "All completed cells reused the preceding frozen checkpoints, normalizers, canonical bases, Protected coordinates, and splits. No backbone/head was trained or refit. Emergence fitting and routing-gate selection used TRAIN subjects only; outer-development labels were used solely for evaluation/oracle diagnostics. Final-heldout data were not accessed.", "", "## Completion", "", f"Cells: {sum(c.get('status')=='COMPLETE' for c in cells)} COMPLETE; {sum(c.get('status')=='EMPTY_PROTECTED' for c in cells)} EMPTY_PROTECTED; {sum(c.get('status')=='FAIL_CLOSED' for c in cells)} FAIL_CLOSED.", "", "## Per model/task", ""]
    for row in summary:
        if row.get("status") != "COMPLETE":
            report.append(f"- {row['model']} / {row['task']}: {row['status']}.")
        else:
            report.append(f"- {row['model']} / {row['task']}: onset={row['emergence_onset']}; oracle BA gain={row['oracle_BA_gain']:.4f}; TRAIN-only routing BA delta={row['routing_delta_BA']:.4f} [{row['routing_delta_BA_CI_low']:.4f}, {row['routing_delta_BA_CI_high']:.4f}].")
    report += ["", "Oracle values are nondeployable label-aware upper bounds. A positive actionability claim requires the TRAIN-only routing gain, its subject bootstrap interval, and the equal-rank random-routing comparison to agree; consult the CSV tables before making any claim."]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("AGGREGATE_COMPLETE", flush=True)


def lock() -> None:
    rows = []
    for model in MODELS:
        for task in TASKS:
            for fold in FOLDS:
                record, stored, ckpt, pathway = PW.previous(model, task, fold)
                data = UP.outer_data(task, fold)
                rows.append({"model": model, "task": task, "fold": fold, "checkpoint_sha256": sha(ckpt), "normalizer_sha256": data["normalizer"]["mean_std_sha256"], "protected_assignment_sha256": hashlib.sha256(json.dumps(stored.get("protected_assignment", []), sort_keys=True).encode()).hexdigest(), "previous_pathway_cell_sha256": sha(PW.path(model, task, fold)), "split_sha256": data["split_sha256"], "previous_status": pathway.get("status"), "final_heldout_accessed": False})
    value = {"schema": "PERSIST_EEG_PROTECTED_EMERGENCE_ROUTING_SEED0_V1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": SEED, "models": MODELS, "tasks": TASKS, "folds": FOLDS, "mapping": {"unit": "trial-level TRAIN activation", "cap_per_subject_session_class": CAP, "subject_crossfit_folds": SUBJECT_FOLDS, "ridge_alphas": RIDGE_ALPHAS, "solver": "standardized dual kernel ridge; alpha=0 stable eigenspace pseudoinverse"}, "routing": {"suppression_alphas": SUPPRESS_ALPHAS, "gate": "L2 logistic regression", "gate_Cs": GATE_CS, "gate_thresholds": GATE_TAUS, "selection": "nested subject-CV subject-equal BA on TRAIN subjects only"}, "random_controls": {"draws": RANDOM_DRAWS, "source": "identical frozen native equal-rank random canonical subsets"}, "final_P_definition": "fixed preceding classifier-input canonical Protected coordinates only", "backbone_training": False, "head_or_probe_refit": False, "final_heldout_accessed": False, "cells": rows}
    write_json(PROTOCOL / "PROVENANCE.json", value)
    (PROTOCOL / "PROVENANCE.sha256").write_text(sha(PROTOCOL / "PROVENANCE.json") + "\n", encoding="utf-8")
    print("PROTOCOL_LOCKED", len(rows), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("lock")
    one = sub.add_parser("cell"); one.add_argument("model", choices=MODELS); one.add_argument("task", choices=TASKS); one.add_argument("fold", type=int, choices=FOLDS)
    sub.add_parser("run-all"); sub.add_parser("aggregate")
    args = parser.parse_args()
    if args.command == "lock": lock()
    elif args.command == "cell": cell(args.model, args.task, args.fold)
    elif args.command == "run-all":
        for model in MODELS:
            for task in TASKS:
                for fold in FOLDS: cell(model, task, fold)
    else: aggregate()


if __name__ == "__main__":
    main()
