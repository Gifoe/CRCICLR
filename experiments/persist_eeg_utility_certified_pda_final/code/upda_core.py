"""Core implementation for the source-only U-PDA experiment.

The code deliberately operates on the frozen CleanRoom representation
archives.  Population logits/features are read-only.  The primary adapter
objective is the label-likelihood objective in :func:`fit_ce_adapter`; the
legacy target-regression adapter is isolated in :func:`fit_previous_pda` and
is used only for the historical comparator.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score


REPO = Path(os.environ.get("U_PDA_REPO", r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC"))
EXP = REPO / "experiments" / "persist_eeg_utility_certified_pda_final"
RESULTS = EXP / "results"
RUNTIME = EXP / "runtime"  # transient only; never packaged
REP_ROOT = REPO / "experiments" / "persist_eeg_scst_competence_generality_v1" / "runtime" / "specialist_representations"
OLD_EXP = REPO / "experiments" / "persist_eeg_persistent_decision_adapter_final"

DATASETS = ("OpenBMI", "WBCIC")
FOLDS = tuple(range(5))
SEEDS = tuple(range(3))
RANKS = (1, 2, 4)
L2S = (1e-3, 1e-2)
ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)
CLASSES = 2
N_BOOTSTRAP = 10_000
FORBIDDEN_FUTURE_TOKENS = ("utility_metrics", "utility_units", "sealed", "outer", "session-2-utility")


def clean(value):
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def subject_sort(values: Iterable[object]) -> list[str]:
    return sorted((str(v).replace("sub-", "") for v in values), key=lambda x: (int(x) if x.isdigit() else 10**9, x))


def load_rep(dataset: str, fold: int, seed: int, role: str, backbone: str = "ATCNet") -> dict[str, np.ndarray]:
    """Load one authorized source representation archive only."""
    path = REP_ROOT / backbone / dataset / f"fold-{fold}" / f"seed-{seed}" / f"{role}.npz"
    lowered = str(path).lower()
    if any(token in lowered for token in FORBIDDEN_FUTURE_TOKENS):
        raise PermissionError(f"future/sealed resource is unavailable: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"missing authorized source archive: {path}")
    with np.load(path, allow_pickle=True) as z:
        out = {key: z[key] for key in z.files}
    required = {"indices", "features", "logits", "labels", "subjects", "sessions"}
    missing = required.difference(out)
    if missing:
        raise RuntimeError(f"{path} missing keys {sorted(missing)}")
    out["features"] = np.asarray(out["features"], np.float64)
    out["logits"] = np.asarray(out["logits"], np.float64)
    out["labels"] = np.asarray(out["labels"], np.int64)
    out["subjects"] = np.asarray(out["subjects"]).astype("U")
    out["sessions"] = np.asarray(out["sessions"], np.int64)
    out["indices"] = np.asarray(out["indices"], np.int64)
    n = len(out["labels"])
    if out["features"].ndim != 2 or out["features"].shape[0] != n or out["logits"].shape != (n, CLASSES):
        raise RuntimeError(f"unexpected archive shape: {path}")
    if len(np.unique(out["indices"])) != n:
        raise RuntimeError(f"duplicate source indices: {path}")
    if not np.isfinite(out["features"]).all() or not np.isfinite(out["logits"]).all():
        raise RuntimeError(f"non-finite source archive: {path}")
    return out


@dataclass(frozen=True)
class Fold:
    model_fit: dict[str, np.ndarray]
    validation: dict[str, np.ndarray]
    outcome: dict[str, np.ndarray]


def load_fold(dataset: str, fold: int, seed: int, backbone: str = "ATCNet") -> Fold:
    return Fold(*(load_rep(dataset, fold, seed, role, backbone) for role in ("model_fit", "validation", "outcome")))


def population_checkpoint_id(rep: dict[str, np.ndarray]) -> str:
    """Stable identity of the frozen population archive (metadata only)."""
    h = hashlib.sha256()
    for key in ("indices", "subjects", "sessions"):
        h.update(np.ascontiguousarray(rep[key]).tobytes())
    h.update(np.asarray(rep["features"].shape, np.int64).tobytes())
    h.update(np.asarray(rep["logits"].shape, np.int64).tobytes())
    return h.hexdigest()


def layer_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-5)


@dataclass(frozen=True)
class SubjectTransition:
    subject: str
    history_blocks: tuple[dict[str, np.ndarray], ...]
    future: dict[str, np.ndarray]


def _subset(rep: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[mask] for k, v in rep.items()}


def make_transitions(rep: dict[str, np.ndarray], n_blocks: int = 4) -> list[SubjectTransition]:
    """Construct deterministic historical blocks and a later-session future.

    With two natural sessions, the first session is split into exactly four
    contiguous index-ordered blocks because four cross-fitting units are
    required.  No trial permutation is performed.
    """
    sessions = sorted(int(x) for x in np.unique(rep["sessions"]))
    if not sessions:
        return []
    subjects = subject_sort(np.unique(rep["subjects"]))
    result: list[SubjectTransition] = []
    for subject in subjects:
        sm = rep["subjects"] == subject
        if len(sessions) >= 2:
            hist_sessions = sessions[:-1]
            future_session = sessions[-1]
        else:
            hist_sessions = [sessions[0]]
            future_session = sessions[0]
        hist_idx = np.flatnonzero(sm & np.isin(rep["sessions"], hist_sessions))
        hist_idx = hist_idx[np.argsort(rep["indices"][hist_idx], kind="stable")]
        future_mask = sm & (rep["sessions"] == future_session)
        if len(hist_idx) < 4 or not np.any(future_mask):
            continue
        chunks = np.array_split(hist_idx, max(4, min(int(n_blocks), len(hist_idx))))
        blocks = tuple(_subset(rep, np.isin(np.arange(len(rep["labels"])), chunk)) for chunk in chunks if len(chunk))
        if len(blocks) < 4:
            continue
        result.append(SubjectTransition(subject, blocks, _subset(rep, future_mask)))
    return result


def class_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, np.int64)
    counts = np.bincount(y, minlength=CLASSES).astype(np.float64)
    weights = np.zeros_like(y, dtype=np.float64)
    for cls in range(CLASSES):
        if counts[cls] > 0:
            weights[y == cls] = len(y) / (CLASSES * counts[cls])
    return weights


def balanced_ce(logits: np.ndarray, y: np.ndarray) -> float:
    logits = np.asarray(logits, np.float64)
    y = np.asarray(y, np.int64)
    p = np.clip(softmax(logits, axis=1), 1e-12, 1.0)
    w = class_weights(y)
    if not np.any(w):
        return float("nan")
    return float(np.sum(w * (-np.log(p[np.arange(len(y)), y]))) / np.sum(w))


def layer_response(rep: dict[str, np.ndarray], basis: "Basis") -> np.ndarray:
    return layer_norm(rep["features"]) @ basis.U


@dataclass(frozen=True)
class Basis:
    U: np.ndarray  # feature dimension x rank
    V: np.ndarray  # classes x rank


def _pad_columns(base: np.ndarray, width: int, seed: int) -> np.ndarray:
    rows = base.shape[0]
    cols = [base[:, i] for i in range(base.shape[1])]
    rng = np.random.default_rng(seed)
    while len(cols) < width:
        v = rng.normal(size=rows)
        for q in cols:
            v = v - float(v @ q) * q
        norm = float(np.linalg.norm(v))
        if norm < 1e-10:
            v = np.zeros(rows)
            v[len(cols) % rows] = 1.0
            for q in cols:
                v = v - float(v @ q) * q
            norm = float(np.linalg.norm(v))
        cols.append(v / max(norm, 1e-12))
    return np.column_stack(cols[:width])


def fit_shared_basis(training_rep: dict[str, np.ndarray], rank: int, ridge: float = 1e-2) -> Basis:
    """Fit a shared low-rank CE basis using model-fit labels only.

    A balanced logistic regression is a direct label-likelihood fit.  Its
    coefficient matrix is factorized into U,V, then deterministically padded
    for rank four.  No validation/outcome sample is read here.
    """
    x = layer_norm(training_rep["features"])
    y = training_rep["labels"]
    clf = LogisticRegression(C=1.0 / max(float(ridge), 1e-8), class_weight="balanced", solver="lbfgs", max_iter=250, random_state=stable_seed("basis", x.shape, rank))
    clf.fit(x, y)
    coef = np.asarray(clf.coef_, np.float64)
    inter = np.asarray(clf.intercept_, np.float64)
    if coef.shape[0] == 1:
        w = np.column_stack([-0.5 * coef[0], 0.5 * coef[0]])
        b = np.asarray([-0.5 * inter[0], 0.5 * inter[0]])
    else:
        w = coef.T
        b = inter
    ud, _, vh = np.linalg.svd(w, full_matrices=False)
    take_u = min(int(rank), ud.shape[1])
    take_v = min(int(rank), vh.shape[0])
    u = _pad_columns(ud[:, :take_u], int(rank), stable_seed("U", rank, x.shape[1]))
    v = _pad_columns(vh[:take_v].T, int(rank), stable_seed("V", rank))
    # Keep a deterministic sign convention so archives and tests are stable.
    for j in range(int(rank)):
        if u[np.argmax(np.abs(u[:, j])), j] < 0:
            u[:, j] *= -1.0
            v[:, j] *= -1.0
    return Basis(u.astype(np.float64), v.astype(np.float64))


def reconstruct(a: np.ndarray, c: np.ndarray, basis: Basis, rep: dict[str, np.ndarray]) -> np.ndarray:
    response = layer_response(rep, basis)
    return (response * np.asarray(a, np.float64)[None, :]) @ basis.V.T + np.asarray(c, np.float64)[None, :]


def _adapter_design(rep: dict[str, np.ndarray], basis: Basis) -> np.ndarray:
    """Return per-sample/per-class Jacobian for (a,c)."""
    response = layer_response(rep, basis)
    n, r = response.shape
    design = np.zeros((n, CLASSES, r + CLASSES), np.float64)
    for j in range(r):
        design[:, :, j] = response[:, j, None] * basis.V[None, :, j]
    for cls in range(CLASSES):
        design[:, cls, r + cls] = 1.0
    return design


def _ce_objective(theta: np.ndarray, rep: dict[str, np.ndarray], basis: Basis, lambda_a: float, lambda_c: float, need_hessian: bool = True):
    r = basis.U.shape[1]
    a = theta[:r]
    c = theta[r:]
    logits = rep["logits"] + reconstruct(a, c, basis, rep)
    p = softmax(logits, axis=1)
    y = rep["labels"]
    w = class_weights(y)
    denom = max(float(np.sum(w)), 1e-12)
    ce = float(np.sum(w * (-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)))) / denom)
    reg = float(lambda_a * np.dot(a, a) + lambda_c * np.dot(c, c))
    design = _adapter_design(rep, basis)
    residual = (p - np.eye(CLASSES)[y]) * (w / denom)[:, None]
    grad = np.einsum("nc,ncm->m", residual, design)
    grad[:r] += 2.0 * lambda_a * a
    grad[r:] += 2.0 * lambda_c * c
    if not need_hessian:
        return ce + reg, grad, None
    hess = np.zeros((r + CLASSES, r + CLASSES), np.float64)
    for i in range(len(y)):
        pi = p[i]
        cov = np.diag(pi) - np.outer(pi, pi)
        hess += (w[i] / denom) * (design[i].T @ cov @ design[i])
    hess[:r, :r] += 2.0 * lambda_a * np.eye(r)
    hess[r:, r:] += 2.0 * lambda_c * np.eye(CLASSES)
    return ce + reg, grad, hess


def fit_ce_adapter(rep: dict[str, np.ndarray], basis: Basis, l2: float) -> dict[str, object]:
    """Fit (a,c) by minimizing class-balanced CE plus explicit L2 penalties."""
    r = basis.U.shape[1]
    theta = np.zeros(r + CLASSES, np.float64)
    best_loss = float("inf")
    for _ in range(40):
        loss, grad, hess = _ce_objective(theta, rep, basis, l2, l2, need_hessian=True)
        if not np.isfinite(loss) or not np.isfinite(grad).all():
            break
        if loss < best_loss:
            best_loss = loss
        if np.linalg.norm(grad) < 1e-7:
            break
        try:
            step = np.linalg.solve(hess + 1e-7 * np.eye(len(theta)), grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess + 1e-5 * np.eye(len(theta))) @ grad
        rate = 1.0
        improved = False
        while rate >= 1e-5:
            cand = theta - rate * step
            cand_loss, _, _ = _ce_objective(cand, rep, basis, l2, l2, need_hessian=False)
            if np.isfinite(cand_loss) and cand_loss <= loss + 1e-10:
                theta = cand
                improved = True
                break
            rate *= 0.5
        if not improved:
            break
    return {"a": theta[:r], "c": theta[r:], "fit_loss": float(best_loss), "objective": "class_balanced_cross_entropy", "future_labels_used_for_fit": False}


def fit_intercept(rep: dict[str, np.ndarray], l2: float) -> dict[str, object]:
    """Fit only the intercept by true class-balanced CE."""
    y = rep["labels"]
    counts = np.bincount(y, minlength=CLASSES).astype(float)
    target = np.log(np.maximum(counts, 1e-12))
    target -= target.mean()
    c = target / (1.0 + 2.0 * float(l2))
    # A small Newton refinement retains the explicit likelihood objective.
    for _ in range(20):
        logits = rep["logits"] + c[None, :]
        p = softmax(logits, axis=1)
        w = class_weights(y); den = max(w.sum(), 1e-12)
        g = (w[:, None] * (p - np.eye(CLASSES)[y])).sum(axis=0) / den + 2*l2*c
        H = np.zeros((CLASSES, CLASSES))
        for i in range(len(y)):
            H += w[i] / den * (np.diag(p[i]) - np.outer(p[i], p[i]))
        H += 2*l2*np.eye(CLASSES)
        try: c -= np.linalg.solve(H + 1e-8*np.eye(CLASSES), g)
        except np.linalg.LinAlgError: break
        if np.linalg.norm(g) < 1e-8: break
    return {"a": np.zeros(0, np.float64), "c": c, "fit_loss": balanced_ce(rep["logits"] + c[None, :], y), "objective": "class_balanced_cross_entropy_intercept", "future_labels_used_for_fit": False}


def fit_previous_pda(rep: dict[str, np.ndarray], basis: Basis, ridge: float) -> dict[str, object]:
    """Legacy full-PDA comparator, isolated from the primary CE implementation."""
    x = layer_norm(rep["features"])
    design = np.column_stack([x, np.ones(len(x))])
    # This is deliberately retained only as the historical comparator.
    target_logits = np.zeros((len(x), CLASSES), np.float64)
    target_logits[np.arange(len(x)), rep["labels"]] = 2.0
    residual = target_logits - rep["logits"]
    reg = float(ridge) * np.eye(design.shape[1])
    reg[-1, -1] *= 0.25
    coef = np.linalg.solve(design.T @ design + reg, design.T @ residual)
    matrix = coef[:-1].T
    intercept = coef[-1]
    # Least-squares projection is only for reproducing the old comparator.
    cols = [np.outer(basis.V[:, j], basis.U[:, j]).reshape(-1) for j in range(basis.U.shape[1])]
    a = np.linalg.lstsq(np.column_stack(cols), matrix.reshape(-1), rcond=None)[0]
    return {"a": a, "c": intercept, "fit_loss": float("nan"), "objective": "legacy_full_pda_comparator", "future_labels_used_for_fit": False}


def concat_blocks(blocks: Iterable[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    blocks = list(blocks)
    return {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}


def metric(rep: dict[str, np.ndarray], a: np.ndarray, c: np.ndarray, basis: Basis) -> dict[str, float]:
    logits = rep["logits"] + reconstruct(a, c, basis, rep)
    y = rep["labels"]
    pred = logits.argmax(axis=1)
    recalls = [float(np.mean(pred[y == cls] == cls)) for cls in range(CLASSES) if np.any(y == cls)]
    ba = float(np.mean(recalls)) if recalls else float("nan")
    f1 = float(f1_score(y, pred, labels=list(range(CLASSES)), average="macro", zero_division=0))
    margins = logits[:, 1] - logits[:, 0]
    signed = np.where(y == 1, margins, -margins)
    ece = expected_calibration_error(logits, y)
    return {"BA": ba, "macro_F1": f1, "margin": float(np.mean(signed)), "CE": balanced_ce(logits, y), "ECE": ece}


def expected_calibration_error(logits: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    p = softmax(logits, axis=1)[:, 1]
    conf = np.maximum(p, 1.0 - p)
    pred = (p >= 0.5).astype(int)
    out = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(0, 1, bins + 1)[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if np.any(m):
            out += float(np.mean(m)) * abs(float(np.mean(pred[m] == y[m])) - float(np.mean(conf[m])))
    return float(out)


def select_one_se(held: list[dict[str, np.ndarray]], deltas: list[dict[str, np.ndarray]], basis: Basis) -> tuple[float, dict[str, object]]:
    curves = []
    for alpha in ALPHAS:
        values = []
        bas = []
        for block, params in zip(held, deltas):
            pop = metric(block, np.zeros(basis.U.shape[1]), np.zeros(CLASSES), basis)
            ad = metric(block, np.asarray(params["a"]), np.asarray(params["c"]), basis)
            # Recompute CE at the fixed scale without changing fit parameters.
            logits = block["logits"] + alpha * reconstruct(np.asarray(params["a"]), np.asarray(params["c"]), basis, block)
            values.append(balanced_ce(logits, block["labels"]))
            bas.append(float(metric(block, alpha*np.asarray(params["a"]), alpha*np.asarray(params["c"]), basis)["BA"]))
        vals = np.asarray(values, float)
        mean = float(np.mean(vals)); se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        curves.append({"alpha": alpha, "mean_CE": mean, "SE_CE": se, "mean_BA": float(np.mean(bas)), "held_CE": vals.tolist()})
    best = min(curves, key=lambda x: (x["mean_CE"], x["alpha"]))
    threshold = float(best["mean_CE"] + best["SE_CE"])
    eligible = [row for row in curves if row["mean_CE"] <= threshold + 1e-12]
    selected = float(min(eligible, key=lambda x: x["alpha"])["alpha"])
    return selected, {"curves": curves, "best_alpha": best["alpha"], "one_se_threshold": threshold, "selected_alpha": selected}


def eb_alpha(one_se_alpha: float, historical_utility: float, utility_se: float, prior: Mapping[str, float]) -> tuple[float, float]:
    """Shrink only weak/negative evidence toward the population fallback."""
    mu = float(prior.get("mu", 0.0)); tau = max(float(prior.get("tau", 0.0)), 1e-8)
    obs = max(float(utility_se), 1e-8)
    weight = tau * tau / (tau * tau + obs * obs)
    posterior = weight * float(historical_utility) + (1.0 - weight) * mu
    return (float(one_se_alpha) if posterior > 0.0 else 0.0), float(posterior)


def fit_subject(tr: SubjectTransition, basis: Basis, l2: float, prior: Mapping[str, float] | None = None, cache: dict | None = None) -> dict[str, object]:
    """Fit all source-only subject quantities needed by U-PDA and controls."""
    keybase = (tr.subject, round(float(l2), 8), basis.U.shape[1])
    def fit(rep):
        key = ("ce", keybase, tuple(rep["indices"].tolist())) if cache is not None else None
        if cache is not None and key in cache: return cache[key]
        out = fit_ce_adapter(rep, basis, l2)
        if cache is not None: cache[key] = out
        return out
    loo_params = []
    held = list(tr.history_blocks)
    for k in range(len(held)):
        train = concat_blocks([held[j] for j in range(len(held)) if j != k])
        loo_params.append(fit(train))
    full = fit(concat_blocks(held))
    ordinary = fit(held[-1])
    intercept = fit_intercept(concat_blocks(held), l2)
    prev = fit_previous_pda(concat_blocks(held), basis, l2)
    alpha, curve = select_one_se(held, loo_params, basis)
    held_utils = []
    for block, params in zip(held, loo_params):
        pop = metric(block, np.zeros(basis.U.shape[1]), np.zeros(CLASSES), basis)
        logits = block["logits"] + reconstruct(np.asarray(params["a"]), np.asarray(params["c"]), basis, block)
        held_utils.append(pop["CE"] - balanced_ce(logits, block["labels"]))
    hu = float(np.mean(held_utils))
    hse = float(np.std(held_utils, ddof=1) / np.sqrt(len(held_utils))) if len(held_utils) > 1 else 0.0
    eb_a, posterior = eb_alpha(alpha, hu, hse, prior or {"mu": 0.0, "tau": 0.0})
    return {"population": {"a": np.zeros(basis.U.shape[1]), "c": np.zeros(CLASSES)},
            "intercept_only": intercept, "ordinary_adapter": ordinary, "previous_full_pda": prev,
            "persistent_ce": full, "u_pda": full, "eb_u_pda": full,
            "loo": loo_params, "alpha": alpha, "eb_alpha": eb_a, "posterior_utility": posterior,
            "alpha_curve": curve, "historical_utility": hu, "historical_utility_se": hse,
            "historical_positive_blocks": int(sum(x > 0 for x in held_utils)),
            "future_labels_used_for_fit": False, "future_session_used_for_fit": False,
            "persistent_norm": float(np.linalg.norm(full["a"]) + np.linalg.norm(full["c"])),
            "transient_norm": float(np.mean([np.linalg.norm(p["a"]) + np.linalg.norm(p["c"]) for p in loo_params])),
            "persistent_transient_ratio": float((np.linalg.norm(full["a"]) + np.linalg.norm(full["c"])) / max(np.mean([np.linalg.norm(p["a"]) + np.linalg.norm(p["c"]) for p in loo_params]), 1e-12))}


def params_for(method: str, methods: dict[str, object], subject: str, subject_ids: list[str], alpha_map: Mapping[str, float], adapter_map: Mapping[str, dict[str, object]], rank: int, seed: int) -> tuple[str, dict[str, object], float]:
    """Resolve a future method and its target alpha."""
    if method == "population": return method, methods["population"], 0.0
    if method == "intercept_only": return method, methods[method], 1.0
    if method == "ordinary_adapter": return method, methods[method], 1.0
    if method == "previous_full_pda": return method, methods[method], 1.0
    if method == "persistent_ce": return method, methods[method], 1.0
    if method == "correct_adapter": return method, methods["persistent_ce"], float(methods["alpha"])
    if method == "u_pda": return method, methods[method], float(methods["alpha"])
    if method == "eb_u_pda": return method, methods[method], float(methods["eb_alpha"])
    if method == "random_gate":
        # Same frozen adapter family as U-PDA, with alpha labels permuted
        # across subjects; only the gate assignment is randomized.
        return method, methods["persistent_ce"], float(alpha_map[subject])
    if method in {"wrong_adapter", "shuffled_adapter"}:
        ids = list(subject_ids)
        pos = ids.index(subject)
        if method == "shuffled_adapter": other = ids[(pos + 1) % len(ids)]
        else:
            # norm-matched wrong direction, deterministic and never self.
            target_norm = float(methods.get("persistent_norm", 0.0))
            candidates = [s for s in ids if s != subject]
            other = min(candidates, key=lambda s: (abs(float(adapter_map[s]["persistent_norm"]) - target_norm), s))
        return method, adapter_map[other]["persistent_ce"], float(methods["alpha"])
    if method == "oracle_alpha":
        return method, methods["persistent_ce"], float(methods.get("oracle_alpha", 0.0))
    raise KeyError(method)


def evaluate_subject(tr: SubjectTransition, methods: dict[str, object], basis: Basis, subject_ids: list[str], adapter_map: Mapping[str, dict[str, object]], role: str, dataset: str, fold: int, seed: int, recipe_id: str, alpha_map: Mapping[str, float]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    method_names = ("population", "intercept_only", "ordinary_adapter", "previous_full_pda", "persistent_ce", "u_pda", "eb_u_pda", "random_gate", "correct_adapter", "wrong_adapter", "shuffled_adapter", "oracle_alpha")
    # Oracle is a diagnostic upper bound only and is never used by alpha selection.
    oracle_values = []
    for a in ALPHAS:
        oracle_values.append(metric(tr.future, a*np.asarray(methods["persistent_ce"]["a"]), a*np.asarray(methods["persistent_ce"]["c"]), basis)["BA"])
    methods["oracle_alpha"] = float(ALPHAS[int(np.argmax(oracle_values))])
    rows = []
    for name in method_names:
        _, params, alpha = params_for(name, methods, tr.subject, subject_ids, alpha_map, adapter_map, basis.U.shape[1], stable_seed(dataset, fold, seed))
        if name == "intercept_only":
            a = np.zeros(basis.U.shape[1]); c = np.asarray(params["c"])
        else:
            a = np.asarray(params["a"]); c = np.asarray(params["c"])
        m = metric(tr.future, alpha*a, alpha*c, basis)
        rows.append({"dataset": dataset, "role": role, "fold": fold, "seed": seed, "subject": tr.subject, "method": name, "recipe": recipe_id,
                     "alpha": float(alpha), **m, "future_session_used_for_fit": False, "future_labels_used_for_fit": False,
                     "population_checkpoint_unchanged": True, "oracle_label": "DIAGNOSTIC_UPPER_BOUND_ONLY" if name == "oracle_alpha" else ""})
    curve_rows = []
    for row in methods["alpha_curve"]["curves"]:
        curve_rows.append({"dataset": dataset, "role": role, "fold": fold, "seed": seed, "subject": tr.subject, "recipe": recipe_id,
                           "alpha": row["alpha"], "mean_CE": row["mean_CE"], "SE_CE": row["SE_CE"], "mean_BA": row["mean_BA"],
                           "selected_one_se": float(methods["alpha"]), "selected_eb": float(methods["eb_alpha"]),
                           "historical_utility": methods["historical_utility"], "historical_utility_se": methods["historical_utility_se"],
                           "positive_held_blocks": methods["historical_positive_blocks"], "future_labels_used_for_selection": False})
    return rows, curve_rows


def aggregate_subject(frame: pd.DataFrame, method: str, dataset: str) -> pd.Series:
    return frame[(frame.dataset == dataset) & (frame.method == method)].groupby("subject", sort=True).BA.mean()


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    values = np.asarray(values, np.float64)
    if values.size == 0: return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    sample = values[rng.integers(0, len(values), size=(int(draws), len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(sample, .025)), float(np.quantile(sample, .975))


def paired_delta(frame: pd.DataFrame, left: str, right: str, dataset: str, tag: str) -> dict[str, object]:
    a = aggregate_subject(frame, left, dataset); b = aggregate_subject(frame, right, dataset)
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).to_numpy(float)
    mean, lo, hi = bootstrap_ci(d, stable_seed("paired", tag, dataset, left, right))
    return {"dataset": dataset, "comparison": f"{left}-{right}", "delta_BA": mean, "CI95_L": lo, "CI95_U": hi,
            "subjects": int(len(d)), "positive_subject_fraction": float(np.mean(d > 0)) if len(d) else float("nan"),
            "paired_subject_unit": True}


def pooled_paired_delta(frame: pd.DataFrame, left: str, right: str, tag: str) -> dict[str, object]:
    pieces=[]
    for ds in DATASETS:
        a=aggregate_subject(frame,left,ds); b=aggregate_subject(frame,right,ds); common=a.index.intersection(b.index)
        pieces.extend((a.loc[common]-b.loc[common]).to_numpy(float))
    d=np.asarray(pieces,float); mean,lo,hi=bootstrap_ci(d,stable_seed("pooled",tag,left,right))
    return {"dataset":"POOLED","comparison":f"{left}-{right}","delta_BA":mean,"CI95_L":lo,"CI95_U":hi,"subjects":int(len(d)),"positive_subject_fraction":float(np.mean(d>0)) if len(d) else float("nan"),"paired_subject_unit":True}


def association(rows: pd.DataFrame) -> dict[str, object]:
    """Predictive association between historical utility and future gain."""
    x = rows["historical_utility"].to_numpy(float); y = rows["future_gain"].to_numpy(float)
    out: dict[str, object] = {"n": int(len(rows))}
    if len(rows) >= 3 and np.std(x) > 0 and np.std(y) > 0:
        out["pearson_r"] = float(pearsonr(x, y).statistic)
        out["spearman_r"] = float(spearmanr(x, y).statistic)
    else: out["pearson_r"] = None; out["spearman_r"] = None
    label = (y > 0).astype(int)
    out["auroc_future_gain_gt_0"] = float(roc_auc_score(label, x)) if len(np.unique(label)) == 2 else None
    pos = rows[x > 0]["future_gain"]; neg = rows[x <= 0]["future_gain"]
    out["positive_hist_mean_future_gain"] = float(pos.mean()) if len(pos) else None
    out["negative_hist_mean_future_gain"] = float(neg.mean()) if len(neg) else None
    return out
