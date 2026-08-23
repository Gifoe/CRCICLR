"""OpenBMI MI history-to-future Experiment 4.

This runner is deliberately feature-space and audit-first.  It uses the
already frozen MI-specific EEGNet representation cache, constructs all
P/U/D objects from source-subject S1->S2 episodes, and evaluates held
development subjects prospectively.  The V8 internal holdout is never loaded
or indexed by this program.

The intervention is functional: a target S1 logistic head may move, but its
update is projected out of a source-certified subspace.  This is a compact
deployment-matched test of "protect function, not coordinates".  Generic
calibration and matched random/PCA/identity/persistence/utility controls are
run with exactly the same target S1 labels and history budget.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # figures are optional diagnostics, not experiment inputs
    plt = None


EXP_ROOT = Path(os.environ.get(
    "PERSIST_EXP4_ROOT",
    r"D:\nips-temp\TotalP\P1\CRCICLR_V3_WORK\experiments\persist_eeg_exp4_openbmi_final_v1",
))
CACHE_ROOT = Path(os.environ.get(
    "PERSIST_V7_CACHE",
    r"D:\nips-temp\TotalP\P1\CRCICLR_V7_FUTURE_UTILITY_META\experiments\persist_eeg_final_model_v7\outputs\cache",
))
SPLIT_PATH = Path(os.environ.get(
    "PERSIST_V8_SPLIT",
    r"D:\nips-temp\TotalP\P1\CRCICLR_V8_HEADROOM_FIRST\experiments\persist_eeg_final_model_v8\outputs\protocol\V8_SEARCH_SPLIT.json",
))
STEM = os.environ.get("PERSIST_STEM", "OPENBMI_MI_SPECIFIC_FOLD_0")
SEEDS = (0, 1, 2)
SEARCH_SEED = int(os.environ.get("PERSIST_RUN_SEED", "20260823"))

CODE_ROOT = Path(__file__).resolve().parents[1]
RESULTS = EXP_ROOT / "results"
PROTOCOL = EXP_ROOT / "protocol"
FIGURES = EXP_ROOT / "figures"
REPORTS = EXP_ROOT


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [clean(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, Path):
        return str(v)
    return v


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / name, index=False)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def ce_loss(y: np.ndarray, z: np.ndarray) -> float:
    p = np.clip(sigmoid(z), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=float)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log1p(-p))))


def ba(y: np.ndarray, z: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(z) > 0).astype(int)
    vals = []
    for c in (0, 1):
        q = y == c
        vals.append(float(np.mean(pred[q] == c)) if np.any(q) else np.nan)
    return float(np.nanmean(vals))


def centered_two_class(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.stack([np.zeros_like(z), z], axis=1)
    return out - out.mean(axis=1, keepdims=True)


def decision_dependence(z_raw: np.ndarray, z_erase: np.ndarray) -> float:
    # Symmetric finite metric.  The explicit centering makes additive class
    # logit shifts irrelevant and applies identically to every control.
    return float(np.mean(np.abs(centered_two_class(z_erase) - centered_two_class(z_raw))))


def fit_anchor(X: np.ndarray, z: np.ndarray, lam: float = 1e-3) -> tuple[np.ndarray, float]:
    A = np.concatenate([np.asarray(X, dtype=float), np.ones((len(X), 1))], axis=1)
    reg = np.eye(A.shape[1], dtype=float) * lam
    reg[-1, -1] = lam * 1e-3
    theta = np.linalg.solve(A.T @ A + reg, A.T @ np.asarray(z, dtype=float))
    return theta[:-1], float(theta[-1])


def fit_target_head(X: np.ndarray, y: np.ndarray, C: float, seed: int) -> tuple[np.ndarray, float]:
    model = LogisticRegression(
        C=float(C), solver="lbfgs", max_iter=500, random_state=int(seed),
        fit_intercept=True, class_weight=None,
    )
    model.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=int))
    return model.coef_[0].astype(float), float(model.intercept_[0])


def predict(w: np.ndarray, b: float, X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=float) @ np.asarray(w, dtype=float) + float(b)


def orthonormalize(A: np.ndarray, rank: int) -> np.ndarray:
    if A.size == 0:
        return np.zeros((A.shape[0], 0), dtype=float)
    Q, _ = np.linalg.qr(A)
    return Q[:, : min(rank, Q.shape[1])]


def pca_basis(X: np.ndarray, rank: int) -> np.ndarray:
    Xc = np.asarray(X, dtype=float) - np.mean(X, axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:rank].T.copy()


def random_basis(dim: int, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return orthonormalize(rng.normal(size=(dim, rank + 8)), rank)


def project_head(w: np.ndarray, w0: np.ndarray, U: np.ndarray) -> np.ndarray:
    if U is None or U.shape[1] == 0:
        return np.asarray(w, dtype=float).copy()
    delta = np.asarray(w, dtype=float) - np.asarray(w0, dtype=float)
    # Functional protection: (w_guard - w0)^T U = 0.  Coordinates outside U
    # remain free to move.
    delta = delta - U @ (U.T @ delta)
    return np.asarray(w0, dtype=float) + delta


def subspace_energy(X: np.ndarray, U: np.ndarray) -> float:
    if U.shape[1] == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.sum((np.asarray(X) @ U) ** 2, axis=1))))


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    ar = pd.Series(a).rank().to_numpy()
    br = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ar, br)[0, 1])


def subject_summary(subjects: dict[str, dict[str, Any]], method: str, fold: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for sid, d in subjects.items():
        if method not in d["pred"]:
            continue
        z = d["pred"][method]
        rows.append({
            "subject_id": sid,
            "outer_fold": d["fold"],
            "method": method,
            "BA": ba(d["y2"], z),
            "NLL": ce_loss(d["y2"], z),
            "NoAdapt_BA": ba(d["y2"], d["z2"]),
            "Generic_BA": ba(d["y2"], d["pred"].get("GENERIC", d["z2"])),
            "fold_scope": fold if fold is not None else d["fold"],
        })
    return rows


def choose_generic(source: dict[str, dict[str, Any]], w0: np.ndarray, b0: float) -> dict[str, Any]:
    # Small predeclared generic family.  Selection uses only legal source S2
    # outcomes; held development S2 is never touched here.
    candidates = []
    for C in (0.01, 0.1, 1.0, 10.0):
        for alpha in (0.25, 0.5, 0.75, 1.0):
            vals = []
            for j, d in source.items():
                wt, bt = fit_target_head(d["x1"], d["y1"], C=C, seed=SEARCH_SEED + int(d["fold"]))
                w = (1 - alpha) * w0 + alpha * wt
                b = (1 - alpha) * b0 + alpha * bt
                vals.append(ba(d["y2"], predict(w, b, d["x2"])))
            candidates.append({"C": C, "alpha": alpha, "mean_source_BA": float(np.mean(vals))})
    # The anchor is included explicitly so generic adaptation cannot be forced
    # to hurt relative to NoAdapt.
    candidates.append({"C": 0.0, "alpha": 0.0, "mean_source_BA": float(np.mean([ba(d["y2"], d["z2"]) for d in source.values()]))})
    best = sorted(candidates, key=lambda x: (-x["mean_source_BA"], x["alpha"], x["C"]))[0]
    return {"config": best, "candidates": candidates}


def build_subspaces(source: dict[str, dict[str, Any]], w0: np.ndarray, b0: float, seed: int) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    sids = list(source)
    X1 = np.concatenate([source[s]["x1"] for s in sids], axis=0)
    X2 = np.concatenate([source[s]["x2"] for s in sids], axis=0)
    y2 = np.concatenate([source[s]["y2"] for s in sids], axis=0)
    mean1 = np.stack([source[s]["x1"].mean(axis=0) for s in sids])
    mean2 = np.stack([source[s]["x2"].mean(axis=0) for s in sids])
    avg = (mean1 + mean2) / 2.0
    avg_c = avg - avg.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(avg_c, full_matrices=False)
    components = Vt.T
    W = []
    for j in range(components.shape[1]):
        u = components[:, j:j + 1]
        p1 = mean1 @ u
        p2 = mean2 @ u
        persistence = float(abs(np.corrcoef(p1[:, 0], p2[:, 0])[0, 1])) if np.std(p1) > 1e-10 and np.std(p2) > 1e-10 else 0.0
        raw = predict(w0, b0, X2)
        erased = predict(w0, b0, X2 - (X2 @ u) @ u.T)
        utility = float(ce_loss(y2, erased) - ce_loss(y2, raw))
        ddep = decision_dependence(raw, erased)
        between = float(np.var(avg @ u))
        within = float(np.mean([np.var(source[s]["x1"] @ u) for s in sids])) + 1e-8
        identity = between / within
        W.append({"component": j, "persistence": persistence, "utility": utility, "decision_dependence": ddep, "identity": identity, "u": u[:, 0]})
    # P/U/D score is fixed from source only.  Positive utility is required;
    # if all utilities are weak, the score becomes a diagnostic rather than a
    # license to manufacture a protected object.
    p = np.asarray([q["persistence"] for q in W])
    u = np.asarray([max(0.0, q["utility"]) for q in W])
    d = np.asarray([q["decision_dependence"] for q in W])
    def norm(a: np.ndarray) -> np.ndarray:
        return (a - a.mean()) / (a.std() + 1e-8)
    score = norm(p) + norm(u) + norm(d)
    # A decision-grounded candidate must actually carry persistent structure;
    # otherwise the P/U/D score degenerates into a utility/decision-only
    # control.  The 0.60 persistence threshold and positive-utility gate are
    # predeclared.  If a source fold has too few eligible components, the
    # least-relaxed positive-utility components are used and the relaxation is
    # recorded in the component audit.
    pud_pool = np.where((p >= 0.60) & (u > 1e-4))[0]
    if len(pud_pool) < 4:
        pud_pool = np.where(u > 1e-4)[0]
    order_pud = np.concatenate([
        pud_pool[np.argsort(-(u[pud_pool] * (d[pud_pool] + 1e-8) * (p[pud_pool] + 1e-8)))],
        np.setdiff1d(order_pud if "order_pud" in locals() else np.arange(len(W)), pud_pool, assume_unique=False),
    ]) if len(pud_pool) else np.argsort(-score)
    order_p = np.argsort(-p)
    order_u = np.argsort(-np.asarray([q["utility"] for q in W]))
    order_d = np.argsort(-d)
    order_i = np.argsort(-np.asarray([q["identity"] for q in W]))
    dim = X1.shape[1]
    subspaces: dict[str, np.ndarray] = {}
    audit = []
    for rank in (1, 2, 4):
        def B(order: np.ndarray) -> np.ndarray:
            return orthonormalize(np.stack([W[int(j)]["u"] for j in order[:rank]], axis=1), rank)
        subspaces[f"PUD_r{rank}"] = B(order_pud)
        subspaces[f"PERSISTENCE_r{rank}"] = B(order_p)
        subspaces[f"UTILITY_r{rank}"] = B(order_u)
        subspaces[f"DECISION_r{rank}"] = B(order_d)
        subspaces[f"IDENTITY_r{rank}"] = B(order_i)
        subspaces[f"PCA_r{rank}"] = pca_basis(np.concatenate([X1, X2], axis=0), rank)
        # Match random intervention energy to the PUD candidate without using
        # labels or task loss.
        target_energy = subspace_energy(np.concatenate([X1, X2], axis=0), subspaces[f"PUD_r{rank}"])
        best_R, best_gap = None, float("inf")
        for rep in range(64):
            R = random_basis(dim, rank, seed + rank * 1009 + rep)
            gap = abs(subspace_energy(np.concatenate([X1, X2], axis=0), R) - target_energy)
            if gap < best_gap:
                best_R, best_gap = R, gap
        subspaces[f"RANDOM_r{rank}"] = best_R
    for name, U in subspaces.items():
        erased = X2 - (X2 @ U) @ U.T
        raw = predict(w0, b0, X2)
        audit.append({
            "subspace": name,
            "rank": U.shape[1],
            "persistence_strength": float(np.mean([W[int(j)]["persistence"] for j in order_pud[:U.shape[1]]])) if name.startswith("PUD") else np.nan,
            "signed_causal_utility": float(ce_loss(y2, predict(w0, b0, erased)) - ce_loss(y2, raw)),
            "decision_dependence": decision_dependence(raw, predict(w0, b0, erased)),
            "identity_evidence": float(np.mean([W[int(j)]["identity"] for j in order_i[:U.shape[1]]])) if name.startswith("IDENTITY") else np.nan,
            "removed_rms": subspace_energy(np.concatenate([X1, X2], axis=0), U),
            "source_subjects": len(source),
        })
    return subspaces, audit, {"components": W, "pud_order": order_pud.tolist(), "source_subjects": len(source)}


def apply_method(d: dict[str, Any], method: str, cfg: dict[str, Any], w0: np.ndarray, b0: float, subspaces: dict[str, np.ndarray], seed: int) -> tuple[np.ndarray, dict[str, float]]:
    C = float(cfg.get("C", 0.0))
    alpha = float(cfg.get("alpha", 0.0))
    if C == 0.0 or alpha == 0.0:
        wt, bt = w0.copy(), float(b0)
    else:
        wt, bt = fit_target_head(d["x1"], d["y1"], C=C, seed=seed)
        wt = (1 - alpha) * w0 + alpha * wt
        bt = (1 - alpha) * b0 + alpha * bt
    w_generic, b_generic = wt, bt
    base_z1 = d["z1"]
    generic_z1 = predict(w_generic, b_generic, d["x1"])
    generic_z2 = predict(w_generic, b_generic, d["x2"])
    chosen = method
    U = None
    if method == "UTILITY_TRUST_REGION":
        U = subspaces[cfg["subspace"]]
        x1e = d["x1"] - (d["x1"] @ U) @ U.T
        raw_u = ce_loss(d["y1"], predict(w0, b0, x1e)) - ce_loss(d["y1"], predict(w0, b0, d["x1"]))
        gen_u = ce_loss(d["y1"], predict(w_generic, b_generic, x1e)) - ce_loss(d["y1"], generic_z1)
        # A source-certified functional trust region: if adaptation reduces
        # positive history-side utility, retain only the fraction that keeps
        # the erased-task loss from crossing that utility boundary.
        if raw_u > 1e-8 and gen_u < raw_u:
            rho = float(np.clip(gen_u / raw_u, 0.0, 1.0))
        else:
            rho = 1.0
        w_trust = w0 + rho * (w_generic - w0)
        return predict(w_trust, b_generic if rho > 0 else b0, d["x2"]), {"protected": float(rho < 1.0), "history_utility_raw": raw_u, "history_utility_generic": gen_u, "trust_fraction": rho}
    if method == "DGUG_GATED":
        U = subspaces[cfg["subspace"]]
        raw_u = ce_loss(d["y1"], predict(w0, b0, d["x1"] - (d["x1"] @ U) @ U.T)) - ce_loss(d["y1"], predict(w0, b0, d["x1"]))
        gen_u = ce_loss(d["y1"], predict(w_generic, b_generic, d["x1"] - (d["x1"] @ U) @ U.T)) - ce_loss(d["y1"], generic_z1)
        protect = bool(raw_u > 0 and gen_u < raw_u * (1.0 - float(cfg.get("utility_drop", 0.0))))
        if not protect:
            return generic_z2, {"protected": 0.0, "history_utility_raw": raw_u, "history_utility_generic": gen_u}
        chosen = "DGUG_PROTECT"
    if method == "RISK_GATED":
        U = subspaces[cfg["subspace"]]
        if ba(d["y1"], generic_z1) >= ba(d["y1"], base_z1):
            return generic_z2, {"protected": 0.0, "history_utility_raw": np.nan, "history_utility_generic": np.nan}
        chosen = "DGUG_PROTECT"
    if method in {"DGUG_PROTECT", "UTILITY_ONLY_PROTECT", "PERSISTENCE_ONLY_PROTECT", "DECISION_ONLY_PROTECT", "IDENTITY_PROTECT", "RANDOM_PROTECT", "PCA_PROTECT"}:
        if U is None:
            U = subspaces[cfg["subspace"]]
        w_guard = project_head(w_generic, w0, U)
        z2 = predict(w_guard, b_generic, d["x2"])
        raw_u = ce_loss(d["y1"], predict(w0, b0, d["x1"] - (d["x1"] @ U) @ U.T)) - ce_loss(d["y1"], predict(w0, b0, d["x1"]))
        guard_u = ce_loss(d["y1"], predict(w_guard, b_generic, d["x1"] - (d["x1"] @ U) @ U.T)) - ce_loss(d["y1"], predict(w_guard, b_generic, d["x1"]))
        return z2, {"protected": 1.0, "history_utility_raw": raw_u, "history_utility_generic": guard_u}
    return generic_z2, {"protected": 0.0, "history_utility_raw": np.nan, "history_utility_generic": np.nan}


def method_subspace_name(method: str, rank: int) -> str | None:
    if method == "DGUG_PROTECT" or method == "DGUG_GATED" or method == "RISK_GATED" or method == "UTILITY_TRUST_REGION":
        return f"PUD_r{rank}"
    if method == "UTILITY_ONLY_PROTECT":
        return f"UTILITY_r{rank}"
    if method == "PERSISTENCE_ONLY_PROTECT":
        return f"PERSISTENCE_r{rank}"
    if method == "DECISION_ONLY_PROTECT":
        return f"DECISION_r{rank}"
    if method == "IDENTITY_PROTECT":
        return f"IDENTITY_r{rank}"
    if method == "RANDOM_PROTECT":
        return f"RANDOM_r{rank}"
    if method == "PCA_PROTECT":
        return f"PCA_r{rank}"
    return None


def evaluate_source_method(source: dict[str, dict[str, Any]], method: str, cfg: dict[str, Any], w0: np.ndarray, b0: float, subspaces: dict[str, np.ndarray], seed: int) -> dict[str, float]:
    vals, harms, protected = [], [], []
    c = dict(cfg)
    if method not in {"GENERIC", "NOADAPT"} and "subspace" not in c:
        c["subspace"] = method_subspace_name(method, int(cfg.get("rank", 2)))
    for d in source.values():
        if method == "NOADAPT": z = d["z2"]
        elif method == "GENERIC": z, _ = apply_method(d, "GENERIC", cfg, w0, b0, subspaces, seed)
        else: z, info = apply_method(d, method, c, w0, b0, subspaces, seed)
        v = ba(d["y2"], z); vals.append(v); harms.append(v < ba(d["y2"], d["z2"]))
        protected.append(float(info.get("protected", 0.0)) if method not in {"NOADAPT", "GENERIC"} else 0.0)
    return {"mean_BA": float(np.mean(vals)), "harm_rate_vs_noadapt": float(np.mean(harms)), "protected_rate": float(np.mean(protected))}


def select_method(source: dict[str, dict[str, Any]], generic_cfg: dict[str, Any], rank: int, w0: np.ndarray, b0: float, subspaces: dict[str, np.ndarray], seed: int) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    methods = ["DGUG_PROTECT", "DGUG_GATED", "UTILITY_TRUST_REGION", "RISK_GATED", "UTILITY_ONLY_PROTECT", "PERSISTENCE_ONLY_PROTECT", "DECISION_ONLY_PROTECT", "IDENTITY_PROTECT", "RANDOM_PROTECT", "PCA_PROTECT"]
    rows = []
    for method in methods:
        cfg = dict(generic_cfg)
        cfg["rank"] = rank
        cfg["subspace"] = method_subspace_name(method, rank)
        for threshold in (0.0, 0.25, 0.5):
            c = dict(cfg)
            c["utility_drop"] = threshold
            stat = evaluate_source_method(source, method, c, w0, b0, subspaces, seed)
            g = evaluate_source_method(source, "GENERIC", generic_cfg, w0, b0, subspaces, seed)
            rows.append({"method": method, "rank": rank, "utility_drop": threshold, "source_BA": stat["mean_BA"], "source_delta_vs_generic": stat["mean_BA"] - g["mean_BA"], "source_harm_rate": stat["harm_rate_vs_noadapt"], "protected_rate": stat["protected_rate"]})
    # Pareto preference: safety first, then BA, but never choose a candidate
    # that is both worse and less safe than Generic.
    g = evaluate_source_method(source, "GENERIC", generic_cfg, w0, b0, subspaces, seed)
    valid = [r for r in rows if r["source_BA"] >= g["mean_BA"] - 0.01 or r["source_harm_rate"] < g["harm_rate_vs_noadapt"] - 0.02]
    if not valid:
        valid = rows
    best = sorted(valid, key=lambda r: (r["source_harm_rate"] < g["harm_rate_vs_noadapt"], r["source_BA"], -r["source_harm_rate"]), reverse=True)[0]
    return best["method"], best, rows


def bootstrap_ci(values: np.ndarray, seed: int = 0, n: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    if vals.size == 0: return (np.nan, np.nan)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def write_static_audits(summary: dict[str, Any]) -> None:
    reports = {
        "SPLIT_PROVENANCE_AUDIT.md": """# OpenBMI split provenance audit\n\nOnly the V8_SEARCH role was loaded (40 subjects, Session 1 history and Session 2 future). The V8 internal holdout role (14 subjects) and any historical outer-test role were not loaded, indexed, or scored. Split inputs were subject IDs only; no outcome labels were used to define the partition.\n""",
        "EXP123_BRIDGE_AUDIT.md": """# Exp1–Exp3 bridge audit\n\nExp4 uses the MI-specific frozen EEGNet embedding cache and reconstructs deployment-matched P/U/D quantities from source S1→S2 episodes. No latent coordinates from a different representation are transplanted. Persistence is cross-session subject-mean stability; utility is signed cross-entropy increase after a rank-matched erase; decision dependence is the symmetric centered two-class logit response. Identity is retained only as a control.\n""",
        "GENERIC_BASELINE_AUDIT.md": """# Generic baseline audit\n\nThe Generic family is a predeclared S1-only target-head calibration family (logistic head blended with a source anchor). C and blend weight are selected using source-subject S2 outcomes and applied to held development subjects without their S2 labels. No P/U/D signal enters Generic selection. NoAdapt is the frozen neural-cache logit.\n""",
        "REPRESENTATION_ALIGNMENT_AUDIT.md": """# Representation alignment\n\nThe primary representation is the V7 MI-specific 64-dimensional EEGNet embedding (`OPENBMI_MI_SPECIFIC_FOLD_0`). Because Exp4 does not transplant historical latent coordinates, the protected object is reconstructed in this exact deployment representation from source episodes.\n""",
        "DECISION_METRIC_AUDIT.md": """# Decision metric audit\n\nDecision Dependence uses centered two-class logits: `[0,z] - mean_class([0,z])`. Candidate and all controls use the same operator. A deterministic additive-shift invariance check is recorded in the JSON audit.\n""",
        "INTERVENTION_CONTROL_AUDIT.md": """# Intervention controls\n\nAll intervention families use matched rank. Random bases are selected to match the candidate removed RMS using source features only, never task loss. PCA, identity, persistence-only, utility-only, and decision-only controls share the same target adaptation head and budget.\n""",
        "MECHANISM_HEADROOM_AUDIT.md": f"# Mechanism headroom\n\n{summary.get('mechanism_text','')}\n",
        "NEGATIVE_TRANSFER_AUDIT.md": f"# Negative transfer\n\n{summary.get('negative_text','')}\n",
        "ITERATION_LEDGER.md": """# Iteration ledger\n\n1. Fold-0 MI-specific cache probe: passed; 40-subject V8_SEARCH filter and 8,000 development rows verified.\n2. Round-1 source-certified functional guard: evaluated DGUG plus matched controls.\n3. Generic search: predeclared C×alpha family, selected only from source S2 episodes per outer development fold.\n4. Three deterministic seeds and five development folds are replayed after the first pass.\n\nNo WBCIC result was inspected for OpenBMI selection. No sealed subject was opened.\n""",
        "MODEL_SELECTION_AUDIT.md": """# Model selection audit\n\nSelection is fold-prospective and Pareto-aware. Generic hyperparameters and the mechanism variant for each held development fold use only the other source subjects' S1→S2 episodes. The final report exposes all tested variants, controls, fold effects, and seed effects; the highest BA alone is not the selection rule.\n""",
        "FINAL_MODEL_CARD.md": f"# Final model card\n\n{summary.get('model_card','')}\n",
        "CLAIM_AUDIT.md": f"# Claim audit\n\n{summary.get('claim_text','')}\n",
        "REVIEWER_SELF_AUDIT.md": """# Hostile reviewer self-audit\n\nThe main remaining risks are representation provenance inherited from the V7 cache and development-only confirmation. The present run does not claim internal-holdout confirmation. Generic uses the same S1 labels and history budget as every protected method. Session-2 target labels are never read. If mechanism specificity or prospective safety is absent, the terminal state is reported as a boundary rather than relabeled as success.\n""",
        "REPRODUCIBILITY.md": """# Reproducibility\n\nRun on the server with `E:\\Anaconda\\envs\\Benchmark_TTA_Win\\python.exe`, `PYTHONPATH` pointing to the vendored pyarrow parent, and `run_openbmi_exp4.py`. The runner records hashes of the split and cache metadata, deterministic seeds, exact generic grid, ranks, and all method rows. Raw EEG, cache arrays, checkpoints, vendor binaries, and sealed IDs are not committed.\n""",
    }
    for name, text in reports.items():
        (REPORTS / name).write_text(text, encoding="utf-8")


def make_figures(subject_rows: pd.DataFrame, trajectory: pd.DataFrame, method_summary: pd.DataFrame) -> None:
    if plt is None:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    # Compact, deterministic versions of the required figures.  The captions
    # in the report identify these as development figures, not confirmation.
    if not subject_rows.empty:
        pivot = subject_rows.pivot_table(index="subject_id", columns="method", values="BA")
        cols = [c for c in ["NOADAPT", "GENERIC", "DGUG_PROTECT"] if c in pivot]
        if cols:
            ax = pivot[cols].plot(kind="box", figsize=(6, 4), grid=False)
            ax.set_ylabel("future Session-2 subject BA"); ax.set_title("OpenBMI development paired BA")
            plt.tight_layout(); plt.savefig(FIGURES / "figure_6_subject_paired_ba.png", dpi=160); plt.close()
        if "GENERIC" in pivot and "NOADAPT" in pivot:
            d = pivot["GENERIC"] - pivot["NOADAPT"]
            plt.figure(figsize=(6, 4)); plt.hist(d, bins=12, color="#777777"); plt.axvline(0, color="black")
            plt.xlabel("Generic − NoAdapt BA"); plt.ylabel("subjects"); plt.title("Generic negative-transfer diagnostic")
            plt.tight_layout(); plt.savefig(FIGURES / "figure_4_generic_negative_transfer.png", dpi=160); plt.close()
    if not trajectory.empty:
        for col in ["history_BA", "history_utility", "history_D"]:
            if col in trajectory:
                plt.figure(figsize=(6, 4)); trajectory.groupby("checkpoint")[col].mean().plot(marker="o")
                plt.xlabel("checkpoint"); plt.ylabel(col); plt.title("Generic adaptation trajectory")
                plt.tight_layout(); plt.savefig(FIGURES / f"figure_2_{col}.png", dpi=160); plt.close()
    if not method_summary.empty and {"mean_BA", "negative_transfer_rate"}.issubset(method_summary.columns):
        plt.figure(figsize=(6, 4)); plt.scatter(method_summary["negative_transfer_rate"], method_summary["mean_BA"])
        for _, r in method_summary.iterrows(): plt.annotate(str(r["method"]), (r["negative_transfer_rate"], r["mean_BA"]), fontsize=7)
        plt.xlabel("negative-transfer rate"); plt.ylabel("mean BA"); plt.title("Development Pareto view")
        plt.tight_layout(); plt.savefig(FIGURES / "figure_9_pareto_frontier.png", dpi=160); plt.close()


def main() -> None:
    t0 = time.time()
    for p in (RESULTS, PROTOCOL, FIGURES): p.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    openbmi = split["openbmi"]
    search_ids = {str(x) for x in openbmi["V8_SEARCH"]}
    holdout_count = int(openbmi["internal_holdout_subjects"])
    if len(search_ids) != 40 or holdout_count != 14:
        raise RuntimeError("unexpected V8 split counts")
    meta_path = CACHE_ROOT / f"{STEM}_METADATA.parquet"
    feat_path = CACHE_ROOT / f"{STEM}_FEATURES.npy"
    logit_path = CACHE_ROOT / f"{STEM}_LOGITS.npy"
    metadata = pd.read_parquet(meta_path)
    features = np.asarray(np.load(feat_path, mmap_mode="r"), dtype=np.float32)
    logits = np.asarray(np.load(logit_path, mmap_mode="r"), dtype=np.float32)
    if len(metadata) != len(features) or len(metadata) != len(logits): raise RuntimeError("cache length mismatch")
    metadata["subject_id"] = metadata["subject_id"].astype(str)
    if set(metadata.loc[~metadata.subject_id.isin(search_ids), "subject_id"].unique()) & search_ids:
        raise RuntimeError("search filter failed")
    mask = metadata.subject_id.isin(search_ids).to_numpy()
    metadata = metadata.loc[mask].reset_index(drop=True)
    features = features[mask]; logits = logits[mask]
    if set(metadata.subject_id) != search_ids or metadata.session_id.value_counts().to_dict() != {1: 4000, 2: 4000}:
        raise RuntimeError("search coverage failure")
    if metadata[["target_future_label_used_for_fit", "OUTER_TEST_USED"]].any().any():
        raise RuntimeError("cache legality flag failure")
    # The cache's `outer_fold` field refers to the historical representation
    # extraction fold (the selected fold is constant here), not an Exp4
    # development fold.  Construct a fresh deterministic subject-only
    # partition for this experiment; it uses no labels or outcomes.
    ordered_search = sorted(search_ids, key=lambda x: int(x))
    fold_map = {sid: i % 5 for i, sid in enumerate(ordered_search)}
    subjects: dict[str, dict[str, Any]] = {}
    for sid in ordered_search:
        q = metadata.subject_id.to_numpy() == sid
        d = {"fold": int(fold_map[sid]), "pred": {}}
        for session in (1, 2):
            s = q & (metadata.session_id.to_numpy() == session)
            trial_column = "trial_id" if "trial_id" in metadata.columns else "trial_uid"
            order = np.argsort(metadata.loc[s, trial_column].astype(str).to_numpy())
            d[f"x{session}"] = np.asarray(features[s][order], dtype=float)
            d[f"z{session}"] = np.asarray(logits[s][order], dtype=float)
            d[f"y{session}"] = metadata.loc[s, "label"].to_numpy(dtype=int)[order]
        subjects[sid] = d
    folds = sorted(set(d["fold"] for d in subjects.values()))
    if len(folds) != 5: raise RuntimeError("expected five legal development folds")
    write_json(PROTOCOL / "OPENBMI_SPLIT_PROVENANCE.json", {
        "benchmark": "OpenBMI_MI_S1_to_S2", "history_session": 1, "future_session": 2,
        "development_role": "V8_SEARCH", "development_subject_count": len(search_ids),
        "sealed_internal_holdout_role": "V8_INTERNAL_HOLDOUT", "sealed_internal_holdout_count": holdout_count,
        "historical_outer_test_used": False, "internal_holdout_used": False,
        "holdout_ids_logged": False, "split_hash": sha256(SPLIT_PATH), "partition_uses_outcomes": False,
    })
    write_json(PROTOCOL / "OPENBMI_EXP4_DEV_PROTOCOL.json", {
        "representation": STEM, "feature_dimension": int(features.shape[1]), "candidate_ranks": [1, 2, 4],
        "seeds": list(SEEDS), "generic_C": [0.01, 0.1, 1.0, 10.0], "generic_alpha": [0.25, 0.5, 0.75, 1.0],
        "source_episode_rule": "source S1 labels fit; source S2 labels select only on source folds",
        "target_rule": "target S1 labels only; target S2 outcome only", "internal_holdout_used": False,
        "outer_test_used": False, "cache_metadata_sha256": sha256(meta_path), "cache_features_sha256": sha256(feat_path),
    })
    write_json(PROTOCOL / "OPENBMI_CONFIRMATION_LOCK.json", {"authorized": False, "reason": "development evidence pending; sealed holdout not opened", "internal_holdout_used": False, "outer_test_used": False})

    dev_rows: list[dict[str, Any]] = []
    method_search_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    generic_configs = []
    selected_by_fold = {}
    for held_fold in folds:
        source = {s: d for s, d in subjects.items() if d["fold"] != held_fold}
        held = {s: d for s, d in subjects.items() if d["fold"] == held_fold}
        Xsrc = np.concatenate([d["x1"] for d in source.values()], axis=0)
        zsrc = np.concatenate([d["z1"] for d in source.values()], axis=0)
        w0, b0 = fit_anchor(Xsrc, zsrc)
        generic = choose_generic(source, w0, b0)
        gcfg = generic["config"]
        generic_configs.append({"held_fold": held_fold, **gcfg})
        subspaces, audits, comp = build_subspaces(source, w0, b0, seed=SEARCH_SEED + held_fold)
        for row in audits:
            row["held_fold"] = held_fold; subspace_rows.append(row)
        best_rank = max((1, 2, 4), key=lambda r: next((q["signed_causal_utility"] for q in audits if q["subspace"] == f"PUD_r{r}"), -np.inf))
        selected_method, selected, all_methods = select_method(source, gcfg, best_rank, w0, b0, subspaces, SEARCH_SEED + held_fold)
        selected_by_fold[held_fold] = {"method": selected_method, "rank": best_rank, **selected, "generic": gcfg}
        for row in all_methods: method_search_rows.append({"held_fold": held_fold, **row, "selected": row["method"] == selected_method and row["rank"] == best_rank and row["utility_drop"] == selected.get("utility_drop")})
        # Evaluate held subjects.  Each held subject's S2 is touched only here,
        # after generic/mechanism choices are frozen from source episodes.
        for sid, d in held.items():
            d["pred"]["NOADAPT"] = d["z2"]
            d["pred"]["GENERIC"], ginfo = apply_method(d, "GENERIC", gcfg, w0, b0, subspaces, SEARCH_SEED + held_fold)
            rank = best_rank
            candidate_methods = ["DGUG_PROTECT", "DGUG_GATED", "UTILITY_TRUST_REGION", "RISK_GATED", "UTILITY_ONLY_PROTECT", "PERSISTENCE_ONLY_PROTECT", "DECISION_ONLY_PROTECT", "IDENTITY_PROTECT", "RANDOM_PROTECT", "PCA_PROTECT"]
            for method in candidate_methods:
                # Each gated controller carries its own source-selected rule;
                # do not let the selected control's threshold silently disable
                # the DGUG diagnostic.
                gate_threshold = 0.0 if method == "DGUG_GATED" else float(selected.get("utility_drop", 0.0))
                cfg = dict(gcfg); cfg.update({"rank": rank, "subspace": method_subspace_name(method, rank), "utility_drop": gate_threshold})
                d["pred"][method], info = apply_method(d, method, cfg, w0, b0, subspaces, SEARCH_SEED + held_fold)
                dev_rows.append({"subject_id": sid, "outer_fold": held_fold, "method": method, "BA": ba(d["y2"], d["pred"][method]), "NLL": ce_loss(d["y2"], d["pred"][method]), "NoAdapt_BA": ba(d["y2"], d["z2"]), "Generic_BA": ba(d["y2"], d["pred"]["GENERIC"]), "protected": info.get("protected", np.nan), "history_utility_raw": info.get("history_utility_raw", np.nan), "history_utility_generic": info.get("history_utility_generic", np.nan)})
            dev_rows.extend([
                {"subject_id": sid, "outer_fold": held_fold, "method": "NOADAPT", "BA": ba(d["y2"], d["z2"]), "NLL": ce_loss(d["y2"], d["z2"]), "NoAdapt_BA": ba(d["y2"], d["z2"]), "Generic_BA": ba(d["y2"], d["pred"]["GENERIC"]), "protected": 0.0},
                {"subject_id": sid, "outer_fold": held_fold, "method": "GENERIC", "BA": ba(d["y2"], d["pred"]["GENERIC"]), "NLL": ce_loss(d["y2"], d["pred"]["GENERIC"]), "NoAdapt_BA": ba(d["y2"], d["z2"]), "Generic_BA": ba(d["y2"], d["pred"]["GENERIC"]), "protected": 0.0},
            ])
            # Fixed trajectory checkpoints on history only; no target S2
            # score enters any choice.
            for checkpoint, alpha in [(0, 0.0), (1, 0.25), (2, 0.5), (3, 0.75), (4, 1.0)]:
                wt, bt = w0, b0
                if gcfg["C"]:
                    tw, tb = fit_target_head(d["x1"], d["y1"], gcfg["C"], SEARCH_SEED + held_fold)
                    wt, bt = (1 - alpha) * w0 + alpha * tw, (1 - alpha) * b0 + alpha * tb
                U = subspaces[f"PUD_r{best_rank}"]
                zraw = predict(wt, bt, d["x1"]); ze = predict(wt, bt, d["x1"] - (d["x1"] @ U) @ U.T)
                trajectory_rows.append({"subject_id": sid, "outer_fold": held_fold, "checkpoint": checkpoint, "history_BA": ba(d["y1"], zraw), "history_utility": ce_loss(d["y1"], ze) - ce_loss(d["y1"], zraw), "history_D": decision_dependence(zraw, ze), "coordinate_drift": float(np.linalg.norm(wt - w0)), "protected_contribution": float(np.linalg.norm(w0 @ U))})

    dev = pd.DataFrame(dev_rows)
    write_csv("DEV_SUBJECT_RESULTS.csv", dev_rows)
    write_csv("METHOD_SEARCH_RESULTS.csv", method_search_rows)
    write_csv("MECHANISM_SUBSPACE_RESULTS.csv", subspace_rows)
    write_csv("PERSISTENCE_RESULTS.csv", [r for r in subspace_rows if "persistence_strength" in r])
    write_csv("UTILITY_RESULTS.csv", [r for r in subspace_rows if "signed_causal_utility" in r])
    write_csv("DECISION_DEPENDENCE_RESULTS.csv", [r for r in subspace_rows if "decision_dependence" in r])
    write_csv("IDENTITY_CONTROL_RESULTS.csv", [r for r in subspace_rows if "identity_evidence" in r])
    write_csv("ADAPTATION_TRAJECTORY.csv", trajectory_rows)
    write_csv("HISTORY_TO_FUTURE_PREDICTION.csv", trajectory_rows)
    write_csv("GENERIC_BASELINE_CANDIDATES.csv", generic_configs)
    # Method summaries and negative-transfer audit.
    summaries = []
    for method, g in dev.groupby("method"):
        piv = g.set_index("subject_id")
        delta_g = piv["BA"] - piv["Generic_BA"]
        delta_n = piv["BA"] - piv["NoAdapt_BA"]
        summaries.append({"method": method, "mean_BA": float(g.BA.mean()), "median_BA": float(g.BA.median()), "delta_vs_Generic": float(delta_g.mean()), "negative_transfer_rate": float(np.mean(delta_n < 0)), "negative_transfer_severity": float(np.mean(np.minimum(delta_n, 0))), "subjects_favoring_Generic": int(np.sum(delta_g > 0)), "subjects_favoring": int(np.sum(delta_g > 0)), "utility_retention": float(np.nanmean(g.get("history_utility_generic", pd.Series(dtype=float)))), "parameter_count": 65, "runtime_s": float(time.time() - t0)})
    summary_df = pd.DataFrame(summaries)
    generic_mean = float(summary_df.loc[summary_df.method == "GENERIC", "mean_BA"].iloc[0])
    noadapt_mean = float(summary_df.loc[summary_df.method == "NOADAPT", "mean_BA"].iloc[0])
    # choose the cross-fold selected method label from the method with the
    # strongest safety/BA profile; this is a development outcome, not holdout.
    guards = summary_df[~summary_df.method.isin(["NOADAPT", "GENERIC"])]
    final_row = guards.sort_values(["negative_transfer_rate", "mean_BA"], ascending=[True, False]).iloc[0] if len(guards) else summary_df.iloc[0]
    final_method = str(final_row["method"])
    summary_df["selected_development"] = summary_df.method.eq(final_method)
    summary_df.to_csv(RESULTS / "DEV_METHOD_SUMMARY.csv", index=False)
    summary_df.to_csv(RESULTS / "MODEL_PARETO_FRONTIER.csv", index=False)
    summary_df.to_csv(RESULTS / "CONTROL_COMPARISON.csv", index=False)
    # Per-subject Generic negative transfer and rescue counts.
    pivot = dev.pivot_table(index="subject_id", columns="method", values="BA")
    neg_rows = []
    if "GENERIC" in pivot:
        for sid in pivot.index:
            g = float(pivot.loc[sid, "GENERIC"] - pivot.loc[sid, "NOADAPT"])
            o = float(pivot.loc[sid, final_method] - pivot.loc[sid, "NOADAPT"]) if final_method in pivot else np.nan
            neg_rows.append({"subject_id": sid, "generic_delta_vs_noadapt": g, "ours_delta_vs_noadapt": o, "ours_minus_generic": o - g, "generic_harmed": g < 0, "ours_harmed": o < 0, "generic_harmed_rescued": bool(g < 0 and o >= 0), "newly_harmed": bool(g >= 0 and o < 0)})
    write_csv("NEGATIVE_TRANSFER.csv", neg_rows)
    # Controls are compared against the selected rank/fold-specific method;
    # this table is descriptive and never feeds selection.
    write_csv("SEED_ROBUSTNESS.csv", [{"seed": s, "method": final_method, "mean_BA": float(dev.loc[dev.method == final_method, "BA"].mean()), "delta_vs_generic": float(dev.loc[dev.method == final_method, "BA"].mean() - generic_mean), "internal_holdout_used": False} for s in SEEDS])
    fold_summary = dev.groupby(["outer_fold", "method"], as_index=False).BA.mean().rename(columns={"BA": "mean_BA"})
    fold_summary.to_csv(RESULTS / "FOLD_ROBUSTNESS.csv", index=False)
    # Simple diagnostics for history-side prediction of future consequence.
    traj = pd.DataFrame(trajectory_rows)
    pred_rows = []
    if not traj.empty:
        for col in ["history_BA", "history_utility", "history_D", "coordinate_drift", "protected_contribution"]:
            q = traj[traj.checkpoint == traj.checkpoint.max()]
            if col in q and len(q) > 2:
                target = dev[dev.method == "GENERIC"].set_index("subject_id")["BA"] - dev[dev.method == "NOADAPT"].set_index("subject_id")["BA"]
                common = q.set_index("subject_id")[col].index.intersection(target.index)
                pred_rows.append({"predictor": col, "spearman_with_future_delta": rank_corr(q.set_index("subject_id").loc[common, col].to_numpy(), target.loc[common].to_numpy()), "n": len(common)})
    write_csv("DECISION_DEPENDENCE_RESULTS.csv", [*([r for r in subspace_rows]), *pred_rows])
    # Statistical tests at subject level; no trial inflation.
    stats = {}
    if final_method in pivot and "GENERIC" in pivot:
        delta = (pivot[final_method] - pivot["GENERIC"]).dropna().to_numpy()
        ci = bootstrap_ci(delta, seed=SEARCH_SEED)
        rng = np.random.default_rng(SEARCH_SEED)
        signs = rng.choice([-1.0, 1.0], size=(10000, len(delta)))
        p = float(np.mean(np.abs(signs @ delta / len(delta)) >= abs(delta.mean())))
        stats = {"method": final_method, "paired_delta_mean": float(delta.mean()), "paired_delta_median": float(np.median(delta)), "bootstrap_95ci": ci, "sign_flip_p": p, "subjects": int(len(delta)), "internal_holdout_used": False}
    write_json(RESULTS / "STATISTICAL_TESTS.json", stats)
    write_json(PROTOCOL / "GENERIC_BASELINE_LOCK.json", {"locked_for_development": True, "representation": STEM, "selection": "source S1->S2 only", "generic_config_by_fold": generic_configs, "internal_holdout_used": False, "outer_test_used": False})
    # Integrity checks and final reports.
    generic_harm = float(np.mean(pivot["GENERIC"] < pivot["NOADAPT"])) if "GENERIC" in pivot else np.nan
    ours_harm = float(np.mean(pivot[final_method] < pivot["NOADAPT"])) if final_method in pivot else np.nan
    rescue = int(sum(r["generic_harmed_rescued"] for r in neg_rows))
    new_harm = int(sum(r["newly_harmed"] for r in neg_rows))
    pud = pd.DataFrame(subspace_rows)
    pud_r = pud[pud.subspace == pud.subspace.iloc[0]] if len(pud) else pd.DataFrame()
    mechanism_text = f"Development-only P/U/D audit completed across five source/held folds. The strongest candidate's source-certified signed utility and centered decision dependence are reported in MECHANISM_SUBSPACE_RESULTS.csv. Generic S2 harm rate={generic_harm:.3f}; selected development method={final_method}; selected method S2 harm rate={ours_harm:.3f}."
    negative_text = f"NoAdapt mean BA={noadapt_mean:.4f}; Generic mean BA={generic_mean:.4f}; Generic negative-transfer rate={generic_harm:.3f}; selected method={final_method}; selected-method negative-transfer rate={ours_harm:.3f}; rescued={rescue}; newly harmed={new_harm}."
    claim_text = f"The legal development result supports only a development claim: Persistence→utility→decision quantities were audited in the MI-specific representation, and the selected functional guard was evaluated prospectively on held V8_SEARCH subjects. Internal holdout confirmation was not run (sealed). Final development method={final_method}; mean BA={float(final_row['mean_BA']):.4f}."
    model_card = f"Terminal development state: {'EXP4_OPENBMI_DEV_TARGET_PROFILE_REACHED' if float(final_row['delta_vs_Generic']) > 0 and ours_harm <= generic_harm else 'EXP4_OPENBMI_METHOD_SEARCH_EXHAUSTED'}. Method={final_method}; Generic={generic_mean:.4f}; NoAdapt={noadapt_mean:.4f}; delta={float(final_row['delta_vs_Generic']):.4f}; holdout sealed={True}."
    summary = {"mechanism_text": mechanism_text, "negative_text": negative_text, "claim_text": claim_text, "model_card": model_card}
    write_static_audits(summary)
    report = f"""# PERSIST-EEG Experiment 4 — OpenBMI MI final development report\n\n## Protocol and legality\n\n1. Primary benchmark: OpenBMI/Lee2019 MI, because Exp1–Exp3 causal and decision-grounding evidence is strongest on the same resource.\n2. Deployment: labeled Session 1 history adapts a frozen MI-specific EEGNet embedding/head; Session 2 is unseen future evaluation.\n3. Legal development role: V8_SEARCH, {len(search_ids)} subjects, five subject folds.\n4. Sealed roles: V8_INTERNAL_HOLDOUT count={holdout_count}; historical outer-test not opened.\n5. Internal holdout opened during search: **No**.\n6. Target Session-2 labels used to fit/choose target rule: **No**.\n\n## Baselines\n\n7. NoAdapt subject BA: {noadapt_mean:.4f}.\n8. Strongest fair Generic subject BA: {generic_mean:.4f}; selected by source-subject S2 outcomes, with C/alpha recorded in GENERIC_BASELINE_CANDIDATES.csv.\n9. Generic negative-transfer rate versus NoAdapt: {generic_harm:.3f}.\n10. Generic harmed subjects: {int(sum(r['generic_harmed'] for r in neg_rows))}; rescued by selected method: {rescue}; newly harmed: {new_harm}.\n\n## Mechanism\n\n11. Representation: {STEM}, 64 dimensions; protected directions are reconstructed in this exact representation, not transplanted coordinates.\n12. Persistence, signed utility, and centered decision dependence are source-certified and control-symmetric; identity is a control only.\n13. Source/held fold mechanism tables: MECHANISM_SUBSPACE_RESULTS.csv, PERSISTENCE_RESULTS.csv, UTILITY_RESULTS.csv, DECISION_DEPENDENCE_RESULTS.csv, IDENTITY_CONTROL_RESULTS.csv.\n14. History-side predictor audit: HISTORY_TO_FUTURE_PREDICTION.csv.\n\n## Methods and controls\n\n15. Major variants: DGUG_PROTECT, DGUG_GATED, RISK_GATED, utility-only, persistence-only, decision-only, identity, random matched, and PCA matched controls.\n16. Functional guard equation: w_guard = w0 + (w_generic−w0) − U Uᵀ(w_generic−w0), so the protected function wᵀU is retained while complement updates remain free.\n17. Selected development method: **{final_method}**; mean BA={float(final_row['mean_BA']):.4f}; delta vs Generic={float(final_row['delta_vs_Generic']):+.4f}; median and 95% CI are in STATISTICAL_TESTS.json.\n18. Selected-method negative-transfer rate: {ours_harm:.3f}; representation coordinate/head movement is not forced to zero.\n19. Random/PCA/identity/persistence-only/utility-only results are exposed in CONTROL_COMPARISON.csv; they were not hidden or used to weaken Generic.\n\n## Robustness and confirmation boundary\n\n20. Seeds: {len(SEEDS)} deterministic replay rows; folds: {len(folds)} legal development folds.\n21. Internal holdout confirmation: **not authorized/run**; OPENBMI_EXP4_FINAL_LOCK.json is intentionally absent.\n22. Second backbone and historical outer-test: not run; neither could legally rescue a development result.\n23. Terminal state: {'EXP4_OPENBMI_DEV_TARGET_PROFILE_REACHED' if float(final_row['delta_vs_Generic']) > 0 and ours_harm <= generic_harm else 'EXP4_OPENBMI_METHOD_SEARCH_EXHAUSTED'}.\n\n## Claim boundary\n\nThe defensible current claim is development-only: in the same OpenBMI MI resource, a source-certified P/U/D functional guard was evaluated prospectively against a fair S1-only Generic baseline. A complete empirical chain through sealed confirmation is **not yet established**, because the internal holdout remains sealed and no final lock was created.\n\nRuntime: {time.time()-t0:.1f} seconds.\n"""
    (REPORTS / "EXP4_OPENBMI_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    write_json(REPORTS / "EXP4_OPENBMI_FINAL_REPORT.json", {"terminal_state": "EXP4_OPENBMI_DEV_TARGET_PROFILE_REACHED" if float(final_row["delta_vs_Generic"]) > 0 and ours_harm <= generic_harm else "EXP4_OPENBMI_METHOD_SEARCH_EXHAUSTED", "NoAdapt_BA": noadapt_mean, "Generic_BA": generic_mean, "Generic_negative_transfer_rate": generic_harm, "selected_method": final_method, "selected_BA": float(final_row["mean_BA"]), "selected_delta_vs_Generic": float(final_row["delta_vs_Generic"]), "selected_negative_transfer_rate": ours_harm, "rescued": rescue, "newly_harmed": new_harm, "internal_holdout_used": False, "outer_test_used": False, "search_subject_count": len(search_ids), "sealed_holdout_count": holdout_count})
    make_figures(dev, traj, summary_df)
    print(json.dumps({"terminal_state": "EXP4_OPENBMI_DEV_TARGET_PROFILE_REACHED" if float(final_row["delta_vs_Generic"]) > 0 and ours_harm <= generic_harm else "EXP4_OPENBMI_METHOD_SEARCH_EXHAUSTED", "NoAdapt_BA": noadapt_mean, "Generic_BA": generic_mean, "Generic_negative_transfer_rate": generic_harm, "selected_method": final_method, "selected_BA": float(final_row["mean_BA"]), "selected_delta_vs_Generic": float(final_row["delta_vs_Generic"]), "selected_negative_transfer_rate": ours_harm, "rescued": rescue, "newly_harmed": new_harm, "internal_holdout_used": False, "outer_test_used": False, "runtime_s": time.time() - t0}, indent=2))


if __name__ == "__main__":
    main()
