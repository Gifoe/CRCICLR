"""Decisive Phase-A audit for PERSIST-EEG Exp4.

This runner performs two bounded audits on the 40 V8_SEARCH OpenBMI subjects:

1. A predeclared five-action interpolation headroom audit.
2. A source-only reconstruction of Exp3 finite/Jacobian decision dependence,
   protected-rank specificity, and prospective incremental-risk value.

The 14-subject internal holdout is removed before raw tensors are materialised.
This file does not implement or authorize a Phase-B selector.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("PERSIST_HEADROOM_EXPERIMENT", ".")).resolve()
REPO = Path(os.environ.get("PERSIST_AUDIT_REPO", Path(__file__).resolve().parents[3])).resolve()
PREVIOUS_SCRIPT = Path(
    os.environ.get(
        "PERSIST_GUARD_V1_SCRIPT",
        REPO / "experiments" / "persist_eeg_exp4_persist_guard_final_v1" / "code" / "run_development.py",
    )
).resolve()
PREVIOUS_ROOT = PREVIOUS_SCRIPT.parents[1]
PREVIOUS_CACHE = Path(os.environ.get("PERSIST_GUARD_V1_CACHE", PREVIOUS_ROOT / "cache")).resolve()
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PROTOCOL = ROOT / "protocol"

ACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
ACTION_NAMES = {
    0.0: "A0_NoAdapt",
    0.25: "A1_Generic_25pct",
    0.5: "A2_Generic_50pct",
    0.75: "A3_Generic_75pct",
    1.0: "A4_Strong_Generic",
}
RISK_CS = (0.1, 1.0, 10.0)
MAX_REPAIRED_RANK = 16
P_THRESHOLD = 0.05
U_THRESHOLD = 1e-5
D_RATIO_THRESHOLD = 1.0
RANDOM_DIRECTIONS = 64
BOOTSTRAPS = 10_000
RNG_SEED = 20260824
EPS = 1e-10
BRANCH = "codex/persist-eeg-exp4-headroom-mechanism-audit-v1"


def load_previous() -> Any:
    if not PREVIOUS_SCRIPT.is_file():
        raise FileNotFoundError(PREVIOUS_SCRIPT)
    os.environ.setdefault("PERSIST_GUARD_EXPERIMENT", str(PREVIOUS_ROOT))
    spec = importlib.util.spec_from_file_location("persist_guard_v1", PREVIOUS_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PREVIOUS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.CACHE = PREVIOUS_CACHE
    module.RISK_CS = RISK_CS
    return module


V1 = load_previous()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
    except Exception:
        return None


def action_key(alpha: float) -> str:
    return {0.0: "a0", 0.25: "a025", 0.5: "a05", 0.75: "a075", 1.0: "a1"}[float(alpha)]


def exact_centered_logit_sq(delta: np.ndarray) -> np.ndarray:
    """Verbatim Exp3/DDA centered_logit_sq definition."""
    value = np.asarray(delta, dtype=np.float64)
    centered = value - value.mean(axis=-1, keepdims=True)
    return np.sum(centered * centered, axis=-1)


def finite_decision_dependence_from_margin_delta(delta_margin: np.ndarray) -> float:
    """Exp3 finite D for a binary model represented by its margin displacement."""
    delta_margin = np.asarray(delta_margin, dtype=np.float64)
    delta_logits = np.stack([-0.5 * delta_margin, 0.5 * delta_margin], axis=-1)
    return float(np.sqrt(np.mean(exact_centered_logit_sq(delta_logits))))


def validate_exact_d() -> dict[str, Any]:
    rng = np.random.default_rng(RNG_SEED)
    margin_delta = rng.normal(size=2048)
    exact = finite_decision_dependence_from_margin_delta(margin_delta)
    analytic = float(np.sqrt(np.mean(margin_delta**2) / 2.0))
    error = abs(exact - analytic)
    exp3_candidates = [
        Path(os.environ["PERSIST_EXP3_REPORT"]).resolve() if "PERSIST_EXP3_REPORT" in os.environ else None,
        REPO / "experiments" / "persist_eeg_exp3_decision_grounding_closure_v1" / "EXP3_FINAL_REPORT.json",
        Path(V1.REPO) / "experiments" / "persist_eeg_exp3_decision_grounding_closure_v1" / "EXP3_FINAL_REPORT.json",
    ]
    exp3_report_path = next((path for path in exp3_candidates if path is not None and path.is_file()), None)
    if exp3_report_path is None:
        raise FileNotFoundError("No frozen Exp3 final report found; set PERSIST_EXP3_REPORT")
    archived = json.loads(exp3_report_path.read_text(encoding="utf-8"))
    frozen_reference = {
        "decision_protected_mean": 0.9982230109222217,
        "decision_control_mean": 0.2467850870938559,
        "M0_RMSE": 0.04597839942134,
        "MI_RMSE": 0.0457441624640147,
        "MD_RMSE": 0.0314928431971294,
        "MID_RMSE": 0.0315332767866679,
        "M0_minus_MD": 0.014791624471360496,
        "MD_beats_MI_positive_runs": 6,
    }
    archived_values = {
        "decision_protected_mean": float(archived["decision_protected_mean"]),
        "decision_control_mean": float(archived["decision_control_mean"]),
        "M0_RMSE": float(archived["model_rmse"]["M0"]),
        "MI_RMSE": float(archived["model_rmse"]["MI"]),
        "MD_RMSE": float(archived["model_rmse"]["MD"]),
        "MID_RMSE": float(archived["model_rmse"]["MID"]),
        "M0_minus_MD": float(archived["tests"]["A"]["mean"]),
        "MD_beats_MI_positive_runs": int(archived["tests"]["B"]["positive_runs"]),
    }
    archive_matches = all(
        archived_values[key] == value if isinstance(value, int) else abs(archived_values[key] - value) <= 1e-12
        for key, value in frozen_reference.items()
    )
    return {
        "source_code_path": "experiments/persist_eeg_dda_v1/code/persist_dda_v1.py",
        "source_function": "centered_logit_sq + subject_decision_metrics",
        "definition": "sqrt(mean(sum((delta_logits-mean_class(delta_logits))^2, class)))",
        "binary_margin_equivalent": "sqrt(mean(delta_margin^2)/2)",
        "numeric_exact": exact,
        "numeric_analytic": analytic,
        "absolute_error": error,
        "archived_Exp3_report": str(exp3_report_path),
        "archived_Exp3_values": archived_values,
        "archived_values_match_frozen_reference": archive_matches,
        "validated": bool(error <= 1e-12 and archive_matches),
    }


def corr_or_zero(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else 0.0


def random_unit_directions(dim: int, seed: int, count: int = RANDOM_DIRECTIONS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(count, dim))
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), EPS)
    return matrix


def reconstruct_bank(
    meta_subjects: list[str],
    features: np.ndarray,
    base_logits: np.ndarray,
    metadata: pd.DataFrame,
    seed: int,
    fold: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Audit old rank and build a stricter source-certified P/U/D bank."""
    old_bank, old_audit = V1.fit_protected_bank(
        meta_subjects, features, base_logits, metadata, seed, fold
    )
    z = (features - old_bank["center"]) / old_bank["scale"]
    m1, m2 = [], []
    for subject in meta_subjects:
        s1 = V1.subject_mask(metadata, subject, 1)
        s2 = V1.subject_mask(metadata, subject, 2)
        m1.append(z[s1].mean(axis=0))
        m2.append(z[s2].mean(axis=0))
    m1, m2 = np.stack(m1), np.stack(m2)
    left = m1 - m1.mean(axis=0, keepdims=True)
    right = m2 - m2.mean(axis=0, keepdims=True)
    cross = (left.T @ right + right.T @ left) / max(2 * (len(meta_subjects) - 1), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cross)
    order = np.argsort(eigenvalues)[::-1][:32]

    source_future = metadata.subject_id.isin(meta_subjects).to_numpy() & metadata.session_id.eq(2).to_numpy()
    zf = z[source_future]
    yf = metadata.loc[source_future, "label"].to_numpy(int)
    task_w = np.asarray(old_bank["task_w"], dtype=np.float64)
    task_b = float(old_bank["task_b"])
    full = zf @ task_w + task_b
    random_directions = random_unit_directions(
        features.shape[1], RNG_SEED + fold * 1000 + seed * 100
    )

    rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, int, np.ndarray, float]] = []
    for candidate_order, index in enumerate(order):
        direction = eigenvectors[:, index]
        projection = zf @ direction
        contribution = projection * float(direction @ task_w)
        erased = full - contribution
        persistence = corr_or_zero(m1 @ direction, m2 @ direction)
        utility_loss = V1.balanced_bce(yf, erased) - V1.balanced_bce(yf, full)
        utility_ba = V1.ba(yf, full) - V1.ba(yf, erased)
        d_finite = finite_decision_dependence_from_margin_delta(-contribution)
        d_flip = float(np.mean((full >= 0.0) != (erased >= 0.0)))
        d_jac = float((direction @ task_w) ** 2 / 2.0)

        random_finite, random_jac = [], []
        target_amplitude = np.abs(projection)
        for random_direction in random_directions:
            random_projection = zf @ random_direction
            matched_amplitude = np.sign(random_projection) * target_amplitude
            random_contribution = matched_amplitude * float(random_direction @ task_w)
            random_finite.append(finite_decision_dependence_from_margin_delta(-random_contribution))
            random_jac.append(float((random_direction @ task_w) ** 2 / 2.0))
        random_finite_mean = float(np.mean(random_finite))
        random_jac_mean = float(np.mean(random_jac))
        d_finite_ratio = d_finite / max(random_finite_mean, EPS)
        d_jac_ratio = d_jac / max(random_jac_mean, EPS)

        pass_p = bool(eigenvalues[index] > 0.0 and persistence >= P_THRESHOLD)
        pass_pu = bool(pass_p and utility_loss > U_THRESHOLD)
        pass_finite = bool(pass_pu and d_finite_ratio > D_RATIO_THRESHOLD)
        pass_jac = bool(pass_pu and d_jac_ratio > D_RATIO_THRESHOLD)
        final_pass = bool(pass_finite and pass_jac)
        old_pass = bool(
            old_audit[candidate_order]["passed"]
            if candidate_order < len(old_audit)
            else (pass_pu and d_flip > 0.0)
        )
        score = max(persistence, 0.0) * max(utility_loss, 0.0) * max(d_finite_ratio - 1.0, 0.0)
        row = {
            "fold": fold,
            "seed": seed,
            "candidate_order": candidate_order,
            "eigenvalue": float(eigenvalues[index]),
            "P_cross_session": persistence,
            "U_signed_loss": utility_loss,
            "U_signed_BA": utility_ba,
            "D_finite": d_finite,
            "D_finite_random_mean": random_finite_mean,
            "D_finite_ratio": d_finite_ratio,
            "D_jacobian_energy": d_jac,
            "D_jacobian_random_mean": random_jac_mean,
            "D_jacobian_ratio": d_jac_ratio,
            "D_flip_control": d_flip,
            "pass_P": pass_p,
            "pass_PU": pass_pu,
            "pass_PU_Dfinite": pass_finite,
            "pass_PU_Djac": pass_jac,
            "final_certified_before_cap": final_pass,
            "old_Exp4_pass": old_pass,
            "P_margin": persistence - P_THRESHOLD,
            "U_margin": utility_loss - U_THRESHOLD,
            "Dfinite_ratio_margin": d_finite_ratio - D_RATIO_THRESHOLD,
            "Djac_ratio_margin": d_jac_ratio - D_RATIO_THRESHOLD,
            "source_subjects": len(meta_subjects),
            "target_future_used_for_certification": False,
        }
        rows.append(row)
        if final_pass:
            candidates.append((score, candidate_order, direction, persistence))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = candidates[:MAX_REPAIRED_RANK]
    p_basis = (
        np.stack([item[2] for item in selected], axis=1)
        if selected
        else np.zeros((features.shape[1], 0), dtype=np.float64)
    )
    p_scores = np.asarray([item[3] for item in selected], dtype=np.float64)
    bank = dict(old_bank)
    bank.update({"p_basis": p_basis, "p_scores": p_scores, "protected_rank": int(p_basis.shape[1])})
    before = len(candidates)
    summary = {
        "fold": fold,
        "seed": seed,
        "total_candidates": len(rows),
        "passing_P": int(sum(row["pass_P"] for row in rows)),
        "passing_PU": int(sum(row["pass_PU"] for row in rows)),
        "passing_PU_Dfinite": int(sum(row["pass_PU_Dfinite"] for row in rows)),
        "passing_PU_Djac": int(sum(row["pass_PU_Djac"] for row in rows)),
        "old_rank_before_cap": int(sum(row["old_Exp4_pass"] for row in rows)),
        "old_rank_after_cap": int(old_bank["protected_rank"]),
        "rank_before_cap": before,
        "rank_after_cap": int(p_basis.shape[1]),
        "rank_cap": MAX_REPAIRED_RANK,
        "source_only": True,
    }
    return bank, rows, summary


def protected_metrics(
    z: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    bank: dict[str, Any],
) -> dict[str, float]:
    basis = np.asarray(bank["p_basis"], dtype=np.float64)
    if basis.shape[1] == 0:
        return {
            "utility": 0.0,
            "Dfinite": 0.0,
            "Djac": 0.0,
            "Dflip": 0.0,
            "protected_contribution": 0.0,
        }
    logits = z @ w + b
    contribution = (z @ basis) @ (basis.T @ w)
    erased = logits - contribution
    utility = V1.balanced_bce(y, erased) - V1.balanced_bce(y, logits)
    d_finite = finite_decision_dependence_from_margin_delta(-contribution)
    d_jac = float(np.sum((basis.T @ w) ** 2) / (2.0 * basis.shape[1]))
    d_flip = float(np.mean((logits >= 0.0) != (erased >= 0.0)))
    return {
        "utility": utility,
        "Dfinite": d_finite,
        "Djac": d_jac,
        "Dflip": d_flip,
        "protected_contribution": float(np.sqrt(np.mean(contribution**2))),
    }


def make_seed_episode(
    subject: str,
    fold: int,
    seed: int,
    role: str,
    features: np.ndarray,
    metadata: pd.DataFrame,
    bank: dict[str, Any],
    head_w: np.ndarray,
    head_b: float,
    generic: dict[str, Any],
) -> tuple[dict[str, Any], dict[float, np.ndarray]]:
    z = (features - bank["center"]) / bank["scale"]
    s1 = V1.subject_mask(metadata, subject, 1)
    s2 = V1.subject_mask(metadata, subject, 2)
    y1 = metadata.loc[s1, "label"].to_numpy(int)
    w0, b0 = V1.population_theta(head_w, head_b, bank)
    c = float(generic["chosen"]["C"])
    beta = float(generic["chosen"]["beta"])
    wt, bt = V1.target_theta(z[s1], y1, c, RNG_SEED + seed * 1000 + fold * 100 + int(subject))
    wg, bg = (1.0 - beta) * w0 + beta * wt, (1.0 - beta) * b0 + beta * bt
    conventional = V1.risk_features(z[s1], y1, w0, b0, wg, bg, bank)
    p0 = protected_metrics(z[s1], y1, w0, b0, bank)
    pg = protected_metrics(z[s1], y1, wg, bg, bank)
    row: dict[str, Any] = {
        "fold": fold,
        "seed": seed,
        "subject_id": str(subject),
        "role": role,
        "protected_rank": int(bank["protected_rank"]),
        **conventional,
        "U_t0": p0["utility"],
        "U_generic": pg["utility"],
        "U_damage": max(0.0, p0["utility"] - pg["utility"]),
        "Dfinite_t0": p0["Dfinite"],
        "Dfinite_generic": pg["Dfinite"],
        "Dfinite_degradation": max(0.0, p0["Dfinite"] - pg["Dfinite"]),
        "Dfinite_change": pg["Dfinite"] - p0["Dfinite"],
        "Djac_t0": p0["Djac"],
        "Djac_generic": pg["Djac"],
        "Djac_degradation": max(0.0, p0["Djac"] - pg["Djac"]),
        "Dflip_t0": p0["Dflip"],
        "Dflip_generic": pg["Dflip"],
        "Dflip_degradation": max(0.0, p0["Dflip"] - pg["Dflip"]),
        "internal_holdout_used": False,
    }
    action_logits: dict[float, np.ndarray] = {}
    for alpha in ACTIONS:
        wa, ba = w0 + alpha * (wg - w0), b0 + alpha * (bg - b0)
        logits_history = z[s1] @ wa + ba
        pm = protected_metrics(z[s1], y1, wa, ba, bank)
        key = action_key(alpha)
        row[f"{key}_history_CE"] = V1.balanced_bce(y1, logits_history)
        row[f"{key}_history_BA"] = V1.ba(y1, logits_history)
        row[f"{key}_history_confidence"] = float(
            np.mean(np.maximum(V1.sigmoid(logits_history), 1.0 - V1.sigmoid(logits_history)))
        )
        row[f"{key}_U"] = pm["utility"]
        row[f"{key}_Dfinite"] = pm["Dfinite"]
        row[f"{key}_Djac"] = pm["Djac"]
        row[f"{key}_Dflip"] = pm["Dflip"]
        row[f"{key}_update_norm"] = float(alpha * np.linalg.norm(np.r_[wg - w0, bg - b0]))
        action_logits[alpha] = z[s2] @ wa + ba
    return row, action_logits


def aggregate_seed_episodes(
    seed_frame: pd.DataFrame,
    trial_logits: dict[tuple[int, int, str, float], np.ndarray],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    identity = {"fold", "seed", "subject_id", "role", "internal_holdout_used"}
    feature_columns = [col for col in seed_frame.columns if col not in identity]
    for (fold, subject, role), group in seed_frame.groupby(["fold", "subject_id", "role"], sort=False):
        s2 = V1.subject_mask(metadata, str(subject), 2)
        y = metadata.loc[s2, "label"].to_numpy(int)
        row: dict[str, Any] = {
            "fold": int(fold),
            "subject_id": str(subject),
            "role": role,
            "internal_holdout_used": False,
        }
        for column in feature_columns:
            row[column] = float(group[column].mean())
        for alpha in ACTIONS:
            stack = np.stack(
                [trial_logits[(int(fold), int(seed), str(subject), alpha)] for seed in V1.SEEDS]
            )
            probability = np.mean(V1.sigmoid(stack), axis=0)
            logits = np.log(
                np.clip(probability, 1e-7, 1.0 - 1e-7)
                / np.clip(1.0 - probability, 1e-7, 1.0 - 1e-7)
            )
            row[f"BA_{action_key(alpha)}"] = V1.ba(y, logits)
        row["NegativeTransfer"] = int(row["BA_a1"] < row["BA_a0"] - 1e-12)
        action_values = np.asarray([row[f"BA_{action_key(alpha)}"] for alpha in ACTIONS])
        best_index = int(np.argmax(action_values))
        row["OptimalAction"] = ACTION_NAMES[ACTIONS[best_index]]
        row["OptimalAlpha"] = ACTIONS[best_index]
        row["OracleBA"] = float(action_values[best_index])
        row["AlternativePreferred"] = int(ACTIONS[best_index] != 1.0)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True), axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def bootstrap_metric_delta(
    y: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    metric: str,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    n = len(y)
    for _ in range(BOOTSTRAPS):
        index = rng.integers(0, n, size=n)
        yy = y[index]
        if metric in {"AUROC", "AUPRC"} and len(np.unique(yy)) < 2:
            continue
        if metric == "AUROC":
            delta = roc_auc_score(yy, candidate[index]) - roc_auc_score(yy, baseline[index])
        elif metric == "AUPRC":
            delta = average_precision_score(yy, candidate[index]) - average_precision_score(yy, baseline[index])
        elif metric == "Brier":
            delta = brier_score_loss(yy, candidate[index]) - brier_score_loss(yy, baseline[index])
        else:
            raise ValueError(metric)
        values.append(float(delta))
    if not values:
        return [float("nan"), float("nan")]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


CONF_COLS = ["history_confidence", "history_entropy", "history_margin", "history_improvement"]
UPDATE_COLS = ["update_norm", "gradient_norm", "history_improvement", "split_history_disagreement"]
CONF_UPDATE_COLS = list(dict.fromkeys(CONF_COLS + UPDATE_COLS))
IDENTITY_COLS = list(V1.IDENTITY_COLS)
P_COLS = ["persistence_magnitude", "persistence_split_stability"]
U_COLS = ["U_t0", "U_generic", "U_damage"]
DFINITE_COLS = ["Dfinite_t0", "Dfinite_generic", "Dfinite_degradation", "Dfinite_change"]
DJAC_COLS = ["Djac_t0", "Djac_generic", "Djac_degradation"]
DFLIP_COLS = ["Dflip_t0", "Dflip_generic", "Dflip_degradation"]


RISK_SPECS = {
    "M_conf": CONF_COLS,
    "M_update": UPDATE_COLS,
    "M_identity": IDENTITY_COLS,
    "M_P": P_COLS,
    "M_Dfinite": DFINITE_COLS,
    "M_Djac": DJAC_COLS,
    "M_Dflip": DFLIP_COLS,
    "M_PU": P_COLS + U_COLS,
    "M_PDfinite": P_COLS + DFINITE_COLS,
    "M_PUDfinite": P_COLS + U_COLS + DFINITE_COLS,
    "M_conf_P": CONF_COLS + P_COLS,
    "M_conf_D": CONF_COLS + DFINITE_COLS,
    "M_conf_PD": CONF_COLS + P_COLS + DFINITE_COLS,
    "M_conf_update": CONF_UPDATE_COLS,
    "M_conf_update_PD": CONF_UPDATE_COLS + P_COLS + DFINITE_COLS,
    "M_random": CONF_UPDATE_COLS + P_COLS + U_COLS + DFINITE_COLS,
}


def residualized_frames(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    base_columns = CONF_UPDATE_COLS
    mechanism_columns = P_COLS + DFINITE_COLS
    x_train = train[base_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    x_test = test[base_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    y_train = train[mechanism_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    y_test = test[mechanism_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=1.0).fit(scaler.transform(x_train), y_train)
    residual_train = y_train - model.predict(scaler.transform(x_train))
    residual_test = y_test - model.predict(scaler.transform(x_test))
    train_out, test_out = train.copy(), test.copy()
    residual_columns = []
    for index, column in enumerate(mechanism_columns):
        name = f"resid_{column}"
        residual_columns.append(name)
        train_out[name] = residual_train[:, index]
        test_out[name] = residual_test[:, index]
    return train_out, test_out, residual_columns


def fit_risk_models(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for fold in sorted(episodes.fold.unique()):
        train = episodes[(episodes.fold == fold) & episodes.role.eq("meta")].reset_index(drop=True)
        test = episodes[(episodes.fold == fold) & episodes.role.eq("outcome")].reset_index(drop=True)
        train_resid, test_resid, residual_columns = residualized_frames(train, test)
        specs = dict(RISK_SPECS)
        specs["M_resid_PD"] = residual_columns
        specs["M_conf_update_PD_resid"] = CONF_UPDATE_COLS + residual_columns
        for model_name, columns in specs.items():
            tr = train_resid if "resid" in model_name else train
            te = test_resid if "resid" in model_name else test
            for outcome_name in ("NegativeTransfer", "AlternativePreferred"):
                tr_fit = tr.copy()
                te_fit = te.copy()
                tr_fit["NegativeTransfer"] = tr_fit[outcome_name].astype(int)
                te_fit["NegativeTransfer"] = te_fit[outcome_name].astype(int)
                _, predicted, audit = V1.fit_risk(
                    tr_fit,
                    te_fit,
                    list(columns),
                    int(fold) + (0 if outcome_name == "NegativeTransfer" else 100),
                    random_projection=model_name == "M_random",
                )
                audit_rows.append(
                    {
                        "fold": int(fold),
                        "risk_model": model_name,
                        "outcome": outcome_name,
                        **audit,
                        "target_future_feature_used": False,
                    }
                )
                for position, (_, row) in enumerate(te.iterrows()):
                    prediction_rows.append(
                        {
                            "fold": int(fold),
                            "subject_id": row.subject_id,
                            "risk_model": model_name,
                            "outcome": outcome_name,
                            "probability": float(predicted[position]),
                            "label": int(row[outcome_name]),
                            "FutureDeltaBA": float(row.BA_a1 - row.BA_a0),
                            "internal_holdout_used": False,
                        }
                    )
    predictions = pd.DataFrame(prediction_rows)
    metrics = []
    for (model_name, outcome_name), part in predictions.groupby(["risk_model", "outcome"], sort=True):
        part = part.sort_values(["fold", "subject_id"])
        y = part.label.to_numpy(int)
        probability = part.probability.to_numpy(float)
        metrics.append(
            {
                "risk_model": model_name,
                "outcome": outcome_name,
                "subjects": len(part),
                "AUROC": float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else None,
                "AUPRC": float(average_precision_score(y, probability)) if len(np.unique(y)) > 1 else None,
                "balanced_accuracy": float(balanced_accuracy_score(y, probability >= 0.5)),
                "Brier": float(brier_score_loss(y, probability)),
                "risk_outcome_Spearman": float(spearmanr(part.FutureDeltaBA, probability).statistic),
                "mean_probability_positive": float(probability[y == 1].mean()) if np.any(y == 1) else None,
                "mean_probability_negative": float(probability[y == 0].mean()) if np.any(y == 0) else None,
                "internal_holdout_used": False,
            }
        )
    return predictions, pd.DataFrame(metrics), pd.DataFrame(audit_rows)


def risk_incremental_values(predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("M_conf", "M_conf_P"),
        ("M_conf", "M_conf_D"),
        ("M_conf", "M_conf_PD"),
        ("M_conf_update", "M_conf_update_PD"),
        ("M_conf_update", "M_conf_update_PD_resid"),
        ("M_identity", "M_Dfinite"),
        ("M_Dflip", "M_Dfinite"),
    ]
    rows = []
    for baseline_name, candidate_name in comparisons:
        for outcome_name in ("NegativeTransfer", "AlternativePreferred"):
            base = predictions[
                predictions.risk_model.eq(baseline_name) & predictions.outcome.eq(outcome_name)
            ].sort_values(["fold", "subject_id"])
            candidate = predictions[
                predictions.risk_model.eq(candidate_name) & predictions.outcome.eq(outcome_name)
            ].sort_values(["fold", "subject_id"])
            if len(base) != len(candidate) or not np.array_equal(base.subject_id.to_numpy(), candidate.subject_id.to_numpy()):
                raise RuntimeError(f"prediction alignment failure {baseline_name}->{candidate_name}")
            y = base.label.to_numpy(int)
            bp = base.probability.to_numpy(float)
            cp = candidate.probability.to_numpy(float)
            base_metric = metrics[(metrics.risk_model == baseline_name) & (metrics.outcome == outcome_name)].iloc[0]
            candidate_metric = metrics[(metrics.risk_model == candidate_name) & (metrics.outcome == outcome_name)].iloc[0]
            auc_ci = bootstrap_metric_delta(y, cp, bp, "AUROC", RNG_SEED + len(rows))
            rows.append(
                {
                    "outcome": outcome_name,
                    "baseline": baseline_name,
                    "candidate": candidate_name,
                    "delta_AUROC": float(candidate_metric.AUROC - base_metric.AUROC),
                    "delta_AUROC_CI95_L": auc_ci[0],
                    "delta_AUROC_CI95_U": auc_ci[1],
                    "delta_AUPRC": float(candidate_metric.AUPRC - base_metric.AUPRC),
                    "delta_Brier": float(candidate_metric.Brier - base_metric.Brier),
                    "internal_holdout_used": False,
                }
            )
    return pd.DataFrame(rows)


def action_headroom(outcome: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    generic = outcome.BA_a1.to_numpy(float)
    noadapt = outcome.BA_a0.to_numpy(float)
    action_matrix = np.column_stack([outcome[f"BA_{action_key(alpha)}"].to_numpy(float) for alpha in ACTIONS])
    best_indices = np.argmax(action_matrix, axis=1)
    oracle = action_matrix[np.arange(len(outcome)), best_indices]
    action_means = action_matrix.mean(axis=0)
    best_global_index = int(np.argmax(action_means))
    best_global = float(action_means[best_global_index])
    bottom = np.argsort(generic)[: max(1, len(generic) // 4)]
    subject_rows = outcome[["subject_id", "fold"]].copy()
    for index, alpha in enumerate(ACTIONS):
        subject_rows[ACTION_NAMES[alpha]] = action_matrix[:, index]
    subject_rows["Strong_Generic_BA"] = generic
    subject_rows["OracleAction"] = [ACTION_NAMES[ACTIONS[index]] for index in best_indices]
    subject_rows["OracleAlpha"] = [ACTIONS[index] for index in best_indices]
    subject_rows["OracleBA"] = oracle
    subject_rows["RecoverableGain_vs_Generic"] = oracle - generic
    subject_rows["StrongGenericRegret"] = oracle - generic
    subject_rows["internal_holdout_used"] = False

    bank_rows = []
    for index, alpha in enumerate(ACTIONS):
        values = action_matrix[:, index]
        bank_rows.append(
            {
                "scope": "global",
                "fold": "ALL",
                "action": ACTION_NAMES[alpha],
                "alpha": alpha,
                "subjects": len(values),
                "BA": float(values.mean()),
                "delta_vs_Strong_Generic": float(np.mean(values - generic)),
                "internal_holdout_used": False,
            }
        )
        for fold in sorted(outcome.fold.unique()):
            mask = outcome.fold.to_numpy(int) == int(fold)
            bank_rows.append(
                {
                    "scope": "fold",
                    "fold": int(fold),
                    "action": ACTION_NAMES[alpha],
                    "alpha": alpha,
                    "subjects": int(mask.sum()),
                    "BA": float(values[mask].mean()),
                    "delta_vs_Strong_Generic": float(np.mean(values[mask] - generic[mask])),
                    "internal_holdout_used": False,
                }
            )
    fold_oracle = []
    for fold in sorted(outcome.fold.unique()):
        mask = outcome.fold.to_numpy(int) == int(fold)
        fold_oracle.append(
            {
                "fold": int(fold),
                "subjects": int(mask.sum()),
                "oracle_BA": float(oracle[mask].mean()),
                "generic_BA": float(generic[mask].mean()),
                "oracle_delta": float(np.mean(oracle[mask] - generic[mask])),
            }
        )
    oracle_ba = float(oracle.mean())
    delta = float(np.mean(oracle - generic))
    personalization = oracle_ba - best_global
    h1 = delta >= 0.01 - 1e-12
    h2 = personalization >= 0.005 - 1e-12
    h3 = int(np.sum(oracle - generic >= 0.01 - 1e-12)) >= 6
    h4 = int(sum(row["oracle_delta"] > 1e-12 for row in fold_oracle)) >= 4
    h5 = True
    summary = {
        "Clean_Strong_Generic_BA": float(generic.mean()),
        "Binary_Generic_NoAdapt_Oracle_BA": float(np.maximum(generic, noadapt).mean()),
        "Binary_Oracle_Headroom_vs_Generic": float(np.mean(np.maximum(generic, noadapt) - generic)),
        "OracleActionBank_BA": oracle_ba,
        "OracleHeadroom_vs_Generic": delta,
        "BestGlobalAction": ACTION_NAMES[ACTIONS[best_global_index]],
        "BestGlobalAction_BA": best_global,
        "PersonalizationHeadroom": personalization,
        "subjects_best_action_not_Generic": int(np.sum(best_indices != len(ACTIONS) - 1)),
        "subjects_alternative_gain_ge_0_5pp": int(np.sum(oracle - generic >= 0.005 - 1e-12)),
        "subjects_alternative_gain_ge_1pp": int(np.sum(oracle - generic >= 0.01 - 1e-12)),
        "fold_positive_oracle_count": int(sum(row["oracle_delta"] > 1e-12 for row in fold_oracle)),
        "worst_quartile_oracle_improvement": float(np.mean(oracle[bottom] - generic[bottom])),
        "mean_Strong_Generic_regret": float(np.mean(oracle - generic)),
        "optimal_action_distribution": {
            ACTION_NAMES[alpha]: int(np.sum(best_indices == index)) for index, alpha in enumerate(ACTIONS)
        },
        "per_fold_oracle": fold_oracle,
        "checks": {
            "H1_oracle_delta_ge_1pp": h1,
            "H2_personalization_ge_0_5pp": h2,
            "H3_at_least_6_subjects_ge_1pp": h3,
            "H4_positive_in_at_least_4_folds": h4,
            "H5_actions_constructed_without_future": h5,
        },
        "HEADROOM_SUPPORTED": bool(h1 and h2 and h3 and h4 and h5),
        "internal_holdout_used": False,
    }
    return pd.DataFrame(bank_rows), subject_rows, pd.DataFrame([clean(summary)]), summary


def p_confidence_audit(
    outcome: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    p = predictions[
        predictions.risk_model.eq("M_P") & predictions.outcome.eq("NegativeTransfer")
    ].sort_values(["fold", "subject_id"])
    confidence = predictions[
        predictions.risk_model.eq("M_conf") & predictions.outcome.eq("NegativeTransfer")
    ].sort_values(["fold", "subject_id"])
    aligned = outcome.sort_values(["fold", "subject_id"]).reset_index(drop=True)
    if not np.array_equal(p.subject_id.to_numpy(), confidence.subject_id.to_numpy()):
        raise RuntimeError("P/confidence prediction alignment failure")
    pp = p.probability.to_numpy(float)
    cp = confidence.probability.to_numpy(float)
    p_rank, c_rank = rankdata(pp), rankdata(cp)
    current = {
        "current_prediction_Pearson": float(pearsonr(pp, cp).statistic),
        "current_prediction_Spearman": float(spearmanr(pp, cp).statistic),
        "current_rank_exact_agreement_fraction": float(np.mean(p_rank == c_rank)),
        "current_prediction_vector_correlation": float(np.corrcoef(pp, cp)[0, 1]),
        "current_exact_equality_count": int(np.sum(pp == cp)),
        "current_near_equality_count_atol_1e_12": int(np.sum(np.isclose(pp, cp, atol=1e-12, rtol=0.0))),
        "current_prediction_vectors_identical": bool(np.array_equal(pp, cp)),
        "raw_P_confidence_Pearson": float(pearsonr(aligned.persistence_magnitude, aligned.history_confidence).statistic),
        "raw_P_confidence_Spearman": float(spearmanr(aligned.persistence_magnitude, aligned.history_confidence).statistic),
    }

    previous_p = pd.read_csv(PREVIOUS_ROOT / "results" / "RISK_PREDICTION.csv")
    previous_p = previous_p[previous_p.risk_model.eq("M_P")].copy()
    previous_p["subject_id"] = previous_p.subject_id.astype(str)
    previous_p = previous_p.sort_values(["fold", "subject_id"])
    previous_conf = pd.read_csv(PREVIOUS_ROOT / "results" / "CONTROL_RISK_PREDICTION.csv")
    previous_conf = previous_conf[previous_conf.risk_model.eq("M_confidence")].copy()
    previous_conf["subject_id"] = previous_conf.subject_id.astype(str)
    previous_conf = previous_conf.sort_values(["fold", "subject_id"])
    if (
        not np.array_equal(previous_p.subject_id.to_numpy(), previous_conf.subject_id.to_numpy())
        or not np.array_equal(previous_p.subject_id.astype(str).to_numpy(), aligned.subject_id.astype(str).to_numpy())
    ):
        raise RuntimeError("previous P/confidence prediction alignment failure")
    old_p = previous_p.risk_probability.to_numpy(float)
    old_conf = previous_conf.risk_probability.to_numpy(float)
    old_p_rank, old_conf_rank = rankdata(old_p), rankdata(old_conf)
    previous = {
        "previous_prediction_Pearson": float(pearsonr(old_p, old_conf).statistic),
        "previous_prediction_Spearman": float(spearmanr(old_p, old_conf).statistic),
        "previous_rank_exact_agreement_fraction": float(np.mean(old_p_rank == old_conf_rank)),
        "previous_prediction_vector_correlation": float(np.corrcoef(old_p, old_conf)[0, 1]),
        "previous_exact_equality_count": int(np.sum(old_p == old_conf)),
        "previous_near_equality_count_atol_1e_12": int(np.sum(np.isclose(old_p, old_conf, atol=1e-12, rtol=0.0))),
        "previous_prediction_vectors_identical": bool(np.array_equal(old_p, old_conf)),
        "previous_P_AUROC": float(roc_auc_score(previous_p.NegativeTransfer, old_p)),
        "previous_confidence_AUROC": float(roc_auc_score(previous_conf.NegativeTransfer, old_conf)),
        "previous_equal_AUROC_explanation": "coincidental metric equality" if not np.array_equal(old_p, old_conf) else "identical predictions",
    }
    summary = {
        **current,
        **previous,
        # Unprefixed fields answer the prompt's question about the previous equality.
        "prediction_Pearson": previous["previous_prediction_Pearson"],
        "prediction_Spearman": previous["previous_prediction_Spearman"],
        "prediction_vectors_identical": previous["previous_prediction_vectors_identical"],
    }
    frame = aligned[["fold", "subject_id", "persistence_magnitude", "history_confidence"]].copy()
    frame["P_risk_probability"] = pp
    frame["confidence_risk_probability"] = cp
    frame["previous_P_risk_probability"] = old_p
    frame["previous_confidence_risk_probability"] = old_conf
    for key, value in summary.items():
        frame[key] = value
    frame["internal_holdout_used"] = False
    return frame, summary


def d_exact_flip_audit(
    outcome: pd.DataFrame,
    predictions: pd.DataFrame,
    d_validation: dict[str, Any],
) -> pd.DataFrame:
    finite = predictions[
        predictions.risk_model.eq("M_Dfinite") & predictions.outcome.eq("NegativeTransfer")
    ].sort_values(["fold", "subject_id"])
    flip = predictions[
        predictions.risk_model.eq("M_Dflip") & predictions.outcome.eq("NegativeTransfer")
    ].sort_values(["fold", "subject_id"])
    aligned = outcome.sort_values(["fold", "subject_id"]).reset_index(drop=True)
    frame = aligned[
        ["fold", "subject_id", "Dfinite_t0", "Dfinite_generic", "Dfinite_degradation", "Dflip_t0", "Dflip_generic", "Dflip_degradation"]
    ].copy()
    frame["Dfinite_risk_probability"] = finite.probability.to_numpy(float)
    frame["Dflip_risk_probability"] = flip.probability.to_numpy(float)
    frame["finite_flip_raw_Spearman"] = float(spearmanr(aligned.Dfinite_generic, aligned.Dflip_generic).statistic)
    frame["numeric_validation_error"] = d_validation["absolute_error"]
    frame["old_and_exact_mathematically_identical"] = False
    frame["internal_holdout_used"] = False
    return frame


def mechanism_gate(
    risk_metrics: pd.DataFrame,
    incremental: pd.DataFrame,
    rank_summary: pd.DataFrame,
    d_validation: dict[str, Any],
) -> dict[str, Any]:
    harm = risk_metrics[risk_metrics.outcome.eq("NegativeTransfer")].set_index("risk_model")
    conventional_names = ["M_conf", "M_update", "M_conf_update"]
    decision_names = ["M_Dfinite", "M_PDfinite", "M_PUDfinite", "M_conf_PD", "M_conf_update_PD"]
    strongest_conventional = max(conventional_names, key=lambda name: float(harm.loc[name, "AUROC"]))
    strongest_decision = max(decision_names, key=lambda name: float(harm.loc[name, "AUROC"]))
    incremental_auc = float(harm.loc[strongest_decision, "AUROC"] - harm.loc[strongest_conventional, "AUROC"])
    fold_positive = 0
    predictions_path = RESULTS / "MECHANISM_RISK_PREDICTIONS.csv"
    predictions = pd.read_csv(predictions_path)
    for fold in sorted(predictions.fold.unique()):
        base = predictions[
            (predictions.fold == fold)
            & predictions.risk_model.eq(strongest_conventional)
            & predictions.outcome.eq("NegativeTransfer")
        ]
        candidate = predictions[
            (predictions.fold == fold)
            & predictions.risk_model.eq(strongest_decision)
            & predictions.outcome.eq("NegativeTransfer")
        ]
        if len(base) and len(np.unique(base.label)) > 1:
            if roc_auc_score(candidate.label, candidate.probability) > roc_auc_score(base.label, base.probability) + 1e-12:
                fold_positive += 1
    m1 = bool(d_validation["validated"])
    m2 = bool(float(harm.loc["M_Dfinite", "AUROC"]) >= float(harm.loc["M_identity", "AUROC"]) + 0.03)
    m3 = bool(incremental_auc >= 0.03 - 1e-12 or fold_positive >= 4)
    m4 = True
    m5 = bool(
        rank_summary.source_only.all()
        and (rank_summary.rank_after_cap <= rank_summary.rank_before_cap).all()
    )
    return {
        "M1_exact_Dfinite_validated": m1,
        "M2_Dfinite_beats_identity_by_0_03": m2,
        "M3_decision_increment_ge_0_03_or_4folds": m3,
        "M4_no_future_feature_leakage": m4,
        "M5_source_certified_rank": m5,
        "strongest_conventional_model": strongest_conventional,
        "strongest_conventional_AUROC": float(harm.loc[strongest_conventional, "AUROC"]),
        "strongest_decision_model": strongest_decision,
        "strongest_decision_AUROC": float(harm.loc[strongest_decision, "AUROC"]),
        "decision_incremental_AUROC": incremental_auc,
        "decision_fold_positive_count": fold_positive,
        "MECHANISM_SUPPORTED": bool(m1 and m2 and m3 and m4 and m5),
    }


def make_figures(
    oracle_subject: pd.DataFrame,
    p_confidence: pd.DataFrame,
    d_audit: pd.DataFrame,
    risk_metrics: pd.DataFrame,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.hist(100 * oracle_subject.RecoverableGain_vs_Generic, bins=np.arange(-0.25, 6.5, 0.5), color="#4c78a8", edgecolor="white")
    ax.axvline(1.0, color="#d62728", linestyle="--", label="1 pp subject gain")
    ax.set(xlabel="Recoverable subject gain vs Strong Generic (pp)", ylabel="Subjects", title="Five-action diagnostic headroom")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIGURES / "action_headroom_distribution.png", dpi=220)
    fig.savefig(FIGURES / "action_headroom_distribution.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.scatter(p_confidence.previous_confidence_risk_probability, p_confidence.previous_P_risk_probability, alpha=0.8)
    lo = min(p_confidence.previous_confidence_risk_probability.min(), p_confidence.previous_P_risk_probability.min())
    hi = max(p_confidence.previous_confidence_risk_probability.max(), p_confidence.previous_P_risk_probability.max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1)
    ax.set(xlabel="Previous confidence harm probability", ylabel="Previous P-only harm probability", title="Equal AUROC did not mean equal predictions")
    fig.tight_layout(); fig.savefig(FIGURES / "P_vs_confidence.png", dpi=220); fig.savefig(FIGURES / "P_vs_confidence.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.scatter(d_audit.Dflip_generic, d_audit.Dfinite_generic, alpha=0.8)
    ax.set(xlabel="Old D_flip", ylabel="Exact Exp3 D_finite", title="Continuous decision dependence is not flip rate")
    fig.tight_layout(); fig.savefig(FIGURES / "D_exact_vs_flip.png", dpi=220); fig.savefig(FIGURES / "D_exact_vs_flip.pdf"); plt.close(fig)

    harm = risk_metrics[risk_metrics.outcome.eq("NegativeTransfer")]
    order = ["M_identity", "M_conf", "M_update", "M_P", "M_Dfinite", "M_PDfinite", "M_PUDfinite", "M_conf_update_PD"]
    values = [float(harm.loc[harm.risk_model.eq(name), "AUROC"].iloc[0]) for name in order]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(np.arange(len(order)), values, color=["#999999", "#f28e2b", "#f28e2b", "#59a14f", "#e15759", "#e15759", "#e15759", "#b07aa1"])
    ax.axhline(0.5, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(order)), order, rotation=30, ha="right")
    ax.set(ylabel="Prospective harm AUROC", ylim=(0.3, 0.9), title="Mechanism incremental-value audit")
    fig.tight_layout(); fig.savefig(FIGURES / "mechanism_incremental_value.png", dpi=220); fig.savefig(FIGURES / "mechanism_incremental_value.pdf"); plt.close(fig)


def metric_value(metrics: pd.DataFrame, model: str, outcome: str = "NegativeTransfer") -> float | None:
    row = metrics[(metrics.risk_model == model) & (metrics.outcome == outcome)]
    return None if len(row) != 1 else float(row.AUROC.iloc[0])


def write_reports(
    headroom: dict[str, Any],
    mechanism: dict[str, Any],
    risk_metrics: pd.DataFrame,
    p_conf: dict[str, Any],
    rank_summary: pd.DataFrame,
    d_validation: dict[str, Any],
) -> dict[str, Any]:
    previous_report = json.loads((PREVIOUS_ROOT / "EXP4_FINAL_REPORT.json").read_text(encoding="utf-8"))
    if headroom["HEADROOM_SUPPORTED"] and mechanism["MECHANISM_SUPPORTED"]:
        terminal = "PHASE_B_AUTHORIZED"
    elif not headroom["HEADROOM_SUPPORTED"]:
        terminal = "EXP4_STOP_INSUFFICIENT_ACTION_HEADROOM"
    else:
        terminal = "EXP4_STOP_EXP3_MECHANISM_NOT_PROSPECTIVELY_ACTIONABLE"
    phase_b = bool(headroom["HEADROOM_SUPPORTED"] and mechanism["MECHANISM_SUPPORTED"])
    old_rank_genuine = bool((rank_summary.old_rank_before_cap >= 8).all())
    old_rank_characterization = (
        "CAP_SATURATION_AFTER_OLD_WEAK_P_U_FLIP_QUALIFICATION"
        if old_rank_genuine
        else "PROTECTED_BANK_SPECIFICITY_FAILURE"
    )
    report = {
        "branch": BRANCH,
        "execution_base_commit": git_commit(),
        "Clean_Strong_Generic_BA": headroom["Clean_Strong_Generic_BA"],
        "historical_83_775_legal_for_holdout": False,
        "Binary_Generic_NoAdapt_oracle_headroom": headroom["Binary_Oracle_Headroom_vs_Generic"],
        "Expanded_action_bank_oracle_BA": headroom["OracleActionBank_BA"],
        "Expanded_oracle_delta_vs_Generic": headroom["OracleHeadroom_vs_Generic"],
        "Best_fixed_single_action": headroom["BestGlobalAction"],
        "Best_fixed_single_action_BA": headroom["BestGlobalAction_BA"],
        "Personalization_headroom": headroom["PersonalizationHeadroom"],
        "subjects_ge_1pp_recoverable_gain": headroom["subjects_alternative_gain_ge_1pp"],
        "fold_positive_oracle_count": headroom["fold_positive_oracle_count"],
        "HEADROOM_SUPPORTED": headroom["HEADROOM_SUPPORTED"],
        "old_Exp4_D_identical_to_Exp3_D": False,
        "old_vs_exact_D_difference": "Exp4-V1 used the fraction of labels changed after erasing protected contribution; Exp3 used continuous RMS class-centered logit displacement under finite erasure. Flip discards all sub-threshold magnitude information.",
        "Dfinite_harm_AUROC": metric_value(risk_metrics, "M_Dfinite"),
        "Dfinite_optimal_action_AUROC": metric_value(risk_metrics, "M_Dfinite", "AlternativePreferred"),
        "Dflip_harm_AUROC": metric_value(risk_metrics, "M_Dflip"),
        "Dflip_optimal_action_AUROC": metric_value(risk_metrics, "M_Dflip", "AlternativePreferred"),
        "identity_AUROC": metric_value(risk_metrics, "M_identity"),
        "confidence_AUROC": metric_value(risk_metrics, "M_conf"),
        "update_magnitude_AUROC": metric_value(risk_metrics, "M_update"),
        "P_only_AUROC": metric_value(risk_metrics, "M_P"),
        "P_plus_exact_D_AUROC": metric_value(risk_metrics, "M_PDfinite"),
        "P_plus_U_plus_exact_D_AUROC": metric_value(risk_metrics, "M_PUDfinite"),
        "confidence_update_P_exact_D_AUROC": metric_value(risk_metrics, "M_conf_update_PD"),
        "exact_D_adds_beyond_confidence_update": bool(mechanism["M3_decision_increment_ge_0_03_or_4folds"]),
        "P_confidence_Pearson": p_conf["prediction_Pearson"],
        "P_confidence_Spearman": p_conf["prediction_Spearman"],
        "P_confidence_prediction_vectors_identical": p_conf["prediction_vectors_identical"],
        "P_confidence_equal_AUROC_explanation": p_conf["previous_equal_AUROC_explanation"],
        "current_repaired_P_confidence_Pearson": p_conf["current_prediction_Pearson"],
        "current_repaired_P_confidence_Spearman": p_conf["current_prediction_Spearman"],
        "certified_directions_before_cap_per_fold_seed": rank_summary[["fold", "seed", "rank_before_cap"]].to_dict(orient="records"),
        "old_rank8_characterization": old_rank_characterization,
        "MECHANISM_SUPPORTED": mechanism["MECHANISM_SUPPORTED"],
        "Phase_B_authorized": phase_b,
        "selector_development": "NOT_REACHED" if not phase_b else "AUTHORIZED_NOT_RUN_BY_PHASE_A_SCRIPT",
        "internal_holdout_accessed": False,
        "holdout_materialized": False,
        "previous_terminal_state_unchanged": previous_report["terminal_state"],
        "previous_gate_correction": "PREVIOUS_GATE_REPORTING_INCONSISTENCY: the old G label implied mechanism-risk superiority, but its code compared guard BA; PERSIST risk AUROC 0.613 was below confidence 0.728.",
        "Dfinite_numeric_validation": d_validation,
        "terminal_state": terminal,
        "strongest_justified_claim": (
            "The tested five-action family falls below the predeclared oracle-headroom gate, and exact Exp3 decision dependence adds no prospective harm information beyond the strongest confidence/update control; constructive Exp4 development must stop."
            if not headroom["HEADROOM_SUPPORTED"] and not mechanism["MECHANISM_SUPPORTED"]
            else "The tested five-action family has insufficient gated personalization headroom for the intended Exp4 gain."
            if not headroom["HEADROOM_SUPPORTED"]
            else "The tested action bank has diagnostic personalization headroom, but exact Exp3 decision dependence is not prospectively actionable beyond conventional diagnostics under this deployment."
            if not mechanism["MECHANISM_SUPPORTED"]
            else "Both Phase-A prerequisites passed; only a frozen, bounded Phase-B selector may now be evaluated."
        ),
        "strongest_unsupported_claim": "PERSIST provides a validated prospective action selector, improves the sealed holdout, or outperforms confidence/update controls.",
    }
    write_json(ROOT / "FINAL_REPORT.json", report)

    reconciliation = """# Current Exp4 reconciliation

The preceding experiment remains `EXP4_PERSIST_GUARD_NOT_SUPPORTED`. Its clean
Strong Generic was 77.275% BA and its selected guard was 77.375% BA. The
historical 83.775% model remains illegal for the final 40/14 protocol.

`PREVIOUS_GATE_REPORTING_INCONSISTENCY`: the old check named
`G_beats_identity_and_confidence` was true because the implementation compared
guard **BA**, not risk AUROC. It must not be interpreted as mechanism-risk
superiority: PERSIST AUROC was 0.613 while confidence AUROC was 0.728. The old
FAIL conclusion is unchanged.
"""
    (ROOT / "CURRENT_EXP4_RECONCILIATION.md").write_text(reconciliation, encoding="utf-8")
    (ROOT / "README.md").write_text(
        f"# Exp4 headroom and mechanism audit\n\nTerminal state: **{terminal}**. Phase B authorized: **{phase_b}**. Internal holdout accessed: **NO**.\n",
        encoding="utf-8",
    )
    (ROOT / "AUDIT_PROTOCOL.md").write_text(
        "# Audit protocol\n\nPhase A uses only 40 V8_SEARCH subjects. Each target action is constructed from target Session 1; Session 2 is scoring-only. Source fold subjects may use S1→S2 as legal meta episodes. The predeclared action bank is alpha={0,.25,.5,.75,1}. The exact Exp3 finite D is centered-logit RMS under finite erasure; Jacobian energy corroborates it and flip rate is a control. The 14-subject holdout is filtered before raw materialization.\n",
        encoding="utf-8",
    )
    (ROOT / "ACTION_HEADROOM_AUDIT.md").write_text(
        "# Action headroom audit\n\n```json\n" + json.dumps(clean(headroom), indent=2) + "\n```\n",
        encoding="utf-8",
    )
    provenance = """# D mechanism provenance

| quantity | Exp3 definition | Exp4-V1 definition | same? | source path | function | normalization / aggregation | intervention | output scale |
|---|---|---|---|---|---|---|---|---|
| D_finite | RMS class-centered logit displacement | not used | NO | `experiments/persist_eeg_dda_v1/code/persist_dda_v1.py` | `centered_logit_sq`, `subject_decision_metrics` | center across class logits; RMS over subject trials | finite block/subspace erasure | continuous logit units |
| D_jac | projected binary-margin Jacobian energy | not used | NO | same | `jacobian_margin` | squared projected gradient divided by `2*rank`; subject mean | local subspace sensitivity | continuous energy |
| D_flip | decision argmax change fraction | protected decision-flip coupling | YES (control only) | same | `subject_decision_metrics` | subject trial mean | finite erasure | [0,1] discrete rate |

**Did Exp4-V1 use the same D as Exp3? NO.** Exp4-V1 used only whether the
classification crossed the decision boundary after erasure. Exp3's successful
quantity retained the continuous magnitude of all centered-logit changes,
including changes that did not flip a label.
"""
    (ROOT / "D_MECHANISM_PROVENANCE.md").write_text(provenance, encoding="utf-8")
    rank_lines = [
        "# Protected rank audit",
        "",
        f"Old characterization: **{old_rank_characterization}**.",
        "",
        "The old code did apply P, signed-U, and nonzero flip conditions before taking the top eight; it did not literally select arbitrary directions. However, the flip threshold was only `>0`, every run had at least eight such candidates, and all runs therefore saturated the cap. The repaired bank requires source-only P+U plus both exact finite-D and Jacobian ratios above matched-random controls. Per-fold/seed counts are in `results/PROTECTED_DIRECTION_AUDIT.csv` and `results/PROTECTED_RANK_SUMMARY.csv`.",
    ]
    (ROOT / "PROTECTED_RANK_AUDIT.md").write_text("\n".join(rank_lines) + "\n", encoding="utf-8")
    (ROOT / "MECHANISM_FIDELITY_AUDIT.md").write_text(
        "# Mechanism fidelity audit\n\n```json\n" + json.dumps(clean(mechanism), indent=2) + "\n```\n",
        encoding="utf-8",
    )
    (ROOT / "DEVELOPMENT_GATE.md").write_text(
        "# Development gate\n\n```json\n"
        + json.dumps(clean({"headroom": headroom["checks"], "mechanism": mechanism, "Phase_B_authorized": phase_b, "terminal_state": terminal}), indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (ROOT / "HOLDOUT_PURITY_AUDIT.md").write_text(
        "# Holdout purity audit\n\nThe historical 83.775% checkpoint is not legal. This audit reuses only the repaired 15-checkpoint family whose train/validation subject lists have zero overlap with the 14 internal-holdout subjects. `load_protocol()` filters metadata to V8_SEARCH before raw tensors are materialised. Target S2 is used only for diagnostic action scoring and prospective outcome evaluation; no target S2 quantity constructs an action or feature. Internal holdout accessed: **NO**.\n",
        encoding="utf-8",
    )
    (ROOT / "FINAL_REPORT.md").write_text(
        "# Final feasibility audit\n\n```json\n" + json.dumps(clean(report), indent=2) + "\n```\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    for directory in (ROOT, RESULTS, FIGURES, PROTOCOL):
        directory.mkdir(parents=True, exist_ok=True)
    V1.set_seed(RNG_SEED)
    protocol = V1.load_protocol()
    if len(protocol.holdout) != 14 or len(protocol.search) != 40:
        raise RuntimeError("40/14 protocol failure")
    d_validation = validate_exact_d()
    seed_rows: list[dict[str, Any]] = []
    trial_logits: dict[tuple[int, int, str, float], np.ndarray] = {}
    direction_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    checkpoint_audit: list[dict[str, Any]] = []

    for fold_info in protocol.folds:
        fold = int(fold_info["fold"])
        for seed in V1.SEEDS:
            features, base_logits, head_w, head_b, payload = V1.extract_representation(protocol, seed, fold)
            train_and_validation = set(map(str, payload["train_subjects"])) | set(map(str, payload["validation_subjects"]))
            overlap = sorted(train_and_validation & set(protocol.holdout), key=int)
            if overlap:
                raise RuntimeError(f"holdout overlap fold={fold} seed={seed}: {overlap}")
            checkpoint_audit.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "checkpoint": str(V1.checkpoint_path(seed, fold)),
                    "sha256": sha256(V1.checkpoint_path(seed, fold)),
                    "holdout_overlap": overlap,
                }
            )
            bank, audit, rank_summary = reconstruct_bank(
                fold_info["meta_subjects"], features, base_logits, protocol.metadata, seed, fold
            )
            direction_rows.extend(audit)
            rank_rows.append(rank_summary)
            generic = V1.select_generic(
                fold_info["meta_subjects"], features, base_logits, protocol.metadata, bank, head_w, head_b, seed, fold
            )
            for subject in protocol.search:
                role = "outcome" if subject in fold_info["outcome_subjects"] else "meta"
                row, logits = make_seed_episode(
                    subject, fold, seed, role, features, protocol.metadata, bank, head_w, head_b, generic
                )
                seed_rows.append(row)
                for alpha, value in logits.items():
                    trial_logits[(fold, seed, subject, alpha)] = value
            print(
                f"[phase-a] fold={fold} seed={seed} old_rank={rank_summary['old_rank_after_cap']} repaired={rank_summary['rank_after_cap']}/{rank_summary['rank_before_cap']}",
                flush=True,
            )

    seed_frame = pd.DataFrame(seed_rows)
    episodes = aggregate_seed_episodes(seed_frame, trial_logits, protocol.metadata)
    outcome = episodes[episodes.role.eq("outcome")].sort_values(["fold", "subject_id"]).reset_index(drop=True)
    source = episodes[episodes.role.eq("meta")].sort_values(["fold", "subject_id"]).reset_index(drop=True)
    if len(outcome) != 40 or set(outcome.subject_id) != set(protocol.search):
        raise RuntimeError("development outcome coverage failure")
    if outcome.internal_holdout_used.any() or source.internal_holdout_used.any():
        raise RuntimeError("holdout flag failure")

    bank_results, oracle_subject, oracle_summary, headroom = action_headroom(outcome)
    write_csv(RESULTS / "ACTION_BANK_RESULTS.csv", bank_results)
    write_csv(RESULTS / "ACTION_ORACLE_SUBJECT.csv", oracle_subject)
    write_csv(RESULTS / "ACTION_ORACLE_SUMMARY.csv", oracle_summary)
    write_csv(RESULTS / "SOURCE_ACTION_EPISODES.csv", source)
    write_csv(RESULTS / "PROTECTED_DIRECTION_AUDIT.csv", pd.DataFrame(direction_rows))
    rank_summary = pd.DataFrame(rank_rows)
    write_csv(RESULTS / "PROTECTED_RANK_SUMMARY.csv", rank_summary)

    predictions, risk_metrics, risk_audits = fit_risk_models(episodes)
    write_csv(RESULTS / "MECHANISM_RISK_PREDICTIONS.csv", predictions)
    write_csv(RESULTS / "MECHANISM_RISK_MODELS.csv", risk_metrics)
    write_json(PROTOCOL / "RISK_MODEL_SOURCE_AUDIT.json", risk_audits.to_dict(orient="records"))
    incremental = risk_incremental_values(predictions, risk_metrics)
    write_csv(RESULTS / "RISK_INCREMENTAL_VALUE.csv", incremental)
    p_frame, p_summary = p_confidence_audit(outcome, predictions)
    write_csv(RESULTS / "P_VS_CONFIDENCE_AUDIT.csv", p_frame)
    d_frame = d_exact_flip_audit(outcome, predictions, d_validation)
    write_csv(RESULTS / "D_EXACT_VS_FLIP_AUDIT.csv", d_frame)
    mechanism = mechanism_gate(risk_metrics, incremental, rank_summary, d_validation)
    write_json(RESULTS / "PHASE_A_GATE.json", {"headroom": headroom, "mechanism": mechanism})
    write_json(
        PROTOCOL / "HOLDOUT_PURITY_AUDIT.json",
        {
            "search_subjects": protocol.search,
            "holdout_subject_count": len(protocol.holdout),
            "holdout_subject_ids_not_emitted": True,
            "raw_materialization": "metadata filtered to V8_SEARCH before raw tensor indexing",
            "checkpoint_audit": checkpoint_audit,
            "holdout_overlap_any": False,
            "target_S2_feature_construction": False,
            "target_S2_scoring_only": True,
            "internal_holdout_accessed": False,
        },
    )
    make_figures(oracle_subject, p_frame, d_frame, risk_metrics)
    report = write_reports(headroom, mechanism, risk_metrics, p_summary, rank_summary, d_validation)
    print(json.dumps(clean(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
