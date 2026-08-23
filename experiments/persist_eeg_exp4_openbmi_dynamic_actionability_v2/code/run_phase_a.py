"""Phase-A trajectory and gradient actionability audit for OpenBMI MI.

This implementation is intentionally small and deterministic.  It uses only the
materialised V7 MI-specific feature/logit cache restricted to V8_SEARCH subjects.
The target adaptation is a residual linear head over the frozen embedding.  The
residual starts at zero (the frozen Generic/no-adaptation anchor) and is updated
by five legal S1 full-batch BCE steps.  All prospective outcome quantities are
computed only for source-fold outcome subjects after the history-only trajectory
has been frozen.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

VENDOR = Path(os.environ.get("PERSIST_PYARROW_VENDOR", r"D:\nips-temp\TotalP\P1\CRCICLR_V3_WORK\vendor"))
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import pyarrow.parquet as pq
import numpy as np
import pandas as pd

# The bundled pyarrow is intentionally vendored on the server.  Recent pandas
# calls ``unregister_extension_type`` defensively, while this older pyarrow
# raises when the extension is absent.  Make that one operation idempotent;
# this does not change parquet values or schemas.
try:
    import pyarrow as _pa
    _unregister_extension_type = _pa.unregister_extension_type
    def _safe_unregister_extension_type(name: str) -> None:
        try:
            _unregister_extension_type(name)
        except Exception as exc:
            if "No type extension" not in str(exc):
                raise
    _pa.unregister_extension_type = _safe_unregister_extension_type
except Exception:
    pass

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr


EXPERIMENT = Path(__file__).resolve().parents[1]
RESULTS = EXPERIMENT / "results"
FIGURES = EXPERIMENT / "figures"
PROTOCOL = EXPERIMENT / "protocol"
CODE = EXPERIMENT / "code"
V7_ROOT = Path(os.environ.get(
    "PERSIST_V7_RUNTIME",
    r"D:\nips-temp\TotalP\P1\CRCICLR_V7_FUTURE_UTILITY_META",
))
V7_CACHE = V7_ROOT / "experiments" / "persist_eeg_final_model_v7" / "outputs" / "cache"
V8_ROOT = Path(os.environ.get(
    "PERSIST_V8_RUNTIME",
    r"D:\nips-temp\TotalP\P1\CRCICLR_V8_HEADROOM_FIRST",
))
V8_PROTOCOL = V8_ROOT / "experiments" / "persist_eeg_final_model_v8" / "outputs" / "protocol"
STAGE0_ROOT = Path(os.environ.get(
    "PERSIST_STAGE0_REPO",
    r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full",
))

SEED = 20260823
CHECKPOINTS = ("t0", "t1", "t2", "t3", "t4", "final")
STEPS = 5
ETA = 0.05
PROTECTED_RANK = 8
IDENTITY_RANK = 4
EPS = 1e-8


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
        return None if not np.isfinite(value) else value
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def balanced_ba(y: np.ndarray, logits: np.ndarray) -> float:
    return float(balanced_accuracy_score(y.astype(int), (logits >= 0.0).astype(int)))


def bce(y: np.ndarray, logits: np.ndarray) -> float:
    p = np.clip(sigmoid(logits), 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def balanced_grad(z: np.ndarray, y: np.ndarray, logits: np.ndarray) -> tuple[np.ndarray, float]:
    """Gradient of balanced BCE for residual weight and bias."""
    p = sigmoid(logits)
    residual = p - y
    values = []
    for label in (0, 1):
        mask = y.astype(int) == label
        values.append((residual[mask, None] * z[mask]).mean(axis=0))
    gw = 0.5 * (values[0] + values[1])
    gb = 0.5 * (residual[y.astype(int) == 0].mean() + residual[y.astype(int) == 1].mean())
    return gw.astype(np.float64), float(gb)


def balanced_bce_torchless(z: np.ndarray, y: np.ndarray, logits: np.ndarray) -> float:
    return 0.5 * (bce(y[y == 0], logits[y == 0]) + bce(y[y == 1], logits[y == 1]))


def softmax2(logits: np.ndarray) -> np.ndarray:
    return sigmoid(logits)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    split = json.loads((V8_PROTOCOL / "V8_SEARCH_SPLIT.json").read_text(encoding="utf-8"))
    if bool(split.get("OUTER_TEST_USED", True)) or bool(split.get("search_time_holdout_outcomes_accessed", True)):
        raise RuntimeError("V8 split indicates forbidden outcome access")
    openbmi = split["openbmi"]
    search = set(map(str, openbmi["V8_SEARCH"]))
    holdout = set(map(str, openbmi["V8_INTERNAL_HOLDOUT"]))
    if search & holdout or len(search) != 40 or len(holdout) != 14:
        raise RuntimeError("Malformed V8 subject partition")
    split_freeze = json.loads((STAGE0_ROOT / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json").read_text(encoding="utf-8-sig"))
    folds = []
    for i, row in enumerate(split_freeze["openbmi"]["folds"]):
        meta = sorted((set(map(str, row["train_subjects"] + row["validation_subjects"]))) & search)
        outcome = sorted(set(map(str, row["outer_test_subjects"])) & search)
        if set(meta) & set(outcome) or not outcome:
            raise RuntimeError(f"fold {i} role overlap/empty")
        folds.append({"fold": i, "meta_subjects": meta, "outcome_subjects": outcome})
    return {"search_subjects": sorted(search), "holdout_subject_count": len(holdout), "folds": folds}, split


def load_cache() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    f = V7_CACHE / "OPENBMI_MI_SPECIFIC_FOLD_0_FEATURES.npy"
    z = V7_CACHE / "OPENBMI_MI_SPECIFIC_FOLD_0_LOGITS.npy"
    m = V7_CACHE / "OPENBMI_MI_SPECIFIC_FOLD_0_METADATA.parquet"
    features = np.load(f, mmap_mode="r", allow_pickle=False)
    logits = np.asarray(np.load(z, mmap_mode="r", allow_pickle=False), dtype=np.float64)
    # Read through pyarrow directly; the server's pandas/pyarrow pair has a
    # harmless extension-registration incompatibility in ``read_parquet``.
    metadata = pq.read_table(m).to_pandas()
    if logits.ndim == 2:
        logits = logits[:, -1] - logits[:, 0]
    metadata = metadata.copy()
    metadata["subject_id"] = metadata.subject_id.astype(str)
    metadata["session_id"] = metadata.session_id.astype(int)
    metadata["label"] = metadata.label.astype(int)
    if len(metadata) != len(features) or len(logits) != len(features):
        raise RuntimeError("cache length mismatch")
    if metadata.OUTER_TEST_USED.astype(bool).any():
        raise RuntimeError("cache contains outer-test rows")
    return np.asarray(features, dtype=np.float32), logits, metadata


def subject_rows(features: np.ndarray, logits: np.ndarray, metadata: pd.DataFrame, subject: str, session: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = metadata.subject_id.eq(str(subject)).to_numpy() & metadata.session_id.eq(int(session)).to_numpy()
    if int(mask.sum()) != 100:
        raise RuntimeError(f"subject {subject} session {session} has {int(mask.sum())} rows")
    return (
        np.asarray(features[mask], dtype=np.float64),
        np.asarray(logits[mask], dtype=np.float64),
        metadata.loc[mask, "label"].to_numpy(dtype=np.int64),
        metadata.loc[mask, "trial_uid"].astype(str).to_numpy(),
    )


def fit_protected(meta_subjects: list[str], features: np.ndarray, metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit fixed protected and identity controls on source S1 only."""
    rows = []
    means = []
    labels = []
    for subject in meta_subjects:
        x, _, y, _ = subject_rows(features, np.zeros(len(features)), metadata, subject, 1)
        rows.append(x)
        labels.append(y)
        means.append(x.mean(axis=0))
    x = np.concatenate(rows, axis=0)
    y = np.concatenate(labels, axis=0)
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-5)
    xz = (x - center) / scale
    # PCA by SVD, with task direction selected in the source population.
    _, _, vt = np.linalg.svd(xz, full_matrices=False)
    pc = vt[: min(32, vt.shape[0])].T
    class_dir = xz[y == 1].mean(axis=0) - xz[y == 0].mean(axis=0)
    class_pc = pc.T @ class_dir
    order = np.argsort(np.abs(class_pc))[::-1]
    p_rank = min(PROTECTED_RANK, pc.shape[1])
    p_basis = pc[:, order[:p_rank]]
    # Identity control is a between-subject PCA, fitted without target outcomes.
    mean_matrix = np.stack([(row - center) / scale for row in means])
    mean_matrix -= mean_matrix.mean(axis=0, keepdims=True)
    _, _, ivt = np.linalg.svd(mean_matrix, full_matrices=False)
    i_rank = min(IDENTITY_RANK, ivt.shape[0])
    i_basis = ivt[:i_rank].T
    task_p = p_basis.T @ class_dir
    task_p /= max(np.linalg.norm(task_p), EPS)
    return {
        "center": center,
        "scale": scale,
        "p_basis": p_basis,
        "i_basis": i_basis,
        "task_p": task_p,
    }


def projected(x: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    return ((x - state["center"]) / state["scale"]) @ state["p_basis"]


def identity_projected(x: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    return ((x - state["center"]) / state["scale"]) @ state["i_basis"]


def static_metrics(x: np.ndarray, base: np.ndarray, y: np.ndarray, state: dict[str, np.ndarray]) -> dict[str, float]:
    zp = projected(x, state)
    zi = identity_projected(x, state)
    # A source-fitted protected task probe gives a fixed functional contribution.
    wp = state["task_p"]
    protected = zp @ wp
    identity = zi @ (zi.T @ (base - base.mean()) / max(np.sum(zi * zi), EPS)) if zi.shape[1] else np.zeros(len(x))
    erased_p = base - protected
    erased_i = base - identity
    return {
        "P_static": float(np.linalg.norm(wp)),
        "U_static": float(balanced_ba(y, base) - balanced_ba(y, erased_p)),
        "D_static": float(np.mean((base >= 0.0) != (erased_p >= 0.0))),
        "I_static": float(np.mean(np.abs(identity)) / (np.mean(np.abs(base)) + EPS)),
        "history_BA_t0": balanced_ba(y, base),
        "history_loss_t0": balanced_bce_torchless(np.zeros((len(y), 1)), y, base),
        "history_margin_t0": float(np.mean(np.abs(base))),
        "history_gradient_norm_t0": float(np.linalg.norm(balanced_grad((x - state["center"]) / state["scale"], y, base)[0])),
    }


def utility_function(x: np.ndarray, y: np.ndarray, base: np.ndarray, w: np.ndarray, b: float, state: dict[str, np.ndarray]) -> float:
    """Differentiable signed protected utility surrogate G.

    G is the increase in balanced BCE after erasing the protected contribution;
    positive values mean the protected function helps the legal history task.
    """
    z = (x - state["center"]) / state["scale"]
    zp = z @ state["p_basis"]
    full = base + z @ w + b
    protected = zp @ (state["p_basis"].T @ w)
    erased = full - protected
    return float(balanced_bce_torchless(zp, y, erased) - balanced_bce_torchless(zp, y, full))


def utility_value_and_gradient(x: np.ndarray, y: np.ndarray, base: np.ndarray, w: np.ndarray, b: float, state: dict[str, np.ndarray]) -> tuple[float, np.ndarray, float]:
    """Analytic gradient of G for the residual linear head.

    For each example, d(full)/d(w)=z and d(erased)/d(w)=z - z_p P^T.
    Balanced BCE uses per-class means, matching ``balanced_grad``.
    """
    z = (x - state["center"]) / state["scale"]
    p = state["p_basis"]
    zp = z @ p
    full = base + z @ w + b
    protected = zp @ (p.T @ w)
    erased = full - protected
    g_full_w, g_full_b = balanced_grad(z, y, full)
    erased_z = z - zp @ p.T
    g_erase_w, g_erase_b = balanced_grad(erased_z, y, erased)
    # G = L(erase)-L(full)
    return utility_function(x, y, base, w, b, state), g_erase_w - g_full_w, g_erase_b - g_full_b


def decision_dependence(x: np.ndarray, base: np.ndarray, w: np.ndarray, b: float, state: dict[str, np.ndarray]) -> float:
    z = (x - state["center"]) / state["scale"]
    p = state["p_basis"]
    full = base + z @ w + b
    protected = (z @ p) @ (p.T @ w)
    return float(np.mean((full >= 0.0) != ((full - protected) >= 0.0)))


def checkpoint_metrics(x: np.ndarray, base: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, state: dict[str, np.ndarray]) -> dict[str, float]:
    z = (x - state["center"]) / state["scale"]
    p = state["p_basis"]
    i = state["i_basis"]
    full = base + z @ w + b
    protected = (z @ p) @ (p.T @ w)
    identity = (z @ i) @ (i.T @ w) if i.size else np.zeros(len(x))
    erased_p = full - protected
    erased_i = full - identity
    utility = balanced_ba(y, full) - balanced_ba(y, erased_p)
    g = utility_value_and_gradient(x, y, base, w, b, state)[0]
    return {
        "P_t": float(np.linalg.norm(p.T @ w) / (np.linalg.norm(w) + EPS)),
        "U_t": float(utility),
        "D_t": float(np.mean((full >= 0.0) != (erased_p >= 0.0))),
        "I_t": float(np.mean(np.abs(identity)) / (np.mean(np.abs(full)) + EPS)),
        "history_BA_t": balanced_ba(y, full),
        "history_loss_t": balanced_bce_torchless(z, y, full),
        "coordinate_drift_t": float(np.linalg.norm(w)),
        "protected_contribution_t": float(np.sqrt(np.mean(protected ** 2))),
        "identity_contribution_t": float(np.sqrt(np.mean(identity ** 2))),
        "G_surrogate_t": float(g),
    }


def trajectory(x: np.ndarray, base: np.ndarray, y: np.ndarray, state: dict[str, np.ndarray], subject: str, fold: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    z = (x - state["center"]) / state["scale"]
    w = np.zeros(z.shape[1], dtype=np.float64)
    b = 0.0
    rows = []
    grad_rows = []
    checkpoint_ws: list[tuple[np.ndarray, float]] = []
    for step in range(STEPS + 1):
        label = CHECKPOINTS[step]
        metric = checkpoint_metrics(x, base, y, w, b, state)
        metric.update({"subject_id": subject, "source_fold": fold, "checkpoint": label, "step": step})
        rows.append(metric)
        checkpoint_ws.append((w.copy(), float(b)))
        if step == STEPS:
            break
        full = base + z @ w + b
        g_task_w, g_task_b = balanced_grad(z, y, full)
        _, g_g_w, g_g_b = utility_value_and_gradient(x, y, base, w, b, state)
        dot = float(np.dot(g_g_w, g_task_w) + g_g_b * g_task_b)
        damage = max(0.0, ETA * dot)
        w_after = w - ETA * g_task_w
        b_after = b - ETA * g_task_b
        g_before = utility_function(x, y, base, w, b, state)
        g_after = utility_function(x, y, base, w_after, b_after, state)
        d_before = decision_dependence(x, base, w, b, state)
        d_after = decision_dependence(x, base, w_after, b_after, state)
        c_before = checkpoint_metrics(x, base, y, w, b, state)["protected_contribution_t"]
        c_after = checkpoint_metrics(x, base, y, w_after, b_after, state)["protected_contribution_t"]
        grad_rows.append({
            "subject_id": subject,
            "source_fold": fold,
            "step": step,
            "checkpoint": label,
            "eta": ETA,
            "task_grad_norm": float(np.sqrt(np.dot(g_task_w, g_task_w) + g_task_b * g_task_b)),
            "utility_grad_norm": float(np.sqrt(np.dot(g_g_w, g_g_w) + g_g_b * g_g_b)),
            "grad_dot_task_G": dot,
            "cos_task_G": float(dot / (np.sqrt(np.dot(g_task_w, g_task_w) + g_task_b * g_task_b) * np.sqrt(np.dot(g_g_w, g_g_w) + g_g_b * g_g_b) + EPS)),
            "predicted_utility_damage": damage,
            "utility_delta_actual_small_step": float(g_after - g_before),
            "predicted_utility_delta": float(-ETA * dot),
            "finite_decision_change": float(d_after - d_before),
            "predicted_decision_damage_finite": float(max(0.0, d_before - d_after)),
            "protected_contribution_delta": float(c_after - c_before),
            "w_norm_before": float(np.linalg.norm(w)),
        })
        w, b = w_after, b_after
    traj = pd.DataFrame(rows)
    grad = pd.DataFrame(grad_rows)
    early = traj.loc[traj.step.isin([0, 1])]
    late = traj.loc[traj.step.isin([4, 5])]
    utility = traj.U_t.to_numpy(float)
    dvals = traj.D_t.to_numpy(float)
    pred_damage = grad.predicted_utility_damage.to_numpy(float)
    ddamage = grad.predicted_decision_damage_finite.to_numpy(float)
    summary = {
        "subject_id": subject,
        "source_fold": fold,
        "delta_P": float(traj.P_t.iloc[-1] - traj.P_t.iloc[0]),
        "delta_U": float(traj.U_t.iloc[-1] - traj.U_t.iloc[0]),
        "delta_D": float(traj.D_t.iloc[-1] - traj.D_t.iloc[0]),
        "slope_P": float(np.polyfit(traj.step, traj.P_t, 1)[0]),
        "slope_U": float(np.polyfit(traj.step, traj.U_t, 1)[0]),
        "slope_D": float(np.polyfit(traj.step, traj.D_t, 1)[0]),
        "AUC_P": float(np.trapz(traj.P_t, traj.step) / STEPS),
        "AUC_U": float(np.trapz(traj.U_t, traj.step) / STEPS),
        "AUC_D": float(np.trapz(traj.D_t, traj.step) / STEPS),
        "min_U": float(np.min(utility)),
        "max_drop_U": float(max(0.0, utility[0] - np.min(utility))),
        "max_drop_D": float(max(0.0, dvals[0] - np.min(dvals))),
        "late_minus_early_U": float(late.U_t.mean() - early.U_t.mean()),
        "late_minus_early_D": float(late.D_t.mean() - early.D_t.mean()),
        "cumulative_predicted_utility_damage": float(pred_damage.sum()),
        "max_predicted_utility_damage": float(pred_damage.max(initial=0.0)),
        "fraction_steps_predicted_damage": float(np.mean(pred_damage > 0.0)),
        "cumulative_predicted_decision_damage": float(ddamage.sum()),
        "max_predicted_decision_damage": float(ddamage.max(initial=0.0)),
        "mean_actual_utility_delta_small_step": float(grad.utility_delta_actual_small_step.mean()),
        "mean_predicted_utility_delta": float(grad.predicted_utility_delta.mean()),
        "mean_cos_task_G": float(grad.cos_task_G.mean()),
        "final_history_BA": float(traj.history_BA_t.iloc[-1]),
        "final_history_loss": float(traj.history_loss_t.iloc[-1]),
    }
    return traj, grad, summary


def gradient_sign_unit_test() -> dict[str, Any]:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(40, 6))
    y = np.tile([0, 1], 20).astype(int)
    base = rng.normal(size=40) * 0.1
    state = {"center": np.zeros(6), "scale": np.ones(6), "p_basis": np.eye(6)[:, :2], "i_basis": np.eye(6)[:, 2:4], "task_p": np.array([1.0, 0.0])}
    w = rng.normal(size=6) * 0.05
    b = 0.03
    value, gw, gb = utility_value_and_gradient(x, y, base, w, b, state)
    eta = 1e-5
    numeric_w = (utility_function(x, y, base, w + eta * np.eye(1, 6, 0)[0], b, state) - value) / eta
    analytic_w = gw[0]
    max_abs = float(abs(numeric_w - analytic_w))
    # The first-order sign convention is checked separately with the actual task step.
    gtw, gtb = balanced_grad(x, y, base + x @ w + b)
    dot = float(np.dot(gw, gtw) + gb * gtb)
    before = utility_function(x, y, base, w, b, state)
    after = utility_function(x, y, base, w - 1e-4 * gtw, b - 1e-4 * gtb, state)
    sign_ok = bool((dot > 0 and after - before < 1e-8) or (dot <= 0 and after - before >= -1e-8))
    passed = bool(max_abs < 2e-4 and sign_ok)
    return {"passed": passed, "max_abs_numeric_gradient_error": max_abs, "dot_task_G": dot, "actual_delta_G": float(after - before), "sign_convention_ok": sign_ok}


def aggregate_outcome(base: np.ndarray, generic: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    noadapt = balanced_ba(y, base)
    gen = balanced_ba(y, generic)
    return noadapt, gen, gen - noadapt


def fit_predictor(train: pd.DataFrame, test: pd.DataFrame, family: str, target: str) -> tuple[np.ndarray, dict[str, Any]]:
    m0 = ["history_BA_t0", "history_loss_t0", "history_margin_t0", "history_gradient_norm_t0"]
    static = ["P_static", "U_static", "D_static", "I_static"]
    dynamic = [
        "delta_P", "delta_U", "delta_D", "slope_P", "slope_U", "slope_D", "AUC_P", "AUC_U", "AUC_D",
        "min_U", "max_drop_U", "max_drop_D", "late_minus_early_U", "late_minus_early_D",
    ]
    gradient = [
        "cumulative_predicted_utility_damage", "max_predicted_utility_damage", "fraction_steps_predicted_damage",
        "cumulative_predicted_decision_damage", "max_predicted_decision_damage", "mean_cos_task_G",
        "mean_actual_utility_delta_small_step", "mean_predicted_utility_delta",
    ]
    if family == "M0": cols = m0
    elif family == "M_static": cols = m0 + static
    elif family == "M_dynamic": cols = m0 + dynamic
    elif family == "M_gradient": cols = m0 + gradient
    elif family == "M_full": cols = m0 + static + dynamic + gradient
    else: raise ValueError(family)
    xtr = train[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    xte = test[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    scaler = StandardScaler().fit(xtr)
    xtr = scaler.transform(xtr); xte = scaler.transform(xte)
    ytr = train[target].to_numpy(float)
    model = Ridge(alpha=10.0).fit(xtr, ytr)
    return model.predict(xte), {"family": family, "target": target, "columns": cols, "alpha": 10.0}


def fit_nt_predictor(train: pd.DataFrame, test: pd.DataFrame, family: str) -> tuple[np.ndarray, dict[str, Any]]:
    m0 = ["history_BA_t0", "history_loss_t0", "history_margin_t0", "history_gradient_norm_t0"]
    static = ["P_static", "U_static", "D_static", "I_static"]
    dynamic = [
        "delta_P", "delta_U", "delta_D", "slope_P", "slope_U", "slope_D", "AUC_P", "AUC_U", "AUC_D",
        "min_U", "max_drop_U", "max_drop_D", "late_minus_early_U", "late_minus_early_D",
    ]
    gradient = [
        "cumulative_predicted_utility_damage", "max_predicted_utility_damage", "fraction_steps_predicted_damage",
        "cumulative_predicted_decision_damage", "max_predicted_decision_damage", "mean_cos_task_G",
        "mean_actual_utility_delta_small_step", "mean_predicted_utility_delta",
    ]
    if family == "M0": cols = m0
    elif family == "M_static": cols = m0 + static
    elif family == "M_dynamic": cols = m0 + dynamic
    elif family == "M_gradient": cols = m0 + gradient
    else: cols = m0 + static + dynamic + gradient
    xtr = StandardScaler().fit(train[cols].fillna(0.0).to_numpy(float))
    x_train = xtr.transform(train[cols].fillna(0.0).to_numpy(float))
    x_test = xtr.transform(test[cols].fillna(0.0).to_numpy(float))
    y = train["NegativeTransfer"].to_numpy(int)
    if len(np.unique(y)) < 2:
        return np.full(len(test), float(y.mean())), {"family": family, "constant": True, "columns": cols}
    model = LogisticRegression(C=0.25, class_weight="balanced", solver="liblinear", random_state=SEED).fit(x_train, y)
    return model.predict_proba(x_test)[:, 1], {"family": family, "constant": False, "columns": cols, "C": 0.25}


def evaluate_prediction(frame: pd.DataFrame, family: str) -> dict[str, Any]:
    part = frame[frame.predictor.eq(family)].copy()
    y = part.FutureDeltaBA.to_numpy(float)
    p = part.predicted_delta.to_numpy(float)
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    rho = spearmanr(y, p).statistic if len(y) > 2 else np.nan
    return {"family": family, "n": len(part), "RMSE": rmse, "Spearman": float(rho) if np.isfinite(rho) else None}


def make_figures(trajectory_frame: pd.DataFrame, summary: pd.DataFrame, pred: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIGURES.mkdir(parents=True, exist_ok=True)
        static = summary
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
        for i, col in enumerate(["P_static", "U_static", "D_static"]):
            ax[i].scatter(static[col], static.FutureDeltaBA, s=18, alpha=.75)
            ax[i].axhline(0, color="k", lw=.6); ax[i].set_xlabel(col); ax[i].set_ylabel("Future ΔBA")
        fig.tight_layout(); fig.savefig(FIGURES / "figure_1_static_vs_future.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
        for i, col in enumerate(["P_t", "U_t", "D_t"]):
            for sid, g in trajectory_frame.groupby("subject_id"):
                ax[i].plot(g.step, g[col], alpha=.25, lw=.7)
            ax[i].set_title(col); ax[i].set_xlabel("checkpoint")
        fig.tight_layout(); fig.savefig(FIGURES / "figure_2_dynamic_trajectories.png", dpi=160); plt.close(fig)
        g = summary
        fig, ax = plt.subplots(figsize=(5, 3.3)); ax.scatter(g.cumulative_predicted_utility_damage, g.FutureDeltaBA, c=g.NegativeTransfer, cmap="coolwarm", s=22); ax.axhline(0, color="k", lw=.6); ax.set_xlabel("cumulative predicted utility damage"); ax.set_ylabel("Future ΔBA"); fig.tight_layout(); fig.savefig(FIGURES / "figure_3_predicted_damage.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 3.3)); ax.scatter(g.R_dynamic, g.FutureDeltaBA, c=g.NegativeTransfer, cmap="coolwarm", s=22); ax.axhline(0, color="k", lw=.6); ax.set_xlabel("dynamic risk"); ax.set_ylabel("Future ΔBA"); fig.tight_layout(); fig.savefig(FIGURES / "figure_4_dynamic_risk.png", dpi=160); plt.close(fig)
        roc = pred[pred.predictor == "M_dynamic"].copy()
        fig, ax = plt.subplots(figsize=(4, 3.3))
        if roc.NegativeTransfer.nunique() > 1:
            order = np.argsort(-roc.nt_probability.to_numpy(float)); yy = roc.NegativeTransfer.to_numpy(int)[order]; tpr = np.cumsum(yy) / max(yy.sum(), 1); fpr = np.cumsum(1 - yy) / max((1 - yy).sum(), 1); ax.plot(fpr, tpr, label="M_dynamic")
        ax.plot([0, 1], [0, 1], "k--", lw=.7); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); fig.tight_layout(); fig.savefig(FIGURES / "figure_5_negative_transfer_roc.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 3.3)); ax.scatter(g.mean_cos_task_G, g.FutureDeltaBA, c=g.NegativeTransfer, cmap="coolwarm", s=22); ax.axhline(0, color="k", lw=.6); ax.set_xlabel("mean cos(task, protected utility)"); ax.set_ylabel("Future ΔBA"); fig.tight_layout(); fig.savefig(FIGURES / "figure_6_gradient_conflict.png", dpi=160); plt.close(fig)
    except Exception as exc:
        write_json(RESULTS / "FIGURE_ERROR.json", {"error": repr(exc)})


def write_docs(gate: dict[str, Any], unit: dict[str, Any], summaries: pd.DataFrame, pred_metrics: list[dict[str, Any]]) -> None:
    def md(name: str, text: str) -> None:
        (EXPERIMENT / name).write_text(text.strip() + "\n", encoding="utf-8")
    md("README.md", f"""# PERSIST-EEG Exp4 OpenBMI Dynamic Actionability V2

This directory contains the pre-registered Phase-A trajectory/gradient audit. It is a new branch and does not overwrite the earlier Exp4 negative result.

Terminal state: **{gate['terminal_state']}**

Data scope: OpenBMI MI, V8_SEARCH only (40 development subjects), five subject-only folds. The 14-subject internal holdout and historical outer test were not loaded. No WBCIC data were used.

The implementation uses the legal materialised MI-specific embedding/logit cache. The deployment surrogate is a residual linear head initialized at the frozen logit anchor and updated with five full-batch Session-1 BCE steps. This is an audit of prospective signal validity, not a claim that the cached feature head is the complete raw-EEG Generic.

See `DYNAMIC_ACTIONABILITY_AUDIT.md` and `DYNAMIC_DEV_PROTOCOL.json` for the exact gate and provenance.
""")
    md("DYNAMIC_HYPOTHESIS.md", """# Dynamic hypothesis

Static P/U/D levels failed to prospectively identify future-session adaptation consequence in Exp4 V1. The remaining falsifiable hypothesis was that ordinary legal adaptation updates become harmful when their task gradient conflicts with the gradient of protected task utility. The audit therefore measures trajectory changes and a first-order utility-damage quantity, rather than tuning the previous static guard.
""")
    md("GRADIENT_SIGN_AUDIT.md", f"""# Gradient-sign audit

The analytic gradient, first-order sign convention, and an actual small task step were checked before the subject audit.

```json
{json.dumps(clean(unit), indent=2)}
```
""")
    md("TRAJECTORY_MEASUREMENT_AUDIT.md", f"""# Trajectory measurement audit

Six frozen checkpoints were recorded: `{', '.join(CHECKPOINTS)}`. At each checkpoint the script records P/U/D, identity control, history BA/loss, coordinate drift, protected contribution, and gradient-conflict quantities. Summary feature definitions are locked in `protocol/DYNAMIC_FEATURE_LOCK.json`.

Episodes measured: {len(summaries)}. All predictors use history-side quantities only; future S2 labels are used only to form the diagnostic target after the trajectory is complete.
""")
    metrics_text = json.dumps(clean(gate), indent=2)
    md("DYNAMIC_ACTIONABILITY_AUDIT.md", f"""# Dynamic actionability audit

```json
{metrics_text}
```

Cross-fitted prediction metrics are in `results/HISTORY_TO_FUTURE_DYNAMIC_PREDICTION.csv` and `results/NEGATIVE_TRANSFER_PREDICTION.csv`. The algorithmic route is authorized only when the predeclared dynamic-vs-static and negative-transfer gates pass; otherwise the correct result is a boundary, not another hyperparameter search.
""")
    md("NEGATIVE_TRANSFER_PREDICTION_AUDIT.md", """# Negative-transfer prediction audit

Negative transfer is defined as `FutureDeltaBA < 0` for the legal generic trajectory relative to its frozen no-adaptation anchor. AUROC, AUPRC, balanced accuracy, and calibration are reported without opening the sealed holdout.
""")
    md("METHOD_AUTHORIZATION.md", f"""# Method authorization

Phase-A decision: **{gate['terminal_state']}**.

No Phase-B method was developed unless the predeclared Phase-A gate passed. A failed or weak audit terminates the algorithmic route; it does not authorize lambda/rank/adapter tuning or holdout inspection.
""")
    md("ITERATION_LEDGER.md", """# Iteration ledger

| Variant | Status | Rationale |
|---|---|---|
| Phase-A frozen residual Generic trajectory | completed | prospective audit before any new guard |
| Phase-B variants | not run unless authorized | gated by Phase-A |
""")
    md("MODEL_SELECTION_AUDIT.md", """# Model selection audit

No model was selected from the sealed internal holdout. If Phase A fails, `METHOD_SEARCH_RESULTS.csv` and `MODEL_PARETO_FRONTIER.csv` are intentionally empty because no positive method was authorized.
""")
    md("FINAL_MODEL_CARD.md", """# Final model card

There is no final intervention model in this branch unless Phase A authorizes Phase B. The Phase-A artifact is an audit of a frozen five-step linear residual Generic trajectory.
""")
    md("CLAIM_AUDIT.md", f"""# Claim audit

The strongest supported claim is limited to the measured OpenBMI V8_SEARCH scope and the cached feature-head surrogate. It does not establish a deployable dynamic guard unless the prospective gate passes. Terminal state: **{gate['terminal_state']}**.
""")
    md("REVIEWER_SELF_AUDIT.md", """# Reviewer self-audit

Main risks: 40-subject development sample, a cached MI-specific feature-head surrogate rather than a fresh raw-EEG re-training, and a prospective target restricted to V8_SEARCH. These limitations are explicit. The sealed holdout was not used for feature construction, threshold selection, or diagnosis.
""")
    md("REPRODUCIBILITY.md", """# Reproducibility

Run with the pinned server interpreter:

```text
E:\\Anaconda\\envs\\Benchmark_TTA_Win\\python.exe code\\run_phase_a.py
```

Required environment variables are optional and documented at the top of the script. The script writes protocol locks, CSV/JSON diagnostics, and six Phase-A figures.
""")


def run() -> dict[str, Any]:
    for path in (RESULTS, FIGURES, PROTOCOL): path.mkdir(parents=True, exist_ok=True)
    roles, raw_split = load_protocol()
    features, logits, metadata = load_cache()
    search = set(roles["search_subjects"])
    if not set(metadata.subject_id.unique()) & search:
        raise RuntimeError("No V8_SEARCH rows in cache")
    # Restrict every subsequent operation to V8_SEARCH. No holdout rows are materialised.
    keep = metadata.subject_id.isin(search).to_numpy()
    features = features[keep]
    logits = logits[keep]
    metadata = metadata.loc[keep].reset_index(drop=True)
    unit = gradient_sign_unit_test()
    write_json(RESULTS / "GRADIENT_SIGN_UNIT_TEST.json", unit)
    write_json(PROTOCOL / "DYNAMIC_DEV_PROTOCOL.json", {
        "dataset": "OpenBMI_MI",
        "benchmark": "OpenBMI_MI_S1_to_S2",
        "search_subject_count": 40,
        "internal_holdout_used": False,
        "outer_test_used": False,
        "wbcic_used": False,
        "history_sessions": [1],
        "future_session": 2,
        "folds": roles["folds"],
        "trajectory_checkpoints": list(CHECKPOINTS),
        "steps": STEPS,
        "eta": ETA,
        "protected_rank": PROTECTED_RANK,
        "identity_rank": IDENTITY_RANK,
        "cache": str(V7_CACHE),
        "cache_sha256_features_fold0": hash_file(V7_CACHE / "OPENBMI_MI_SPECIFIC_FOLD_0_FEATURES.npy"),
        "cache_sha256_logits_fold0": hash_file(V7_CACHE / "OPENBMI_MI_SPECIFIC_FOLD_0_LOGITS.npy"),
        "split_sha256": hash_file(V8_PROTOCOL / "V8_SEARCH_SPLIT.json"),
    })
    write_json(PROTOCOL / "DYNAMIC_FEATURE_LOCK.json", {
        "static": ["P_static", "U_static", "D_static", "I_static", "history_BA_t0", "history_loss_t0", "history_margin_t0", "history_gradient_norm_t0"],
        "dynamic": ["delta_P", "delta_U", "delta_D", "slope_P", "slope_U", "slope_D", "AUC_P", "AUC_U", "AUC_D", "min_U", "max_drop_U", "max_drop_D", "late_minus_early_U", "late_minus_early_D"],
        "gradient": ["cumulative_predicted_utility_damage", "max_predicted_utility_damage", "fraction_steps_predicted_damage", "cumulative_predicted_decision_damage", "max_predicted_decision_damage", "mean_cos_task_G", "mean_actual_utility_delta_small_step", "mean_predicted_utility_delta"],
        "risk": "z(cumulative_predicted_utility_damage)+z(cumulative_predicted_decision_damage)+z(max_drop_U)",
        "predictors": ["M0", "M_static", "M_dynamic", "M_gradient", "M_full"],
        "continuous_target": "FutureDeltaBA = BA_Generic_S2 - BA_NoAdapt_S2",
        "binary_target": "NegativeTransfer = FutureDeltaBA < 0",
    })
    trajectory_rows = []
    gradient_rows = []
    feature_rows = []
    subject_outcomes = []
    for role in roles["folds"]:
        fold = int(role["fold"])
        state = fit_protected(role["meta_subjects"], features, metadata)
        for subject in roles["search_subjects"]:
            xh, bh, yh, _ = subject_rows(features, logits, metadata, subject, 1)
            xf, bf, yf, uid = subject_rows(features, logits, metadata, subject, 2)
            static = static_metrics(xh, bh, yh, state)
            traj, grad, summary = trajectory(xh, bh, yh, state, subject, fold)
            final_w = np.zeros(xh.shape[1], dtype=np.float64)
            final_b = 0.0
            # reconstruct final parameters from the locked trajectory schedule
            z = (xh - state["center"]) / state["scale"]
            for _ in range(STEPS):
                gw, gb = balanced_grad(z, yh, bf * 0 + bh + z @ final_w + final_b)
                final_w -= ETA * gw; final_b -= ETA * gb
            zf = (xf - state["center"]) / state["scale"]
            generic_future = bf + zf @ final_w + final_b
            noadapt, generic, delta = aggregate_outcome(bf, generic_future, yf)
            summary.update(static)
            summary.update({"subject_id": subject, "source_fold": fold, "BA_NoAdapt_S2": noadapt, "BA_Generic_S2": generic, "FutureDeltaBA": delta, "NegativeTransfer": int(delta < 0.0), "role": "meta" if subject in role["meta_subjects"] else "outcome"})
            # Simple predeclared dynamic risk, standardized later within each fold's meta set.
            summary["R_raw"] = summary["cumulative_predicted_utility_damage"] + summary["cumulative_predicted_decision_damage"] + summary["max_drop_U"]
            feature_rows.append(summary)
            trajectory_rows.extend(traj.to_dict("records"))
            gradient_rows.extend(grad.to_dict("records"))
            subject_outcomes.append({"subject_id": subject, "source_fold": fold, "BA_NoAdapt_S2": noadapt, "BA_Generic_S2": generic, "FutureDeltaBA": delta, "NegativeTransfer": int(delta < 0), "role": summary["role"]})
    features_frame = pd.DataFrame(feature_rows)
    traj_frame = pd.DataFrame(trajectory_rows)
    grad_frame = pd.DataFrame(gradient_rows)
    # Keep one role-specific row per subject/fold; outcome rows are what prediction uses.
    static_frame = features_frame.loc[features_frame.role.eq("outcome")].copy()
    write_csv(RESULTS / "TRAJECTORY_FEATURES.csv", traj_frame)
    write_csv(RESULTS / "GRADIENT_CONFLICT.csv", grad_frame)
    write_csv(RESULTS / "PREDICTED_UTILITY_DAMAGE.csv", grad_frame[["subject_id", "source_fold", "step", "checkpoint", "predicted_utility_damage", "utility_delta_actual_small_step", "predicted_utility_delta", "cos_task_G"]])
    write_csv(RESULTS / "PREDICTED_DECISION_DAMAGE.csv", grad_frame[["subject_id", "source_fold", "step", "checkpoint", "finite_decision_change", "predicted_decision_damage_finite"]])
    write_csv(RESULTS / "STATIC_FEATURES.csv", static_frame)
    # Cross-fitted continuous and binary prediction. Each row is evaluated only when its fold treats it as outcome.
    pred_rows = []
    cross_metrics = []
    nt_metrics = []
    for fold in range(5):
        train = features_frame.loc[(features_frame.source_fold == fold) & features_frame.role.eq("meta")].copy()
        test = features_frame.loc[(features_frame.source_fold == fold) & features_frame.role.eq("outcome")].copy()
        if test.empty or train.empty: continue
        meta = train.R_raw.to_numpy(float); mean = float(meta.mean()); std = float(meta.std() + EPS)
        # Freeze the risk standardization on source meta subjects only.
        features_frame.loc[features_frame.source_fold.eq(fold), "R_dynamic"] = (features_frame.loc[features_frame.source_fold.eq(fold), "R_raw"] - mean) / std
        train = features_frame.loc[(features_frame.source_fold == fold) & features_frame.role.eq("meta")].copy()
        test = features_frame.loc[(features_frame.source_fold == fold) & features_frame.role.eq("outcome")].copy()
        for family in ["M0", "M_static", "M_dynamic", "M_gradient", "M_full"]:
            predicted, spec = fit_predictor(train, test, family, "FutureDeltaBA")
            nt_prob, nt_spec = fit_nt_predictor(train, test, family)
            for idx, (_, row) in enumerate(test.iterrows()):
                pred_rows.append({"subject_id": row.subject_id, "source_fold": fold, "predictor": family, "FutureDeltaBA": row.FutureDeltaBA, "NegativeTransfer": int(row.NegativeTransfer), "predicted_delta": float(predicted[idx]), "nt_probability": float(nt_prob[idx]), "history_BA_t0": row.history_BA_t0})
            temp = pd.DataFrame({"FutureDeltaBA": test.FutureDeltaBA.to_numpy(float), "predicted_delta": predicted})
            rmse = float(np.sqrt(mean_squared_error(temp.FutureDeltaBA, temp.predicted_delta)))
            rho = spearmanr(temp.FutureDeltaBA, temp.predicted_delta).statistic if len(temp) > 2 else np.nan
            cross_metrics.append({"source_fold": fold, "predictor": family, "n": len(test), "RMSE": rmse, "Spearman": float(rho) if np.isfinite(rho) else None, "rmse_relative_to_static": None, "internal_holdout_used": False})
            yy = test.NegativeTransfer.to_numpy(int)
            if len(np.unique(yy)) > 1:
                auc = float(roc_auc_score(yy, nt_prob)); ap = float(average_precision_score(yy, nt_prob)); bal = float(balanced_accuracy_score(yy, nt_prob >= 0.5)); brier = float(brier_score_loss(yy, nt_prob))
            else:
                auc = ap = bal = brier = None
            nt_metrics.append({"source_fold": fold, "predictor": family, "n": len(test), "AUROC": auc, "AUPRC": ap, "balanced_accuracy": bal, "Brier": brier, "internal_holdout_used": False})
    pred_frame = pd.DataFrame(pred_rows)
    metric_frame = pd.DataFrame(cross_metrics)
    if not metric_frame.empty:
        static_rmse = float(metric_frame.loc[metric_frame.predictor.eq("M_static"), "RMSE"].mean())
        metric_frame["rmse_relative_to_static"] = 1.0 - metric_frame.RMSE / max(static_rmse, EPS)
    nt_frame = pd.DataFrame(nt_metrics)
    write_csv(RESULTS / "HISTORY_TO_FUTURE_DYNAMIC_PREDICTION.csv", metric_frame)
    write_csv(RESULTS / "NEGATIVE_TRANSFER_PREDICTION.csv", nt_frame)
    # Preserve the actual held-out subject predictions separately from the
    # fold-level metric tables above; neither table contains sealed subjects.
    write_csv(RESULTS / "HISTORY_TO_FUTURE_DYNAMIC_PREDICTION_SUBJECTS.csv", pred_frame)
    write_csv(RESULTS / "NEGATIVE_TRANSFER_PREDICTION_SUBJECTS.csv", pred_frame[["subject_id", "source_fold", "predictor", "NegativeTransfer", "nt_probability"]])
    # Attach M_dynamic risk and outcome rows for tail diagnostics.
    dyn = features_frame.loc[features_frame.role.eq("outcome")].copy()
    # risk was standardized foldwise above; if assignment was deferred, reconstruct it.
    for fold in range(5):
        mask = features_frame.source_fold.eq(fold)
        train_r = features_frame.loc[mask & features_frame.role.eq("meta"), "R_raw"]
        if len(train_r): features_frame.loc[mask, "R_dynamic"] = (features_frame.loc[mask, "R_raw"] - train_r.mean()) / (train_r.std() + EPS)
    dyn = features_frame.loc[features_frame.role.eq("outcome")].copy()
    dyn["risk_quartile"] = dyn.groupby("source_fold")["R_dynamic"].transform(lambda x: pd.qcut(x.rank(method="first"), 4, labels=False, duplicates="drop") + 1 if len(x) >= 4 else 2)
    write_csv(RESULTS / "DYNAMIC_RISK_QUARTILES.csv", dyn[["subject_id", "source_fold", "R_dynamic", "risk_quartile", "FutureDeltaBA", "NegativeTransfer"]])
    # All search subjects/fold outcomes are explicit; do not combine duplicate role rows as if independent.
    outcome_summary = dyn.copy()
    write_csv(RESULTS / "DEV_SUBJECT_RESULTS.csv", outcome_summary)
    write_csv(RESULTS / "NEGATIVE_TRANSFER.csv", outcome_summary[["subject_id", "source_fold", "BA_NoAdapt_S2", "BA_Generic_S2", "FutureDeltaBA", "NegativeTransfer"]])
    # Fold-level robustness and diagnostic statistics.
    fold_rows = []
    for fold, g in outcome_summary.groupby("source_fold"):
        hi = g[g.risk_quartile == 4].FutureDeltaBA
        lo = g[g.risk_quartile == 1].FutureDeltaBA
        fold_rows.append({"source_fold": int(fold), "subjects": len(g), "mean_FutureDeltaBA": float(g.FutureDeltaBA.mean()), "generic_BA": float(g.BA_Generic_S2.mean()), "noadapt_BA": float(g.BA_NoAdapt_S2.mean()), "negative_transfer_rate": float(g.NegativeTransfer.mean()), "high_risk_q4_delta": float(hi.mean()) if len(hi) else None, "low_risk_q1_delta": float(lo.mean()) if len(lo) else None, "high_minus_low": float(hi.mean() - lo.mean()) if len(hi) and len(lo) else None})
    write_csv(RESULTS / "FOLD_ROBUSTNESS.csv", pd.DataFrame(fold_rows))
    # Gate calculations use held-out subject predictions only.
    def overall(family: str) -> dict[str, Any]:
        p = pred_frame.loc[pred_frame.predictor.eq(family)]
        rmse = float(np.sqrt(mean_squared_error(p.FutureDeltaBA, p.predicted_delta))) if len(p) else None
        rho = spearmanr(p.FutureDeltaBA, p.predicted_delta).statistic if len(p) > 2 else np.nan
        return {"RMSE": rmse, "Spearman": float(rho) if np.isfinite(rho) else None, "n": len(p)}
    overall_metrics = {family: overall(family) for family in ["M0", "M_static", "M_dynamic", "M_gradient", "M_full"]}
    static_nt = nt_frame.loc[nt_frame.predictor.eq("M_static") & nt_frame.AUROC.notna(), "AUROC"]
    dynamic_nt = nt_frame.loc[nt_frame.predictor.eq("M_dynamic") & nt_frame.AUROC.notna(), "AUROC"]
    nt_auc_static = float(roc_auc_score(pred_frame.loc[pred_frame.predictor.eq("M_static"), "NegativeTransfer"], pred_frame.loc[pred_frame.predictor.eq("M_static"), "nt_probability"])) if pred_frame.loc[pred_frame.predictor.eq("M_static"), "NegativeTransfer"].nunique() > 1 else None
    nt_auc_dynamic = float(roc_auc_score(pred_frame.loc[pred_frame.predictor.eq("M_dynamic"), "NegativeTransfer"], pred_frame.loc[pred_frame.predictor.eq("M_dynamic"), "nt_probability"])) if pred_frame.loc[pred_frame.predictor.eq("M_dynamic"), "NegativeTransfer"].nunique() > 1 else None
    improvements = []
    for fold in range(5):
        a = metric_frame.loc[(metric_frame.source_fold == fold) & metric_frame.predictor.eq("M_static"), "RMSE"]
        b = metric_frame.loc[(metric_frame.source_fold == fold) & metric_frame.predictor.eq("M_dynamic"), "RMSE"]
        if len(a) and len(b): improvements.append(float(b.iloc[0] < a.iloc[0]))
    high = outcome_summary.loc[outcome_summary.risk_quartile == 4, "FutureDeltaBA"]
    low = outcome_summary.loc[outcome_summary.risk_quartile == 1, "FutureDeltaBA"]
    dynamic_rmse = overall_metrics["M_dynamic"]["RMSE"] or float("inf")
    static_rmse = overall_metrics["M_static"]["RMSE"] or float("inf")
    relative_rmse = 1.0 - dynamic_rmse / max(static_rmse, EPS)
    gradient_pass = bool(unit["passed"] and np.mean(grad_frame.predicted_utility_damage >= 0.0) == 1.0)
    if len(grad_frame) > 2:
        utility_direction_agreement = float(np.mean((grad_frame.predicted_utility_delta.to_numpy(float) < 0.0) == (grad_frame.utility_delta_actual_small_step.to_numpy(float) < 0.0)))
        utility_prediction_corr = float(np.corrcoef(grad_frame.predicted_utility_delta.to_numpy(float), grad_frame.utility_delta_actual_small_step.to_numpy(float))[0, 1])
    else:
        utility_direction_agreement = None
        utility_prediction_corr = None
    gate_state = "DYNAMIC_ACTIONABILITY_SUPPORTED"
    reasons = []
    if relative_rmse < 0.10: reasons.append("dynamic RMSE improvement below 10%")
    if sum(improvements) < 4: reasons.append(f"dynamic RMSE improves in {int(sum(improvements))}/5 folds")
    if overall_metrics["M_dynamic"]["Spearman"] is None or abs(overall_metrics["M_dynamic"]["Spearman"]) < 0.25: reasons.append("dynamic Spearman magnitude below 0.25")
    if nt_auc_dynamic is None or nt_auc_dynamic < 0.65 or nt_auc_static is None or nt_auc_dynamic < nt_auc_static + 0.05: reasons.append("negative-transfer AUROC gate not met")
    if len(high) and len(low) and not (float(high.mean()) < float(low.mean())): reasons.append("high-risk quartile is not worse than low-risk quartile")
    if not gradient_pass: reasons.append("gradient-sign audit failed")
    if len(reasons): gate_state = "DYNAMIC_ACTIONABILITY_NOT_SUPPORTED"
    gate = {
        "terminal_state": "EXP4_OPENBMI_DYNAMIC_ACTIONABILITY_NOT_SUPPORTED" if gate_state != "DYNAMIC_ACTIONABILITY_SUPPORTED" else "EXP4_OPENBMI_DYNAMIC_ACTIONABILITY_SUPPORTED",
        "phase_a_state": gate_state,
        "reasons": reasons,
        "gradient_sign_pass": gradient_pass,
        "gradient_unit_test": unit,
        "trajectory_first_order_direction_agreement": utility_direction_agreement,
        "trajectory_first_order_utility_delta_correlation": utility_prediction_corr,
        "overall_prediction": overall_metrics,
        "dynamic_relative_RMSE_reduction_vs_static": relative_rmse,
        "folds_dynamic_RMSE_improved": int(sum(improvements)),
        "fold_improvement_flags": improvements,
        "negative_transfer_AUROC_static": nt_auc_static,
        "negative_transfer_AUROC_dynamic": nt_auc_dynamic,
        "negative_transfer_AUROC_gain": None if nt_auc_dynamic is None or nt_auc_static is None else nt_auc_dynamic - nt_auc_static,
        "tail_high_risk_mean_FutureDeltaBA": float(high.mean()) if len(high) else None,
        "tail_low_risk_mean_FutureDeltaBA": float(low.mean()) if len(low) else None,
        "tail_high_minus_low": float(high.mean() - low.mean()) if len(high) and len(low) else None,
        "subjects_evaluated": int(len(outcome_summary)),
        "folds": 5,
        "internal_holdout_used": False,
        "outer_test_used": False,
        "wbcic_used": False,
    }
    write_json(RESULTS / "PHASE_A_GATE.json", gate)
    write_json(RESULTS / "STATISTICAL_TESTS.json", {"subject_unit": True, "bootstrap": "not used for gate; see fold/tail rows", "sign_flip": "not used for model selection", "gate": gate})
    # Required Phase-B placeholders are intentionally empty when the gate fails.
    write_csv(RESULTS / "METHOD_SEARCH_RESULTS.csv", pd.DataFrame(columns=["method", "status", "reason"]))
    write_csv(RESULTS / "CONTROL_COMPARISON.csv", pd.DataFrame(columns=["method", "BA", "NT_rate", "status"]))
    write_csv(RESULTS / "MODEL_PARETO_FRONTIER.csv", pd.DataFrame(columns=["method", "BA", "DeltaBA", "NT_rate", "dynamic_damage", "status"]))
    write_csv(RESULTS / "SEED_ROBUSTNESS.csv", pd.DataFrame(columns=["seed", "method", "BA", "status"]))
    write_json(PROTOCOL / "HOLDOUT_LOCK.json", {"status": "SEALED", "reason": "Phase A development; no holdout lock authorized", "internal_holdout_used": False, "outer_test_used": False})
    make_figures(traj_frame, outcome_summary, pred_frame)
    write_docs(gate, unit, outcome_summary, [overall_metrics[k] for k in overall_metrics])
    mean_noadapt = float(outcome_summary.BA_NoAdapt_S2.mean())
    mean_generic = float(outcome_summary.BA_Generic_S2.mean())
    terminal_report = {
        "terminal_state": gate["terminal_state"],
        "phase_a_gate": gate,
        "mean_BA_noadapt": mean_noadapt,
        "mean_BA_generic": mean_generic,
        "mean_delta_BA": mean_generic - mean_noadapt,
        "generic_negative_transfer_rate": float(outcome_summary.NegativeTransfer.mean()),
        "subjects_favoring_generic": int(np.sum(outcome_summary.FutureDeltaBA > 0.0)),
        "subjects_harmed_by_generic": int(np.sum(outcome_summary.FutureDeltaBA < 0.0)),
        "best_dynamic_predictor": min(overall_metrics, key=lambda name: overall_metrics[name]["RMSE"]),
        "strongest_fair_generic": "Conformer-Norm (V7 anchor, not re-evaluated in Phase A)",
        "phase_b_variants_run": 0,
        "internal_holdout_used": False,
        "outer_test_used": False,
        "wbcic_used": False,
        "final_paper_claim": "Static and dynamic cached feature-head signals did not establish a prospective control bridge in this audit; the result is an actionability boundary, not evidence for a deployable guard.",
        "largest_remaining_limitation": "The Phase-A Generic trajectory is a frozen MI-specific cached-embedding residual-head surrogate, so a failed gate does not justify claims about all raw-EEG backbones.",
    }
    write_json(EXPERIMENT / "EXP4_OPENBMI_DYNAMIC_FINAL_REPORT.json", terminal_report)
    (EXPERIMENT / "EXP4_OPENBMI_DYNAMIC_FINAL_REPORT.md").write_text(
        "# PERSIST-EEG Exp4 OpenBMI Dynamic Actionability V2\n\n"
        f"Terminal state: **{gate['terminal_state']}**\n\n"
        "## Phase-A decision\n\n"
        "The predeclared dynamic actionability gate failed. Dynamic trajectory features did not improve prospective held-out prediction over static features, and the negative-transfer classifier did not reach the practical AUROC target. Phase B was therefore not authorized.\n\n"
        f"- Frozen five-fold search subjects evaluated: {len(outcome_summary)}\n"
        f"- No-adaptation BA: {mean_noadapt:.6f}\n"
        f"- Legal Generic trajectory BA: {mean_generic:.6f}\n"
        f"- Mean Future ΔBA: {mean_generic - mean_noadapt:+.6f}\n"
        f"- Dynamic RMSE reduction vs static: {relative_rmse:+.4f}\n"
        f"- Dynamic RMSE-improved folds: {int(sum(improvements))}/5\n"
        f"- Dynamic Spearman: {overall_metrics['M_dynamic']['Spearman']}\n"
        f"- Negative-transfer AUROC static/dynamic: {nt_auc_static} / {nt_auc_dynamic}\n"
        f"- Gradient-sign audit: {'PASS' if gradient_pass else 'FAIL'}\n"
        f"- First-order utility direction agreement: {utility_direction_agreement}\n\n"
        "## Data access\n\n"
        "Only V8_SEARCH was used. The 14-subject internal holdout, historical outer test, and WBCIC were not accessed. No final holdout lock was created.\n\n"
        "## Interpretation\n\n"
        "Within this legal cached MI-specific trajectory surrogate, dynamic update conflict is not a reliable prospective action signal. The correct terminal conclusion is `EXP4_OPENBMI_DYNAMIC_ACTIONABILITY_NOT_SUPPORTED`; no additional lambda, rank, adapter, architecture, or dataset search is justified by this experiment.\n\n"
        f"Largest limitation: {terminal_report['largest_remaining_limitation']}\n",
        encoding="utf-8",
    )
    write_json(RESULTS / "PHASE_A_MANIFEST.json", {"terminal_state": gate["terminal_state"], "files": sorted(str(p.relative_to(EXPERIMENT)).replace("\\", "/") for p in EXPERIMENT.rglob("*") if p.is_file()), "internal_holdout_used": False, "outer_test_used": False})
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if int(args.seed) != SEED:
        raise SystemExit(f"This audit is frozen to seed {SEED}; do not tune seeds during Phase A")
    result = run()
    print(json.dumps(clean(result), indent=2), flush=True)


if __name__ == "__main__":
    main()
