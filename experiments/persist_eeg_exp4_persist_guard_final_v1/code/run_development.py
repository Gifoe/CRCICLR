"""Final bounded PERSIST-Guard development on the clean OpenBMI 40-subject split.

The runner never materialises the 14-subject internal holdout.  It reuses the
three-seed/five-fold EEGNet checkpoints trained by the repaired vanilla
baseline, reconstructs a source-only Generic target head, certifies a protected
P/U/D bank in the same 64-dimensional representation, and evaluates a bounded
set of risk-gated interpolation policies.  Session 1 is legal history and
Session 2 is outcome-only for each evaluated subject.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

VENDOR = Path(os.environ.get("PERSIST_PYARROW_VENDOR", r"D:\nips-temp\TotalP\P1\CRCICLR_V3_WORK\vendor"))
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
import pyarrow.parquet as pq

from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("PERSIST_GUARD_EXPERIMENT", ".")).resolve()
REPO = Path(os.environ.get("PERSIST_V8_RUNTIME", r"D:\nips-temp\TotalP\P1\CRCICLR_V8_HEADROOM_FIRST"))
V7_ROOT = Path(os.environ.get("PERSIST_V7_RUNTIME", r"D:\nips-temp\TotalP\P1\CRCICLR_V7_FUTURE_UTILITY_META"))
STAGE0_ROOT = Path(os.environ.get("PERSIST_STAGE0_REPO", r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full"))
BASE_EXP = REPO / "experiments" / "persist_eeg_openbmi_foldcorrect_baseline_and_dynamic_v3"

RESULTS = ROOT / "results"
PROTOCOL = ROOT / "protocol"
FIGURES = ROOT / "figures"
CACHE = ROOT / "cache"
SEEDS = (0, 1, 2)
ALPHAS = (0.0, 0.5, 0.75, 1.0)
GENERIC_CS = (0.1, 1.0, 10.0)
GENERIC_BETAS = (0.25, 0.5, 0.75, 1.0)
RISK_CS = (0.1, 1.0, 10.0)
TAUS = (0.35, 0.5, 0.65)
MAX_PROTECTED_RANK = 8
IDENTITY_RANK = 4
RNG_SEED = 20260824
EPS = 1e-10
BOOTSTRAPS = 10_000

GENERIC_COLS = [
    "history_CE", "history_BA", "history_margin", "history_entropy",
    "history_confidence", "update_norm", "gradient_norm",
    "history_improvement", "split_history_disagreement", "sample_count",
]
IDENTITY_COLS = ["identity_projection_norm", "identity_nearest_distance", "identity_confidence"]
P_COLS = ["persistence_magnitude", "persistence_split_stability"]
STATIC_PUD_COLS = ["persistence_magnitude", "protected_utility_t0", "decision_coupling_t0"]
PERSIST_COLS = [
    "protected_functional_retention", "cumulative_protected_damage",
    "maximum_protected_damage", "damaging_step_fraction",
    "decision_coupling_degradation", "protected_contribution_change",
    "task_protected_gradient_conflict", "task_protected_gradient_cosine",
    "protected_function_slope", "protected_split_disagreement",
]


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def balanced_bce(y: np.ndarray, logits: np.ndarray) -> float:
    p = np.clip(sigmoid(logits), 1e-7, 1.0 - 1e-7)
    losses = []
    for label in (0, 1):
        mask = y.astype(int) == label
        if mask.any():
            losses.append(float(-np.mean(y[mask] * np.log(p[mask]) + (1.0 - y[mask]) * np.log(1.0 - p[mask]))))
    return float(np.mean(losses))


def ba(y: np.ndarray, logits: np.ndarray) -> float:
    return float(balanced_accuracy_score(y.astype(int), (logits >= 0.0).astype(int)))


def classification_metrics_for_actions(
    outcome: pd.DataFrame,
    actions: np.ndarray,
    ensemble_logits: dict[tuple[int, str, float], np.ndarray],
    metadata: pd.DataFrame,
) -> tuple[float, float]:
    """Return subject-mean macro-F1 and accuracy for a fixed action vector."""
    macro_f1, accuracy = [], []
    for position, (_, row) in enumerate(outcome.iterrows()):
        alpha = float(actions[position])
        y = metadata.loc[subject_mask(metadata, str(row.subject_id), 2), "label"].to_numpy(int)
        prediction = (ensemble_logits[(int(row.fold), str(row.subject_id), alpha)] >= 0.0).astype(int)
        macro_f1.append(float(f1_score(y, prediction, average="macro", zero_division=0)))
        accuracy.append(float(accuracy_score(y, prediction)))
    return float(np.mean(macro_f1)), float(np.mean(accuracy))


def entropy_from_logits(logits: np.ndarray) -> float:
    p = np.clip(sigmoid(logits), 1e-7, 1.0 - 1e-7)
    return float(np.mean(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))))


def balanced_grad(z: np.ndarray, y: np.ndarray, logits: np.ndarray) -> tuple[np.ndarray, float]:
    residual = sigmoid(logits) - y
    gw, gb = [], []
    for label in (0, 1):
        mask = y.astype(int) == label
        gw.append((residual[mask, None] * z[mask]).mean(axis=0))
        gb.append(float(residual[mask].mean()))
    return 0.5 * (gw[0] + gw[1]), 0.5 * (gb[0] + gb[1])


class StandardEEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = nn.Conv2d(1, 8, (1, 64), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.spatial = nn.Conv2d(8, 16, (62, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.25)
        self.depth = nn.Conv2d(16, 16, (1, 16), padding="same", groups=16, bias=False)
        self.point = nn.Conv2d(16, 16, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.25)
        self.embedding = nn.Sequential(nn.Linear(16 * 31, 64), nn.ELU(), nn.LayerNorm(64))
        self.head = nn.Linear(64, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.bn1(self.temporal(x))
        x = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(x)))))
        x = self.drop2(self.pool2(F.elu(self.bn3(self.point(self.depth(x))))))
        return self.embedding(x.flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


@dataclass
class Protocol:
    metadata: pd.DataFrame
    raw: torch.Tensor
    folds: list[dict[str, Any]]
    search: list[str]
    holdout: list[str]
    device: torch.device


def load_protocol() -> Protocol:
    split_path = REPO / "experiments" / "persist_eeg_final_model_v8" / "outputs" / "protocol" / "V8_SEARCH_SPLIT.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if bool(split.get("OUTER_TEST_USED", True)):
        raise RuntimeError("V8 split is not sealed")
    search = sorted(map(str, split["openbmi"]["V8_SEARCH"]), key=lambda x: int(x))
    holdout = sorted(map(str, split["openbmi"]["V8_INTERNAL_HOLDOUT"]), key=lambda x: int(x))
    if len(search) != 40 or len(holdout) != 14 or set(search) & set(holdout):
        raise RuntimeError("malformed 40/14 split")
    freeze_path = STAGE0_ROOT / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json"
    frozen = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
    folds = []
    covered: set[str] = set()
    for fold, row in enumerate(frozen["openbmi"]["folds"]):
        train = sorted(set(map(str, row["train_subjects"])) & set(search), key=lambda x: int(x))
        validation = sorted(set(map(str, row["validation_subjects"])) & set(search), key=lambda x: int(x))
        outcome = sorted(set(map(str, row["outer_test_subjects"])) & set(search), key=lambda x: int(x))
        if set(train) & set(validation) or (set(train) | set(validation)) & set(outcome):
            raise RuntimeError(f"fold {fold} role overlap")
        covered |= set(outcome)
        folds.append({"fold": fold, "train_subjects": train, "validation_subjects": validation,
                      "meta_subjects": sorted(set(train) | set(validation), key=lambda x: int(x)),
                      "outcome_subjects": outcome})
    if covered != set(search):
        raise RuntimeError("outcome folds do not cover V8_SEARCH")
    metadata_path = V7_ROOT / "experiments" / "persist_eeg_final_model_v7" / "outputs" / "cache" / "OPENBMI_RAW_METADATA.parquet"
    metadata_all = pq.read_table(metadata_path).to_pandas()
    metadata_all["subject_id"] = metadata_all.subject_id.astype(str)
    metadata_all["session_id"] = metadata_all.session_id.astype(int)
    metadata_all["label"] = metadata_all.label.astype(int)
    keep = metadata_all.subject_id.isin(search).to_numpy()
    metadata = metadata_all.loc[keep].reset_index(drop=True)
    if len(metadata) != 8000 or not metadata.groupby(["subject_id", "session_id", "label"]).size().eq(50).all():
        raise RuntimeError("development metadata malformed")
    raw_path = V7_ROOT / "experiments" / "persist_eeg_final_model_v7" / "outputs" / "cache" / "OPENBMI_RAW_EPOCHS_FLOAT16.npy"
    raw_disk = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    if raw_disk.shape != (10800, 62, 1000):
        raise RuntimeError("raw cache shape mismatch")
    # This is the only materialisation: the boolean index excludes every holdout row first.
    raw_np = np.asarray(raw_disk[keep], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.from_numpy(raw_np).to(device, non_blocking=True)
    del raw_np, raw_disk, metadata_all
    return Protocol(metadata, raw, folds, search, holdout, device)


def checkpoint_path(seed: int, fold: int) -> Path:
    return BASE_EXP / "checkpoints" / "S1S2_SOURCE_TO_S2" / f"seed-{seed}" / f"fold-{fold}.pt"


def extract_representation(protocol: Protocol, seed: int, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    feature_path = CACHE / f"openbmi_clean_seed{seed}_fold{fold}_features.npy"
    logit_path = CACHE / f"openbmi_clean_seed{seed}_fold{fold}_logits.npy"
    head_path = CACHE / f"openbmi_clean_seed{seed}_fold{fold}_head.npz"
    ckpt_path = checkpoint_path(seed, fold)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if set(map(str, payload["train_subjects"])) & set(protocol.holdout) or set(map(str, payload["validation_subjects"])) & set(protocol.holdout):
        raise RuntimeError("clean checkpoint contains internal-holdout subjects")
    state = payload["model"]
    head_w = (state["head.weight"][1] - state["head.weight"][0]).detach().cpu().numpy().astype(np.float64)
    head_b = float((state["head.bias"][1] - state["head.bias"][0]).detach().cpu())
    if feature_path.is_file() and logit_path.is_file() and head_path.is_file():
        features = np.load(feature_path, allow_pickle=False)
        logits = np.load(logit_path, allow_pickle=False)
        saved = np.load(head_path, allow_pickle=False)
        return features.astype(np.float64), logits.astype(np.float64), saved["head_w"].astype(np.float64), float(saved["head_b"]), payload
    subjects = protocol.metadata.subject_id.to_numpy(str)
    sessions = protocol.metadata.session_id.to_numpy(int)
    train_mask = np.isin(subjects, list(map(str, payload["train_subjects"]))) & np.isin(sessions, [1, 2])
    train_idx = np.flatnonzero(train_mask)
    mean = protocol.raw[torch.as_tensor(train_idx, dtype=torch.long, device=protocol.device)].mean(dim=(0, 2))
    std = protocol.raw[torch.as_tensor(train_idx, dtype=torch.long, device=protocol.device)].std(dim=(0, 2), unbiased=False).clamp_min(1e-6)
    model = StandardEEGNet().to(protocol.device)
    model.load_state_dict(state)
    model.eval()
    features, logits = [], []
    with torch.inference_mode():
        for start in range(0, len(protocol.metadata), 512):
            xb = (protocol.raw[start:start + 512] - mean[None, :, None]) / std[None, :, None]
            h = model.forward_features(xb)
            z = model.head(h)
            features.append(h.detach().float().cpu().numpy())
            logits.append((z[:, 1] - z[:, 0]).detach().float().cpu().numpy())
    feature_array = np.concatenate(features).astype(np.float32)
    logit_array = np.concatenate(logits).astype(np.float32)
    np.save(feature_path, feature_array, allow_pickle=False)
    np.save(logit_path, logit_array, allow_pickle=False)
    np.savez(head_path, head_w=head_w, head_b=np.asarray(head_b))
    return feature_array.astype(np.float64), logit_array.astype(np.float64), head_w, head_b, payload


def subject_mask(metadata: pd.DataFrame, subject: str, session: int) -> np.ndarray:
    mask = metadata.subject_id.eq(str(subject)).to_numpy() & metadata.session_id.eq(int(session)).to_numpy()
    if int(mask.sum()) != 100:
        raise RuntimeError(f"expected 100 trials for {subject}/S{session}")
    return mask


def corr_or_zero(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else 0.0


def fit_protected_bank(meta_subjects: list[str], features: np.ndarray, logits: np.ndarray,
                       metadata: pd.DataFrame, seed: int, fold: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_mask = metadata.subject_id.isin(meta_subjects).to_numpy()
    center = features[source_mask].mean(axis=0)
    scale = np.maximum(features[source_mask].std(axis=0), 1e-5)
    z = (features - center) / scale
    source_y = metadata.loc[source_mask, "label"].to_numpy(int)
    task = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=RNG_SEED + seed * 10 + fold)
    task.fit(z[source_mask], source_y)
    task_w = task.coef_[0].astype(np.float64)
    task_b = float(task.intercept_[0])
    m1, m2, mall = [], [], []
    for subject in meta_subjects:
        s1 = subject_mask(metadata, subject, 1)
        s2 = subject_mask(metadata, subject, 2)
        m1.append(z[s1].mean(axis=0)); m2.append(z[s2].mean(axis=0)); mall.append(z[s1 | s2].mean(axis=0))
    m1 = np.stack(m1); m2 = np.stack(m2); mall = np.stack(mall)
    a = m1 - m1.mean(axis=0, keepdims=True)
    b = m2 - m2.mean(axis=0, keepdims=True)
    cross = (a.T @ b + b.T @ a) / max(2 * (len(meta_subjects) - 1), 1)
    values, vectors = np.linalg.eigh(cross)
    order = np.argsort(values)[::-1]
    future_mask = source_mask & metadata.session_id.eq(2).to_numpy()
    zf = z[future_mask]
    yf = metadata.loc[future_mask, "label"].to_numpy(int)
    full = zf @ task_w + task_b
    audit = []
    accepted = []
    for rank_order, index in enumerate(order[:32]):
        direction = vectors[:, index]
        persistence = corr_or_zero(m1 @ direction, m2 @ direction)
        contribution = (zf @ direction) * float(direction @ task_w)
        erased = full - contribution
        utility_loss = balanced_bce(yf, erased) - balanced_bce(yf, full)
        utility_ba = ba(yf, full) - ba(yf, erased)
        decision = float(np.mean((full >= 0.0) != (erased >= 0.0)))
        score = max(persistence, 0.0) * max(utility_loss, 0.0) * math.sqrt(max(decision, 0.0) + EPS)
        passed = bool(values[index] > 0.0 and persistence >= 0.05 and utility_loss > 1e-5 and decision > 0.0)
        audit.append({"seed": seed, "fold": fold, "candidate_order": rank_order, "eigenvalue": float(values[index]),
                      "P_cross_session": persistence, "U_signed_loss": utility_loss, "U_signed_BA": utility_ba,
                      "D_decision_flip": decision, "PUD_score": score, "passed": passed,
                      "identity_required": False, "source_subjects": len(meta_subjects)})
        if passed:
            accepted.append((score, rank_order, direction, persistence, utility_loss, decision))
    accepted.sort(key=lambda row: (-row[0], row[1]))
    selected = accepted[:MAX_PROTECTED_RANK]
    p_basis = np.stack([row[2] for row in selected], axis=1) if selected else np.zeros((features.shape[1], 0), dtype=np.float64)
    p_scores = np.asarray([row[3] for row in selected], dtype=np.float64)
    identity_matrix = mall - mall.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(identity_matrix, full_matrices=False)
    i_basis = vt[:min(IDENTITY_RANK, len(vt))].T
    id_centroids = mall @ i_basis
    bank = {"center": center, "scale": scale, "p_basis": p_basis, "p_scores": p_scores,
            "task_w": task_w, "task_b": task_b, "i_basis": i_basis, "id_centroids": id_centroids,
            "protected_rank": int(p_basis.shape[1]), "meta_subjects": list(meta_subjects)}
    return bank, audit


def population_theta(head_w: np.ndarray, head_b: float, bank: dict[str, Any]) -> tuple[np.ndarray, float]:
    w = head_w * bank["scale"]
    b = float(head_b + bank["center"] @ head_w)
    return w, b


def target_theta(z_history: np.ndarray, y_history: np.ndarray, c: float, seed: int) -> tuple[np.ndarray, float]:
    model = LogisticRegression(C=float(c), class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed)
    model.fit(z_history, y_history)
    return model.coef_[0].astype(np.float64), float(model.intercept_[0])


def select_generic(meta_subjects: list[str], features: np.ndarray, logits: np.ndarray, metadata: pd.DataFrame,
                   bank: dict[str, Any], head_w: np.ndarray, head_b: float, seed: int, fold: int) -> dict[str, Any]:
    z = (features - bank["center"]) / bank["scale"]
    w0, b0 = population_theta(head_w, head_b, bank)
    rows = []
    for c in GENERIC_CS:
        target_heads = {}
        for subject in meta_subjects:
            s1 = subject_mask(metadata, subject, 1)
            target_heads[subject] = target_theta(z[s1], metadata.loc[s1, "label"].to_numpy(int), c, RNG_SEED + seed * 100 + fold)
        for beta in GENERIC_BETAS:
            bas, gens, deltas = [], [], []
            for subject in meta_subjects:
                s2 = subject_mask(metadata, subject, 2)
                y = metadata.loc[s2, "label"].to_numpy(int)
                wt, bt = target_heads[subject]
                wg, bg = (1.0 - beta) * w0 + beta * wt, (1.0 - beta) * b0 + beta * bt
                b0_s, g_s = ba(y, z[s2] @ w0 + b0), ba(y, z[s2] @ wg + bg)
                bas.append(b0_s); gens.append(g_s); deltas.append(g_s - b0_s)
            rows.append({"seed": seed, "fold": fold, "C": c, "beta": beta, "source_subjects": len(meta_subjects),
                         "source_noadapt_BA": float(np.mean(bas)), "source_generic_BA": float(np.mean(gens)),
                         "source_delta": float(np.mean(deltas)), "source_negative_transfer_rate": float(np.mean(np.asarray(deltas) < 0.0))})
    chosen = max(rows, key=lambda row: (row["source_generic_BA"], -row["source_negative_transfer_rate"], -row["beta"], -row["C"]))
    return {"chosen": chosen, "candidates": rows}


def utility_and_gradient(z: np.ndarray, y: np.ndarray, w: np.ndarray, b: float,
                         p_basis: np.ndarray) -> tuple[float, np.ndarray, float, float, float]:
    full = z @ w + b
    if p_basis.shape[1] == 0:
        return 0.0, np.zeros_like(w), 0.0, 0.0, 0.0
    protected = (z @ p_basis) @ (p_basis.T @ w)
    erased = full - protected
    value = balanced_bce(y, erased) - balanced_bce(y, full)
    gw_full, gb_full = balanced_grad(z, y, full)
    z_erased = z - (z @ p_basis) @ p_basis.T
    gw_erased, gb_erased = balanced_grad(z_erased, y, erased)
    decision = float(np.mean((full >= 0.0) != (erased >= 0.0)))
    contribution = float(np.sqrt(np.mean(protected ** 2)))
    return value, gw_erased - gw_full, gb_erased - gb_full, decision, contribution


def stratified_halves(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first, second = [], []
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        first.extend(idx[::2]); second.extend(idx[1::2])
    return np.asarray(sorted(first), dtype=int), np.asarray(sorted(second), dtype=int)


def protected_damage(z: np.ndarray, y: np.ndarray, w0: np.ndarray, b0: float, wg: np.ndarray, bg: float,
                     p_basis: np.ndarray, alpha: float) -> float:
    grid = [value for value in (0.0, 0.5, 0.75, 1.0) if value <= alpha + 1e-9]
    if grid[-1] != alpha:
        grid.append(alpha)
    values = []
    for value in grid:
        w, b = w0 + value * (wg - w0), b0 + value * (bg - b0)
        values.append(utility_and_gradient(z, y, w, b, p_basis)[0])
    return float(sum(max(0.0, values[i - 1] - values[i]) for i in range(1, len(values))))


def risk_features(z: np.ndarray, y: np.ndarray, w0: np.ndarray, b0: float, wg: np.ndarray, bg: float,
                  bank: dict[str, Any]) -> dict[str, float]:
    p = bank["p_basis"]
    logits0 = z @ w0 + b0
    logitsg = z @ wg + bg
    theta_grid = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        w, b = w0 + alpha * (wg - w0), b0 + alpha * (bg - b0)
        value, guw, gub, decision, contribution = utility_and_gradient(z, y, w, b, p)
        theta_grid.append({"alpha": alpha, "G": value, "D": decision, "contribution": contribution})
    g_values = np.asarray([row["G"] for row in theta_grid])
    d_values = np.asarray([row["D"] for row in theta_grid])
    c_values = np.asarray([row["contribution"] for row in theta_grid])
    damage = np.maximum(0.0, g_values[:-1] - g_values[1:])
    gtask_w, gtask_b = balanced_grad(z, y, logits0)
    _, guw, gub, _, _ = utility_and_gradient(z, y, w0, b0, p)
    dot = float(np.dot(gtask_w, guw) + gtask_b * gub)
    cosine = float(dot / ((np.linalg.norm(np.r_[gtask_w, gtask_b]) * np.linalg.norm(np.r_[guw, gub])) + EPS))
    half_a, half_b = stratified_halves(y)
    split_a = protected_damage(z[half_a], y[half_a], w0, b0, wg, bg, p, 1.0)
    split_b = protected_damage(z[half_b], y[half_b], w0, b0, wg, bg, p, 1.0)
    prob0 = sigmoid(logits0)
    identity_projection = z @ bank["i_basis"]
    identity_mean = identity_projection.mean(axis=0)
    distances = np.sqrt(np.sum((bank["id_centroids"] - identity_mean[None, :]) ** 2, axis=1))
    identity_soft = np.exp(-(distances - distances.min()))
    identity_conf = float(identity_soft.max() / max(identity_soft.sum(), EPS))
    p_projection = z @ p if p.shape[1] else np.zeros((len(z), 0))
    split_p = 0.0
    if p.shape[1]:
        split_p = float(1.0 / (1.0 + np.linalg.norm(p_projection[half_a].mean(axis=0) - p_projection[half_b].mean(axis=0))))
    persistence = float(np.linalg.norm(p_projection.mean(axis=0)) * (bank["p_scores"].mean() if len(bank["p_scores"]) else 0.0))
    split_generic = abs(ba(y[half_a], logitsg[half_a]) - ba(y[half_b], logitsg[half_b]))
    return {
        "history_CE": balanced_bce(y, logits0), "history_BA": ba(y, logits0),
        "history_margin": float(np.mean(np.abs(logits0))), "history_entropy": entropy_from_logits(logits0),
        "history_confidence": float(np.mean(np.maximum(prob0, 1.0 - prob0))),
        "update_norm": float(np.linalg.norm(np.r_[wg - w0, bg - b0])),
        "gradient_norm": float(np.linalg.norm(np.r_[gtask_w, gtask_b])),
        "history_improvement": ba(y, logitsg) - ba(y, logits0),
        "split_history_disagreement": split_generic, "sample_count": float(len(y)),
        "identity_projection_norm": float(np.linalg.norm(identity_mean)),
        "identity_nearest_distance": float(distances.min()), "identity_confidence": identity_conf,
        "persistence_magnitude": persistence, "persistence_split_stability": split_p,
        "protected_utility_t0": float(g_values[0]), "decision_coupling_t0": float(d_values[0]),
        "protected_functional_retention": float(g_values[-1] / (abs(g_values[0]) + 1e-6)),
        "cumulative_protected_damage": float(damage.sum()), "maximum_protected_damage": float(damage.max(initial=0.0)),
        "damaging_step_fraction": float(np.mean(damage > 0.0)),
        "decision_coupling_degradation": float(max(0.0, d_values[0] - d_values[-1])),
        "protected_contribution_change": float(c_values[-1] - c_values[0]),
        "task_protected_gradient_conflict": dot, "task_protected_gradient_cosine": cosine,
        "protected_function_slope": float(np.polyfit(np.asarray([row["alpha"] for row in theta_grid]), g_values, 1)[0]),
        "protected_split_disagreement": float(abs(split_a - split_b)),
        "damage_alpha_0": 0.0, "damage_alpha_05": protected_damage(z, y, w0, b0, wg, bg, p, 0.5),
        "damage_alpha_075": protected_damage(z, y, w0, b0, wg, bg, p, 0.75),
        "damage_alpha_1": protected_damage(z, y, w0, b0, wg, bg, p, 1.0),
    }


def make_episode(subject: str, fold: int, seed: int, role: str, features: np.ndarray, metadata: pd.DataFrame,
                 bank: dict[str, Any], head_w: np.ndarray, head_b: float, generic: dict[str, Any]) -> tuple[dict, dict[float, np.ndarray]]:
    z = (features - bank["center"]) / bank["scale"]
    s1, s2 = subject_mask(metadata, subject, 1), subject_mask(metadata, subject, 2)
    y1, y2 = metadata.loc[s1, "label"].to_numpy(int), metadata.loc[s2, "label"].to_numpy(int)
    w0, b0 = population_theta(head_w, head_b, bank)
    c, beta = float(generic["chosen"]["C"]), float(generic["chosen"]["beta"])
    wt, bt = target_theta(z[s1], y1, c, RNG_SEED + seed * 1000 + fold * 100 + int(subject))
    wg, bg = (1.0 - beta) * w0 + beta * wt, (1.0 - beta) * b0 + beta * bt
    history = risk_features(z[s1], y1, w0, b0, wg, bg, bank)
    alpha_logits = {alpha: z[s2] @ (w0 + alpha * (wg - w0)) + (b0 + alpha * (bg - b0)) for alpha in ALPHAS}
    row = {"subject_id": subject, "fold": fold, "seed": seed, "role": role, "protected_rank": bank["protected_rank"],
           "generic_C": c, "generic_beta": beta, "BA_NoAdapt": ba(y2, alpha_logits[0.0]),
           "BA_Strong_Generic": ba(y2, alpha_logits[1.0]),
           "FutureDeltaBA": ba(y2, alpha_logits[1.0]) - ba(y2, alpha_logits[0.0]),
           "NegativeTransfer": int(ba(y2, alpha_logits[1.0]) < ba(y2, alpha_logits[0.0]) - 1e-12),
           "target_S1_used": True, "target_S2_used_for_decision": False, "internal_holdout_used": False,
           **history}
    for alpha in ALPHAS:
        row[f"BA_alpha_{str(alpha).replace('.', '_')}"] = ba(y2, alpha_logits[alpha])
    return row, alpha_logits


def aggregate_seed_episodes(seed_rows: pd.DataFrame, trial_logits: dict[tuple[int, int, str, float], np.ndarray],
                            metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[int, str, float], np.ndarray]]:
    rows, ensemble_logits = [], {}
    feature_cols = GENERIC_COLS + IDENTITY_COLS + P_COLS + ["protected_utility_t0", "decision_coupling_t0"] + PERSIST_COLS + [
        "damage_alpha_0", "damage_alpha_05", "damage_alpha_075", "damage_alpha_1"]
    for (fold, subject, role), group in seed_rows.groupby(["fold", "subject_id", "role"], sort=False):
        s2 = subject_mask(metadata, str(subject), 2)
        y = metadata.loc[s2, "label"].to_numpy(int)
        alpha_values = {}
        for alpha in ALPHAS:
            stack = np.stack([trial_logits[(int(fold), int(seed), str(subject), alpha)] for seed in SEEDS])
            # Average probabilities, then convert back to logits for metric convenience.
            probability = np.mean(sigmoid(stack), axis=0)
            value = np.log(np.clip(probability, 1e-7, 1 - 1e-7) / np.clip(1.0 - probability, 1e-7, 1 - 1e-7))
            ensemble_logits[(int(fold), str(subject), alpha)] = value
            alpha_values[alpha] = ba(y, value)
        row = {"fold": int(fold), "subject_id": str(subject), "role": role,
               "protected_rank": int(round(group.protected_rank.mean())),
               "BA_NoAdapt": alpha_values[0.0], "BA_Strong_Generic": alpha_values[1.0],
               "FutureDeltaBA": alpha_values[1.0] - alpha_values[0.0],
               "NegativeTransfer": int(alpha_values[1.0] < alpha_values[0.0] - 1e-12),
               "internal_holdout_used": False}
        for col in feature_cols:
            row[col] = float(group[col].mean())
        for alpha in ALPHAS:
            row[f"BA_alpha_{str(alpha).replace('.', '_')}"] = alpha_values[alpha]
        rows.append(row)
    return pd.DataFrame(rows), ensemble_logits


def cv_probabilities(x: np.ndarray, y: np.ndarray, c: float, seed: int) -> np.ndarray:
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
        return np.full(len(y), float(np.mean(y)))
    splits = min(5, int(min(np.bincount(y))))
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    out = np.zeros(len(y), dtype=float)
    for train, test in skf.split(x, y):
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed)
        model.fit(scaler.transform(x[train]), y[train])
        out[test] = model.predict_proba(scaler.transform(x[test]))[:, 1]
    return out


def fit_risk(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], fold: int,
             random_projection: bool = False) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    xtr = train[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    xte = test[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    if random_projection:
        rng = np.random.default_rng(RNG_SEED + fold)
        matrix = rng.normal(size=(xtr.shape[1], len(PERSIST_COLS))) / math.sqrt(max(xtr.shape[1], 1))
        xtr, xte = xtr @ matrix, xte @ matrix
    y = train.NegativeTransfer.to_numpy(int)
    candidates = []
    for c in RISK_CS:
        oof = cv_probabilities(xtr, y, c, RNG_SEED + fold)
        auc = float(roc_auc_score(y, oof)) if len(np.unique(y)) > 1 else 0.5
        brier = float(brier_score_loss(y, oof))
        candidates.append({"C": c, "AUROC": auc, "Brier": brier, "oof": oof})
    chosen = max(candidates, key=lambda row: (row["AUROC"], -row["Brier"], -row["C"]))
    if len(np.unique(y)) < 2:
        predicted = np.full(len(test), float(np.mean(y)))
    else:
        scaler = StandardScaler().fit(xtr)
        model = LogisticRegression(C=chosen["C"], class_weight="balanced", solver="liblinear", max_iter=2000, random_state=RNG_SEED + fold)
        model.fit(scaler.transform(xtr), y)
        predicted = model.predict_proba(scaler.transform(xte))[:, 1]
    audit = {"C": chosen["C"], "source_oof_AUROC": chosen["AUROC"], "source_oof_Brier": chosen["Brier"],
             "source_subjects": len(train), "features": cols, "random_projection": random_projection}
    return chosen["oof"], predicted, audit


def alpha_col(alpha: float) -> str:
    return f"BA_alpha_{str(alpha).replace('.', '_')}"


def policy_metrics(frame: pd.DataFrame, probability: np.ndarray, tau: float, safe_alpha: float,
                   trust_epsilon: float | None = None) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    actions, values = [], []
    for i, (_, row) in enumerate(frame.iterrows()):
        if probability[i] < tau or int(row.protected_rank) == 0:
            alpha = 1.0
        elif trust_epsilon is None:
            alpha = safe_alpha
        else:
            candidates = []
            for value, damage_col in ((0.0, "damage_alpha_0"), (0.5, "damage_alpha_05"), (0.75, "damage_alpha_075")):
                if float(row[damage_col]) <= trust_epsilon + 1e-12:
                    candidates.append(value)
            alpha = max(candidates) if candidates else 0.0
        actions.append(alpha)
        values.append(float(row[alpha_col(alpha)]))
    actions = np.asarray(actions, dtype=float)
    values = np.asarray(values, dtype=float)
    base = frame.BA_NoAdapt.to_numpy(float)
    generic = frame.BA_Strong_Generic.to_numpy(float)
    harmed = generic < base - 1e-12
    guard_harmed = values < base - 1e-12
    rescued = harmed & ~guard_harmed
    new_harms = ~harmed & guard_harmed
    metrics = {"mean_BA": float(values.mean()), "delta_vs_generic": float(np.mean(values - generic)),
               "negative_transfer_rate": float(guard_harmed.mean()), "generic_harmed": int(harmed.sum()),
               "rescued_harms": int(rescued.sum()), "new_harms": int(new_harms.sum()),
               "guarded_fraction": float(np.mean(actions < 1.0)), "mean_alpha": float(actions.mean())}
    return metrics, actions, values


def select_policy(train: pd.DataFrame, oof_probability: np.ndarray, allowed_alphas: Iterable[float],
                  trust: bool = False) -> dict[str, Any]:
    candidates = []
    epsilon_grid = [None]
    if trust:
        positive = train.loc[train.damage_alpha_1 > 0, "damage_alpha_1"]
        epsilon_grid = [0.0, float(positive.median()) if len(positive) else 0.0]
    for tau in TAUS:
        for alpha in allowed_alphas:
            for epsilon in epsilon_grid:
                metric, _, _ = policy_metrics(train, oof_probability, tau, alpha, epsilon)
                candidates.append({"tau": tau, "safe_alpha": alpha, "trust_epsilon": epsilon, **metric})
    chosen = max(candidates, key=lambda row: (row["mean_BA"], -row["new_harms"], -row["negative_transfer_rate"], row["mean_alpha"]))
    return {"chosen": chosen, "candidates": candidates}


def bootstrap_delta(values: np.ndarray, seed: int, n_boot: int = BOOTSTRAPS) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def method_metrics(frame: pd.DataFrame, method: str, ba_values: np.ndarray, generic: np.ndarray,
                   noadapt: np.ndarray, actions: np.ndarray | None = None) -> dict[str, Any]:
    delta = ba_values - generic
    harmed = generic < noadapt - 1e-12
    current_harm = ba_values < noadapt - 1e-12
    ci = bootstrap_delta(delta, RNG_SEED + sum(map(ord, method)))
    folds = []
    for fold, index in frame.groupby("fold").groups.items():
        idx = np.asarray(list(index), dtype=int)
        folds.append(float(np.mean(delta[idx])))
    return {"method": method, "subjects": len(frame), "BA": float(np.mean(ba_values)),
            "delta_vs_strong_generic": float(np.mean(delta)), "paired_CI95_L": ci[0], "paired_CI95_U": ci[1],
            "wins": int(np.sum(delta > 1e-12)), "losses": int(np.sum(delta < -1e-12)), "ties": int(np.sum(np.abs(delta) <= 1e-12)),
            "fold_positive_count": int(np.sum(np.asarray(folds) > 1e-12)), "negative_transfer_rate": float(np.mean(current_harm)),
            "generic_harmed": int(harmed.sum()), "rescued_harms": int(np.sum(harmed & ~current_harm)),
            "new_harms": int(np.sum(~harmed & current_harm)), "mean_alpha": None if actions is None else float(np.mean(actions)),
            "internal_holdout_used": False}


def generate_figures(outcome: pd.DataFrame, risk_frame: pd.DataFrame, performance: pd.DataFrame,
                     actions: pd.DataFrame, selected_method: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    selected_risk = risk_frame.loc[risk_frame.risk_model.eq("M_PERSIST")].copy()
    selected_actions = actions.loc[actions.method.eq(selected_method)].copy()
    merged = selected_risk.merge(selected_actions[["subject_id", "fold", "action_alpha", "rescued", "new_harm"]], on=["subject_id", "fold"], how="left")
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    colors = np.where(merged.rescued.fillna(False), "#2ca02c", np.where(merged.new_harm.fillna(False), "#d62728", "#4c78a8"))
    ax.scatter(merged.risk_probability, merged.FutureDeltaBA, c=colors, s=45, alpha=0.85)
    ax.axhline(0, color="black", lw=1); ax.set_xlabel("PERSIST predicted harm probability"); ax.set_ylabel("Generic future DeltaBA")
    fig.tight_layout(); fig.savefig(FIGURES / "risk_vs_future_deltaBA.png", dpi=220); fig.savefig(FIGURES / "risk_vs_future_deltaBA.pdf"); plt.close(fig)

    subset = performance.loc[performance.method.isin(["Strong Generic", "confidence guard", "identity guard", selected_method])].copy()
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.bar(subset.method, subset.negative_transfer_rate * 100, color=["#777777", "#f28e2b", "#b07aa1", "#2ca02c"][:len(subset)])
    ax.set_ylabel("Negative-transfer rate (%)"); ax.tick_params(axis="x", rotation=18)
    fig.tight_layout(); fig.savefig(FIGURES / "negative_transfer_comparison.png", dpi=220); fig.savefig(FIGURES / "negative_transfer_comparison.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    shown = performance.loc[performance.method.isin(["Vanilla EEGNet", "NoAdapt", "Strong Generic", "confidence guard", "identity guard", selected_method])]
    ax.bar(shown.method, shown.BA * 100, color="#4c78a8")
    ax.set_ylabel("Mean subject BA (%)"); ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(FIGURES / "method_performance.png", dpi=220); fig.savefig(FIGURES / "method_performance.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    for harmed, label, color in ((0, "not harmed", "#4c78a8"), (1, "Generic harmed", "#d62728")):
        part = outcome.loc[outcome.NegativeTransfer.eq(harmed)]
        if len(part):
            means = [part[col].mean() for col in ("damage_alpha_0", "damage_alpha_05", "damage_alpha_075", "damage_alpha_1")]
            ax.plot(ALPHAS, means, marker="o", label=label, color=color)
    ax.set_xlabel("Generic interpolation alpha"); ax.set_ylabel("Cumulative protected damage"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "protected_damage_trajectory.png", dpi=220); fig.savefig(FIGURES / "protected_damage_trajectory.pdf"); plt.close(fig)


def main() -> None:
    for path in (RESULTS, PROTOCOL, FIGURES, CACHE):
        path.mkdir(parents=True, exist_ok=True)
    set_seed(RNG_SEED)
    protocol = load_protocol()
    metadata = protocol.metadata
    bank_audit_rows, generic_search_rows, seed_episode_rows = [], [], []
    trial_logits: dict[tuple[int, int, str, float], np.ndarray] = {}
    checkpoint_audit = []
    for fold_info in protocol.folds:
        fold = int(fold_info["fold"])
        for seed in SEEDS:
            features, base_logits, head_w, head_b, payload = extract_representation(protocol, seed, fold)
            checkpoint_audit.append({"seed": seed, "fold": fold, "path": str(checkpoint_path(seed, fold)),
                                     "sha256": sha256_file(checkpoint_path(seed, fold)),
                                     "train_subjects": payload["train_subjects"], "validation_subjects": payload["validation_subjects"],
                                     "holdout_overlap": sorted((set(map(str, payload["train_subjects"])) | set(map(str, payload["validation_subjects"]))) & set(protocol.holdout)),
                                     "internal_holdout_used": False})
            bank, audit = fit_protected_bank(fold_info["meta_subjects"], features, base_logits, metadata, seed, fold)
            bank_audit_rows.extend(audit)
            generic = select_generic(fold_info["meta_subjects"], features, base_logits, metadata, bank, head_w, head_b, seed, fold)
            generic_search_rows.extend(generic["candidates"])
            for subject in protocol.search:
                role = "outcome" if subject in fold_info["outcome_subjects"] else "meta"
                row, logits_by_alpha = make_episode(subject, fold, seed, role, features, metadata, bank, head_w, head_b, generic)
                seed_episode_rows.append(row)
                for alpha, values in logits_by_alpha.items():
                    trial_logits[(fold, seed, subject, alpha)] = values
            print(f"[exp4] fold={fold} seed={seed} rank={bank['protected_rank']} generic={generic['chosen']}", flush=True)
    seed_frame = pd.DataFrame(seed_episode_rows)
    ensemble_frame, ensemble_logits = aggregate_seed_episodes(seed_frame, trial_logits, metadata)
    outcome = ensemble_frame.loc[ensemble_frame.role.eq("outcome")].reset_index(drop=True)
    if len(outcome) != 40 or set(outcome.subject_id) != set(protocol.search):
        raise RuntimeError("development outcome coverage failure")
    write_csv(RESULTS / "DEV_SUBJECT_RESULTS.csv", outcome)
    write_csv(RESULTS / "SOURCE_META_EPISODES.csv", ensemble_frame.loc[ensemble_frame.role.eq("meta")])
    write_csv(RESULTS / "PROTECTED_BANK_AUDIT.csv", pd.DataFrame(bank_audit_rows))
    write_csv(RESULTS / "GENERIC_SOURCE_SEARCH.csv", pd.DataFrame(generic_search_rows))

    risk_specs = {
        "M_generic": GENERIC_COLS,
        "M_confidence": ["history_confidence", "history_entropy", "history_margin", "history_improvement"],
        "M_update": ["update_norm", "gradient_norm", "history_improvement", "split_history_disagreement"],
        "M_identity": IDENTITY_COLS,
        "M_P": P_COLS,
        "M_static_PUD": STATIC_PUD_COLS,
        "M_PERSIST": PERSIST_COLS,
        "M_full": GENERIC_COLS + PERSIST_COLS,
        "M_random": GENERIC_COLS + PERSIST_COLS,
    }
    risk_rows, risk_audits, model_context = [], [], {}
    for fold_info in protocol.folds:
        fold = int(fold_info["fold"])
        train = ensemble_frame.loc[(ensemble_frame.fold == fold) & ensemble_frame.role.eq("meta")].reset_index(drop=True)
        test = ensemble_frame.loc[(ensemble_frame.fold == fold) & ensemble_frame.role.eq("outcome")].reset_index(drop=True)
        for name, cols in risk_specs.items():
            oof, predicted, audit = fit_risk(train, test, cols, fold, random_projection=name == "M_random")
            audit.update({"fold": fold, "risk_model": name}); risk_audits.append(audit)
            model_context[(fold, name)] = {"train": train, "test": test, "oof": oof, "predicted": predicted, "audit": audit}
            for i, (_, row) in enumerate(test.iterrows()):
                risk_rows.append({"subject_id": row.subject_id, "fold": fold, "risk_model": name,
                                  "risk_probability": float(predicted[i]), "NegativeTransfer": int(row.NegativeTransfer),
                                  "FutureDeltaBA": float(row.FutureDeltaBA), "protected_rank": int(row.protected_rank),
                                  "internal_holdout_used": False})
    risk_frame = pd.DataFrame(risk_rows)
    write_csv(RESULTS / "RISK_PREDICTION.csv", risk_frame.loc[risk_frame.risk_model.str.contains("PERSIST|full|static_PUD|M_P$")])
    write_csv(RESULTS / "CONTROL_RISK_PREDICTION.csv", risk_frame.loc[~risk_frame.risk_model.str.contains("PERSIST|full|static_PUD|M_P$")])
    write_json(PROTOCOL / "RISK_MODEL_AUDIT.json", risk_audits)

    # Four bounded major variants. Controls receive the same {0,.5,.75,1} action budget.
    variants = [
        ("V1_PERSIST_ROLLBACK", "M_PERSIST", (0.0,), False),
        ("V2_PERSIST_FIXED_SHRINK", "M_PERSIST", (0.5, 0.75), False),
        ("V3_PERSIST_TRUST_REGION", "M_PERSIST", (0.0,), True),
        ("V4_FULL_FIXED_SHRINK", "M_full", (0.0, 0.5, 0.75), False),
    ]
    controls = [
        ("confidence guard", "M_confidence"), ("update-magnitude guard", "M_update"),
        ("identity guard", "M_identity"), ("P-only guard", "M_P"),
        ("generic-diagnostic guard", "M_generic"), ("random guard", "M_random"),
    ]
    method_values: dict[str, np.ndarray] = {}
    method_actions: dict[str, np.ndarray] = {}
    action_rows, ledger_rows = [], []
    for method, risk_model, allowed, trust in variants:
        values = np.zeros(len(outcome)); actions = np.ones(len(outcome)); fold_policies = []
        for fold_info in protocol.folds:
            fold = int(fold_info["fold"]); context = model_context[(fold, risk_model)]
            policy = select_policy(context["train"], context["oof"], allowed, trust=trust)
            metric, action, value = policy_metrics(context["test"], context["predicted"], policy["chosen"]["tau"],
                                                    policy["chosen"]["safe_alpha"], policy["chosen"]["trust_epsilon"])
            idx = outcome.index[outcome.fold.eq(fold)].to_numpy()
            values[idx], actions[idx] = value, action
            fold_policies.append({"fold": fold, **policy["chosen"]})
        method_values[method], method_actions[method] = values, actions
        metric = method_metrics(outcome, method, values, outcome.BA_Strong_Generic.to_numpy(float), outcome.BA_NoAdapt.to_numpy(float), actions)
        ledger_rows.append({"variant": method, "risk_model": risk_model, "action_family": "trust-region" if trust else str(tuple(allowed)),
                            "development_BA": metric["BA"], "delta_vs_generic": metric["delta_vs_strong_generic"],
                            "negative_transfer_rate": metric["negative_transfer_rate"], "status": "ATTEMPTED_BOUNDED_DEVELOPMENT",
                            "fold_policies": json.dumps(clean(fold_policies), sort_keys=True)})
        for i, row in outcome.iterrows():
            harmed = row.BA_Strong_Generic < row.BA_NoAdapt - 1e-12
            guard_harmed = values[i] < row.BA_NoAdapt - 1e-12
            action_rows.append({"subject_id": row.subject_id, "fold": row.fold, "method": method, "action_alpha": actions[i],
                                "BA_NoAdapt": row.BA_NoAdapt, "BA_Strong_Generic": row.BA_Strong_Generic, "BA_Guard": values[i],
                                "rescued": bool(harmed and not guard_harmed), "new_harm": bool((not harmed) and guard_harmed),
                                "internal_holdout_used": False})
    for method, risk_model in controls:
        values = np.zeros(len(outcome)); actions = np.ones(len(outcome))
        for fold_info in protocol.folds:
            fold = int(fold_info["fold"]); context = model_context[(fold, risk_model)]
            policy = select_policy(context["train"], context["oof"], (0.0, 0.5, 0.75), trust=False)
            _, action, value = policy_metrics(context["test"], context["predicted"], policy["chosen"]["tau"], policy["chosen"]["safe_alpha"])
            idx = outcome.index[outcome.fold.eq(fold)].to_numpy(); values[idx], actions[idx] = value, action
        method_values[method], method_actions[method] = values, actions
        for i, row in outcome.iterrows():
            harmed = row.BA_Strong_Generic < row.BA_NoAdapt - 1e-12
            guard_harmed = values[i] < row.BA_NoAdapt - 1e-12
            action_rows.append({"subject_id": row.subject_id, "fold": row.fold, "method": method, "action_alpha": actions[i],
                                "BA_NoAdapt": row.BA_NoAdapt, "BA_Strong_Generic": row.BA_Strong_Generic, "BA_Guard": values[i],
                                "rescued": bool(harmed and not guard_harmed), "new_harm": bool((not harmed) and guard_harmed),
                                "internal_holdout_used": False})
    actions_frame = pd.DataFrame(action_rows)
    write_csv(RESULTS / "GUARD_ACTIONS.csv", actions_frame)

    generic_values = outcome.BA_Strong_Generic.to_numpy(float); noadapt_values = outcome.BA_NoAdapt.to_numpy(float)
    variant_metrics = [method_metrics(outcome, name, method_values[name], generic_values, noadapt_values, method_actions[name]) for name, *_ in variants]
    # Development selection is explicit and adaptive; no holdout has been touched.
    selected = max(variant_metrics, key=lambda row: (row["BA"], -row["new_harms"], -row["negative_transfer_rate"], row["mean_alpha"]))
    selected_method = selected["method"]
    control_metrics = [method_metrics(outcome, name, method_values[name], generic_values, noadapt_values, method_actions[name]) for name, _ in controls]
    baseline_metrics = [
        method_metrics(outcome, "NoAdapt", noadapt_values, generic_values, noadapt_values),
        method_metrics(outcome, "Strong Generic", generic_values, generic_values, noadapt_values),
    ]
    performance = pd.DataFrame(baseline_metrics + control_metrics + variant_metrics)
    vanilla_summary = pd.read_csv(BASE_EXP / "results" / "VANILLA_EEGNET_SUMMARY.csv").loc[
        lambda x: x.variant.eq("S1S2_SOURCE_TO_S2")
    ].iloc[0]
    vanilla = float(vanilla_summary["Mean BA"])
    performance["dataset"] = "OpenBMI"; performance["backbone"] = "clean StandardEEGNet ensemble"
    performance["seed"] = "0,1,2"; performance["delta_vs_vanilla"] = performance.BA - vanilla
    performance["delta_vs_noadapt"] = performance.BA - float(noadapt_values.mean())
    performance["worst_quartile_BA"] = np.nan; bottom = np.argsort(generic_values)[:max(1, len(generic_values) // 4)]
    for index, row in performance.iterrows():
        if row.method == "NoAdapt": values = noadapt_values
        elif row.method == "Strong Generic": values = generic_values
        else: values = method_values[row.method]
        performance.loc[index, "worst_quartile_BA"] = float(values[bottom].mean())
    performance["macro_F1"] = np.nan; performance["accuracy"] = np.nan
    for index, row in performance.iterrows():
        if row.method == "NoAdapt":
            actions = np.zeros(len(outcome), dtype=float)
        elif row.method == "Strong Generic":
            actions = np.ones(len(outcome), dtype=float)
        else:
            actions = method_actions[row.method]
        macro_f1, accuracy = classification_metrics_for_actions(outcome, actions, ensemble_logits, metadata)
        performance.loc[index, "macro_F1"] = macro_f1
        performance.loc[index, "accuracy"] = accuracy
    performance["fold_correct"] = True; performance["strict_holdout"] = False
    # These are adaptive development rows.  A failed gate cannot support a main/confirmatory claim.
    performance["valid_for_main_claim"] = False
    vanilla_row = {col: None for col in performance.columns}
    vanilla_row.update({"method": "Vanilla EEGNet", "subjects": 40, "BA": vanilla, "dataset": "OpenBMI", "backbone": "StandardEEGNet",
                        "seed": "0,1,2", "delta_vs_vanilla": 0.0,
                        "macro_F1": float(vanilla_summary["Macro-F1"]), "accuracy": float(vanilla_summary["Accuracy"]),
                        "fold_correct": True, "strict_holdout": False,
                        "valid_for_main_claim": False, "internal_holdout_used": False})
    performance = pd.DataFrame(
        [vanilla_row] + performance.to_dict(orient="records"),
        columns=performance.columns,
    )
    write_csv(RESULTS / "EXP4_FINAL_PERFORMANCE.csv", performance)

    mechanism_rows = []
    for risk_model in risk_specs:
        part = risk_frame.loc[risk_frame.risk_model.eq(risk_model)].sort_values(["fold", "subject_id"])
        y, p = part.NegativeTransfer.to_numpy(int), part.risk_probability.to_numpy(float)
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
        auprc = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None
        rho = spearmanr(part.FutureDeltaBA, p).statistic if len(part) > 2 else np.nan
        mapped_method = {"M_confidence": "confidence guard", "M_update": "update-magnitude guard", "M_identity": "identity guard",
                         "M_P": "P-only guard", "M_generic": "generic-diagnostic guard", "M_random": "random guard",
                         "M_PERSIST": selected_method, "M_full": "V4_FULL_FIXED_SHRINK"}.get(risk_model)
        metric = next((row for row in variant_metrics + control_metrics if row["method"] == mapped_method), None)
        mechanism_rows.append({"risk_model": risk_model, "AUROC": auc, "AUPRC": auprc,
                               "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)), "Brier": float(brier_score_loss(y, p)),
                               "risk_outcome_Spearman": float(rho) if np.isfinite(rho) else None,
                               "mean_risk_harmed": float(p[y == 1].mean()) if np.any(y == 1) else None,
                               "mean_risk_non_harmed": float(p[y == 0].mean()) if np.any(y == 0) else None,
                               "rescued_harms": None if metric is None else metric["rescued_harms"],
                               "new_harms": None if metric is None else metric["new_harms"], "BA_after_guard": None if metric is None else metric["BA"]})
    mechanism = pd.DataFrame(mechanism_rows)
    write_csv(RESULTS / "EXP4_MECHANISM_SPECIFICITY.csv", mechanism)

    # Ablations use identical action budgets and are not relabelled as full PERSIST.
    ablation_map = {"w/o P": "M_generic", "w/o U": "M_P", "w/o D": "M_static_PUD",
                    "identity instead of decision": "M_identity", "confidence-only guard": "M_confidence",
                    "update-magnitude guard": "M_update", "random guard": "M_random"}
    ablations = []
    for label, risk_model in ablation_map.items():
        mapped = {"M_generic": "generic-diagnostic guard", "M_P": "P-only guard", "M_identity": "identity guard",
                  "M_confidence": "confidence guard", "M_update": "update-magnitude guard", "M_random": "random guard"}.get(risk_model)
        if mapped is not None:
            metric = next(row for row in control_metrics if row["method"] == mapped)
            ablations.append({"ablation": label, **metric})
        else:
            # static PUD gets the same source-only fixed-shrink policy.
            values = np.zeros(len(outcome)); actions = np.ones(len(outcome))
            for fold_info in protocol.folds:
                fold = int(fold_info["fold"]); context = model_context[(fold, risk_model)]
                policy = select_policy(context["train"], context["oof"], (0.0, 0.5, 0.75))
                _, action, value = policy_metrics(context["test"], context["predicted"], policy["chosen"]["tau"], policy["chosen"]["safe_alpha"])
                idx = outcome.index[outcome.fold.eq(fold)].to_numpy(); values[idx], actions[idx] = value, action
            ablations.append({"ablation": label, **method_metrics(outcome, label, values, generic_values, noadapt_values, actions)})
    oracle = np.maximum(generic_values, noadapt_values)
    ablations.append({"ablation": "Generic/NoAdapt oracle (diagnostic only)", **method_metrics(outcome, "oracle", oracle, generic_values, noadapt_values)})
    write_csv(RESULTS / "ABLATION_RESULTS.csv", pd.DataFrame(ablations))
    write_csv(RESULTS / "ITERATION_RESULTS.csv", pd.DataFrame(ledger_rows))

    # Fold and seed robustness for the selected development candidate.
    fold_rows = []
    for fold, index in outcome.groupby("fold").groups.items():
        idx = np.asarray(list(index), dtype=int)
        fold_rows.append({"fold": fold, "subjects": len(idx), "NoAdapt_BA": float(noadapt_values[idx].mean()),
                          "Strong_Generic_BA": float(generic_values[idx].mean()), "PERSIST_Guard_BA": float(method_values[selected_method][idx].mean()),
                          "delta_vs_generic": float(np.mean(method_values[selected_method][idx] - generic_values[idx])), "internal_holdout_used": False})
    write_csv(RESULTS / "DEV_FOLD_RESULTS.csv", pd.DataFrame(fold_rows))
    seed_rows = []
    for seed in SEEDS:
        seed_outcome = seed_frame.loc[
            seed_frame.role.eq("outcome") & seed_frame.seed.eq(seed)
        ].sort_values(["fold", "subject_id"]).reset_index(drop=True)
        if len(seed_outcome) != len(outcome) or set(seed_outcome.subject_id) != set(protocol.search):
            raise RuntimeError(f"seed {seed} development outcome coverage failure")
        action_lookup = actions_frame.loc[actions_frame.method.eq(selected_method)].set_index(["fold", "subject_id"]).action_alpha
        guarded = []
        for _, row in seed_outcome.iterrows():
            alpha = float(action_lookup.loc[(row.fold, row.subject_id)])
            guarded.append(float(row[alpha_col(alpha)]))
        seed_rows.append({"seed": seed, "subjects": len(seed_outcome), "NoAdapt_BA": float(seed_outcome.BA_NoAdapt.mean()),
                          "Strong_Generic_BA": float(seed_outcome.BA_Strong_Generic.mean()), "PERSIST_Guard_BA": float(np.mean(guarded)),
                          "delta_vs_generic": float(np.mean(np.asarray(guarded) - seed_outcome.BA_Strong_Generic.to_numpy(float))),
                          "internal_holdout_used": False})
    seed_results = pd.DataFrame(seed_rows)
    write_csv(RESULTS / "DEV_SEED_RESULTS.csv", seed_results)
    seed_summary = {
        "NoAdapt_BA_mean": float(seed_results.NoAdapt_BA.mean()),
        "NoAdapt_BA_SD": float(seed_results.NoAdapt_BA.std(ddof=1)),
        "Strong_Generic_BA_mean": float(seed_results.Strong_Generic_BA.mean()),
        "Strong_Generic_BA_SD": float(seed_results.Strong_Generic_BA.std(ddof=1)),
        "PERSIST_Guard_BA_mean": float(seed_results.PERSIST_Guard_BA.mean()),
        "PERSIST_Guard_BA_SD": float(seed_results.PERSIST_Guard_BA.std(ddof=1)),
        "delta_vs_generic_mean": float(seed_results.delta_vs_generic.mean()),
        "delta_vs_generic_SD": float(seed_results.delta_vs_generic.std(ddof=1)),
    }

    selected_risk_row = mechanism.loc[mechanism.risk_model.eq("M_PERSIST")].iloc[0]
    confidence_auc = mechanism.loc[mechanism.risk_model.eq("M_confidence"), "AUROC"].iloc[0]
    identity_auc = mechanism.loc[mechanism.risk_model.eq("M_identity"), "AUROC"].iloc[0]
    selected_perf = next(row for row in variant_metrics if row["method"] == selected_method)
    confidence_perf = next(row for row in control_metrics if row["method"] == "confidence guard")
    identity_perf = next(row for row in control_metrics if row["method"] == "identity guard")
    generic_harms = int(np.sum(generic_values < noadapt_values - 1e-12))
    relative_harm_reduction = 0.0 if generic_harms == 0 else selected_perf["rescued_harms"] / generic_harms
    worst_generic = float(generic_values[bottom].mean()); worst_guard = float(method_values[selected_method][bottom].mean())
    checks = {
        "A_delta_at_least_0_5pp": selected_perf["delta_vs_strong_generic"] >= 0.005 - 1e-12,
        "B_positive_in_4_of_5_folds": selected_perf["fold_positive_count"] >= 4,
        "C_harm_reduction_at_least_35pct": relative_harm_reduction >= 0.35,
        "D_rescue_at_least_half_harms": relative_harm_reduction >= 0.50,
        "E_new_harms_at_most_2": selected_perf["new_harms"] <= 2,
        "F_worst_quartile_non_decreasing": worst_guard >= worst_generic - 1e-12,
        "G_beats_identity_and_confidence": selected_perf["BA"] > max(identity_perf["BA"], confidence_perf["BA"]) + 1e-12,
        "H_PERSIST_AUROC_at_least_0_65": selected_risk_row.AUROC is not None and float(selected_risk_row.AUROC) >= 0.65,
        "I_harmed_risk_higher": selected_risk_row.mean_risk_harmed is not None and float(selected_risk_row.mean_risk_harmed) > float(selected_risk_row.mean_risk_non_harmed),
        "J_all_three_seed_deltas_nonnegative": bool((seed_results.delta_vs_generic >= -1e-12).all()),
    }
    mandatory = list(checks.values())
    gate_pass = bool(all(mandatory))
    terminal = "EXP4_PERSIST_GUARD_DEV_SUPPORTED" if gate_pass else "EXP4_PERSIST_GUARD_NOT_SUPPORTED"
    gate = {"terminal_state": terminal, "development_gate_pass": gate_pass, "selected_development_variant": selected_method,
            "checks": checks, "checks_passed": int(sum(checks.values())), "checks_total": len(checks),
            "Strong_Generic_BA": float(generic_values.mean()), "PERSIST_Guard_BA": selected_perf["BA"],
            "delta_vs_generic": selected_perf["delta_vs_strong_generic"], "paired_CI95": [selected_perf["paired_CI95_L"], selected_perf["paired_CI95_U"]],
            "generic_negative_transfer_rate": float(generic_harms / len(outcome)), "guard_negative_transfer_rate": selected_perf["negative_transfer_rate"],
            "generic_harmed_subjects": generic_harms, "rescued_harms": selected_perf["rescued_harms"], "new_harms": selected_perf["new_harms"],
            "PERSIST_AUROC": selected_risk_row.AUROC, "confidence_AUROC": confidence_auc, "identity_AUROC": identity_auc,
            "worst_quartile_delta": worst_guard - worst_generic, "fold_positive_count": selected_perf["fold_positive_count"],
            "internal_holdout_used": False, "holdout_access_authorized": gate_pass, "WBCIC_used": False}
    write_json(RESULTS / "DEVELOPMENT_GATE.json", gate)

    purity = {"historical_83_775_legal_for_final_holdout": False,
              "reason": "V6 checkpoint training/architecture/adaptation selection predated the 40/14 split and used original-fold subjects drawn from all 54; some later internal-holdout subjects contributed to population fitting or selection.",
              "clean_population_model": "repaired vanilla StandardEEGNet checkpoints, 3 seeds x 5 folds",
              "checkpoint_audit": checkpoint_audit, "holdout_overlap_any": False,
              "normalization_fit": "fold train subjects from V8_SEARCH only", "protected_bank_fit": "fold meta V8_SEARCH only",
              "risk_model_fit": "fold meta V8_SEARCH source history->future episodes only", "threshold_alpha_selection": "source OOF only",
              "internal_holdout_used": False, "holdout_outcomes_accessed": False}
    write_json(PROTOCOL / "HOLDOUT_PURITY_AUDIT.json", purity)
    write_json(ROOT / "FINAL_METHOD_LOCK.json", {
        "status": "DEVELOPMENT_CANDIDATE_LOCKED" if gate_pass else "NOT_AUTHORIZED_FOR_HOLDOUT",
        "READY_FOR_INTERNAL_HOLDOUT": gate_pass, "selected_variant": selected_method,
        "architecture": "StandardEEGNet F1=8,D=2,F2=16,64-d embedding; three-seed ensemble",
        "generic": "source-selected L2 target-history logistic head blended with population head",
        "generic_C_grid": GENERIC_CS, "generic_beta_grid": GENERIC_BETAS,
        "protected_bank": "source-only symmetric cross-session covariance eigenvectors passing signed P/U/D criteria",
        "protected_rank_rule": f"up to {MAX_PROTECTED_RANK}; rank zero defaults to Generic",
        "risk_features": PERSIST_COLS, "risk_model": "ridge logistic regression", "risk_C_grid": RISK_CS,
        "tau_grid": TAUS, "action_grid": ALPHAS, "seed_list": SEEDS,
        "evaluation_code_sha256": sha256_file(Path(__file__)), "internal_holdout_used": False,
    })
    generate_figures(outcome, risk_frame, performance, actions_frame, selected_method)

    (ROOT / "README.md").write_text(
        f"# Decision-Grounded PERSIST-Guard final Exp4\n\nThis bounded development experiment rebuilds a holdout-pure Generic adapter on the 40 authorized OpenBMI development subjects and evaluates four predeclared PERSIST guard variants plus capacity/action-matched controls. Terminal state: **{terminal}**. Internal holdout accessed: **NO**.\n",
        encoding="utf-8")
    (ROOT / "METHOD.md").write_text(
        "# Method\n\nPERSIST-Guard is a risk-gated interpolation from a clean population EEGNet head to a source-selected target-history Generic head. Protected directions are estimated in the same 64-dimensional EEGNet representation and must pass source-only cross-session persistence (P), signed utility loss under erasure (U), and decision-flip coupling (D). Identity is measured only as a control. Low-risk subjects use Generic exactly; high-risk subjects receive the smallest source-selected rollback/shrink action.\n",
        encoding="utf-8")
    (ROOT / "PROTOCOL.md").write_text(
        "# Protocol\n\nDevelopment uses exactly 40 V8_SEARCH subjects in the frozen five folds. For an evaluated subject, Session 1 is legal labeled history and Session 2 is scoring-only. Fold source subjects may use S1->S2 as legal meta-training episodes. The 14-subject internal holdout is removed before raw tensors are materialised. The historical 83.775% V6 anchor is excluded from final-holdout use because it predates the 40/14 partition.\n",
        encoding="utf-8")
    (ROOT / "ITERATION_LEDGER.md").write_text("# Iteration ledger\n\n" + pd.DataFrame(ledger_rows).to_markdown(index=False) + "\n", encoding="utf-8")
    (ROOT / "HOLDOUT_PURITY_AUDIT.md").write_text(
        "# Holdout purity audit\n\nThe historical 83.775% V6 checkpoint is **not legal** for the final 14-subject holdout: its training and selection predated the V8 40/14 partition. This experiment instead uses 15 repaired EEGNet checkpoints whose train/validation subject lists are subsets of V8_SEARCH and have zero holdout overlap. Normalization, protected-bank certification, risk fitting, threshold selection, and action selection are development/source-only. Internal holdout data and outcomes were not accessed.\n",
        encoding="utf-8")
    ranks = outcome.groupby("fold").protected_rank.mean().to_dict()
    (ROOT / "PROTECTED_REPRESENTATION_COMPATIBILITY.md").write_text(
        f"# Protected representation compatibility\n\nBoth Generic adaptation and protected certification operate on the same 64-dimensional embedding emitted by each clean fold/seed StandardEEGNet checkpoint. The population head, target logistic head, interpolation update, P/U/D bank, gradients, and decisions are all expressed in that standardized coordinate system. Mean protected ranks by fold: `{json.dumps(clean(ranks), sort_keys=True)}`. A zero-rank fold is forced to Generic. No V6/V7 protected basis is reused.\n",
        encoding="utf-8")
    (ROOT / "DEVELOPMENT_GATE.md").write_text("# Development gate\n\n```json\n" + json.dumps(clean(gate), indent=2) + "\n```\n", encoding="utf-8")
    (ROOT / "CLAIM_AUDIT.md").write_text(
        f"# Claim audit\n\nTerminal state: **{terminal}**. The experiment is adaptive development evidence on 40 subjects. The exact future-DeltaBA regression claim remains unsupported. No confirmatory holdout, second-backbone, or WBCIC claim is made unless the strict development gate passes and a separate frozen run is completed.\n",
        encoding="utf-8")
    report = {
        "final_strongest_clean_Generic_BA": float(generic_values.mean()),
        "historical_83_775_legal": False, "final_PERSIST_Guard_BA": selected_perf["BA"],
        "delta_PERSIST_vs_Generic": selected_perf["delta_vs_strong_generic"],
        "paired_95_CI": [selected_perf["paired_CI95_L"], selected_perf["paired_CI95_U"]],
        "fold_positive_count": selected_perf["fold_positive_count"],
        "Generic_negative_transfer_rate": generic_harms / len(outcome),
        "PERSIST_negative_transfer_rate": selected_perf["negative_transfer_rate"],
        "Generic_harmed_subjects": generic_harms, "rescued_by_PERSIST": selected_perf["rescued_harms"],
        "newly_harmed": selected_perf["new_harms"], "PERSIST_risk_AUROC": selected_risk_row.AUROC,
        "confidence_AUROC": confidence_auc, "identity_AUROC": identity_auc,
        "PERSIST_beats_identity_AUROC": bool(float(selected_risk_row.AUROC) > float(identity_auc)) if selected_risk_row.AUROC is not None else False,
        "PERSIST_beats_confidence_update_performance": bool(selected_perf["BA"] > max(confidence_perf["BA"], next(row for row in control_metrics if row["method"] == "update-magnitude guard")["BA"])),
        "worst_quartile_change": worst_guard - worst_generic,
        "three_seed_results": seed_results.to_dict(orient="records"),
        "three_seed_summary_individual_models": seed_summary,
        "development_gate": "PASS" if gate_pass else "FAIL",
        "internal_holdout_accessed": False, "holdout_Generic_BA": None, "holdout_PERSIST_BA": None, "holdout_harm_reduction": None,
        "second_backbone_result": "NOT_REACHED", "WBCIC_result": "NOT_REACHED", "terminal_state": terminal,
        "strongest_justified_claim": "A bounded, source-only decision-grounded guard was evaluated on the authorized development split; its support is exactly the terminal state and metrics reported here.",
        "strongest_unsupported_claim": "PERSIST-Guard is confirmed on the sealed internal holdout, transfers across backbones, or generalizes to WBCIC.",
    }
    write_json(ROOT / "EXP4_FINAL_REPORT.json", report)
    (ROOT / "EXP4_FINAL_REPORT.md").write_text("# Exp4 final report\n\n```json\n" + json.dumps(clean(report), indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(clean({"gate": gate, "report": report}), indent=2), flush=True)


if __name__ == "__main__":
    main()
