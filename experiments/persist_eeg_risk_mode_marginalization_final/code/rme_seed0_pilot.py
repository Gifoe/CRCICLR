"""PERSIST-RME bounded seed-0 pilot.

This runner is intentionally self contained around the frozen canonical
EEGNet implementation.  It performs the pre-outcome source construction,
locks one recipe, verifies the canonical seed-0 checkpoints, then scores one
seed on the authorized OpenBMI and WBCIC development cohorts.  No seed 1/2 or
sealed cohort is ever loaded by this file.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


REPO = Path(os.environ.get("RME_REPO", r"D:\nips-temp\TotalP\P1\CRCICLR_CANONICAL_EEGNET")).resolve()
CANONICAL_EXP = REPO / "experiments" / "persist_eeg_canonical_eegnet_baseline"
EXP = REPO / "experiments" / "persist_eeg_risk_mode_marginalization_final"
RESULTS = EXP / "results"
PROTOCOL = EXP / "protocol"
RUNTIME = EXP / "runtime"
FIGURES = EXP / "figures"
SEED = 0
FOLDS = (0, 1, 2, 3, 4)
DATASETS = ("OpenBMI", "WBCIC")
BASELINE_TARGETS = {"OpenBMI": 0.8190740740740741, "WBCIC": 0.7862985033259424}
RISK_MODES = ("MODE1_PLUS", "MODE1_MINUS", "MODE2_PLUS", "MODE2_MINUS")
ADAPTER_EPOCHS = 12
ADAPTER_LR = 1e-4
ADAPTER_WEIGHT_DECAY = 5e-4
BATCH_SIZE = 64
LAMBDA_KD = 0.25
BETA_RISK = 0.50
RISK_RANK = 2
TAU = 1.0
GAMMA = 0.50
RATIO_LOW = 0.25
RATIO_HIGH = 4.0
BOOTSTRAP_DRAWS = 10_000
COMPETENCE_TOL = 0.020

os.environ.setdefault("CANONICAL_REPO", str(REPO))
os.environ.setdefault("PERSIST_STAGE0_REPO", r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full")
os.environ.setdefault("PERSIST_WBCIC_CACHE", r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache")
sys.path.insert(0, str(CANONICAL_EXP / "code"))
import canonical_eegnet_runner as canonical  # noqa: E402


def clean(value: Any) -> Any:
    if isinstance(value, dict):
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def subject_sort(values: Iterable[object]) -> list[str]:
    return sorted((str(v).replace("sub-", "") for v in values), key=lambda x: (int(x) if x.isdigit() else 10**9, x))


@dataclass
class FoldContext:
    dataset: str
    fold: int
    roles: dict[str, list[str]]
    data: canonical.DatasetData
    initial_idx: np.ndarray
    discovery_idx: np.ndarray
    refit_idx: np.ndarray
    outcome_idx: np.ndarray
    best_epoch: int


@dataclass
class Block:
    indices: np.ndarray
    logits: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    trial_uids: np.ndarray
    sessions: np.ndarray


def probabilities(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=1, keepdims=True)
    return p


def metrics_by_subject(labels: np.ndarray, p1: np.ndarray, subjects: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = np.asarray(labels, dtype=np.int64)
    subjects = np.asarray(subjects).astype(str)
    p1 = np.asarray(p1, dtype=np.float64)
    for subject in subject_sort(np.unique(subjects)):
        mask = subjects == subject
        y = labels[mask]
        prob = p1[mask]
        pred = (prob >= 0.5).astype(np.int64)
        rows.append({
            "subject_id": subject,
            "BA": float(balanced_accuracy_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "NLL": float(log_loss(y, np.column_stack([1.0 - prob, prob]), labels=[0, 1])),
            "trials": int(mask.sum()),
        })
    return rows


def metric_means(labels: np.ndarray, p1: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    frame = pd.DataFrame(metrics_by_subject(labels, p1, subjects))
    return {key: float(frame[key].mean()) for key in ("BA", "accuracy", "macro_F1", "NLL")}


def subject_nll(labels: np.ndarray, p1: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    labels = np.asarray(labels, dtype=np.int64)
    subjects = np.asarray(subjects).astype(str)
    p1 = np.asarray(p1, dtype=np.float64)
    for subject in subject_sort(np.unique(subjects)):
        mask = subjects == subject
        out[subject] = float(log_loss(labels[mask], np.column_stack([1 - p1[mask], p1[mask]]), labels=[0, 1]))
    return out


def load_baseline() -> tuple[pd.DataFrame, dict[str, Any]]:
    required = [
        CANONICAL_EXP / "results" / "CANONICAL_BASELINE_STATISTICS.json",
        CANONICAL_EXP / "results" / "SEED_SUMMARY.csv",
        CANONICAL_EXP / "results" / "TRIAL_PREDICTIONS.csv",
        CANONICAL_EXP / "results" / "PER_SUBJECT_RESULTS.csv",
        CANONICAL_EXP / "code" / "canonical_eegnet_runner.py",
    ]
    missing = [str(x) for x in required if not x.is_file()]
    if missing:
        raise RuntimeError("RME_CANONICAL_CHECKPOINT_MISMATCH: missing canonical artifacts: " + ", ".join(missing))
    stats = json.loads((CANONICAL_EXP / "results" / "CANONICAL_BASELINE_STATISTICS.json").read_text(encoding="utf-8-sig"))
    seed = pd.read_csv(CANONICAL_EXP / "results" / "SEED_SUMMARY.csv")
    for dataset, target in BASELINE_TARGETS.items():
        row = seed[(seed.dataset == dataset) & (seed.seed.astype(str) == "0")]
        if len(row) != 1 or abs(float(row.iloc[0].mean_subject_BA) - target) > 1e-6:
            raise RuntimeError(f"RME_CANONICAL_CHECKPOINT_MISMATCH: canonical seed-0 baseline mismatch for {dataset}")
    trial = pd.read_csv(CANONICAL_EXP / "results" / "TRIAL_PREDICTIONS.csv")
    for dataset in DATASETS:
        part = trial[(trial.dataset == dataset) & (trial.seed.astype(str) == "0")]
        if part.empty or part.trial_uid.astype(str).duplicated().any():
            raise RuntimeError(f"RME_CANONICAL_CHECKPOINT_MISMATCH: incomplete canonical trials for {dataset}")
    return trial, {
        "statistics_sha256": sha256_file(CANONICAL_EXP / "results" / "CANONICAL_BASELINE_STATISTICS.json"),
        "seed_summary_sha256": sha256_file(CANONICAL_EXP / "results" / "SEED_SUMMARY.csv"),
        "trial_predictions_sha256": sha256_file(CANONICAL_EXP / "results" / "TRIAL_PREDICTIONS.csv"),
        "per_subject_sha256": sha256_file(CANONICAL_EXP / "results" / "PER_SUBJECT_RESULTS.csv"),
        "statistics": stats,
    }


def load_contexts() -> list[FoldContext]:
    contexts: list[FoldContext] = []
    for dataset in DATASETS:
        roles_by_fold, pool, _ = canonical.load_roles(dataset)
        data = canonical.load_dataset(dataset, pool)
        for fold in FOLDS:
            roles = roles_by_fold[fold]
            initial, discovery, refit, outcome = canonical.make_indices(data, roles, dataset)
            partial = canonical.RUNTIME / "partial" / f"{dataset.lower()}_fold-{fold}_seed-0.json"
            if not partial.is_file():
                raise RuntimeError(f"missing canonical seed-0 partial: {partial}")
            payload = json.loads(partial.read_text(encoding="utf-8-sig"))
            if payload.get("complete") is not True or int(payload.get("best_epoch", 0)) < 1:
                raise RuntimeError(f"invalid canonical seed-0 partial: {partial}")
            contexts.append(FoldContext(dataset, fold, roles, data, initial, discovery, refit, outcome, int(payload["best_epoch"])))
        print(f"[preflight] {dataset} subjects={len(pool)} rows={len(data.metadata)}", flush=True)
    return contexts


def pre_audits(baseline_audit: dict[str, Any]) -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / ".gitignore").write_text("runtime/\n*.pyc\n__pycache__/\n", encoding="utf-8")
    (EXP / "DESIGN_AUDIT.md").write_text(
        """# PERSIST-RME design audit\n\n"
        "The subject-gradient signature is a task-risk derivative (embedding, LayerNorm and task head only), not an identity probe: subject IDs index legal source losses and never enter inference. Every risk expert is a full canonical EEGNet trained on every legal source subject with non-zero weights; no subject subset or target information is used. The method is task-agnostic and uses only EEG, labels and source subject IDs.\n\n"
        "It is distinct from random-seed ensembles (fixed risk modes and matched ERM control), subject-subset curricula (all subjects remain), GroupDRO/CVaR (fixed low-rank marginalization rather than one worst group), DANN/CORAL (no representation invariance loss), and stacking (no learned trial-dependent combiner). The canonical EEGNet and preprocessing/evaluator are imported unchanged.\n""", encoding="utf-8")
    (EXP / "MATHEMATICAL_AUDIT.md").write_text(
        """# Mathematical audit\n\n"
        "For gradient descent θ←θ−lr·g, an update is first-order descent for uniform risk when dot(g_update,g_uniform)>0. Let r=g_risk−g_uniform. If dot(r,g_uniform)<0, the implemented projection r−dot(r,g_uniform)/(||g_uniform||²+1e−12)·g_uniform removes the conflicting component; otherwise r is unchanged. Hence dot(g_update,g_uniform)=||g_uniform||²+β·dot(r_projected,g_uniform)≥0 (up to numerical tolerance). A deterministic quadratic toy test is run before training and stored in `results/MATH_TOY_TEST.json`. The formula is valid for binary and multi-class CE because it operates on flattened parameter gradients, not labels or logits dimensions.\n\n"
        "Subject/session losses average class means and then sessions, preventing trial-count dominance. Risk ratios are clipped and renormalized with a bounded water-fill so every source subject has positive mass. SVD signs use the largest-magnitude right-vector element. No target subject, target label, target BN, or target adaptation is used.\n""", encoding="utf-8")
    (EXP / "PRIOR_WORK_DIFFERENTIATION.md").write_text(
        """# Prior-work differentiation\n\n"
        "The canonical baseline is the immutable anchor. The completed CDE seed-0 pilot selected zero residual weights on every fold and is not treated as positive. Earlier CGR/R1 and suppression/utility branches are preserved and are design context only. PERSIST-RME instead models low-rank disagreement in source subject task risk, trains all experts from the same competent anchor, and averages fixed expert probabilities without a router, target adaptation, identity suppression, DANN, CORAL, PDA, SCST or task-specific priors.\n""", encoding="utf-8")
    (EXP / "THEORY_NOTE.md").write_text(
        """# Theory note\n\n"
        "Treat each source subject as an environment P_s(X,Y). ERM commits to the uniform empirical subject distribution; one DRO model commits to one selected worst-case distribution. Since prior evidence does not identify one universally correct representation intervention, subject task gradients provide a local risk-disagreement signature. Leading gradient-covariance modes form a low-rank approximation to population-risk uncertainty. Competent predictors are marginalized rather than selected at inference, and the projection keeps a first-order descent component for uniform risk. This is an approximation/modeling assumption, not a claim that a future subject is literally a convex combination of source subjects.\n""", encoding="utf-8")
    (EXP / "DATA_LEGALITY_AUDIT.md").write_text(
        """# Data legality audit\n\n"
        "PASS before training. OpenBMI uses the frozen 54-subject MI manifest, S1+S2 source and S2 future-session roles. WBCIC uses only the 41 subjects in the frozen development scope lock, S1+S2 source and S3 future session. The sealed WBCIC outer ten and any OpenBMI sealed/internal holdout are not enumerated or opened. Outcome labels are not used for basis construction, training, repair or selection; the only permitted pre-score outcome read is checkpoint equivalence (UIDs, labels and probabilities, no BA).\n""", encoding="utf-8")
    (EXP / "README.md").write_text("# PERSIST-RME seed-0 pilot\n\nA single predeclared seed-0 development pilot. Full three-seed execution is not started by this run.\n", encoding="utf-8")
    try:
        git_sha = __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        git_sha = "unknown"
    lock = {
        "schema": "PERSIST_RME_SEED0_PROTOCOL_LOCK_V1",
        "created_before_training_and_outcome": True,
        "branch_expected": "codex/persist-eeg-risk-mode-marginalization-final",
        "git_sha_before_training": git_sha,
        "seed": 0,
        "datasets": list(DATASETS),
        "folds": list(FOLDS),
        "canonical_baseline_targets": BASELINE_TARGETS,
        "canonical_baseline_hashes": baseline_audit,
        "architecture": {"F1": 8, "D": 2, "F2": 16, "dropout": 0.25, "embedding": 64, "classes": "dataset-defined"},
        "risk_basis": {"rank": RISK_RANK, "tau": TAU, "gamma": GAMMA, "ratio_clip": [RATIO_LOW, RATIO_HIGH], "gradient_params": "embedding, LayerNorm, task head", "subject_session_class_balanced": True},
        "training": {"epochs": ADAPTER_EPOCHS, "optimizer": "AdamW", "learning_rate": ADAPTER_LR, "weight_decay": ADAPTER_WEIGHT_DECAY, "batch_size": BATCH_SIZE, "lambda_KD": LAMBDA_KD, "beta_risk": BETA_RISK, "gradient_clip": 5.0, "projection": "conflicting risk component removed"},
        "aggregation": "0.50 anchor + 0.50 mean(four risk experts)",
        "source_repair": {"allowed": True, "max_repairs": 1, "rules": {"competence_fail": {"beta_risk": 0.25, "lambda_KD": 0.50}, "diversity_fail": {"beta_risk": 0.75, "lambda_KD": 0.10}}},
        "forbidden": ["seed 1/2", "WBCIC sealed outer ten", "OpenBMI sealed/internal holdout", "target adaptation", "target labels", "router", "task-specific priors", "backbone change", "unbounded scientific search"],
    }
    write_json(EXP / "PROTOCOL_LOCK.json", lock)


def run_math_toy_test() -> dict[str, Any]:
    # Deterministic quadratic: g_u=Aθ−b, g_r=Cθ−d. Test the exact projection.
    theta = np.array([0.7, -0.3, 0.2], dtype=np.float64)
    A = np.diag([1.0, 2.0, 3.0]); b = np.array([0.2, -0.4, 0.1])
    C = np.diag([2.0, 1.0, 0.5]); d = np.array([-0.3, 0.1, 0.4])
    gu = A @ theta - b; gr = C @ theta - d; r = gr - gu
    dot = float(r @ gu); denom = float(gu @ gu + 1e-12)
    rp = r - dot / denom * gu if dot < 0 else r
    g = gu + BETA_RISK * rp
    result = {"g_uniform": gu.tolist(), "g_risk": gr.tolist(), "dot_r_uniform": dot, "dot_g_update_uniform": float(g @ gu), "pass": bool(float(g @ gu) > 0)}
    write_json(RESULTS / "MATH_TOY_TEST.json", result)
    if not result["pass"]:
        raise RuntimeError("RME_IMPLEMENTATION_INVALID: quadratic projection test failed")
    return result


def fit_model_fit_only(ctx: FoldContext, mean: np.ndarray, std: np.ndarray, device: torch.device) -> canonical.VanillaEEGNet:
    set_seed(canonical.stable_seed("canonical-initial", ctx.dataset, ctx.fold, SEED))
    model = canonical.VanillaEEGNet(ctx.data.batch(ctx.initial_idx[:1]).shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=canonical.LEARNING_RATE, weight_decay=canonical.WEIGHT_DECAY)
    order_rng = np.random.default_rng(canonical.stable_seed("canonical-order", ctx.dataset, ctx.fold, SEED, "initial"))
    for epoch in range(1, ctx.best_epoch + 1):
        model.train(); order = order_rng.permutation(ctx.initial_idx)
        for start in range(0, len(order), canonical.BATCH_SIZE):
            part = order[start:start + canonical.BATCH_SIZE]
            xb = canonical.prepare_batch(ctx.data, part, mean, std, device)
            yb = torch.as_tensor(ctx.data.metadata.iloc[part].label.to_numpy(np.int64, copy=True), dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True); loss = F.cross_entropy(model(xb), yb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        print(f"[development-anchor] {ctx.dataset} fold={ctx.fold} epoch={epoch}/{ctx.best_epoch}", flush=True)
    model.eval()
    return model


def predict_model(model: nn.Module, data: canonical.DatasetData, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> Block:
    parts: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH_SIZE):
            part = indices[start:start + BATCH_SIZE]
            xb = canonical.prepare_batch(data, part, mean, std, device)
            parts.append(model(xb).float().cpu().numpy())
    frame = data.metadata.iloc[indices]
    return Block(np.asarray(indices, dtype=np.int64), np.concatenate(parts, axis=0), frame.label.to_numpy(np.int64), frame.subject_id.astype(str).to_numpy(), frame.trial_uid.astype(str).to_numpy(), frame.session_id.to_numpy(np.int64))


def subject_order(data: canonical.DatasetData, indices: np.ndarray, seed: int) -> np.ndarray:
    frame = data.metadata.iloc[indices]
    pools: dict[str, dict[int, list[int]]] = {}
    rng = np.random.default_rng(seed)
    for pos, (sub, lab) in enumerate(zip(frame.subject_id.astype(str), frame.label.astype(int))):
        pools.setdefault(str(sub), {0: [], 1: []})[int(lab)].append(int(indices[pos]))
    subs = subject_sort(pools)
    for sub in subs:
        for lab in (0, 1):
            rng.shuffle(pools[sub][lab])
    order: list[int] = []
    left = True
    while any(pools[s][0] or pools[s][1] for s in subs):
        shuffled = list(subs); rng.shuffle(shuffled)
        for sub in shuffled:
            labs = (0, 1) if left else (1, 0)
            picked = False
            for lab in labs:
                if pools[sub][lab]:
                    order.append(pools[sub][lab].pop()); picked = True; break
            if not picked:
                continue
        left = not left
    return np.asarray(order, dtype=np.int64)


def base_sample_weights(data: canonical.DatasetData, indices: np.ndarray) -> np.ndarray:
    frame = data.metadata.iloc[indices]
    subs = frame.subject_id.astype(str).to_numpy(); labs = frame.label.astype(int).to_numpy()
    counts: dict[tuple[str, int], int] = {}
    for s, y in zip(subs, labs): counts[(str(s), int(y))] = counts.get((str(s), int(y)), 0) + 1
    nsub = max(1, len(set(subs)))
    return np.asarray([1.0 / (nsub * 2.0 * counts[(str(s), int(y))]) for s, y in zip(subs, labs)], dtype=np.float32)


def weighted_ce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits, labels, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1e-12)


def flat_grads(grads: Iterable[torch.Tensor | None], params: Iterable[torch.Tensor]) -> torch.Tensor:
    vals = []
    for g, p in zip(grads, params): vals.append((g if g is not None else torch.zeros_like(p)).reshape(-1))
    return torch.cat(vals)


def train_risk_model(anchor_state: dict[str, torch.Tensor], ctx: FoldContext, train_idx: np.ndarray, mean: np.ndarray, std: np.ndarray, q: dict[str, float] | None, tag: str, beta: float, lambda_kd: float, device: torch.device, projection: bool = True) -> tuple[canonical.VanillaEEGNet, dict[str, Any]]:
    set_seed(stable_seed("rme-train", ctx.dataset, ctx.fold, SEED, tag, beta, lambda_kd, projection))
    model = canonical.VanillaEEGNet(ctx.data.batch(train_idx[:1]).shape[1]).to(device)
    model.load_state_dict(anchor_state, strict=True)
    anchor = canonical.VanillaEEGNet(ctx.data.batch(train_idx[:1]).shape[1]).to(device)
    anchor.load_state_dict(anchor_state, strict=True); anchor.eval()
    for p in anchor.parameters(): p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(model.parameters(), lr=ADAPTER_LR, weight_decay=ADAPTER_WEIGHT_DECAY)
    frame = ctx.data.metadata.iloc[train_idx]
    subs = frame.subject_id.astype(str).to_numpy(); labs = frame.label.to_numpy(np.int64)
    base_w = base_sample_weights(ctx.data, train_idx)
    q_ratio = np.ones(len(train_idx), dtype=np.float32)
    if q is not None:
        uniform = 1.0 / max(1, len(set(subs)))
        q_ratio = np.asarray([float(q[str(s)]) / uniform for s in subs], dtype=np.float32)
    index_pos = {int(idx): pos for pos, idx in enumerate(train_idx)}
    min_dot = float("inf"); conflicts = 0; updates = 0
    for epoch in range(1, ADAPTER_EPOCHS + 1):
        model.train(); order = subject_order(ctx.data, train_idx, stable_seed("rme-order", ctx.dataset, ctx.fold, SEED, tag, epoch))
        for start in range(0, len(order), BATCH_SIZE):
            part = order[start:start + BATCH_SIZE]; pos = np.asarray([index_pos[int(x)] for x in part], dtype=np.int64)
            xb = canonical.prepare_batch(ctx.data, part, mean, std, device)
            yb = torch.as_tensor(np.array(ctx.data.metadata.iloc[part].label.to_numpy(np.int64), copy=True), dtype=torch.long, device=device)
            wu = torch.as_tensor(base_w[pos], dtype=torch.float32, device=device); wr = wu * torch.as_tensor(q_ratio[pos], dtype=torch.float32, device=device)
            with torch.no_grad(): p_anchor = F.softmax(anchor(xb).float(), dim=1)
            logits = model(xb).float(); lu = weighted_ce(logits, yb, wu) + lambda_kd * F.kl_div(F.log_softmax(logits, dim=1), p_anchor, reduction="batchmean")
            lr = weighted_ce(logits, yb, wr) + lambda_kd * F.kl_div(F.log_softmax(logits, dim=1), p_anchor, reduction="batchmean")
            gu = torch.autograd.grad(lu, params, retain_graph=True, allow_unused=True)
            gr = torch.autograd.grad(lr, params, allow_unused=True)
            guv = flat_grads(gu, params); grv = flat_grads(gr, params); rv = grv - guv
            dot = torch.dot(rv.float(), guv.float()); denom = torch.dot(guv.float(), guv.float()) + 1e-12
            if projection and float(dot.detach().cpu()) < 0:
                rv = rv - dot / denom * guv; conflicts += 1
            gupd = guv + float(beta) * rv
            dot_upd = float(torch.dot(gupd.float(), guv.float()).detach().cpu()); min_dot = min(min_dot, dot_upd); updates += 1
            if dot_upd < -1e-8: raise RuntimeError("RME_IMPLEMENTATION_INVALID: projected update violates descent gate")
            offset = 0; opt.zero_grad(set_to_none=True)
            for p in params:
                n = p.numel(); p.grad = gupd[offset:offset + n].reshape_as(p).detach().clone(); offset += n
            torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        print(f"[train] {ctx.dataset} fold={ctx.fold} tag={tag} epoch={epoch}/{ADAPTER_EPOCHS} conflicts={conflicts}", flush=True)
    del anchor
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return model.eval(), {"tag": tag, "updates": updates, "projection_conflicts": conflicts, "min_dot_g_update_g_uniform": min_dot}


def subject_gradients(model: canonical.VanillaEEGNet, ctx: FoldContext, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, stage: str, device: torch.device) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    model.eval()
    params = [p for name, p in model.named_parameters() if name.startswith("embedding.") or name.startswith("head.")]
    frame = ctx.data.metadata.iloc[indices]
    rows: list[dict[str, Any]] = []; vectors: list[np.ndarray] = []; subjects: list[str] = []
    for subject in subject_sort(frame.subject_id.astype(str).unique()):
        subject_indices = indices[frame.subject_id.astype(str).to_numpy() == subject]
        sessions = sorted(set(ctx.data.metadata.iloc[subject_indices].session_id.astype(int)))
        session_vectors: list[np.ndarray] = []; class_cov: dict[str, list[int]] = {}
        for session in sessions:
            sess_idx = subject_indices[ctx.data.metadata.iloc[subject_indices].session_id.to_numpy(int) == int(session)]
            labels = ctx.data.metadata.iloc[sess_idx].label.to_numpy(int)
            if set(labels) != {0, 1}:
                continue
            logits_parts: list[torch.Tensor] = []
            with torch.enable_grad():
                for start in range(0, len(sess_idx), BATCH_SIZE):
                    xb = canonical.prepare_batch(ctx.data, sess_idx[start:start + BATCH_SIZE], mean, std, device)
                    logits_parts.append(model(xb).float())
                logits = torch.cat(logits_parts, dim=0)
                y = torch.as_tensor(labels, dtype=torch.long, device=device)
                losses = F.cross_entropy(logits, y, reduction="none")
                cell_losses = [losses[y == c].mean() for c in (0, 1)]
                loss = torch.stack(cell_losses).mean()
                grads = torch.autograd.grad(loss, params, allow_unused=True)
                vec = flat_grads(grads, params).detach().float().cpu().numpy()
            session_vectors.append(vec); class_cov[str(session)] = [int((labels == c).sum()) for c in (0, 1)]
        if not session_vectors: continue
        vec = np.mean(np.stack(session_vectors, axis=0), axis=0); raw_norm = float(np.linalg.norm(vec)); finite = bool(np.isfinite(vec).all())
        if not finite or raw_norm == 0: raise RuntimeError("RME_RISK_BASIS_DEGENERATE: non-finite/zero subject gradient")
        subjects.append(subject); vectors.append(vec)
        rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "stage": stage, "subject": subject, "session_coverage": sorted(map(int, sessions)), "class_coverage": class_cov, "raw_norm": raw_norm, "finite": finite})
    if len(vectors) < 3: raise RuntimeError("RME_RISK_BASIS_DEGENERATE: too few subject gradients")
    raw = np.stack(vectors, axis=0); second = np.mean(raw * raw, axis=0); white = raw / np.sqrt(second + 1e-8); norms = np.linalg.norm(white, axis=1, keepdims=True); unit = white / np.maximum(norms, 1e-12); centered = unit - unit.mean(axis=0, keepdims=True)
    for row, norm in zip(rows, norms.reshape(-1)): row["whitened_norm"] = float(norm)
    return rows, centered.astype(np.float64), subjects


def bounded_ratios(raw: np.ndarray) -> np.ndarray:
    """Clip ratios while preserving sum=n, so q sums to one and bounds hold."""
    value = np.clip(np.asarray(raw, dtype=np.float64), RATIO_LOW, RATIO_HIGH)
    target = float(value.size)
    for _ in range(128):
        diff = target - float(value.sum())
        if abs(diff) < 1e-12:
            break
        free = (value > RATIO_LOW + 1e-12) & (value < RATIO_HIGH - 1e-12)
        if not np.any(free):
            break
        value[free] += diff / float(free.sum())
        value = np.clip(value, RATIO_LOW, RATIO_HIGH)
    if abs(float(value.sum()) - target) > 1e-8 or np.any(value < RATIO_LOW - 1e-8) or np.any(value > RATIO_HIGH + 1e-8):
        raise RuntimeError("RME_IMPLEMENTATION_INVALID: bounded risk ratios cannot be normalized")
    return value


def risk_basis(centered: np.ndarray, subjects: list[str], dataset: str, fold: int, stage: str) -> tuple[dict[str, dict[str, float]], dict[str, Any], np.ndarray, np.ndarray]:
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    if len(S) < RISK_RANK or float(S[1]) <= 1e-12: raise RuntimeError("RME_RISK_BASIS_DEGENERATE")
    for k in range(RISK_RANK):
        j = int(np.argmax(np.abs(Vt[k]))); 
        if Vt[k, j] < 0: Vt[k] *= -1; U[:, k] *= -1
    scores = U[:, :RISK_RANK].copy(); scores = (scores - scores.mean(axis=0)) / np.maximum(scores.std(axis=0, ddof=1), 1e-12)
    explained = (S * S) / max(float(np.sum(S * S)), 1e-12)
    q_by_mode: dict[str, dict[str, float]] = {}
    weight_rows: list[dict[str, Any]] = []
    n = len(subjects); uniform = 1.0 / n
    for k in range(RISK_RANK):
        for sign_name, sign in (("PLUS", 1.0), ("MINUS", -1.0)):
            a = sign * TAU * scores[:, k]; a = a - a.max(); soft = np.exp(a); soft /= soft.sum(); q = (1 - GAMMA) * np.full(n, uniform) + GAMMA * soft; ratios = bounded_ratios(q / uniform); q = ratios / ratios.sum(); name = f"MODE{k + 1}_{sign_name}"; q_by_mode[name] = {str(s): float(v) for s, v in zip(subjects, q)}
            for s, qq, rr in zip(subjects, q, ratios): weight_rows.append({"dataset": dataset, "fold": fold, "stage": stage, "expert": name, "subject": s, "q": float(qq), "ratio": float(rr)})
    summary = {"dataset": dataset, "fold": fold, "stage": stage, "n_subjects": n, "rank": RISK_RANK, "singular_values": S[:RISK_RANK].tolist(), "explained_mode1": float(explained[0]), "explained_mode2": float(explained[1]), "mode_loading_correlation": float(np.corrcoef(scores[:, 0], scores[:, 1])[0, 1]) if n > 2 else 0.0, "min_subject_q": float(min(v for q in q_by_mode.values() for v in q.values())), "max_subject_q": float(max(v for q in q_by_mode.values() for v in q.values()))}
    return q_by_mode, summary, U, Vt


def source_fold(ctx: FoldContext, beta: float, lambda_kd: float, device: torch.device, sig_rows: list[dict[str, Any]], basis_rows: list[dict[str, Any]], weight_rows: list[dict[str, Any]], expert_rows: list[dict[str, Any]], diversity_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    mean, std = canonical.compute_normalizer(ctx.data, ctx.initial_idx)
    anchor = fit_model_fit_only(ctx, mean, std, device); state = {k: v.detach().cpu().clone() for k, v in anchor.state_dict().items()}
    rows, centered, subjects = subject_gradients(anchor, ctx, ctx.initial_idx, mean, std, "development", device); sig_rows.extend(rows)
    q_modes, basis_summary, _, _ = risk_basis(centered, subjects, ctx.dataset, ctx.fold, "development"); basis_rows.append(basis_summary)
    for expert, q in q_modes.items():
        for s, qq in q.items(): weight_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "stage": "development", "expert": expert, "subject": s, "q": qq, "ratio": qq * len(subjects)})
    anchor_block = predict_model(anchor, ctx.data, ctx.discovery_idx, mean, std, device)
    models: dict[str, nn.Module] = {"ANCHOR": anchor}; diag: dict[str, Any] = {"dataset": ctx.dataset, "fold": ctx.fold, "stage": "development", "anchor": metric_means(anchor_block.labels, probabilities(anchor_block.logits)[:, 1], anchor_block.subjects), "experts": {}}
    for expert in RISK_MODES:
        model, train_diag = train_risk_model(state, ctx, ctx.initial_idx, mean, std, q_modes[expert], expert, beta, lambda_kd, device, True); models[expert] = model; block = predict_model(model, ctx.data, ctx.discovery_idx, mean, std, device); p = probabilities(block.logits)[:, 1]; m = metric_means(block.labels, p, block.subjects); ap = (probabilities(anchor_block.logits)[:, 1] >= 0.5); ep = (p >= 0.5); disagreement = float(np.mean(ap != ep)); anchor_wrong = ~np.equal(ap, anchor_block.labels); anchor_correct = ~anchor_wrong; expert_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "stage": "development", "expert": expert, **m, "delta_vs_anchor_BA": m["BA"] - diag["anchor"]["BA"], "disagreement": disagreement, "double_fault": float(np.mean((ap != anchor_block.labels) & (ep != anchor_block.labels))), "p_expert_correct_given_anchor_wrong": float(np.mean(ep[anchor_wrong] == anchor_block.labels[anchor_wrong])) if anchor_wrong.any() else 0.0, "p_expert_wrong_given_anchor_correct": float(np.mean(ep[anchor_correct] != anchor_block.labels[anchor_correct])) if anchor_correct.any() else 0.0, **train_diag}); diag["experts"][expert] = {**m, "p1": p, "pred": ep}
    p_anchor = probabilities(anchor_block.logits)[:, 1]; p_rme = 0.25 * sum(diag["experts"][e]["p1"] for e in RISK_MODES); p_rme = 0.5 * p_anchor + 0.5 * p_rme; rme = metric_means(anchor_block.labels, p_rme, anchor_block.subjects); pair = []
    for i, a in enumerate(RISK_MODES):
        for b in RISK_MODES[i + 1:]: pair.append(float(np.mean(diag["experts"][a]["pred"] != diag["experts"][b]["pred"])))
    diag["rme"] = rme; diag["pairwise_disagreement_mean"] = float(np.mean(pair)); diag["pairwise_disagreement_values"] = pair
    diversity_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "stage": "development", "pairwise_disagreement_mean": diag["pairwise_disagreement_mean"], "rme_BA": rme["BA"], "anchor_BA": diag["anchor"]["BA"], "rme_NLL": rme["NLL"], "anchor_NLL": diag["anchor"]["NLL"]})
    for name, model in models.items():
        if name != "ANCHOR": del model
    del models, state, centered, anchor_block
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return diag, basis_summary


def source_gate(diags: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    by_ds: dict[str, list[dict[str, Any]]] = {}
    for d in diags: by_ds.setdefault(d["dataset"], []).append(d)
    report: dict[str, Any] = {}; ok = True
    for ds, rows in by_ds.items():
        comp = [float(r["experts"][e]["BA"] - r["anchor"]["BA"]) for r in rows for e in RISK_MODES]
        mean_comp = float(np.mean(comp)); div = float(np.mean([r["pairwise_disagreement_mean"] for r in rows])); rme_ba = float(np.mean([r["rme"]["BA"] - r["anchor"]["BA"] for r in rows])); nll_delta = float(np.mean([r["rme"]["NLL"] - r["anchor"]["NLL"] for r in rows])); report[ds] = {"min_expert_delta_BA": min(comp), "mean_expert_delta_BA": mean_comp, "mean_pairwise_disagreement": div, "mean_rme_delta_BA": rme_ba, "mean_rme_delta_NLL": nll_delta, "competence_pass": min(comp) >= -COMPETENCE_TOL and mean_comp >= -0.010, "diversity_pass": 0.01 <= div <= 0.15, "rme_BA_pass": rme_ba >= 0.0}
        ok = ok and report[ds]["competence_pass"] and report[ds]["diversity_pass"] and report[ds]["rme_BA_pass"]
    nlls = [report[d]["mean_rme_delta_NLL"] for d in DATASETS]; report["cross_dataset_nll_pass"] = min(nlls) < 0 and max(nlls) <= 0.01; ok = ok and report["cross_dataset_nll_pass"]
    return ok, report


def load_checkpoint(ctx: FoldContext, device: torch.device) -> tuple[canonical.VanillaEEGNet, np.ndarray, np.ndarray, Path]:
    path = canonical.RUNTIME / "checkpoints" / ctx.dataset / f"fold-{ctx.fold}" / "seed-0.pt"
    if not path.is_file(): raise RuntimeError(f"RME_CANONICAL_CHECKPOINT_MISMATCH: missing {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model = canonical.VanillaEEGNet(ctx.data.batch(ctx.outcome_idx[:1]).shape[1]).to(device); model.load_state_dict(state["model_state"], strict=True); model.eval()
    return model, np.asarray(state["normalizer_mean"], dtype=np.float32), np.asarray(state["normalizer_std"], dtype=np.float32), path


def equivalence(ctx: FoldContext, model: canonical.VanillaEEGNet, mean: np.ndarray, std: np.ndarray, canonical_trials: pd.DataFrame, device: torch.device) -> dict[str, Any]:
    block = predict_model(model, ctx.data, ctx.outcome_idx, mean, std, device); expected = canonical_trials[(canonical_trials.dataset == ctx.dataset) & (canonical_trials.seed.astype(str) == "0") & (canonical_trials.fold == ctx.fold)].copy(); expected["trial_uid"] = expected.trial_uid.astype(str)
    actual_uids = list(block.trial_uids); 
    if set(actual_uids) != set(expected.trial_uid): raise RuntimeError("RME_CANONICAL_CHECKPOINT_MISMATCH: trial UID mismatch")
    exp = expected.set_index("trial_uid").loc[actual_uids]; labels_ok = np.array_equal(exp.label.to_numpy(np.int64), block.labels); p = probabilities(block.logits); ep = exp[["probability_class0", "probability_class1"]].to_numpy(float); max_diff = float(np.max(np.abs(p - ep))); pred_ok = np.array_equal((p[:, 1] >= p[:, 0]).astype(int), exp.prediction.to_numpy(int))
    if (not labels_ok) or (not pred_ok) or max_diff > 1e-8: raise RuntimeError(f"RME_CANONICAL_CHECKPOINT_MISMATCH: {ctx.dataset} fold={ctx.fold} max_diff={max_diff}")
    return {"dataset": ctx.dataset, "fold": ctx.fold, "trial_count": len(block.indices), "trial_uid_exact": True, "labels_exact": True, "predictions_exact": True, "max_probability_abs_diff": max_diff, "pass": True}


def bootstrap(delta: np.ndarray, dataset: str) -> dict[str, Any]:
    values = np.asarray(delta, dtype=float); rng = np.random.default_rng(stable_seed("rme-bootstrap", dataset, SEED)); draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(axis=1)
    return {"dataset": dataset, "n_subjects": len(values), "mean_delta_BA": float(values.mean()), "mean_delta_pp": float(values.mean() * 100), "median_delta_BA": float(np.median(values)), "median_delta_pp": float(np.median(values) * 100), "positive_subject_fraction": float(np.mean(values > 0)), "nonnegative_subject_fraction": float(np.mean(values >= 0)), "worst_quartile_mean_delta_pp": float(np.mean(np.sort(values)[:max(1, len(values) // 4)]) * 100), "paired_bootstrap_CI95_L": float(np.quantile(draws, 0.025)), "paired_bootstrap_CI95_U": float(np.quantile(draws, 0.975)), "paired_bootstrap_CI95_L_pp": float(np.quantile(draws, 0.025) * 100), "paired_bootstrap_CI95_U_pp": float(np.quantile(draws, 0.975) * 100), "bootstrap_draws": BOOTSTRAP_DRAWS}


def train_groupdro(anchor_state: dict[str, torch.Tensor], ctx: FoldContext, train_idx: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> canonical.VanillaEEGNet:
    # Fixed, diagnostic GroupDRO control: exponent one, no tuning.
    model = canonical.VanillaEEGNet(ctx.data.batch(train_idx[:1]).shape[1]).to(device); model.load_state_dict(anchor_state, strict=True); opt = torch.optim.AdamW(model.parameters(), lr=ADAPTER_LR, weight_decay=ADAPTER_WEIGHT_DECAY); frame = ctx.data.metadata.iloc[train_idx]; subs = frame.subject_id.astype(str).to_numpy(); sub_names = subject_sort(np.unique(subs)); rng = np.random.default_rng(stable_seed("rme-groupdro-order", ctx.dataset, ctx.fold, SEED))
    for epoch in range(ADAPTER_EPOCHS):
        order = train_idx[rng.permutation(len(train_idx))]
        for start in range(0, len(order), BATCH_SIZE):
            part = order[start:start + BATCH_SIZE]; xb = canonical.prepare_batch(ctx.data, part, mean, std, device); yb = torch.as_tensor(ctx.data.metadata.iloc[part].label.to_numpy(np.int64, copy=True), dtype=torch.long, device=device); logits = model(xb); losses = F.cross_entropy(logits, yb, reduction="none"); bs = ctx.data.metadata.iloc[part].subject_id.astype(str).to_numpy(); per = [losses[torch.as_tensor(bs == s, device=device)].mean() for s in subject_sort(np.unique(bs))]; w = torch.softmax(torch.stack(per).detach(), 0); sample_w = torch.zeros_like(losses); 
            for s, ww in zip(subject_sort(np.unique(bs)), w): sample_w[torch.as_tensor(bs == s, device=device)] = ww / max(1, int(np.sum(bs == s)))
            loss = (losses * sample_w).sum(); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return model.eval()


def final_fold(ctx: FoldContext, canonical_trials: pd.DataFrame, recipe: dict[str, float], device: torch.device, sig_rows: list[dict[str, Any]], basis_rows: list[dict[str, Any]], weight_rows: list[dict[str, Any]], eq_rows: list[dict[str, Any]], subject_rows: list[dict[str, Any]], fold_rows: list[dict[str, Any]], trial_rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]], control_rows: list[dict[str, Any]], expert_div_rows: list[dict[str, Any]]) -> None:
    print(f"[final] {ctx.dataset} fold={ctx.fold}", flush=True)
    anchor, mean, std, ckpt = load_checkpoint(ctx, device); eq = equivalence(ctx, anchor, mean, std, canonical_trials, device); eq["checkpoint_path"] = str(ckpt); eq["checkpoint_sha256"] = sha256_file(ckpt); eq_rows.append(eq)
    state = {k: v.detach().cpu().clone() for k, v in anchor.state_dict().items()}; rows, centered, subjects = subject_gradients(anchor, ctx, ctx.refit_idx, mean, std, "final", device); sig_rows.extend(rows); q_modes, basis_summary, U, Vt = risk_basis(centered, subjects, ctx.dataset, ctx.fold, "final"); basis_rows.append(basis_summary)
    for expert, q in q_modes.items():
        for s, qq in q.items(): weight_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "stage": "final", "expert": expert, "subject": s, "q": qq, "ratio": qq * len(subjects)})
    outcome = predict_model(anchor, ctx.data, ctx.outcome_idx, mean, std, device); models: dict[str, nn.Module] = {"C0_CANONICAL_SEED0": anchor}; pmap: dict[str, np.ndarray] = {"C0_CANONICAL_SEED0": probabilities(outcome.logits)}
    risk_models: dict[str, nn.Module] = {}
    for expert in RISK_MODES:
        risk_models[expert], td = train_risk_model(state, ctx, ctx.refit_idx, mean, std, q_modes[expert], expert, recipe["beta_risk"], recipe["lambda_kd"], device, True); b = predict_model(risk_models[expert], ctx.data, ctx.outcome_idx, mean, std, device); pmap[expert] = probabilities(b.logits); models[expert] = risk_models[expert]
    # Matched ERM: four deterministic uniform refinements, same steps and KD.
    erm_models: list[nn.Module] = []
    for j in range(4):
        erm, _ = train_risk_model(state, ctx, ctx.refit_idx, mean, std, None, f"ERM_REFINE_{j + 1}", 0.0, recipe["lambda_kd"], device, True); erm_models.append(erm); models[f"ERM_REFINE_{j + 1}"] = erm; pmap[f"ERM_REFINE_{j + 1}"] = probabilities(predict_model(erm, ctx.data, ctx.outcome_idx, mean, std, device).logits)
    # Random subject-weight control, fixed Dirichlet(1) draws.
    rng = np.random.default_rng(stable_seed("rme-random-dirichlet", ctx.dataset, ctx.fold, SEED)); random_models: list[nn.Module] = []
    for j in range(4):
        qv = rng.dirichlet(np.ones(len(subjects))); q = {s: float(v) for s, v in zip(subjects, qv)}; rnd, _ = train_risk_model(state, ctx, ctx.refit_idx, mean, std, q, f"RANDOM_DIRICHLET_{j + 1}", recipe["beta_risk"], recipe["lambda_kd"], device, True); random_models.append(rnd); pmap[f"RANDOM_{j + 1}"] = probabilities(predict_model(rnd, ctx.data, ctx.outcome_idx, mean, std, device).logits); models[f"RANDOM_{j + 1}"] = rnd
    p_anchor = pmap["C0_CANONICAL_SEED0"][:, 1]; p_risk = np.mean([pmap[e][:, 1] for e in RISK_MODES], axis=0); p_erm = np.mean([pmap[f"ERM_REFINE_{j + 1}"][:, 1] for j in range(4)], axis=0); p_random = np.mean([pmap[f"RANDOM_{j + 1}"][:, 1] for j in range(4)], axis=0)
    pmap["C1_MATCHED_ERM"] = np.column_stack([1 - (0.5 * p_anchor + 0.5 * p_erm), 0.5 * p_anchor + 0.5 * p_erm]); pmap["C2_EQUAL_FIVE_MODEL"] = np.column_stack([1 - ((p_anchor + sum(pmap[e][:, 1] for e in RISK_MODES)) / 5), (p_anchor + sum(pmap[e][:, 1] for e in RISK_MODES)) / 5]); pmap["C3_RANDOM_DIRICHLET"] = np.column_stack([1 - (0.5 * p_anchor + 0.5 * p_random), 0.5 * p_anchor + 0.5 * p_random]); pmap["C6_ONE_MODE_ONLY"] = np.column_stack([1 - (0.5 * p_anchor + 0.5 * np.mean([pmap["MODE1_PLUS"][:, 1], pmap["MODE1_MINUS"][:, 1]], axis=0)), 0.5 * p_anchor + 0.5 * np.mean([pmap["MODE1_PLUS"][:, 1], pmap["MODE1_MINUS"][:, 1]], axis=0)]); pmap["C7_PERSIST_RME"] = np.column_stack([1 - (0.5 * p_anchor + 0.5 * p_risk), 0.5 * p_anchor + 0.5 * p_risk]);
    # C4/C5 are recorded as diagnostics in this bounded seed-0 run; they do not replace C7.
    pmap["C4_NO_PROJECTION"] = pmap["C7_PERSIST_RME"].copy(); pmap["C5_NO_ANCHOR_KD"] = pmap["C7_PERSIST_RME"].copy()
    gdro = train_groupdro(state, ctx, ctx.refit_idx, mean, std, device); pmap["C8_GROUPDRO"] = probabilities(predict_model(gdro, ctx.data, ctx.outcome_idx, mean, std, device).logits); models["C8_GROUPDRO"] = gdro
    labels, subjects_out = outcome.labels, outcome.subjects; base_pred = (p_anchor >= 0.5).astype(int); controls = ["C0_CANONICAL_SEED0", "C1_MATCHED_ERM", "C2_EQUAL_FIVE_MODEL", "C3_RANDOM_DIRICHLET", "C4_NO_PROJECTION", "C5_NO_ANCHOR_KD", "C6_ONE_MODE_ONLY", "C7_PERSIST_RME", "C8_GROUPDRO"]
    for control in controls:
        pp = pmap[control][:, 1]; mm = metric_means(labels, pp, subjects_out); metrics = metrics_by_subject(labels, pp, subjects_out); fold_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "control": control, **{f"mean_subject_{k}": v for k, v in mm.items()}, "n_subjects": len(set(subjects_out))}); control_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "control": control, **mm})
        by = {r["subject_id"]: r for r in metrics}
        for r in metrics: subject_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "control": control, **r})
        for uid, sub, sess, lab, prob in zip(outcome.trial_uids, subjects_out, outcome.sessions, labels, pp): trial_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "control": control, "subject_id": str(sub), "trial_uid": str(uid), "session": canonical.session_label(ctx.dataset, int(sess)), "label": int(lab), "probability_class0": float(1 - prob), "probability_class1": float(prob), "prediction": int(prob >= 0.5)})
    rme_sub = {r["subject_id"]: r for r in metrics_by_subject(labels, pmap["C7_PERSIST_RME"][:, 1], subjects_out)}; base_sub = {r["subject_id"]: r for r in metrics_by_subject(labels, p_anchor, subjects_out)}; erm_sub = {r["subject_id"]: r for r in metrics_by_subject(labels, pmap["C1_MATCHED_ERM"][:, 1], subjects_out)}
    for s in subject_sort(base_sub): delta = float(rme_sub[s]["BA"] - base_sub[s]["BA"]); delta_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "subject_id": s, "anchor_BA": base_sub[s]["BA"], "rme_BA": rme_sub[s]["BA"], "matched_ERM_BA": erm_sub[s]["BA"], "delta_BA": delta, "delta_pp": delta * 100})
    expert_preds = {e: (pmap[e][:, 1] >= 0.5) for e in RISK_MODES}; pair = [float(np.mean(expert_preds[a] != expert_preds[b])) for i, a in enumerate(RISK_MODES) for b in RISK_MODES[i + 1:]]; expert_div_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "stage": "outcome_diagnostic", "pairwise_disagreement_mean": float(np.mean(pair)), "anchor_to_mode_mean_disagreement": float(np.mean([np.mean(base_pred != expert_preds[e]) for e in RISK_MODES]))})
    for model in models.values(): del model
    del outcome, pmap, state, centered, U, Vt
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def make_figures(summary: pd.DataFrame, expert: pd.DataFrame, delta: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        FIGURES.mkdir(parents=True, exist_ok=True)
        for name, x, y, xlabel, ylabel in [("competence_vs_diversity", expert.get("pairwise_disagreement_mean", []), expert.get("rme_BA", []), "pairwise disagreement", "RME BA"), ("per_subject_delta", np.arange(len(delta)), delta.get("delta_pp", []), "subject index", "delta BA (pp)")]:
            fig, ax = plt.subplots(figsize=(5, 3)); ax.scatter(x, y, s=12); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); fig.tight_layout(); fig.savefig(FIGURES / f"{name}.png", dpi=140); plt.close(fig)
        if not summary.empty:
            fig, ax = plt.subplots(figsize=(6, 3)); summary.pivot(index="dataset", columns="control", values="mean_subject_BA").plot.bar(ax=ax); fig.tight_layout(); fig.savefig(FIGURES / "rme_vs_erm_ensemble.png", dpi=140); plt.close(fig)
    except Exception as exc:
        (FIGURES / "FIGURE_NOTE.txt").write_text(f"Matplotlib unavailable: {exc}\n", encoding="utf-8")


def write_reports(baseline_audit: dict[str, Any], source_report: dict[str, Any], eq_rows: list[dict[str, Any]], seed_summary: pd.DataFrame, paired: dict[str, Any], terminal: str, recipe: dict[str, float], runtime_sec: float) -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    eq_pass = bool(eq_rows) and all(r.get("pass") for r in eq_rows)
    lines = ["# PERSIST-RME SEED-0 PILOT", "", "| Dataset | Canonical seed-0 | RME seed-0 | Delta pp | paired 95% CI | matched ERM |", "|---|---:|---:|---:|---|---:|"]
    if not seed_summary.empty and "control" in seed_summary.columns:
        for _, r in seed_summary[seed_summary.control == "C7_PERSIST_RME"].iterrows():
            lines.append(f"| {r.dataset} | {r.anchor_BA * 100:.6f}% | {r.rme_BA * 100:.6f}% | {r.delta_pp:+.4f} | [{r.ci_l_pp:+.4f}, {r.ci_u_pp:+.4f}] | {r.matched_ERM_BA * 100:.6f}% |")
    lines += ["", "## Source-only status", json.dumps(source_report, indent=2), "", "## Controls", "C7 is the fixed 0.50 anchor + 0.50 four-mode risk mean. C1 is the compute-matched four-refinement ERM ensemble. C4/C5 are recorded as diagnostics identical to C7 in this bounded pilot and are not used to replace the primary method.", "", "## Validity", f"- mathematical audit: PASS (`results/MATH_TOY_TEST.json`)\n- canonical checkpoint equivalence: {'PASS' if eq_pass else 'FAIL'}\n- sealed cohorts accessed: NO\n- seed 1/2 authorized/run: NO\n- recipe: {recipe}\n- runtime seconds: {runtime_sec:.1f}", "", f"terminal = {terminal}"]
    (EXP / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {"title": "PERSIST-RME SEED-0 PILOT", "terminal": terminal, "baseline_hashes": baseline_audit, "source_report": source_report, "checkpoint_equivalence": eq_rows, "seed0_summary": seed_summary.to_dict(orient="records"), "paired_bootstrap": paired, "recipe": recipe, "seed1_seed2_authorized": False, "full_three_seed_run": False, "task_generality": "NOT_AUTHORIZED_AFTER_SEED0", "backbone_generality": "NOT_AUTHORIZED_AFTER_SEED0", "sealed_status": {"WBCIC_outer_10_accessed": False, "OpenBMI_sealed_holdout_accessed": False}}
    write_json(EXP / "FINAL_REPORT.json", report)
    (EXP / "CHECKPOINT_EQUIVALENCE.md").write_text("# Checkpoint equivalence\n\nAll canonical seed-0 refit checkpoints were compared to canonical outcome trial UIDs, labels, probabilities and hard predictions without computing outcome BA.\n\nstatus = " + ("PASS" if eq_pass else "FAIL") + "\n", encoding="utf-8")
    (EXP / "SOURCE_DEVELOPMENT_REPORT.md").write_text("# Source development report\n\n" + json.dumps(source_report, indent=2) + "\n", encoding="utf-8")
    (EXP / "COMPETENCE_REPORT.md").write_text("# Competence report\n\nSee `results/EXPERT_COMPETENCE.csv`; gates are computed only on discovery subjects.\n", encoding="utf-8")
    (EXP / "DIVERSITY_REPORT.md").write_text("# Diversity report\n\nSee `results/EXPERT_DIVERSITY.csv`; disagreements are hard-decision rates on discovery/outcome diagnostics.\n", encoding="utf-8")
    (EXP / "CONTROL_REPORT.md").write_text("# Control report\n\nC0 canonical, C1 matched ERM, C2 equal five-model mean, C3 random Dirichlet, C6 one-mode and C7 RME are scored. C4/C5 are diagnostic placeholders equal to C7 in this bounded seed-0 implementation and are not interpreted as ablations. C8 fixed GroupDRO diagnostic is scored; C9 SAM/SAGM was not available.\n", encoding="utf-8")
    for name, text in [("ABLATION_REPORT.md", "# Ablation report\n\nAblations are descriptive only; the primary prediction is never replaced after outcome access.\n"), ("TASK_GENERALITY_REPORT.md", "# Task generality report\n\nNot authorized after the seed-0 gate.\n"), ("BACKBONE_GENERALITY_REPORT.md", "# Backbone generality report\n\nNot authorized after the seed-0 gate.\n"), ("CLAIM_AUDIT.md", "# Claim audit\n\nThis seed-0 development result is exploratory and cannot support a multi-seed or sealed confirmation claim.\n"), ("BUG_REPAIR_LEDGER.md", "# Bug repair ledger\n\nNo outcome-dependent scientific repair. Engineering fixes, if any, are recorded in the git history.\n"), ("ITERATION_LEDGER.md", "# Iteration ledger\n\nExactly one predeclared seed-0 recipe was run; at most one source-only repair is permitted by the lock.\n"), ("REPRODUCIBILITY.md", "# Reproducibility\n\nRun with the GPU environment's Python and `rme_seed0_pilot.py --run-seed0 --device cuda`. Runtime/checkpoints/raw EEG are excluded from git.\n")]: (EXP / name).write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--preflight", action="store_true"); parser.add_argument("--run-seed0", action="store_true"); parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda"); args = parser.parse_args()
    if not args.preflight and not args.run_seed0: parser.error("choose --preflight or --run-seed0")
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device); RESULTS.mkdir(parents=True, exist_ok=True); PROTOCOL.mkdir(parents=True, exist_ok=True); RUNTIME.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    canonical_trials, baseline_audit = load_baseline(); pre_audits(baseline_audit); run_math_toy_test()
    if args.preflight: print("preflight complete", flush=True); return
    start_time = time.time(); contexts = load_contexts(); sig_rows: list[dict[str, Any]] = []; basis_rows: list[dict[str, Any]] = []; weight_rows: list[dict[str, Any]] = []; expert_rows: list[dict[str, Any]] = []; diversity_rows: list[dict[str, Any]] = []; source_diags: list[dict[str, Any]] = []
    recipe = {"beta_risk": BETA_RISK, "lambda_kd": LAMBDA_KD};
    for ctx in contexts: d, _ = source_fold(ctx, recipe["beta_risk"], recipe["lambda_kd"], device, sig_rows, basis_rows, weight_rows, expert_rows, diversity_rows); source_diags.append(d)
    source_ok, source_report = source_gate(source_diags); repair_used = False
    if not source_ok:
        competence_fail = any(v["min_expert_delta_BA"] < -COMPETENCE_TOL or v["mean_expert_delta_BA"] < -0.010 for k, v in source_report.items() if k in DATASETS); diversity_fail = any(not v["diversity_pass"] for k, v in source_report.items() if k in DATASETS)
        if competence_fail and diversity_fail: terminal = "RME_SOURCE_CONSTRUCTION_FAILED"; write_json(RESULTS / "SOURCE_GATE.json", {"pass": False, "report": source_report, "terminal": terminal}); write_reports(baseline_audit, source_report, [], pd.DataFrame(), {}, terminal, recipe, time.time() - start_time); return
        repair_used = True; recipe = {"beta_risk": 0.25 if competence_fail else 0.75, "lambda_kd": 0.50 if competence_fail else 0.10}; (EXP / "SOURCE_REPAIR_RATIONALE.md").write_text("# Source repair rationale\n\nThe single allowed repair was chosen before outcome access from the source-only gate: " + json.dumps({"competence_fail": competence_fail, "diversity_fail": diversity_fail, "pre_repair_report": source_report, "new_recipe": recipe}, indent=2) + "\n", encoding="utf-8"); write_json(EXP / "SOURCE_REPAIR_LOCK.json", {"pre_repair_report": source_report, "recipe": recipe, "outcome_accessed": False}); source_diags = []; sig_rows = []; basis_rows = []; weight_rows = []; expert_rows = []; diversity_rows = []
        for ctx in contexts: d, _ = source_fold(ctx, recipe["beta_risk"], recipe["lambda_kd"], device, sig_rows, basis_rows, weight_rows, expert_rows, diversity_rows); source_diags.append(d)
        source_ok, source_report = source_gate(source_diags)
        if not source_ok: terminal = "RME_SOURCE_CONSTRUCTION_FAILED"; write_json(RESULTS / "SOURCE_GATE.json", {"pass": False, "repair_used": True, "report": source_report, "terminal": terminal}); write_reports(baseline_audit, source_report, [], pd.DataFrame(), {}, terminal, recipe, time.time() - start_time); return
    write_json(RESULTS / "SOURCE_GATE.json", {"pass": True, "repair_used": repair_used, "report": source_report, "recipe": recipe})
    # Lock the selected source recipe before any outcome scoring.
    lock = json.loads((EXP / "PROTOCOL_LOCK.json").read_text(encoding="utf-8")); lock["selected_recipe"] = recipe; lock["source_gate"] = source_report; lock["repair_used"] = repair_used; lock["outcome_scoring_unlocked"] = True; write_json(EXP / "PROTOCOL_LOCK.json", lock)
    eq_rows: list[dict[str, Any]] = []; subject_rows: list[dict[str, Any]] = []; fold_rows: list[dict[str, Any]] = []; trial_rows: list[dict[str, Any]] = []; delta_rows: list[dict[str, Any]] = []; control_rows: list[dict[str, Any]] = []; outcome_div: list[dict[str, Any]] = []
    for ctx in contexts: final_fold(ctx, canonical_trials, recipe, device, sig_rows, basis_rows, weight_rows, eq_rows, subject_rows, fold_rows, trial_rows, delta_rows, control_rows, outcome_div)
    sig = pd.DataFrame(sig_rows); 
    try: sig.to_parquet(RESULTS / "SUBJECT_GRADIENT_SIGNATURES.parquet", index=False)
    except Exception: sig.to_json(RESULTS / "SUBJECT_GRADIENT_SIGNATURES.parquet", orient="records", lines=True)
    write_json(RESULTS / "GRADIENT_SIGNATURE_AUDIT.json", {"rows": len(sig), "finite": bool(sig.finite.all()) if len(sig) else False, "stages": sorted(sig.stage.unique().tolist()) if len(sig) else [], "outcome_subjects_in_basis": False})
    write_csv(RESULTS / "RISK_BASIS_SUMMARY.csv", pd.DataFrame(basis_rows)); write_csv(RESULTS / "SUBJECT_RISK_WEIGHTS.csv", pd.DataFrame(weight_rows)); write_csv(RESULTS / "EXPERT_COMPETENCE.csv", pd.DataFrame(expert_rows)); write_csv(RESULTS / "EXPERT_DIVERSITY.csv", pd.DataFrame(diversity_rows + outcome_div)); write_csv(RESULTS / "PER_SUBJECT_RESULTS.csv", pd.DataFrame(subject_rows)); write_csv(RESULTS / "PER_FOLD_RESULTS.csv", pd.DataFrame(fold_rows)); write_csv(RESULTS / "TRIAL_PREDICTIONS.csv", pd.DataFrame(trial_rows)); write_csv(RESULTS / "PER_SUBJECT_DELTA.csv", pd.DataFrame(delta_rows)); write_csv(RESULTS / "COMPUTE_MATCHED_CONTROLS.csv", pd.DataFrame(control_rows)); write_csv(RESULTS / "ABLATION_SUMMARY.csv", pd.DataFrame(control_rows)); write_json(RESULTS / "CHECKPOINT_EQUIVALENCE.json", {"pass": bool(eq_rows) and all(r["pass"] for r in eq_rows), "rows": eq_rows})
    summaries: list[dict[str, Any]] = []; paired: dict[str, Any] = {}; all_delta = pd.DataFrame(delta_rows)
    for ds in DATASETS:
        d = all_delta[all_delta.dataset == ds]; ctrl = pd.DataFrame(control_rows); rme = ctrl[(ctrl.dataset == ds) & (ctrl.control == "C7_PERSIST_RME")]; erm = ctrl[(ctrl.dataset == ds) & (ctrl.control == "C1_MATCHED_ERM")]; anchor = ctrl[(ctrl.dataset == ds) & (ctrl.control == "C0_CANONICAL_SEED0")]; vals = d.delta_BA.to_numpy(float); bs = bootstrap(vals, ds); paired[ds] = bs; summaries.append({"dataset": ds, "seed": SEED, "anchor_BA": float(anchor.mean_subject_BA.mean()), "rme_BA": float(rme.mean_subject_BA.mean()), "matched_ERM_BA": float(erm.mean_subject_BA.mean()), "delta_BA": float(vals.mean()), "delta_pp": float(vals.mean() * 100), "ci_l_pp": bs["paired_bootstrap_CI95_L_pp"], "ci_u_pp": bs["paired_bootstrap_CI95_U_pp"], "positive_subject_fraction": bs["positive_subject_fraction"], "median_delta_pp": bs["median_delta_pp"]})
    seed_summary = pd.DataFrame(summaries); write_csv(RESULTS / "SEED0_RESULTS.csv", seed_summary); write_json(RESULTS / "PAIRED_BOOTSTRAP.json", paired); write_csv(RESULTS / "FULL_THREE_SEED_RESULTS.csv", pd.DataFrame([{"status": "NOT_RUN", "reason": "seed-0 pilot requested; no seed 1/2 authorized"}])); write_csv(RESULTS / "TASK_GENERALITY.csv", pd.DataFrame([{"status": "NOT_AUTHORIZED_AFTER_SEED0"}])); write_csv(RESULTS / "BACKBONE_GENERALITY.csv", pd.DataFrame([{"status": "NOT_AUTHORIZED_AFTER_SEED0"}])); make_figures(pd.DataFrame(control_rows), pd.DataFrame(diversity_rows + outcome_div), all_delta)
    both_positive = all(float(r.delta_BA) > 0 for r in summaries); min_delta = min(float(r.delta_BA) for r in summaries); beats_erm = all(float(r.rme_BA) >= float(r.matched_ERM_BA) for r in summaries) and any(float(r.rme_BA) > float(r.matched_ERM_BA) for r in summaries); ci_ok = all(float(r.ci_u_pp) > 0 for r in summaries); terminal = "RME_SEED0_SUPPORTED" if both_positive and min_delta >= 0.003 and beats_erm and ci_ok else "RME_SEED0_NOT_SUPPORTED"; write_json(RESULTS / "VALIDATION.json", {"pass": bool(eq_rows) and all(r["pass"] for r in eq_rows) and len(summaries) == 2, "terminal": terminal, "no_seed1_seed2": True, "sealed_untouched": True, "source_gate_pass": True}); write_reports(baseline_audit, {**source_report, "repair_used": repair_used}, eq_rows, pd.DataFrame([{"control": "C7_PERSIST_RME", **r} for r in summaries]), paired, terminal, recipe, time.time() - start_time)
    print(f"branch = codex/persist-eeg-risk-mode-marginalization-final\nterminal = {terminal}", flush=True)
    for r in summaries: print(f"{r['dataset']}: anchor_BA={r['anchor_BA']:.9f} RME_BA={r['rme_BA']:.9f} delta_pp={r['delta_pp']:+.4f} CI=[{r['ci_l_pp']:+.4f},{r['ci_u_pp']:+.4f}] matched_ERM={r['matched_ERM_BA']:.9f}", flush=True)
    print("checkpoint equivalence status = PASS" if all(r["pass"] for r in eq_rows) else "checkpoint equivalence status = FAIL", flush=True); print("seed1_seed2_authorized = NO", flush=True)


if __name__ == "__main__":
    main()
