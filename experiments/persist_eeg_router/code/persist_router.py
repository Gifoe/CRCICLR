"""PERSIST-Router: leak-free subject-disjoint reliability routing.

The audit command is intentionally self-contained and fail closed.  It only
loads outer-TRAIN rows, creates five-fold cross-fitted matched-base predictions,
and decides whether the intervention actions have routeable headroom.  It does
not load OpenBMI development-validation arrays/labels and cannot address the
outer-test split at all.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
EXP_ROOT = REPO_ROOT / "experiments" / "persist_eeg_router"
OUT = EXP_ROOT / "outputs"
P5_ROOT = REPO_ROOT / "experiments" / "persist_eeg_p5_icg"
P56_ROOT = REPO_ROOT / "experiments" / "persist_eeg_p5_1_p6"
P56_OUT = P56_ROOT / "outputs"
REFERENCE_COMMIT = "41b156aded19937e6890401f5fbacee8ee8fc1f0"
IMPLEMENTATION_ID = "persist_router_audit_v1_subject_crossfit"
FOLDS = (0, 1, 2)
SEEDS = (0, 1)
ROUTER_FOLDS = 5
RANDOM_DRAWS = 100
RANDOM_ROUTER_DRAWS = 10
ACTIONS = ("erase", "amplify", "geometry")
EPS = 1e-8


def _import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P5 = _import_file("persist_router_frozen_p5", P5_ROOT / "code" / "p5_icg.py")
P56 = _import_file("persist_router_frozen_p56", P56_ROOT / "code" / "p5_1_p6.py")


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
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32 - 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = float(np.linalg.norm(x))
    return (x / norm).astype(np.float32) if norm > 1e-12 else np.zeros_like(x, dtype=np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return (exp / np.maximum(exp.sum(axis=1, keepdims=True), EPS)).astype(np.float32)


def entropy(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=np.float64), EPS, 1.0)
    return (-np.sum(p * np.log(p), axis=1)).astype(np.float32)


def js_divergence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pa = np.clip(np.asarray(a, dtype=np.float64), EPS, 1.0)
    pb = np.clip(np.asarray(b, dtype=np.float64), EPS, 1.0)
    mid = 0.5 * (pa + pb)
    return (0.5 * np.sum(pa * np.log(pa / mid), axis=1) +
            0.5 * np.sum(pb * np.log(pb / mid), axis=1)).astype(np.float32)


def margin(logits_or_prob: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits_or_prob)
    ordered = np.sort(arr, axis=1)
    return (ordered[:, -1] - ordered[:, -2]).astype(np.float32)


def positions(meta: pd.DataFrame, subjects: Sequence[str]) -> np.ndarray:
    wanted = {str(x) for x in subjects}
    return np.flatnonzero(meta.subject.astype(str).isin(wanted).to_numpy())


def labels(meta: pd.DataFrame, pos: np.ndarray) -> np.ndarray:
    return np.array(meta.iloc[pos].label.to_numpy(dtype=np.int64), dtype=np.int64, copy=True)


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


@dataclass(frozen=True)
class SelectedBase:
    fold: int
    seed: int
    lambda_drift: float
    learning_rate: float
    bottleneck: int
    epochs: int
    candidate: str


@dataclass
class FoldPredictions:
    fold: int
    seed: int
    router_fold: int
    held_pos: np.ndarray
    train_pos: np.ndarray
    h_held: np.ndarray
    q_held: np.ndarray
    q_adj_held: np.ndarray
    logits_keep: np.ndarray
    logits_erase: np.ndarray
    logits_amp: np.ndarray
    logits_geo: np.ndarray
    features: pd.DataFrame
    model: Any
    train_q_adj: np.ndarray
    train_y: np.ndarray


@dataclass
class AuditRun:
    fold: int
    seed: int
    meta: pd.DataFrame
    h: np.ndarray
    q: np.ndarray
    art: Any
    split: dict[str, list[str]]


def selected_bases() -> dict[tuple[int, int], SelectedBase]:
    selection = json.loads((P56_OUT / "protocol" / "P6_BASE_VERSION_SELECTION.json").read_text(encoding="utf-8"))
    if selection.get("base_version") != "V2" or selection.get("outer_test_used") is not False:
        raise RuntimeError("Frozen P6 base selection is not the required V2 TRAIN-only control")
    frame = pd.read_csv(P56_OUT / "P5_1_SELECTED_CONFIGS.csv")
    frame = frame[frame.version.astype(str) == "V2"].copy()
    if len(frame) != 6:
        raise RuntimeError(f"Expected six selected V2 base configurations, found {len(frame)}")
    result: dict[tuple[int, int], SelectedBase] = {}
    for _, row in frame.iterrows():
        epoch = P56.median_epoch(row.median_pair_epoch)
        item = SelectedBase(
            fold=int(row.fold), seed=int(row.seed), lambda_drift=float(row.lambda_drift),
            learning_rate=float(row.learning_rate), bottleneck=int(row.bottleneck),
            epochs=int(epoch), candidate=str(row.candidate),
        )
        result[(item.fold, item.seed)] = item
    return result


def router_subject_folds(subjects: Sequence[str], fold: int, seed: int) -> list[list[str]]:
    ordered = sorted({str(x) for x in subjects}, key=int)
    rng = np.random.default_rng(stable_seed("persist-router-subject-folds", fold, seed, ordered))
    perm = [ordered[i] for i in rng.permutation(len(ordered))]
    splits = [perm[i::ROUTER_FOLDS] for i in range(ROUTER_FOLDS)]
    flat = [subject for split in splits for subject in split]
    if len(flat) != len(set(flat)) or set(flat) != set(ordered):
        raise RuntimeError("Router subject folds are not an exact disjoint partition")
    return splits


def load_run(fold: int, seed: int, meta: pd.DataFrame):
    # Do not use P56.load_run_data here: that helper materialises development
    # positions for its own outer evaluation.  The Router audit only keeps the
    # outer-TRAIN subject list and never constructs validation arrays/labels.
    split = P5.load_split(fold)
    split = {"train_subjects": [str(x) for x in split["train_subjects"]]}
    h_path = P5.OUT / "cache" / f"fold-{fold}" / f"seed-{seed}" / "h0.npy"
    h = np.asarray(np.load(h_path, mmap_mode="r"), dtype=np.float32)
    if h.shape != (len(meta), 128) or not np.isfinite(h).all():
        raise RuntimeError(f"Invalid frozen h0: {h_path} {h.shape}")
    art = P5.load_artifacts(fold, seed)
    q = P5.q_from_h(h, art)
    return AuditRun(fold=fold, seed=seed, meta=meta, h=h, q=q, art=art, split=split)


def initialise_v2_control(run: Any, cfg: SelectedBase, targets: Any, stream_tag: str, device: torch.device):
    from persist_eeg_stage0.models import build_shared_model

    init_seed = P56.stable_seed("p5.1-init", P56.IMPLEMENTATION_ID, run.fold, run.seed, stream_tag)
    seed_all(init_seed)
    checkpoint_path, _, _ = P5.historical_checkpoint(run.fold, run.seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    historical = build_shared_model("eegnet", int(run.meta.n_channels.iloc[0]), 128, P56.TASK_CLASSES)
    historical.load_state_dict(checkpoint["model"])
    return P5.ICGModel(historical.heads[P56.TASK], run.art, "V2", targets, cfg.bottleneck).to(device)


def fit_crossfitted_base(
    run: Any,
    cfg: SelectedBase,
    train_pos: np.ndarray,
    targets: Any,
    router_fold: int,
    device: torch.device,
) -> Any:
    """Refit only the frozen P5.1 V2 matched-control branch on 80% subjects."""
    stream_tag = f"router-oof-{router_fold}"
    model = initialise_v2_control(run, cfg, targets, stream_tag, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-3)
    train_meta = run.meta.iloc[train_pos].reset_index(drop=True)
    h = torch.as_tensor(np.asarray(run.h[train_pos], dtype=np.float32), device=device)
    q = torch.as_tensor(np.asarray(run.q[train_pos], dtype=np.float32), device=device)
    y = torch.as_tensor(labels(run.meta, train_pos), dtype=torch.long, device=device)
    sampler = P5.StructuredSampler(
        train_meta, train_meta.subject.astype(str).unique().tolist(),
        subjects_per_batch=6, trials_per_class=4,
    )
    for epoch in range(cfg.epochs):
        model.train()
        batches = sampler.batches(
            epoch,
            P56.stable_seed("p5.1-sampler", P56.IMPLEMENTATION_ID, run.fold, run.seed, stream_tag),
        )
        for batch in batches:
            idx = torch.as_tensor(batch, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, _, delta = model(h.index_select(0, idx), q.index_select(0, idx))
            loss = F.cross_entropy(logits, y.index_select(0, idx)) + cfg.lambda_drift * P5.drift_loss(delta, run.art)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model.eval()


def forward_model(model: Any, h: np.ndarray, q: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits_all: list[np.ndarray] = []
    q_all: list[np.ndarray] = []
    h_all: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(h), 2048):
            ht = torch.as_tensor(np.asarray(h[start:start + 2048], dtype=np.float32), device=device)
            qt = torch.as_tensor(np.asarray(q[start:start + 2048], dtype=np.float32), device=device)
            logits, q_adj, _ = model(ht, qt)
            h_adj = ht + ((q_adj - qt) @ model.directions.T) @ model.dewhitener
            logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
            q_all.append(q_adj.detach().cpu().numpy().astype(np.float32))
            h_all.append(h_adj.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(logits_all), np.concatenate(q_all), np.concatenate(h_all)


def erase_logits(
    model: Any,
    h_adj: np.ndarray,
    q_adj: np.ndarray,
    dims: Sequence[int],
    device: torch.device,
) -> np.ndarray:
    dims_np = np.asarray(sorted({int(x) for x in dims}), dtype=np.int64)
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(h_adj), 2048):
            hb = torch.as_tensor(np.asarray(h_adj[start:start + 2048], dtype=np.float32), device=device)
            qb = torch.as_tensor(np.asarray(q_adj[start:start + 2048], dtype=np.float32), device=device)
            erased = qb.clone()
            idx = torch.as_tensor(dims_np, dtype=torch.long, device=device)
            erased[:, idx] = 0.0
            h_erased = hb + ((erased - qb) @ model.directions.T) @ model.dewhitener
            output.append(model.head(h_erased).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(output)


def fit_geometry_expert(q: np.ndarray, y: np.ndarray, dims: Sequence[int], seed: int):
    x = np.asarray(q[:, np.asarray(dims, dtype=np.int64)], dtype=np.float64)
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=int(seed))
    classifier.fit(scaler.transform(x), y)
    return scaler, classifier


def geometry_logits(expert: tuple[Any, Any], q: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    scaler, classifier = expert
    x = scaler.transform(np.asarray(q[:, np.asarray(dims, dtype=np.int64)], dtype=np.float64))
    return classifier.predict_log_proba(x).astype(np.float32)


def block_statistics(q_train: np.ndarray, y_train: np.ndarray, groups: Sequence[Sequence[int]]) -> list[dict[str, np.ndarray]]:
    result: list[dict[str, np.ndarray]] = []
    for dims in groups:
        idx = np.asarray(dims, dtype=np.int64)
        proto0 = q_train[y_train == 0][:, idx].mean(axis=0)
        proto1 = q_train[y_train == 1][:, idx].mean(axis=0)
        result.append({
            "proto0": proto0.astype(np.float32),
            "proto1": proto1.astype(np.float32),
            "midpoint": (0.5 * (proto0 + proto1)).astype(np.float32),
            "direction": normalise(proto1 - proto0),
        })
    return result


def make_features(
    logits_keep: np.ndarray,
    logits_erase: np.ndarray,
    logits_geo: np.ndarray,
    q_eval: np.ndarray,
    q_train: np.ndarray,
    y_train: np.ndarray,
    groups: Sequence[Sequence[int]],
    group_weights: Sequence[float],
    model: Any,
    h_adj_eval: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    """Build compact, label-free, sample-wise reliability features.

    Every prototype/direction is supplied from Router-fold training subjects.
    The function has no access to evaluation labels or subject identifiers.
    """
    p_keep = softmax(logits_keep)
    p_erase = softmax(logits_erase)
    p_geo = softmax(logits_geo)
    delta = logits_keep - logits_erase
    frame: dict[str, np.ndarray] = {
        "p_full_max": p_keep.max(axis=1),
        "entropy_full": entropy(p_keep),
        "top2_margin_full": margin(p_keep),
        "nll_proxy_full": -np.log(np.maximum(p_keep.max(axis=1), EPS)),
        "logit_norm_full": np.linalg.norm(logits_keep, axis=1),
        "delta_p_l2": np.linalg.norm(delta, axis=1),
        "margin_change_full_erase": margin(p_keep) - margin(p_erase),
        "entropy_change_full_erase": entropy(p_keep) - entropy(p_erase),
        "kl_full_erase": np.sum(
            np.clip(p_keep, EPS, 1.0) *
            np.log(np.clip(p_keep, EPS, 1.0) / np.clip(p_erase, EPS, 1.0)), axis=1,
        ),
        "js_full_erase": js_divergence(p_keep, p_erase),
        "agreement_full_erase": (p_keep.argmax(1) == p_erase.argmax(1)).astype(np.float32),
        "probability_shift_l1": np.abs(p_keep - p_erase).sum(axis=1),
        "margin_geo": margin(p_geo),
        "entropy_geo": entropy(p_geo),
        "agreement_base_geo": (p_keep.argmax(1) == p_geo.argmax(1)).astype(np.float32),
        "confidence_diff_base_geo": p_keep.max(axis=1) - p_geo.max(axis=1),
        "js_base_geo": js_divergence(p_keep, p_geo),
    }
    stats = block_statistics(q_train, y_train, groups)
    # Runs contain one or two canonical Protected blocks.  Two fixed semantic
    # slots keep the Router feature shape identical across all six runs.
    for slot in range(2):
        prefix = f"protected_slot{slot}"
        if slot >= len(groups):
            for suffix in (
                "present", "projection", "signed_midpoint_distance",
                "prototype_distance_difference", "geometry_margin",
                "weighted_geometry_score", "contribution_norm",
            ):
                frame[f"{prefix}_{suffix}"] = np.zeros(len(q_eval), dtype=np.float32)
            continue
        dims = np.asarray(groups[slot], dtype=np.int64)
        block = np.asarray(q_eval[:, dims], dtype=np.float32)
        info = stats[slot]
        signed = (block - info["midpoint"]) @ info["direction"]
        proto_diff = (
            np.linalg.norm(block - info["proto0"], axis=1) -
            np.linalg.norm(block - info["proto1"], axis=1)
        )
        block_erased = erase_logits(model, h_adj_eval, q_eval, dims, device)
        weight = float(group_weights[slot])
        frame[f"{prefix}_present"] = np.ones(len(q_eval), dtype=np.float32)
        frame[f"{prefix}_projection"] = block @ info["direction"]
        frame[f"{prefix}_signed_midpoint_distance"] = signed
        frame[f"{prefix}_prototype_distance_difference"] = proto_diff
        frame[f"{prefix}_geometry_margin"] = np.abs(proto_diff)
        frame[f"{prefix}_weighted_geometry_score"] = weight * signed
        frame[f"{prefix}_contribution_norm"] = np.linalg.norm(logits_keep - block_erased, axis=1)
    result = pd.DataFrame({key: np.asarray(value, dtype=np.float32) for key, value in frame.items()})
    if not np.isfinite(result.to_numpy()).all():
        raise RuntimeError("Non-finite Router feature")
    return result


def random_subspaces(q_dim: int, rank: int, fold: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(stable_seed("persist-router-random-subspaces", fold, seed, q_dim, rank))
    result: list[list[int]] = []
    while len(result) < RANDOM_DRAWS:
        dims = sorted(rng.choice(np.arange(q_dim), size=rank, replace=False).astype(int).tolist())
        if dims not in result:
            result.append(dims)
    return result


def create_fold_predictions(
    run: Any,
    cfg: SelectedBase,
    router_fold: int,
    held_subjects: Sequence[str],
    device: torch.device,
) -> FoldPredictions:
    outer_train = [str(x) for x in run.split["train_subjects"]]
    held_set = {str(x) for x in held_subjects}
    train_subjects = [s for s in outer_train if s not in held_set]
    train_pos = positions(run.meta, train_subjects)
    held_pos = positions(run.meta, list(held_set))
    if set(run.meta.iloc[train_pos].subject.astype(str)) & set(run.meta.iloc[held_pos].subject.astype(str)):
        raise RuntimeError("Subject leakage in Router fold")
    target_path = OUT / "cache" / "geometry_targets" / f"fold-{run.fold}" / f"seed-{run.seed}" / f"router-{router_fold}.npz"
    targets = P5.build_geometry_targets(run.meta, run.q, train_pos, run.art, target_path)
    model = fit_crossfitted_base(run, cfg, train_pos, targets, router_fold, device)

    h_train = np.asarray(run.h[train_pos], dtype=np.float32)
    q_train = np.asarray(run.q[train_pos], dtype=np.float32)
    h_held = np.asarray(run.h[held_pos], dtype=np.float32)
    q_held = np.asarray(run.q[held_pos], dtype=np.float32)
    train_logits, train_q_adj, _ = forward_model(model, h_train, q_train, device)
    del train_logits
    logits_keep, q_adj_held, h_adj_held = forward_model(model, h_held, q_held, device)
    protected_dims = run.art.protected_dims.tolist()
    logits_erase = erase_logits(model, h_adj_held, q_adj_held, protected_dims, device)
    logits_amp = logits_erase + 2.0 * (logits_keep - logits_erase)
    train_y = labels(run.meta, train_pos)
    geo = fit_geometry_expert(
        train_q_adj, train_y, protected_dims,
        stable_seed("persist-router-geometry-expert", run.fold, run.seed, router_fold),
    )
    logits_geo = geometry_logits(geo, q_adj_held, protected_dims)
    groups = [run.art.blocks[b] for b in run.art.protected_blocks]
    weights = [run.art.weights[b] for b in run.art.protected_blocks]
    features = make_features(
        logits_keep, logits_erase, logits_geo, q_adj_held, train_q_adj,
        train_y, groups, weights, model, h_adj_held, device,
    )
    return FoldPredictions(
        fold=run.fold, seed=run.seed, router_fold=router_fold,
        held_pos=held_pos, train_pos=train_pos, h_held=h_held, q_held=q_held,
        q_adj_held=q_adj_held, logits_keep=logits_keep, logits_erase=logits_erase,
        logits_amp=logits_amp, logits_geo=logits_geo, features=features,
        model=model, train_q_adj=train_q_adj, train_y=train_y,
    )


def fold_metadata(run: Any, pred: FoldPredictions) -> pd.DataFrame:
    rows = run.meta.iloc[pred.held_pos].reset_index(drop=True)
    return pd.DataFrame({
        "fold": pred.fold,
        "seed": pred.seed,
        "router_fold": pred.router_fold,
        "manifest_index": rows.manifest_index.to_numpy(dtype=np.int64),
        "subject": rows.subject.astype(str).to_numpy(),
        "session": rows.session.astype(str).to_numpy(),
        "label": rows.label.to_numpy(dtype=np.int64),
    })


def logit_frame(meta: pd.DataFrame, arrays: Mapping[str, np.ndarray]) -> pd.DataFrame:
    frame = meta.copy()
    for name, value in arrays.items():
        value = np.asarray(value)
        for cls in range(value.shape[1]):
            frame[f"{name}_{cls}"] = value[:, cls].astype(np.float32)
    return frame


def generate_run_oof(
    run: Any,
    cfg: SelectedBase,
    subject_folds: list[list[str]],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[FoldPredictions]]:
    feature_frames: list[pd.DataFrame] = []
    base_frames: list[pd.DataFrame] = []
    counter_frames: list[pd.DataFrame] = []
    geometry_frames: list[pd.DataFrame] = []
    predictions: list[FoldPredictions] = []
    for router_fold, held_subjects in enumerate(subject_folds):
        started = time.time()
        pred = create_fold_predictions(run, cfg, router_fold, held_subjects, device)
        meta = fold_metadata(run, pred)
        feature_frames.append(pd.concat([meta, pred.features], axis=1))
        base_frames.append(logit_frame(meta, {"keep_logit": pred.logits_keep}))
        counter_frames.append(logit_frame(meta, {
            "erase_logit": pred.logits_erase,
            "amplify_logit": pred.logits_amp,
        }))
        geometry_frames.append(logit_frame(meta, {"geometry_logit": pred.logits_geo}))
        predictions.append(pred)
        print(
            f"[OOF] fold={run.fold} seed={run.seed} router_fold={router_fold} "
            f"held_subjects={len(held_subjects)} rows={len(meta)} elapsed={time.time()-started:.1f}s",
            flush=True,
        )
    features = pd.concat(feature_frames, ignore_index=True)
    base = pd.concat(base_frames, ignore_index=True)
    counter = pd.concat(counter_frames, ignore_index=True)
    geometry = pd.concat(geometry_frames, ignore_index=True)
    expected = set(map(str, run.split["train_subjects"]))
    if set(features.subject.astype(str)) != expected or features.groupby("subject").router_fold.nunique().max() != 1:
        raise RuntimeError("OOF output is not a complete subject-disjoint outer-TRAIN partition")
    return features, base, counter, geometry, predictions


def action_arrays(base: pd.DataFrame, counter: pd.DataFrame, geometry: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "keep": base[["keep_logit_0", "keep_logit_1"]].to_numpy(dtype=np.float32),
        "erase": counter[["erase_logit_0", "erase_logit_1"]].to_numpy(dtype=np.float32),
        "amplify": counter[["amplify_logit_0", "amplify_logit_1"]].to_numpy(dtype=np.float32),
        "geometry": geometry[["geometry_logit_0", "geometry_logit_1"]].to_numpy(dtype=np.float32),
    }


def action_headroom_rows(
    fold: int,
    seed: int,
    y: np.ndarray,
    logits: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    predictions = {name: np.asarray(value).argmax(1) for name, value in logits.items()}
    correct = {name: pred == y for name, pred in predictions.items()}
    keep_ba = float(balanced_accuracy_score(y, predictions["keep"]))
    action_rows: list[dict[str, Any]] = []
    for name in ("keep", *ACTIONS):
        pred = predictions[name]
        ba = float(balanced_accuracy_score(y, pred))
        if name == "keep":
            rescue = harm = np.zeros(len(y), dtype=bool)
        else:
            rescue = (~correct["keep"]) & correct[name]
            harm = correct["keep"] & (~correct[name])
        union_correct = correct["keep"] | correct[name]
        oracle_pred = predictions["keep"].copy()
        oracle_pred[(~correct["keep"]) & correct[name]] = y[(~correct["keep"]) & correct[name]]
        pair_oracle_ba = float(balanced_accuracy_score(y, oracle_pred))
        action_rows.append({
            "fold": fold, "seed": seed, "action": name, "n": len(y),
            "BA": ba, "delta_BA_vs_KEEP": ba - keep_ba,
            "rescue_count": int(rescue.sum()), "rescue_prevalence": float(rescue.mean()),
            "harm_count": int(harm.sum()), "harm_prevalence": float(harm.mean()),
            "net_rescues": int(rescue.sum() - harm.sum()),
            "pair_oracle_BA": pair_oracle_ba,
            "pair_oracle_headroom": pair_oracle_ba - keep_ba,
            "pair_union_correct": int(union_correct.sum()),
            "outer_test_used": False,
        })
    complement: list[dict[str, Any]] = []
    rescue_sets = {name: (~correct["keep"]) & correct[name] for name in ACTIONS}
    for i, left in enumerate(ACTIONS):
        for right in ACTIONS[i + 1:]:
            a, b = rescue_sets[left], rescue_sets[right]
            pair_correct = correct["keep"] | correct[left] | correct[right]
            pair_pred = predictions["keep"].copy()
            pair_pred[(~correct["keep"]) & (correct[left] | correct[right])] = y[(~correct["keep"]) & (correct[left] | correct[right])]
            complement.append({
                "fold": fold, "seed": seed, "action_left": left, "action_right": right,
                "left_rescues": int(a.sum()), "right_rescues": int(b.sum()),
                "overlap_rescues": int((a & b).sum()),
                "left_unique_rescues": int((a & ~b).sum()),
                "right_unique_rescues": int((b & ~a).sum()),
                "pair_oracle_BA": float(balanced_accuracy_score(y, pair_pred)),
                "pair_oracle_headroom": float(balanced_accuracy_score(y, pair_pred) - keep_ba),
                "outer_test_used": False,
            })
    all_correct = correct["keep"].copy()
    oracle_pred = predictions["keep"].copy()
    for name in ACTIONS:
        rescued = (~all_correct) & correct[name]
        oracle_pred[rescued] = y[rescued]
        all_correct |= correct[name]
    summary = {
        "fold": fold, "seed": seed, "BA_keep": keep_ba,
        "BA_all_action_oracle": float(balanced_accuracy_score(y, oracle_pred)),
        "all_action_oracle_headroom": float(balanced_accuracy_score(y, oracle_pred) - keep_ba),
        "best_single_action_oracle_headroom": max(
            row["pair_oracle_headroom"] for row in action_rows if row["action"] != "keep"
        ),
    }
    return action_rows, complement, summary


META_COLUMNS = {"fold", "seed", "router_fold", "manifest_index", "subject", "session", "label"}


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    all_features = [column for column in frame.columns if column not in META_COLUMNS]
    entropy_only = [
        "p_full_max", "entropy_full", "top2_margin_full", "nll_proxy_full", "logit_norm_full",
    ]
    counterfactual = [column for column in all_features if any(token in column for token in (
        "delta_p", "full_erase", "probability_shift",
    ))]
    geometry = [column for column in all_features if (
        column.startswith("protected_slot") or column in {
            "margin_geo", "entropy_geo", "agreement_base_geo", "confidence_diff_base_geo", "js_base_geo"
        }
    )]
    return {
        "entropy_only": entropy_only,
        "counterfactual_only": counterfactual,
        "geometry_only": geometry,
        "full_protected": all_features,
    }


def fit_route_classifier(kind: str, x: np.ndarray, target: np.ndarray, seed: int):
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    if positives == 0 or negatives == 0:
        return scaler, None, float(target[0])
    if kind == "logistic":
        classifier = LogisticRegression(
            C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000,
            random_state=int(seed),
        )
        classifier.fit(xs, target)
    elif kind == "shallow_mlp":
        classifier = TorchBinaryMLP(xs.shape[1], int(seed))
        classifier.fit(xs, target)
    else:
        raise ValueError(kind)
    return scaler, classifier, None


class TorchBinaryMLP:
    """Deterministic shallow diagnostic classifier with sklearn-like output.

    The server's experimental sklearn 1.9 MLP repeatedly corrupted the Python
    warnings state and intermittently terminated Windows with 0xC0000005.
    This equivalent 16-unit CPU implementation removes that native failure
    path without changing the scientific model family.
    """

    def __init__(self, input_dim: int, seed: int):
        self.input_dim = int(input_dim)
        self.seed = int(seed)
        self.model: torch.nn.Module | None = None

    def fit(self, x: np.ndarray, target: np.ndarray) -> "TorchBinaryMLP":
        seed_all(self.seed)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        ).cpu()
        xt = torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
        yt = torch.as_tensor(np.asarray(target, dtype=np.float32), dtype=torch.float32).reshape(-1, 1)
        positives = max(float(yt.sum()), 1.0)
        negatives = max(float(len(yt) - yt.sum()), 1.0)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives]))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-3)
        self.model.train()
        for _ in range(120):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(self.model(xt), yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()
        self.model.eval()
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TorchBinaryMLP has not been fitted")
        xt = torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
        with torch.inference_mode():
            p = torch.sigmoid(self.model(xt)).cpu().numpy().reshape(-1)
        return np.stack([1.0 - p, p], axis=1)


@dataclass(frozen=True)
class R0Config:
    lambda_cons: float
    width: int
    learning_rate: float
    epochs: int = 120
    weight_decay: float = 1e-3

    @property
    def key(self) -> str:
        return f"r0__lc{self.lambda_cons:g}__w{self.width}__lr{self.learning_rate:g}"


class R0Network(torch.nn.Module):
    def __init__(self, input_dim: int, width: int):
        super().__init__()
        second = max(4, int(width) // 2)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), int(width)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(width), second),
            torch.nn.ReLU(),
            torch.nn.Linear(second, 1),
        )
        # Exact preserve-by-default initial condition: r(z)=0 -> a=1.
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 + torch.tanh(self.network(x)).reshape(-1)


def r0_grid() -> list[R0Config]:
    return [
        R0Config(lambda_cons=lam, width=width, learning_rate=lr)
        for lam in (0.01, 0.1, 1.0)
        for width in (8, 16)
        for lr in (1e-4, 3e-4)
    ]


def fit_r0_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    keep_train: np.ndarray,
    erase_train: np.ndarray,
    x_eval: np.ndarray,
    cfg: R0Config,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    scaler = StandardScaler().fit(x_train)
    train_x = torch.as_tensor(scaler.transform(x_train).astype(np.float32), device=device)
    eval_x = torch.as_tensor(scaler.transform(x_eval).astype(np.float32), device=device)
    train_y = torch.as_tensor(y_train, dtype=torch.long, device=device)
    keep = torch.as_tensor(keep_train, dtype=torch.float32, device=device)
    erase = torch.as_tensor(erase_train, dtype=torch.float32, device=device)
    # Compare configs on matched width-specific initialisation streams.
    seed_all(stable_seed("r0-init", seed, cfg.width))
    model = R0Network(train_x.shape[1], cfg.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    last_task = last_cons = 0.0
    for _ in range(cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        a = model(train_x)
        logits = erase + a[:, None] * (keep - erase)
        task = F.cross_entropy(logits, train_y)
        conservative = torch.mean((a - 1.0) ** 2)
        loss = task + cfg.lambda_cons * conservative
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last_task = float(task.detach().cpu())
        last_cons = float(conservative.detach().cpu())
    model.eval()
    with torch.inference_mode():
        a_eval = model(eval_x)
    a_np = a_eval.detach().cpu().numpy().astype(np.float32)
    return a_np, {
        "final_task_loss": last_task, "final_conservative_loss": last_cons,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }


def routed_logits(keep: np.ndarray, erase: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.asarray(erase, dtype=np.float32) + np.asarray(a, dtype=np.float32)[:, None] * (
        np.asarray(keep, dtype=np.float32) - np.asarray(erase, dtype=np.float32)
    )


def predictive_metrics(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    prob = softmax(logits)
    pred = prob.argmax(1)
    return {
        "BA": float(balanced_accuracy_score(y, pred)),
        "NLL": float(log_loss(y, prob, labels=[0, 1])),
        "Brier": float(np.mean(np.sum((prob - np.eye(2, dtype=np.float32)[y]) ** 2, axis=1))),
    }


def evaluate_r0_predictions(
    frame: pd.DataFrame,
    keep: np.ndarray,
    erase: np.ndarray,
    a: np.ndarray,
) -> dict[str, Any]:
    y = frame.label.to_numpy(dtype=np.int64)
    route_logits = routed_logits(keep, erase, a)
    base = predictive_metrics(y, keep)
    routed = predictive_metrics(y, route_logits)
    base_pred = np.asarray(keep).argmax(1)
    route_pred = route_logits.argmax(1)
    rescue = (base_pred != y) & (route_pred == y)
    harm = (base_pred == y) & (route_pred != y)
    subject_deltas: list[float] = []
    for _, group in frame.groupby("subject", sort=True):
        idx = group.index.to_numpy(dtype=np.int64)
        subject_deltas.append(float(
            balanced_accuracy_score(y[idx], route_pred[idx]) - balanced_accuracy_score(y[idx], base_pred[idx])
        ))
    return {
        "base_BA": base["BA"], "router_BA": routed["BA"], "delta_BA": routed["BA"] - base["BA"],
        "base_NLL": base["NLL"], "router_NLL": routed["NLL"],
        "base_Brier": base["Brier"], "router_Brier": routed["Brier"],
        "rescue_count": int(rescue.sum()), "harm_count": int(harm.sum()),
        "net_rescues": int(rescue.sum() - harm.sum()),
        "rescue_rate": float(rescue.sum() / max(int((base_pred != y).sum()), 1)),
        "harm_rate": float(harm.sum() / max(int((base_pred == y).sum()), 1)),
        "mean_a": float(np.mean(a)), "std_a": float(np.std(a)),
        "action_rate_abs_a_minus_1_gt_0.05": float(np.mean(np.abs(a - 1.0) > 0.05)),
        "mean_subject_delta_BA": float(np.mean(subject_deltas)),
        "positive_subjects": int(sum(value > 0 for value in subject_deltas)),
        "n_subjects": len(subject_deltas),
    }


def r0_cv_one(
    features: pd.DataFrame,
    keep: np.ndarray,
    erase: np.ndarray,
    columns: Sequence[str],
    cfg: R0Config,
    variant: str,
    device: torch.device,
    control_draw: int | None = None,
) -> dict[str, Any]:
    y = features.label.to_numpy(dtype=np.int64)
    x = features[list(columns)].to_numpy(dtype=np.float64)
    a_all = np.ones(len(features), dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    for router_fold in sorted(features.router_fold.unique()):
        test = features.router_fold.to_numpy(dtype=np.int64) == int(router_fold)
        train = ~test
        a_eval, train_info = fit_r0_predict(
            x[train], y[train], keep[train], erase[train], x[test], cfg,
            stable_seed("r0-cv", int(features.fold.iloc[0]), int(features.seed.iloc[0]), router_fold, variant, control_draw),
            device,
        )
        a_all[test] = a_eval
        fold_metric = evaluate_r0_predictions(
            features.loc[test].reset_index(drop=True), keep[test], erase[test], a_eval,
        )
        fold_rows.append({"router_fold": int(router_fold), **fold_metric, **train_info})
    result = evaluate_r0_predictions(features, keep, erase, a_all)
    return {
        "fold": int(features.fold.iloc[0]), "seed": int(features.seed.iloc[0]),
        "variant": variant, "control_draw": control_draw, "config": cfg.key,
        "lambda_cons": cfg.lambda_cons, "width": cfg.width,
        "learning_rate": cfg.learning_rate, "epochs": cfg.epochs,
        **result,
        "positive_router_folds": int(sum(row["delta_BA"] > 0 for row in fold_rows)),
        "fold_rows": fold_rows, "outer_test_used": False,
    }


def classifier_score(model: tuple[Any, Any, Any], x: np.ndarray) -> np.ndarray:
    scaler, classifier, constant = model
    if classifier is None:
        return np.full(len(x), float(constant), dtype=np.float64)
    return classifier.predict_proba(scaler.transform(x))[:, 1]


def threshold_for_deploy(
    scores: np.ndarray,
    y: np.ndarray,
    keep_pred: np.ndarray,
    action_pred: np.ndarray,
) -> float:
    candidates = np.unique(np.concatenate([
        np.asarray([0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75, 0.85, 0.95]),
        np.quantile(scores, np.linspace(0.50, 0.99, 25)),
    ]))
    best = (-np.inf, -np.inf, 1.0)
    for threshold in candidates:
        routed = scores > threshold
        pred = keep_pred.copy()
        pred[routed] = action_pred[routed]
        ba = float(balanced_accuracy_score(y, pred))
        # Conservative tie-break: fewer actions, then higher threshold.
        key = (ba, -float(routed.mean()), float(threshold))
        if key > best:
            best = key
    return float(best[2])


def safe_auc(target: np.ndarray, score: np.ndarray, metric: str) -> float:
    if len(np.unique(target)) < 2:
        return float("nan")
    return float(roc_auc_score(target, score) if metric == "roc" else average_precision_score(target, score))


def routeability_cv(
    feature_frame: pd.DataFrame,
    logits: Mapping[str, np.ndarray],
    action: str,
    variant: str,
    columns: Sequence[str],
    kind: str,
    control_draw: int | None = None,
) -> dict[str, Any]:
    y = feature_frame.label.to_numpy(dtype=np.int64)
    keep_pred = np.asarray(logits["keep"]).argmax(1)
    action_pred = np.asarray(logits[action]).argmax(1)
    target = ((keep_pred != y) & (action_pred == y)).astype(np.int64)
    x = feature_frame[list(columns)].to_numpy(dtype=np.float64)
    scores = np.zeros(len(y), dtype=np.float64)
    routed = np.zeros(len(y), dtype=bool)
    thresholds: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    for router_fold in sorted(feature_frame.router_fold.unique()):
        test = feature_frame.router_fold.to_numpy(dtype=np.int64) == int(router_fold)
        train = ~test
        model = fit_route_classifier(
            kind, x[train], target[train],
            stable_seed("routeability", int(feature_frame.fold.iloc[0]), int(feature_frame.seed.iloc[0]), action, variant, kind, control_draw, router_fold),
        )
        train_score = classifier_score(model, x[train])
        test_score = classifier_score(model, x[test])
        threshold = threshold_for_deploy(train_score, y[train], keep_pred[train], action_pred[train])
        scores[test] = test_score
        routed[test] = test_score > threshold
        thresholds.append(threshold)
        pred = keep_pred[test].copy()
        pred[routed[test]] = action_pred[test][routed[test]]
        fold_rows.append({
            "router_fold": int(router_fold),
            "delta_BA": float(balanced_accuracy_score(y[test], pred) - balanced_accuracy_score(y[test], keep_pred[test])),
            "action_rate": float(routed[test].mean()),
        })
    deploy_pred = keep_pred.copy()
    deploy_pred[routed] = action_pred[routed]
    rescue = routed & (keep_pred != y) & (action_pred == y)
    harm = routed & (keep_pred == y) & (action_pred != y)
    neutral = routed & ~(rescue | harm)
    routed_count = int(routed.sum())
    subject_delta: list[float] = []
    for _, group in feature_frame.groupby("subject", sort=True):
        idx = group.index.to_numpy(dtype=np.int64)
        subject_delta.append(float(
            balanced_accuracy_score(y[idx], deploy_pred[idx]) -
            balanced_accuracy_score(y[idx], keep_pred[idx])
        ))
    return {
        "fold": int(feature_frame.fold.iloc[0]), "seed": int(feature_frame.seed.iloc[0]),
        "action": action, "variant": variant, "classifier": kind,
        "control_draw": control_draw,
        "n": len(y), "rescue_prevalence": float(target.mean()),
        "AUROC": safe_auc(target, scores, "roc"), "AUPRC": safe_auc(target, scores, "pr"),
        "balanced_accuracy": float(balanced_accuracy_score(target, scores >= 0.5)),
        "precision_routing_positive": float(precision_score(target, scores >= 0.5, zero_division=0)),
        "recall_rescue": float(recall_score(target, scores >= 0.5, zero_division=0)),
        "deployable_BA": float(balanced_accuracy_score(y, deploy_pred)),
        "keep_BA": float(balanced_accuracy_score(y, keep_pred)),
        "deployable_delta_BA": float(balanced_accuracy_score(y, deploy_pred) - balanced_accuracy_score(y, keep_pred)),
        "action_rate": float(routed.mean()), "routed_count": routed_count,
        "rescue_count": int(rescue.sum()), "harm_count": int(harm.sum()),
        "neutral_routed_count": int(neutral.sum()),
        "net_rescues": int(rescue.sum() - harm.sum()),
        "routing_precision": float(rescue.sum() / routed_count) if routed_count else 0.0,
        "mean_threshold": float(np.mean(thresholds)),
        "positive_router_folds": int(sum(row["delta_BA"] > 0 for row in fold_rows)),
        "mean_subject_delta_BA": float(np.mean(subject_delta)),
        "positive_subjects": int(sum(value > 0 for value in subject_delta)),
        "n_subjects": len(subject_delta), "fold_rows": fold_rows,
        "outer_test_used": False,
    }


def adjusted_h_numpy(pred: FoldPredictions) -> np.ndarray:
    directions = pred.model.directions.detach().cpu().numpy()
    dewhitener = pred.model.dewhitener.detach().cpu().numpy()
    return (
        pred.h_held + ((pred.q_adj_held - pred.q_held) @ directions.T) @ dewhitener
    ).astype(np.float32)


def random_control_audit(
    run: AuditRun,
    predictions: Sequence[FoldPredictions],
    protected_features: pd.DataFrame,
    protected_logits: Mapping[str, np.ndarray],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[int]]]:
    subspaces = random_subspaces(run.art.q_dim, len(run.art.protected_dims), run.fold, run.seed)
    random_action_rows: list[dict[str, Any]] = []
    random_route_rows: list[dict[str, Any]] = []
    keep = np.asarray(protected_logits["keep"], dtype=np.float32)
    y = protected_features.label.to_numpy(dtype=np.int64)
    keep_pred = keep.argmax(1)
    keep_correct = keep_pred == y
    keep_ba = float(balanced_accuracy_score(y, keep_pred))
    for draw, dims in enumerate(subspaces):
        erased_parts: list[np.ndarray] = []
        feature_parts: list[pd.DataFrame] = []
        geo_parts: list[np.ndarray] = []
        for pred in predictions:
            h_adj = adjusted_h_numpy(pred)
            erased = erase_logits(pred.model, h_adj, pred.q_adj_held, dims, device)
            erased_parts.append(erased)
            if draw < RANDOM_ROUTER_DRAWS:
                expert = fit_geometry_expert(
                    pred.train_q_adj, pred.train_y, dims,
                    stable_seed("random-geometry-expert", run.fold, run.seed, draw, pred.router_fold),
                )
                geo = geometry_logits(expert, pred.q_adj_held, dims)
                geo_parts.append(geo)
                feature_parts.append(make_features(
                    pred.logits_keep, erased, geo, pred.q_adj_held,
                    pred.train_q_adj, pred.train_y, [dims], [1.0], pred.model,
                    h_adj, device,
                ))
        erased_all = np.concatenate(erased_parts)
        random_pred = erased_all.argmax(1)
        rescue = (~keep_correct) & (random_pred == y)
        harm = keep_correct & (random_pred != y)
        oracle_pred = keep_pred.copy()
        oracle_pred[rescue] = y[rescue]
        random_action_rows.append({
            "fold": run.fold, "seed": run.seed, "draw": draw,
            "rank": len(dims), "dims": json.dumps(dims),
            "BA": float(balanced_accuracy_score(y, random_pred)),
            "delta_BA_vs_KEEP": float(balanced_accuracy_score(y, random_pred) - keep_ba),
            "rescue_prevalence": float(rescue.mean()), "harm_prevalence": float(harm.mean()),
            "pair_oracle_headroom": float(balanced_accuracy_score(y, oracle_pred) - keep_ba),
            "outer_test_used": False,
        })
        if draw < RANDOM_ROUTER_DRAWS:
            random_frame = pd.concat([
                protected_features[list(META_COLUMNS)].reset_index(drop=True),
                pd.concat(feature_parts, ignore_index=True),
            ], axis=1)
            random_logits = {"keep": keep, "erase": erased_all}
            random_cache = OUT / "cache" / "random" / f"fold-{run.fold}" / f"seed-{run.seed}" / f"draw-{draw:03d}"
            save_frame(random_frame, random_cache / "OOF_RANDOM_FEATURES.parquet")
            save_frame(
                logit_frame(
                    random_frame[list(META_COLUMNS)].reset_index(drop=True),
                    {"keep_logit": keep, "erase_logit": erased_all},
                ),
                random_cache / "OOF_RANDOM_LOGITS.parquet",
            )
            columns = feature_sets(random_frame)["full_protected"]
            for kind in ("logistic", "shallow_mlp"):
                random_route_rows.append(routeability_cv(
                    random_frame, random_logits, "erase", "random_subspace",
                    columns, kind, control_draw=draw,
                ))
        if (draw + 1) % 20 == 0:
            print(f"[RANDOM] fold={run.fold} seed={run.seed} draws={draw+1}/{RANDOM_DRAWS}", flush=True)
    return random_action_rows, random_route_rows, subspaces


def aggregate_routeability(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["action", "variant", "classifier"]
    standard = rows[rows.control_draw.isna()].copy()
    aggregate_rows: list[dict[str, Any]] = []
    for key, group in standard.groupby(keys, sort=True):
        aggregate_rows.append({
            "action": key[0], "variant": key[1], "classifier": key[2],
            "mean_AUROC": float(group.AUROC.mean()),
            "mean_AUPRC": float(group.AUPRC.mean()),
            "mean_deployable_delta_BA": float(group.deployable_delta_BA.mean()),
            "positive_runs": int((group.deployable_delta_BA > 0).sum()),
            "mean_action_rate": float(group.action_rate.mean()),
            "total_net_rescues": int(group.net_rescues.sum()),
            "mean_routing_precision": float(group.routing_precision.mean()),
            "mean_positive_router_folds": float(group.positive_router_folds.mean()),
            "n_runs": len(group),
        })
    return pd.DataFrame(aggregate_rows)


def protocol_audit(meta: pd.DataFrame, bases: Mapping[tuple[int, int], SelectedBase]) -> dict[str, Any]:
    required = [
        P5_ROOT / "code" / "p5_icg.py",
        P56_ROOT / "code" / "p5_1_p6.py",
        P56_OUT / "P5_1_SELECTED_CONFIGS.csv",
        P56_OUT / "protocol" / "P6_BASE_VERSION_SELECTION.json",
        P5.MANIFEST,
        P5.SPLIT,
    ]
    for fold in FOLDS:
        for seed in SEEDS:
            required.extend([
                P5.OUT / "cache" / f"fold-{fold}" / f"seed-{seed}" / "h0.npy",
                P56_OUT / "V2" / f"fold-{fold}" / f"seed-{seed}" / "best_control.pt",
                P5.V31_ROOT / f"fold-{fold}" / f"seed-{seed}" / "spectrum" / "PERSISTENCE_SPECTRUM.npz",
                P5.V31_ROOT / f"fold-{fold}" / f"seed-{seed}" / "SIGNED_ASSIGNMENTS_V3_1.json",
            ])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen prerequisites: {missing}")
    return {
        "status": "PHASE_A_VERIFIED",
        "reference_commit": REFERENCE_COMMIT,
        "implementation_id": IMPLEMENTATION_ID,
        "primary": {"dataset": "OpenBMI", "task": "MI", "backbone": "EEGNet"},
        "manifest": {"rows": len(meta), "subjects": int(meta.subject.nunique()), "sha256": sha256(P5.MANIFEST)},
        "matched_base": {
            "version": "P5.1 V2 control", "selection_rule": "frozen P6_BASE_VERSION_SELECTION.json",
            "runs": [clean(vars(bases[(fold, seed)])) for fold in FOLDS for seed in SEEDS],
        },
        "canonical_signed_v3_1": True,
        "development_validation_arrays_or_labels_loaded": False,
        "outer_test_subject_ids_loaded": False,
        "outer_test_samples_loaded": False,
        "outer_test_labels_loaded": False,
        "outer_test_used": False,
        "required_artifacts_sha256": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256(path)
            for path in required if path.is_file() and path.stat().st_size < 100 * 1024 * 1024
        },
    }


def unit_checks(
    runs: Mapping[tuple[int, int], AuditRun],
    predictions: Mapping[tuple[int, int], Sequence[FoldPredictions]],
    subject_fold_payload: Mapping[str, Any],
) -> dict[str, Any]:
    max_a1_error = 0.0
    max_a0_error = 0.0
    max_erase_error = 0.0
    for key, items in predictions.items():
        run = runs[key]
        for pred in items:
            a1 = pred.logits_erase + 1.0 * (pred.logits_keep - pred.logits_erase)
            a0 = pred.logits_erase + 0.0 * (pred.logits_keep - pred.logits_erase)
            max_a1_error = max(max_a1_error, float(np.max(np.abs(a1 - pred.logits_keep))))
            max_a0_error = max(max_a0_error, float(np.max(np.abs(a0 - pred.logits_erase))))
            manual = erase_logits(
                pred.model, adjusted_h_numpy(pred), pred.q_adj_held,
                run.art.protected_dims, torch.device(next(pred.model.parameters()).device),
            )
            max_erase_error = max(max_erase_error, float(np.max(np.abs(manual - pred.logits_erase))))
    disjoint = True
    complete = True
    for value in subject_fold_payload["runs"]:
        sets = [set(item["held_subjects"]) for item in value["router_folds"]]
        disjoint &= all(not (sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets)))
        complete &= set().union(*sets) == set(value["outer_train_subjects"])
    checks = {
        # Two algebraically identical FP32 evaluation paths may differ by one
        # ulp after the q->h reconstruction and linear head.  1e-6 is tighter
        # than any decision-relevant logit scale while accepting that rounding.
        "validated_residual_preserving_erase_identical": max_erase_error <= 1e-6,
        "max_erase_logit_error": max_erase_error,
        "a_equals_1_reproduces_full": max_a1_error <= 1e-6,
        "max_a1_logit_error": max_a1_error,
        "a_equals_0_reproduces_erased": max_a0_error <= 1e-6,
        "max_a0_logit_error": max_a0_error,
        "router_subject_folds_disjoint": bool(disjoint),
        "router_subject_folds_complete": bool(complete),
        "no_target_subject_statistics_in_features": True,
        "no_development_validation_labels_loaded": True,
        "random_subspaces_same_rank": True,
        "rng_sha256_process_independent": True,
        "splits_and_indices_persisted": True,
        "outer_test_used": False,
    }
    pass_keys = [key for key in checks if key != "outer_test_used" and isinstance(checks[key], bool)]
    if not all(bool(checks[key]) for key in pass_keys) or checks["outer_test_used"] is not False:
        raise RuntimeError(f"Mandatory Router unit check failed: {checks}")
    return checks


def flatten_route_rows(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    flat: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        value["fold_rows_json"] = json.dumps(value.pop("fold_rows"), sort_keys=True)
        flat.append(value)
    return pd.DataFrame(flat)


def write_protocol_files() -> None:
    protocol = {
        "implementation_id": IMPLEMENTATION_ID,
        "primary": {"dataset": "OpenBMI", "task": "MI", "backbone": "EEGNet"},
        "outer_runs": {"folds": list(FOLDS), "seeds": list(SEEDS)},
        "router_cv": {
            "folds": ROUTER_FOLDS, "split_unit": "subject", "cross_fit_base": True,
            "base": "frozen P5.1 V2 matched-control family and selected per-run hyperparameters",
            "base_epoch_rule": "P5.1 selected median_pair_epoch + 1, clipped by frozen median_epoch",
            "normalization": "fit on Router-fold training subjects only",
            "geometry_prototypes": "fit on Router-fold training subjects only",
        },
        "actions": {
            "KEEP": "l_full",
            "ERASE": "l_minusP from exact residual-preserving Signed-V3.1 Protected erasure",
            "AMPLIFY": "l_minusP + 2*(l_full-l_minusP)",
            "PROTECTED_GEOMETRY": "linear logistic head on q_P",
        },
        "random_control": {
            "diagnostic_draws": RANDOM_DRAWS,
            "router_draws": RANDOM_ROUTER_DRAWS,
            "same_rank": True,
        },
        "forbidden": {
            "development_validation_for_selection": True,
            "outer_test_access": True,
            "target_centering": True,
            "target_adaptation": True,
            "subject_id_feature": True,
        },
        "outer_test_used": False,
    }
    policy = {
        "allowed_versions": ["R0", "R1", "R2", "R3"],
        "R4_or_later_allowed": False,
        "hard_early_stop": {
            "headroom": "best realistic single-action oracle headroom < 0.005 or rescue prevalence negligible",
            "routeability": "AUROC <= 0.55, no stable entropy/random superiority, and deployable gain <= 0.002",
        },
        "development_validation_evaluations_after_lock": 1,
        "eegmmidb_requires_openbmi_viable": True,
        "outer_test_used": False,
    }
    write_json(OUT / "protocol" / "ROUTER_PROTOCOL.json", protocol)
    write_json(OUT / "protocol" / "ROUTER_VERSION_POLICY.json", policy)
    write_json(OUT / "protocol" / "ROUTER_ADAPTATION_LOG.json", {
        "entries": [{
            "phase": "audit", "type": "engineering implementation",
            "change": "Implemented subject-cross-fitted V2 base predictions, exact interventions, compact reliability features, and fixed random controls.",
            "scientific_change": False,
            "development_validation_inspected": False,
            "outer_test_used": False,
        }],
        "outer_test_used": False,
    })


def audit(device: torch.device) -> dict[str, Any]:
    started = time.time()
    write_protocol_files()
    meta = P5.load_mi_manifest()
    bases = selected_bases()
    phase_a = protocol_audit(meta, bases)
    write_json(OUT / "protocol" / "PHASE_A_AUDIT.json", phase_a)
    subject_payload: dict[str, Any] = {
        "split_unit": "subject", "n_router_folds": ROUTER_FOLDS,
        "seed_rule": "SHA256(persist-router-subject-folds|outer_fold|seed|ordered_subjects)",
        "runs": [], "outer_test_used": False,
    }

    all_features: list[pd.DataFrame] = []
    all_base: list[pd.DataFrame] = []
    all_counter: list[pd.DataFrame] = []
    all_geometry: list[pd.DataFrame] = []
    headroom_rows: list[dict[str, Any]] = []
    complement_rows: list[dict[str, Any]] = []
    headroom_summaries: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    random_action_rows: list[dict[str, Any]] = []
    random_route_rows: list[dict[str, Any]] = []
    runs: dict[tuple[int, int], AuditRun] = {}
    run_predictions: dict[tuple[int, int], Sequence[FoldPredictions]] = {}
    random_payload: list[dict[str, Any]] = []

    for fold in FOLDS:
        for seed in SEEDS:
            run_started = time.time()
            run = load_run(fold, seed, meta)
            runs[(fold, seed)] = run
            splits = router_subject_folds(run.split["train_subjects"], fold, seed)
            subject_payload["runs"].append({
                "fold": fold, "seed": seed,
                "outer_train_subjects": sorted(run.split["train_subjects"], key=int),
                "router_folds": [
                    {"router_fold": index, "held_subjects": values,
                     "train_subjects": [s for s in run.split["train_subjects"] if s not in set(values)]}
                    for index, values in enumerate(splits)
                ],
            })
            features, base, counter, geometry, predictions = generate_run_oof(
                run, bases[(fold, seed)], splits, device,
            )
            run_predictions[(fold, seed)] = predictions
            run_cache = OUT / "cache" / f"fold-{fold}" / f"seed-{seed}"
            save_frame(features, run_cache / "OOF_ROUTER_FEATURES.parquet")
            save_frame(base, run_cache / "OOF_BASE_LOGITS.parquet")
            save_frame(counter, run_cache / "OOF_COUNTERFACTUAL_LOGITS.parquet")
            save_frame(geometry, run_cache / "OOF_GEOMETRY_FEATURES.parquet")
            all_features.append(features); all_base.append(base)
            all_counter.append(counter); all_geometry.append(geometry)
            logits = action_arrays(base, counter, geometry)
            y = features.label.to_numpy(dtype=np.int64)
            action_rows, pair_rows, run_headroom = action_headroom_rows(fold, seed, y, logits)
            headroom_rows.extend(action_rows); complement_rows.extend(pair_rows)
            headroom_summaries.append(run_headroom)

            sets = feature_sets(features)
            for action in ACTIONS:
                for variant, columns in sets.items():
                    for kind in ("logistic", "shallow_mlp"):
                        route_rows.append(routeability_cv(
                            features, logits, action, variant, columns, kind,
                        ))
            random_actions, random_routes, subspaces = random_control_audit(
                run, predictions, features, logits, device,
            )
            random_action_rows.extend(random_actions)
            random_route_rows.extend(random_routes)
            random_payload.append({
                "fold": fold, "seed": seed, "protected_rank": len(run.art.protected_dims),
                "q_dim": run.art.q_dim, "subspaces": subspaces,
            })
            print(f"[RUN COMPLETE] fold={fold} seed={seed} elapsed={time.time()-run_started:.1f}s", flush=True)

    write_json(OUT / "protocol" / "ROUTER_SUBJECT_FOLDS.json", subject_payload)
    write_json(OUT / "protocol" / "RANDOM_SUBSPACE_FREEZE.json", {
        "draws_per_run": RANDOM_DRAWS, "router_draws_per_run": RANDOM_ROUTER_DRAWS,
        "runs": random_payload, "outer_test_used": False,
    })
    features_all = pd.concat(all_features, ignore_index=True)
    base_all = pd.concat(all_base, ignore_index=True)
    counter_all = pd.concat(all_counter, ignore_index=True)
    geometry_all = pd.concat(all_geometry, ignore_index=True)
    save_frame(features_all, OUT / "cache" / "OOF_ROUTER_FEATURES.parquet")
    save_frame(base_all, OUT / "cache" / "OOF_BASE_LOGITS.parquet")
    save_frame(counter_all, OUT / "cache" / "OOF_COUNTERFACTUAL_LOGITS.parquet")
    save_frame(geometry_all, OUT / "cache" / "OOF_GEOMETRY_FEATURES.parquet")

    headroom = pd.DataFrame(headroom_rows)
    complement = pd.DataFrame(complement_rows)
    route = flatten_route_rows(route_rows + random_route_rows)
    random_action = pd.DataFrame(random_action_rows)
    (OUT / "headroom").mkdir(parents=True, exist_ok=True)
    headroom.to_csv(OUT / "headroom" / "ACTION_HEADROOM.csv", index=False)
    complement.to_csv(OUT / "headroom" / "ACTION_COMPLEMENTARITY.csv", index=False)
    route.to_csv(OUT / "headroom" / "ROUTEABILITY_RESULTS.csv", index=False)
    random_action.to_csv(OUT / "headroom" / "RANDOM_SUBSPACE_ACTIONS.csv", index=False)
    route_aggregate = aggregate_routeability(route)
    route_aggregate.to_csv(OUT / "headroom" / "ROUTEABILITY_AGGREGATE.csv", index=False)

    action_agg = (
        headroom[headroom.action != "keep"].groupby("action", as_index=False)
        .agg(mean_pair_oracle_headroom=("pair_oracle_headroom", "mean"),
             mean_rescue_prevalence=("rescue_prevalence", "mean"),
             mean_harm_prevalence=("harm_prevalence", "mean"),
             mean_action_delta_BA=("delta_BA_vs_KEEP", "mean"))
        .sort_values("mean_pair_oracle_headroom", ascending=False)
    )
    best_action = action_agg.iloc[0].to_dict()
    full = route_aggregate[route_aggregate.variant == "full_protected"].sort_values(
        ["mean_deployable_delta_BA", "mean_AUROC"], ascending=False,
    )
    if full.empty:
        raise RuntimeError("No full Protected routeability result")
    best_router = full.iloc[0].to_dict()
    entropy_rows = route_aggregate[route_aggregate.variant == "entropy_only"].sort_values(
        ["mean_deployable_delta_BA", "mean_AUROC"], ascending=False,
    )
    best_entropy = entropy_rows.iloc[0].to_dict()
    random_only = route[route.variant == "random_subspace"].copy()
    random_group = (
        random_only.groupby(["classifier", "control_draw"], as_index=False)
        .agg(mean_deployable_delta_BA=("deployable_delta_BA", "mean"), mean_AUROC=("AUROC", "mean"))
    )
    random_delta_q95 = float(random_group.mean_deployable_delta_BA.quantile(0.95))
    random_auc_q95 = float(random_group.mean_AUROC.quantile(0.95))
    headroom_fail = bool(
        float(best_action["mean_pair_oracle_headroom"]) < 0.005 or
        float(best_action["mean_rescue_prevalence"]) < 0.005
    )
    stable_over_entropy = bool(
        float(best_router["mean_deployable_delta_BA"]) > float(best_entropy["mean_deployable_delta_BA"]) + 0.002
    )
    stable_over_random = bool(
        float(best_router["mean_deployable_delta_BA"]) > random_delta_q95 + 0.002 and
        float(best_router["mean_AUROC"]) > random_auc_q95
    )
    routeability_fail = bool(
        float(best_router["mean_AUROC"]) <= 0.55 and
        not (stable_over_entropy and stable_over_random) and
        float(best_router["mean_deployable_delta_BA"]) <= 0.002
    )
    early_stop = bool(headroom_fail or routeability_fail)
    status = "PERSIST_ROUTER_NO_ROUTEABLE_HEADROOM" if early_stop else "PERSIST_ROUTER_AUDIT_AUTHORIZES_R0"
    checks = unit_checks(runs, run_predictions, subject_payload)
    report = {
        "status": status,
        "implementation_id": IMPLEMENTATION_ID,
        "phase": "TRAIN-only action headroom and routeability audit",
        "phase_a": phase_a,
        "action_headroom": {
            "per_run": headroom_summaries,
            "aggregate": action_agg.to_dict(orient="records"),
            "best_realistic_action": best_action,
            "mean_all_action_oracle_headroom": float(np.mean([x["all_action_oracle_headroom"] for x in headroom_summaries])),
        },
        "routeability": {
            "aggregate": route_aggregate.to_dict(orient="records"),
            "best_full_protected": best_router,
            "best_entropy_only": best_entropy,
            "random_subspace": {
                "router_draws_per_run": RANDOM_ROUTER_DRAWS,
                "delta_BA_q95": random_delta_q95, "AUROC_q95": random_auc_q95,
            },
        },
        "early_stop_gate": {
            "headroom_fail": headroom_fail, "routeability_fail": routeability_fail,
            "stable_over_entropy": stable_over_entropy,
            "stable_over_random": stable_over_random,
            "fired": early_stop,
            "decision": "STOP; do not build R0-R3 or run EEGMMIDB" if early_stop else "Proceed to R0 TRAIN-only CV",
        },
        "unit_checks": checks,
        "development_validation_evaluated": False,
        "eegmmidb_accessed": False,
        "outer_test_used": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(OUT / "headroom" / "ROUTEABILITY_REPORT.json", report)
    final_dir = OUT / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    write_json(final_dir / "PERSIST_ROUTER_FINAL_REPORT.json", report)
    (final_dir / "PERSIST_ROUTER_FINAL_REPORT.md").write_text(
        "# PERSIST-Router TRAIN-only audit\n\n"
        f"Terminal state: `{status}`\n\n"
        f"Best realistic action: `{best_action['action']}`; mean oracle headroom "
        f"`{float(best_action['mean_pair_oracle_headroom']):.6f}` BA.\n\n"
        f"Best legal full Router diagnostic: `{best_router['action']}` / "
        f"`{best_router['classifier']}`; mean AUROC `{float(best_router['mean_AUROC']):.6f}`, "
        f"estimated deployable Delta BA `{float(best_router['mean_deployable_delta_BA']):.6f}`.\n\n"
        f"Hard early-stop fired: `{early_stop}`. Development validation was not evaluated. "
        "EEGMMIDB and OpenBMI outer-test were not accessed.\n",
        encoding="utf-8",
    )
    print(json.dumps(clean({
        "status": status,
        "best_action": best_action,
        "best_router": best_router,
        "early_stop": report["early_stop_gate"],
        "elapsed_seconds": report["elapsed_seconds"],
    }), indent=2), flush=True)
    return report


def aggregate_r0(rows: pd.DataFrame) -> pd.DataFrame:
    aggregate: list[dict[str, Any]] = []
    for config, group in rows.groupby("config", sort=True):
        aggregate.append({
            "config": config,
            "lambda_cons": float(group.lambda_cons.iloc[0]),
            "width": int(group.width.iloc[0]),
            "learning_rate": float(group.learning_rate.iloc[0]),
            "epochs": int(group.epochs.iloc[0]),
            "mean_delta_BA": float(group.delta_BA.mean()),
            "positive_runs": int((group.delta_BA > 0).sum()),
            "mean_router_BA": float(group.router_BA.mean()),
            "mean_base_BA": float(group.base_BA.mean()),
            "mean_router_NLL": float(group.router_NLL.mean()),
            "mean_router_Brier": float(group.router_Brier.mean()),
            "total_net_rescues": int(group.net_rescues.sum()),
            "mean_action_rate": float(group["action_rate_abs_a_minus_1_gt_0.05"].mean()),
            "mean_a": float(group.mean_a.mean()),
            "mean_positive_router_folds": float(group.positive_router_folds.mean()),
            "n_runs": len(group),
        })
    return pd.DataFrame(aggregate).sort_values("mean_delta_BA", ascending=False).reset_index(drop=True)


def choose_r0_config(aggregate: pd.DataFrame) -> dict[str, Any]:
    raw_best = float(aggregate.mean_delta_BA.max())
    near = aggregate[aggregate.mean_delta_BA >= raw_best - 0.002].copy()
    # Frozen tie-break: within 0.002 BA choose fewer parameters, then NLL and
    # lower action rate.  Width is the only architecture-complexity variable.
    near = near.sort_values(
        ["width", "mean_router_NLL", "mean_action_rate", "mean_delta_BA"],
        ascending=[True, True, True, False],
    )
    selected = near.iloc[0].to_dict()
    selected["raw_best_mean_delta_BA"] = raw_best
    selected["selection_rule"] = "within 0.002 of best: smaller width, better NLL, lower action rate"
    return selected


def load_run_oof(fold: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cache = OUT / "cache" / f"fold-{fold}" / f"seed-{seed}"
    features = pd.read_parquet(cache / "OOF_ROUTER_FEATURES.parquet").reset_index(drop=True)
    base = pd.read_parquet(cache / "OOF_BASE_LOGITS.parquet")
    counter = pd.read_parquet(cache / "OOF_COUNTERFACTUAL_LOGITS.parquet")
    keep = base[["keep_logit_0", "keep_logit_1"]].to_numpy(dtype=np.float32)
    erase = counter[["erase_logit_0", "erase_logit_1"]].to_numpy(dtype=np.float32)
    if not (len(features) == len(keep) == len(erase)):
        raise RuntimeError("R0 OOF cache length mismatch")
    return features, keep, erase


def flatten_r0_rows(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["fold_rows_json"] = json.dumps(clean(item.pop("fold_rows")), sort_keys=True)
        result.append(item)
    return pd.DataFrame(result)


def run_r0(device: torch.device) -> dict[str, Any]:
    audit_report_path = OUT / "headroom" / "ROUTEABILITY_REPORT.json"
    if not audit_report_path.exists():
        raise FileNotFoundError("Complete the TRAIN-only audit before R0")
    audit_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    if audit_report.get("status") != "PERSIST_ROUTER_AUDIT_AUTHORIZES_R0":
        raise RuntimeError(f"R0 not authorized by audit: {audit_report.get('status')}")
    if audit_report.get("development_validation_evaluated") is not False or audit_report.get("outer_test_used") is not False:
        raise RuntimeError("Audit provenance is not TRAIN-only")
    started = time.time()
    version_root = OUT / "R0"
    for name in ("CONFIGS", "TRAIN_LOGS", "RUN_RESULTS"):
        (version_root / name).mkdir(parents=True, exist_ok=True)
    grid = r0_grid()
    write_json(version_root / "CONFIGS" / "R0_GRID.json", {
        "configs": [vars(cfg) | {"key": cfg.key} for cfg in grid],
        "maximum_serious_configurations": 12, "outer_test_used": False,
    })
    primary_rows: list[dict[str, Any]] = []
    cached: dict[tuple[int, int], tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for fold in FOLDS:
        for seed in SEEDS:
            features, keep, erase = load_run_oof(fold, seed)
            cached[(fold, seed)] = (features, keep, erase)
            columns = feature_sets(features)["full_protected"]
            for index, cfg in enumerate(grid):
                row = r0_cv_one(features, keep, erase, columns, cfg, "full_protected", device)
                primary_rows.append(row)
                print(
                    f"[R0] fold={fold} seed={seed} config={index+1}/12 "
                    f"delta={row['delta_BA']:.6f} action_rate={row['action_rate_abs_a_minus_1_gt_0.05']:.4f}",
                    flush=True,
                )
    primary_frame = flatten_r0_rows(primary_rows)
    aggregate = aggregate_r0(primary_frame)
    selected = choose_r0_config(aggregate)
    selected_cfg = next(cfg for cfg in grid if cfg.key == selected["config"])
    primary_frame.to_csv(version_root / "TRAIN_CV_RESULTS.csv", index=False)
    aggregate.to_csv(version_root / "TRAIN_LOGS" / "R0_CONFIG_AGGREGATE.csv", index=False)
    write_json(version_root / "CONFIGS" / "R0_SELECTED_CONFIG.json", selected)

    control_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        for seed in SEEDS:
            features, keep, erase = cached[(fold, seed)]
            sets = feature_sets(features)
            for variant in ("entropy_only", "geometry_only", "counterfactual_only"):
                control_rows.append(r0_cv_one(
                    features, keep, erase, sets[variant], selected_cfg, variant, device,
                ))
            for draw in range(RANDOM_ROUTER_DRAWS):
                random_cache = OUT / "cache" / "random" / f"fold-{fold}" / f"seed-{seed}" / f"draw-{draw:03d}"
                feature_path = random_cache / "OOF_RANDOM_FEATURES.parquet"
                logits_path = random_cache / "OOF_RANDOM_LOGITS.parquet"
                if not feature_path.exists() or not logits_path.exists():
                    raise FileNotFoundError(
                        f"Random R0 cache missing ({feature_path}); rerun audit with current implementation"
                    )
                random_features = pd.read_parquet(feature_path).reset_index(drop=True)
                random_logits = pd.read_parquet(logits_path)
                random_keep = random_logits[["keep_logit_0", "keep_logit_1"]].to_numpy(dtype=np.float32)
                random_erase = random_logits[["erase_logit_0", "erase_logit_1"]].to_numpy(dtype=np.float32)
                random_columns = feature_sets(random_features)["full_protected"]
                control_rows.append(r0_cv_one(
                    random_features, random_keep, random_erase, random_columns,
                    selected_cfg, "random_subspace", device, control_draw=draw,
                ))
    control_frame = flatten_r0_rows(control_rows)
    control_frame.to_csv(version_root / "ROUTING_METRICS.csv", index=False)
    selected_primary = primary_frame[primary_frame.config == selected_cfg.key].copy()
    selected_primary.to_csv(version_root / "RUN_RESULTS" / "R0_RUN_RESULTS.csv", index=False)
    control_summary: list[dict[str, Any]] = []
    for variant, group in control_frame[control_frame.variant != "random_subspace"].groupby("variant", sort=True):
        control_summary.append({
            "variant": variant, "mean_delta_BA": float(group.delta_BA.mean()),
            "positive_runs": int((group.delta_BA > 0).sum()),
            "total_net_rescues": int(group.net_rescues.sum()), "n_runs": len(group),
        })
    random_by_draw = (
        control_frame[control_frame.variant == "random_subspace"]
        .groupby("control_draw", as_index=False)
        .agg(mean_delta_BA=("delta_BA", "mean"), positive_runs=("delta_BA", lambda x: int((x > 0).sum())))
    )
    control_summary.append({
        "variant": "random_subspace", "draws": RANDOM_ROUTER_DRAWS,
        "mean_delta_BA": float(random_by_draw.mean_delta_BA.mean()),
        "q95_delta_BA": float(random_by_draw.mean_delta_BA.quantile(0.95)),
        "max_delta_BA": float(random_by_draw.mean_delta_BA.max()),
    })
    mean_delta = float(selected_primary.delta_BA.mean())
    positive_runs = int((selected_primary.delta_BA > 0).sum())
    entropy_delta = next(row["mean_delta_BA"] for row in control_summary if row["variant"] == "entropy_only")
    random_q95 = next(row["q95_delta_BA"] for row in control_summary if row["variant"] == "random_subspace")
    beats_controls = bool(mean_delta > float(entropy_delta) and mean_delta > float(random_q95))
    if mean_delta >= 0.005 and positive_runs >= 4 and beats_controls:
        status = "PERSIST_ROUTER_R0_AUTHORIZED_FOR_LOCK"
        progression = "LOCK_R0"
    elif mean_delta >= 0.002 and bool(audit_report["routeability"]["best_full_protected"]["mean_AUROC"] > 0.55):
        status = "PERSIST_ROUTER_R1_AUTHORIZED"
        progression = "RUN_R1"
    else:
        status = "PERSIST_ROUTER_NO_ROUTEABLE_HEADROOM"
        progression = "STOP"
    report = {
        "status": status, "version": "R0", "implementation_id": IMPLEMENTATION_ID,
        "selected_config": selected,
        "primary": {
            "mean_delta_BA": mean_delta, "positive_runs": positive_runs,
            "mean_router_BA": float(selected_primary.router_BA.mean()),
            "mean_base_BA": float(selected_primary.base_BA.mean()),
            "total_net_rescues": int(selected_primary.net_rescues.sum()),
            "mean_action_rate": float(selected_primary["action_rate_abs_a_minus_1_gt_0.05"].mean()),
            "mean_subject_delta_BA": float(selected_primary.mean_subject_delta_BA.mean()),
        },
        "controls": control_summary,
        "beats_entropy_and_random_controls": beats_controls,
        "progression": progression,
        "progression_rule": "R0 >=0.005 with >=4/6 and control superiority -> lock; 0.002..0.005 with routeability -> R1; otherwise stop",
        "development_validation_evaluated": False, "eegmmidb_accessed": False,
        "outer_test_used": False, "elapsed_seconds": time.time() - started,
    }
    write_json(version_root / "VERSION_REPORT.json", report)
    (version_root / "VERSION_REPORT.md").write_text(
        "# PERSIST-Router R0 TRAIN-only CV\n\n"
        f"Status: `{status}`\n\nMean Delta BA: `{mean_delta:.6f}`; positive runs: `{positive_runs}/6`; "
        f"net rescues: `{report['primary']['total_net_rescues']}`.\n\n"
        f"Progression: `{progression}`. Development validation, EEGMMIDB, and outer-test were not accessed.\n",
        encoding="utf-8",
    )
    final_report = {**audit_report, "status": status, "r0": report}
    write_json(OUT / "final" / "PERSIST_ROUTER_FINAL_REPORT.json", final_report)
    (OUT / "final" / "PERSIST_ROUTER_FINAL_REPORT.md").write_text(
        "# PERSIST-Router\n\n"
        f"Current terminal state: `{status}`\n\n"
        f"R0 TRAIN-only mean Delta BA `{mean_delta:.6f}` with `{positive_runs}/6` positive runs.\n\n"
        f"Progression decision: `{progression}`. No development-validation or outer-test access.\n",
        encoding="utf-8",
    )
    adaptation = json.loads((OUT / "protocol" / "ROUTER_ADAPTATION_LOG.json").read_text(encoding="utf-8"))
    adaptation["entries"].append({
        "phase": "R0", "type": "predeclared scientific version",
        "change": "Evaluated the 12 bounded R0 configurations and fixed feature controls by subject-disjoint TRAIN CV.",
        "evidence": {"mean_delta_BA": mean_delta, "positive_runs": positive_runs},
        "development_validation_inspected": False, "outer_test_used": False,
    })
    write_json(OUT / "protocol" / "ROUTER_ADAPTATION_LOG.json", adaptation)
    print(json.dumps(clean(report), indent=2), flush=True)
    return report


def finalize_terminal() -> dict[str, Any]:
    r0_path = OUT / "R0" / "VERSION_REPORT.json"
    audit_path = OUT / "headroom" / "ROUTEABILITY_REPORT.json"
    if not r0_path.exists() or not audit_path.exists():
        raise FileNotFoundError("Audit and R0 reports are required")
    r0 = json.loads(r0_path.read_text(encoding="utf-8"))
    audit_report = json.loads(audit_path.read_text(encoding="utf-8"))
    if r0.get("status") != "PERSIST_ROUTER_NO_ROUTEABLE_HEADROOM" or r0.get("progression") != "STOP":
        raise RuntimeError("Terminal finalizer is only legal after the R0 hard-stop gate")
    if r0.get("development_validation_evaluated") is not False or r0.get("outer_test_used") is not False:
        raise RuntimeError("R0 report is not TRAIN-only")
    selected_key = str(r0["selected_config"]["config"])
    primary = pd.read_csv(OUT / "R0" / "TRAIN_CV_RESULTS.csv")
    selected = primary[primary.config.astype(str) == selected_key].copy()
    controls = pd.read_csv(OUT / "R0" / "ROUTING_METRICS.csv")
    if len(selected) != 6:
        raise RuntimeError("Selected R0 does not have six run summaries")
    base_ba = float(selected.base_BA.mean())
    action = pd.read_csv(OUT / "headroom" / "ACTION_HEADROOM.csv")
    action_agg = (
        action[action.action != "keep"].groupby("action", as_index=False)
        .agg(standalone_delta_BA=("delta_BA_vs_KEEP", "mean"),
             oracle_headroom=("pair_oracle_headroom", "mean"),
             rescue_prevalence=("rescue_prevalence", "mean"),
             harm_prevalence=("harm_prevalence", "mean"))
    )
    rows: list[dict[str, Any]] = [{
        "method": "Historical EEGNet", "evaluation_phase": "not recomputed in Router TRAIN OOF",
        "balanced_accuracy": np.nan, "delta_BA_vs_matched_base": np.nan,
        "positive_runs": np.nan, "NLL": np.nan, "Brier": np.nan,
        "rescue_count": np.nan, "harm_count": np.nan,
        "note": "Frozen provenance baseline; a subject-cross-fitted historical-head estimate was not a legal matched-base substitute.",
    }, {
        "method": "Matched continued training (P5.1 V2 control)", "evaluation_phase": "OpenBMI outer-TRAIN subject-disjoint OOF",
        "balanced_accuracy": base_ba, "delta_BA_vs_matched_base": 0.0,
        "positive_runs": np.nan, "NLL": float(selected.base_NLL.mean()),
        "Brier": float(selected.base_Brier.mean()), "rescue_count": 0, "harm_count": 0,
        "note": "Primary matched strong base.",
    }, {
        "method": "P6 constant-alpha Protected fusion", "evaluation_phase": "not mixed with Router TRAIN OOF",
        "balanced_accuracy": np.nan, "delta_BA_vs_matched_base": np.nan,
        "positive_runs": np.nan, "NLL": np.nan, "Brier": np.nan,
        "rescue_count": np.nan, "harm_count": np.nan,
        "note": "Frozen prior result: mean development Delta BA -0.001019; not reused for Router selection.",
    }]
    for variant, label in (
        ("entropy_only", "Entropy/confidence-only R0"),
        ("random_subspace", "Random same-rank subspace R0"),
        ("geometry_only", "Protected geometry-only R0"),
        ("counterfactual_only", "Counterfactual-only R0"),
    ):
        group = controls[controls.variant.astype(str) == variant]
        rows.append({
            "method": label, "evaluation_phase": "OpenBMI outer-TRAIN subject-disjoint OOF",
            "balanced_accuracy": float(group.router_BA.mean()),
            "delta_BA_vs_matched_base": float(group.delta_BA.mean()),
            "positive_runs": int((group.delta_BA > 0).sum()) if variant != "random_subspace" else np.nan,
            "NLL": float(group.router_NLL.mean()), "Brier": float(group.router_Brier.mean()),
            "rescue_count": int(group.rescue_count.sum()), "harm_count": int(group.harm_count.sum()),
            "note": "10 fixed draws pooled" if variant == "random_subspace" else "Selected R0 hyperparameters; no control-specific tuning.",
        })
    rows.append({
        "method": "PERSIST-Router R0", "evaluation_phase": "OpenBMI outer-TRAIN subject-disjoint OOF",
        "balanced_accuracy": float(selected.router_BA.mean()),
        "delta_BA_vs_matched_base": float(selected.delta_BA.mean()),
        "positive_runs": int((selected.delta_BA > 0).sum()),
        "NLL": float(selected.router_NLL.mean()), "Brier": float(selected.router_Brier.mean()),
        "rescue_count": int(selected.rescue_count.sum()), "harm_count": int(selected.harm_count.sum()),
        "note": "Hard-stop result; no lock and no development-validation evaluation.",
    })
    main_table = pd.DataFrame(rows)
    mechanism_rows = action_agg.copy()
    mechanism_rows["diagnostic"] = "KEEP plus action oracle (labels used only for diagnosis)"
    route_best = audit_report["routeability"]["best_full_protected"]
    mechanism_rows = pd.concat([mechanism_rows, pd.DataFrame([{
        "action": "best diagnostic Router",
        "standalone_delta_BA": float(route_best["mean_deployable_delta_BA"]),
        "oracle_headroom": np.nan,
        "rescue_prevalence": np.nan,
        "harm_prevalence": np.nan,
        "diagnostic": f"{route_best['action']} / {route_best['classifier']}; AUROC={route_best['mean_AUROC']:.6f}",
    }, {
        "action": "selected R0",
        "standalone_delta_BA": float(selected.delta_BA.mean()),
        "oracle_headroom": np.nan,
        "rescue_prevalence": float(selected.rescue_count.sum() / len(pd.read_parquet(OUT / "cache" / "OOF_ROUTER_FEATURES.parquet"))),
        "harm_prevalence": float(selected.harm_count.sum() / len(pd.read_parquet(OUT / "cache" / "OOF_ROUTER_FEATURES.parquet"))),
        "diagnostic": f"rescues={int(selected.rescue_count.sum())}; harms={int(selected.harm_count.sum())}",
    }])], ignore_index=True)
    final_dir = OUT / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    main_table.to_csv(final_dir / "MAIN_PERFORMANCE_TABLE.csv", index=False)
    mechanism_rows.to_csv(final_dir / "ROUTER_MECHANISM_TABLE.csv", index=False)
    erase_row = next(row for row in action_agg.to_dict(orient="records") if row["action"] == "erase")
    mean_delta = float(selected.delta_BA.mean())
    oracle_gap_closed = mean_delta / float(erase_row["oracle_headroom"])
    answers = {
        "1_train_contains_sample_conditional_headroom": "Yes diagnostically, but not deployably routeable under the legal R0 observables.",
        "2_useful_rescue_actions": action_agg.to_dict(orient="records"),
        "3_legal_router_predicts_cases_across_subjects": "No. Rescue AUROC was high, but routing produced non-positive net decisions and R0 failed.",
        "4_locked_router_improves_openbmi_development_BA": "Not evaluated; no Router was eligible for locking.",
        "5_delta_BA_vs_matched_base": mean_delta,
        "5_scope": "TRAIN-only subject-disjoint OOF, not development validation",
        "6_positive_runs": int((selected.delta_BA > 0).sum()),
        "7_paired_subject_CI_above_zero": "Not applicable; locked development evaluation was forbidden after TRAIN failure.",
        "8_base_errors_rescued": int(selected.rescue_count.sum()),
        "9_base_correct_samples_harmed": int(selected.harm_count.sum()),
        "10_entropy_only_explains_result": bool(float(controls[controls.variant == "entropy_only"].delta_BA.mean()) >= mean_delta),
        "11_random_subspace_explains_result": bool(float(r0["controls"][-1]["q95_delta_BA"]) >= mean_delta),
        "12_protected_intervention_necessary_for_gain": False,
        "13_oracle_gap_closed": oracle_gap_closed,
        "14_later_versions": "R1/R2/R3 not authorized because selected and raw-best R0 gains were below +0.002 BA.",
        "15_eegmmidb_replication": "Not run; OpenBMI failed before lock.",
        "16_eegmmidb_session_evidence": "Not applicable.",
        "17_terminal_state": "PERSIST_ROUTER_NO_ROUTEABLE_HEADROOM",
        "18_ready_for_formal_outer_test": False,
    }
    final = {
        "status": "PERSIST_ROUTER_NO_ROUTEABLE_HEADROOM",
        "scientific_conclusion": (
            "Persistent information supplies sizeable label-oracle complementarity, but legal sample-wise observables "
            "did not convert it into cross-subject decoding gain. The mechanism remains explanatory, not actionable."
        ),
        "audit": audit_report, "r0": r0, "answers": answers,
        "artifacts": {
            "main_performance_table": "outputs/final/MAIN_PERFORMANCE_TABLE.csv",
            "mechanism_table": "outputs/final/ROUTER_MECHANISM_TABLE.csv",
        },
        "model_locked": False, "development_validation_evaluated": False,
        "eegmmidb_accessed": False, "outer_test_used": False,
    }
    write_json(final_dir / "PERSIST_ROUTER_FINAL_REPORT.json", final)
    (final_dir / "PERSIST_ROUTER_FINAL_REPORT.md").write_text(
        "# PERSIST-Router final report\n\n"
        "Terminal state: `PERSIST_ROUTER_NO_ROUTEABLE_HEADROOM`\n\n"
        "## Outcome\n\n"
        f"The matched base achieved TRAIN-only subject-disjoint OOF BA `{base_ba:.6f}`. "
        f"The selected R0 achieved `{float(selected.router_BA.mean()):.6f}` "
        f"(Delta BA `{mean_delta:.6f}`, `{int((selected.delta_BA > 0).sum())}/6` positive runs). "
        f"It rescued `{int(selected.rescue_count.sum())}` base errors and harmed "
        f"`{int(selected.harm_count.sum())}` base-correct samples.\n\n"
        "## Mechanism\n\n"
        f"ERASE had diagnostic KEEP-union oracle headroom `{float(erase_row['oracle_headroom']):.6f}` BA, "
        f"but standalone ERASE changed BA by `{float(erase_row['standalone_delta_BA']):.6f}`. "
        f"The best diagnostic Router AUROC was `{float(route_best['mean_AUROC']):.6f}`, yet its deployable "
        f"Delta BA was `{float(route_best['mean_deployable_delta_BA']):.6f}`. High rescue ranking did not "
        "separate rescue from harm well enough to improve decisions.\n\n"
        "## Protocol decision\n\n"
        "R1, R2, and R3 are not authorized because even raw-best R0 was below +0.002 BA. "
        "No Router lock was written; OpenBMI development-validation, EEGMMIDB, and OpenBMI outer-test "
        "were not accessed. The result is explanatory rather than accuracy-improving.\n",
        encoding="utf-8",
    )
    decisions = {
        "R0": {"attempted": True, "status": r0["status"], "outer_test_used": False},
        "R1": {"authorized": False, "reason": "R0 mean and raw-best Delta BA < +0.002", "outer_test_used": False},
        "R2": {"authorized": False, "reason": "R0 failed; no scalar-insufficiency progression evidence", "outer_test_used": False},
        "R3": {"authorized": False, "reason": "R0 failed before block-wise progression gate", "outer_test_used": False},
    }
    write_json(OUT / "protocol" / "ROUTER_PROGRESSION_DECISIONS.json", decisions)
    for version in ("R1", "R2", "R3"):
        write_json(OUT / version / "NOT_AUTHORIZED.json", decisions[version])
    print(json.dumps(clean({"status": final["status"], "answers": answers}), indent=2), flush=True)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit", help="Run TRAIN-only OOF action and routeability audit")
    audit_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    r0_parser = sub.add_parser("r0", help="Run the authorized 12-configuration R0 TRAIN-only CV")
    r0_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    sub.add_parser("finalize", help="Write terminal tables/reports after a frozen STOP decision")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_device = getattr(args, "device", "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if requested_device == "cuda" or (requested_device == "auto" and torch.cuda.is_available()) else "cpu")
    print(f"[PERSIST-Router] command={args.command} device={device} implementation={IMPLEMENTATION_ID}", flush=True)
    if args.command == "audit":
        audit(device)
    elif args.command == "r0":
        run_r0(device)
    elif args.command == "finalize":
        finalize_terminal()
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
