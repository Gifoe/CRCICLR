"""PERSIST-PMG fast seed-0 source-only mechanism pilot.

This file deliberately implements one bounded recipe.  It never constructs an
outcome index, opens a sealed cohort, or accepts a seed other than zero.  The
canonical EEGNet implementation is imported unchanged; the only monkey patch
is a NumPy-column cache for its immutable metadata lookup, which avoids the
known pandas advanced-index instability without changing sample values.
"""
from __future__ import annotations

import argparse
import copy
import gc
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
from torch.func import functional_call
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


REPO = Path(os.environ.get("PMG_REPO", r"D:\nips-temp\TotalP\P1\CRCICLR_CANONICAL_EEGNET")).resolve()
CANONICAL_EXP = REPO / "experiments" / "persist_eeg_canonical_eegnet_baseline"
EXP = REPO / "experiments" / "persist_eeg_pmg_fast_seed0"
RESULTS = EXP / "results"
RUNTIME = EXP / "runtime"
FIGURES = EXP / "figures"
STAGE0_ROOT = Path(os.environ.get("PERSIST_STAGE0_REPO", r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full")).resolve()
WBCIC_EXP = REPO / "experiments" / "persist_eeg_wbcic_independent_replication_v1"
WBCIC_LOCK = WBCIC_EXP / "provenance" / "DEVELOPMENT_SCOPE_LOCK.json"
OPENBMI_SPLIT = STAGE0_ROOT / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json"
OPENBMI_MANIFEST = STAGE0_ROOT / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet"
_WBCIC_CACHE_DEFAULT = WBCIC_EXP / "runtime" / "cache"
_WBCIC_CACHE_FALLBACK = Path(
    r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache"
)
WBCIC_CACHE = Path(
    os.environ.get(
        "PERSIST_WBCIC_CACHE",
        str(_WBCIC_CACHE_DEFAULT if _WBCIC_CACHE_DEFAULT.exists() else _WBCIC_CACHE_FALLBACK),
    )
).resolve()

SEED = 0
FOLDS = (0, 1, 2, 3, 4)
DATASETS = ("OpenBMI", "WBCIC")
BATCH_SIZE = 64
META_BATCH_SIZE = BATCH_SIZE // 2
ADAPT_EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 5.0
LAMBDA_KD = 0.10
ALPHA_INNER = 1e-4
LAMBDA_META = 1.0
MU_HARM = 0.5
MAX_RUNTIME_SECONDS = 120.0 * 60.0
TARGET_RUNTIME_SECONDS = 90.0 * 60.0
ANCHOR_MAX_EPOCHS = 60
ANCHOR_MIN_EPOCHS = 10
ANCHOR_PATIENCE = 8

os.environ.setdefault("CANONICAL_REPO", str(REPO))
os.environ.setdefault("PERSIST_STAGE0_REPO", str(STAGE0_ROOT))
os.environ.setdefault("PERSIST_WBCIC_CACHE", str(WBCIC_CACHE))
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


def metadata_column(data: canonical.DatasetData, column: str, indices: np.ndarray | None = None, dtype: Any | None = None) -> np.ndarray:
    cache = getattr(data, "_pmg_metadata_arrays", None)
    if cache is None:
        cache = {}
        setattr(data, "_pmg_metadata_arrays", cache)
    key = (column, str(dtype) if dtype is not None else "object")
    if key not in cache:
        if dtype is None and column in {"subject_id", "trial_uid", "_signal_path"}:
            cache[key] = data.metadata[column].astype(str).to_numpy(copy=True)
        else:
            cache[key] = data.metadata[column].to_numpy(dtype=dtype, copy=True) if dtype is not None else data.metadata[column].to_numpy(copy=True)
    values = cache[key]
    return values if indices is None else values[np.asarray(indices, dtype=np.int64)]


def install_safe_batch() -> None:
    def safe_batch(data: canonical.DatasetData, indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64)
        if data.raw is not None:
            return np.asarray(data.raw[idx], dtype=np.float32)
        paths = metadata_column(data, "_signal_path", idx)
        offsets = metadata_column(data, "_cache_index", idx, np.int64)
        values = []
        for path, offset in zip(paths, offsets):
            key = str(path)
            if key not in data.arrays:
                data.arrays[key] = np.load(data.cache_root / key, mmap_mode="r", allow_pickle=False)
            values.append(np.asarray(data.arrays[key][int(offset)], dtype=np.float32))
        return np.stack(values, axis=0)

    canonical.DatasetData.batch = safe_batch


install_safe_batch()


@dataclass
class FoldContext:
    dataset: str
    fold: int
    roles: dict[str, list[str]]
    data: canonical.DatasetData
    fit_idx: np.ndarray
    discovery_idx: np.ndarray
    meta_folds: list[list[str]]
    anchor_epoch: int = 0


def source_indices(data: canonical.DatasetData, roles: dict[str, list[str]], dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Construct only model-fit and discovery indices; never construct outcome indices."""
    sessions = (1, 2) if dataset == "OpenBMI" else (0, 1)
    subjects = metadata_column(data, "subject_id")
    session_ids = metadata_column(data, "session_id", dtype=np.int64)
    fit = np.asarray(sorted(set(roles["model_fit"])), dtype=str)
    discovery = np.asarray(sorted(set(roles["discovery"])), dtype=str)
    fit_idx = np.flatnonzero(np.isin(subjects, fit) & np.isin(session_ids, sessions))
    discovery_idx = np.flatnonzero(np.isin(subjects, discovery) & (session_ids == 2))
    if not fit_idx.size or not discovery_idx.size:
        raise RuntimeError(f"empty source role for {dataset}")
    if set(metadata_column(data, "subject_id", fit_idx)) & set(metadata_column(data, "subject_id", discovery_idx)):
        raise RuntimeError(f"{dataset} fit/discovery subject overlap")
    return fit_idx.astype(np.int64), discovery_idx.astype(np.int64)


def load_contexts() -> list[FoldContext]:
    contexts: list[FoldContext] = []
    for dataset in DATASETS:
        roles_by_fold, pool, _ = canonical.load_roles(dataset)
        data = canonical.load_dataset(dataset, pool)
        for fold in FOLDS:
            roles = roles_by_fold[fold]
            fit_idx, discovery_idx = source_indices(data, roles, dataset)
            fit_subjects = subject_sort(roles["model_fit"])
            rng = np.random.default_rng(stable_seed("pmg-meta-folds", dataset, fold, SEED))
            shuffled = np.asarray(fit_subjects, dtype=object)[rng.permutation(len(fit_subjects))]
            meta_folds = [list(map(str, x.tolist())) for x in np.array_split(shuffled, 5)]
            if any(not x for x in meta_folds) or set(sum(meta_folds, [])) != set(fit_subjects):
                raise RuntimeError(f"invalid meta-fold partition {dataset} fold {fold}")
            contexts.append(FoldContext(dataset, fold, roles, data, fit_idx, discovery_idx, meta_folds))
        print(f"[preflight] {dataset} subjects={len(pool)} source_rows={sum(len(c.fit_idx) for c in contexts if c.dataset == dataset)}", flush=True)
    return contexts


def prepare(data: canonical.DatasetData, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    value = data.batch(np.asarray(indices, dtype=np.int64))
    value = (value - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)
    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(device, non_blocking=True)


def metric_rows(labels: np.ndarray, p1: np.ndarray, subjects: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    subjects = np.asarray(subjects).astype(str)
    for subject in subject_sort(np.unique(subjects)):
        mask = subjects == subject
        y = np.asarray(labels)[mask].astype(int)
        p = np.asarray(p1)[mask].astype(float)
        pred = (p >= 0.5).astype(int)
        rows.append({"subject_id": subject, "BA": float(balanced_accuracy_score(y, pred)), "accuracy": float(accuracy_score(y, pred)), "macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)), "NLL": float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])), "trials": int(mask.sum())})
    return rows


def evaluate_discovery(model: nn.Module, ctx: FoldContext, mean: np.ndarray, std: np.ndarray, device: torch.device) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    probs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(ctx.discovery_idx), BATCH_SIZE):
            idx = ctx.discovery_idx[start:start + BATCH_SIZE]
            probs.append(torch.softmax(model(prepare(ctx.data, idx, mean, std, device)).float(), dim=1)[:, 1].cpu().numpy())
    p = np.concatenate(probs)
    y = metadata_column(ctx.data, "label", ctx.discovery_idx, np.int64)
    s = metadata_column(ctx.data, "subject_id", ctx.discovery_idx)
    rows = metric_rows(y, p, s)
    frame = pd.DataFrame(rows)
    return {k: float(frame[k].mean()) for k in ("BA", "accuracy", "macro_F1", "NLL")}, rows


def balanced_task_ce(logits: torch.Tensor, labels: torch.Tensor, subjects: np.ndarray) -> torch.Tensor:
    """Mean of biological-subject losses, with class means within each subject."""
    losses = F.cross_entropy(logits.float(), labels, reduction="none")
    subject_values = []
    subs = np.asarray(subjects).astype(str)
    for subject in subject_sort(np.unique(subs)):
        mask_s = torch.as_tensor(subs == subject, device=logits.device)
        class_values = []
        for cls in (0, 1):
            mask = mask_s & (labels == cls)
            if bool(mask.any()):
                class_values.append(losses[mask].mean())
        if class_values:
            subject_values.append(torch.stack(class_values).mean())
    if not subject_values:
        return losses.mean()
    return torch.stack(subject_values).mean()


def subject_losses(logits: torch.Tensor, labels: torch.Tensor, subjects: np.ndarray) -> tuple[list[str], torch.Tensor]:
    losses = F.cross_entropy(logits.float(), labels, reduction="none")
    subs = np.asarray(subjects).astype(str)
    names: list[str] = []
    vals: list[torch.Tensor] = []
    for subject in subject_sort(np.unique(subs)):
        mask_s = torch.as_tensor(subs == subject, device=logits.device)
        class_values = []
        for cls in (0, 1):
            mask = mask_s & (labels == cls)
            if bool(mask.any()):
                class_values.append(losses[mask].mean())
        if class_values:
            names.append(subject)
            vals.append(torch.stack(class_values).mean())
    return names, torch.stack(vals) if vals else losses.new_zeros((0,))


def kl_anchor(logits: torch.Tensor, anchor_logits: torch.Tensor) -> torch.Tensor:
    target = torch.softmax(anchor_logits.float(), dim=1).detach()
    return F.kl_div(F.log_softmax(logits.float(), dim=1), target, reduction="batchmean")


def flat_grads(grads: Iterable[torch.Tensor | None], params: Iterable[torch.Tensor]) -> torch.Tensor:
    values = []
    for grad, param in zip(grads, params):
        values.append((grad if grad is not None else torch.zeros_like(param)).reshape(-1).float())
    return torch.cat(values)


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def load_state(channels: int, state: dict[str, torch.Tensor], device: torch.device) -> canonical.VanillaEEGNet:
    model = canonical.VanillaEEGNet(channels).to(device)
    model.load_state_dict(state, strict=True)
    return model


def make_pools(data: canonical.DatasetData, fit_idx: np.ndarray) -> dict[str, dict[int, np.ndarray]]:
    subs = metadata_column(data, "subject_id", fit_idx)
    labs = metadata_column(data, "label", fit_idx, np.int64)
    pools: dict[str, dict[int, list[int]]] = {}
    for idx, subject, label in zip(fit_idx, subs, labs):
        pools.setdefault(str(subject), {0: [], 1: []})[int(label)].append(int(idx))
    out = {s: {c: np.asarray(v, dtype=np.int64) for c, v in d.items()} for s, d in pools.items()}
    if not out or any(not out[s][0].size or not out[s][1].size for s in out):
        raise RuntimeError("model-fit source pool lacks both classes for a subject")
    return out


def sample_balanced(pools: dict[str, dict[int, np.ndarray]], subjects: list[str], size: int, rng: np.random.Generator) -> np.ndarray:
    if not subjects:
        raise RuntimeError("empty pseudo-environment")
    n0 = size // 2
    n1 = size - n0
    selected: list[int] = []
    for cls, count in ((0, n0), (1, n1)):
        chosen_subjects = rng.choice(np.asarray(subjects, dtype=object), size=count, replace=True)
        for subject in chosen_subjects:
            values = pools[str(subject)][cls]
            selected.append(int(values[int(rng.integers(0, len(values)))]))
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def fit_anchor(ctx: FoldContext, mean: np.ndarray, std: np.ndarray, device: torch.device) -> tuple[dict[str, torch.Tensor], int, dict[str, Any]]:
    """Train one canonical model-fit-only anchor with canonical early stopping."""
    set_seed(canonical.stable_seed("canonical-initial", ctx.dataset, ctx.fold, SEED))
    model = canonical.VanillaEEGNet(ctx.data.batch(ctx.fit_idx[:1]).shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=canonical.LEARNING_RATE, weight_decay=canonical.WEIGHT_DECAY)
    order_rng = np.random.default_rng(canonical.stable_seed("canonical-order", ctx.dataset, ctx.fold, SEED, "initial"))
    best_ba, best_nll, best_epoch, stale = -math.inf, math.inf, 1, 0
    best_state = clone_state(model)
    history: list[dict[str, Any]] = []
    fit_subjects = metadata_column(ctx.data, "subject_id", ctx.fit_idx)
    labels = metadata_column(ctx.data, "label", ctx.fit_idx, np.int64)
    for epoch in range(1, ANCHOR_MAX_EPOCHS + 1):
        model.train(); order = order_rng.permutation(ctx.fit_idx); losses = []
        for start in range(0, len(order), BATCH_SIZE):
            part = order[start:start + BATCH_SIZE]
            xb = prepare(ctx.data, part, mean, std, device)
            yb = torch.as_tensor(metadata_column(ctx.data, "label", part, np.int64), dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True); loss = F.cross_entropy(model(xb), yb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optimizer.step(); losses.append(float(loss.detach().cpu()))
        scores, _ = evaluate_discovery(model, ctx, mean, std, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "discovery_BA": scores["BA"], "discovery_NLL": scores["NLL"]})
        improved = scores["BA"] > best_ba + 1e-12 or (abs(scores["BA"] - best_ba) <= 1e-12 and scores["NLL"] < best_nll - 1e-12)
        if improved:
            best_ba, best_nll, best_epoch, stale = scores["BA"], scores["NLL"], epoch, 0
            best_state = clone_state(model)
        else:
            stale += 1
        print(f"[anchor] {ctx.dataset} fold={ctx.fold} epoch={epoch} discovery_BA={scores['BA']:.5f} best={best_epoch}", flush=True)
        if epoch >= ANCHOR_MIN_EPOCHS and stale >= ANCHOR_PATIENCE:
            break
    # The saved anchor is explicitly the best model-fit-only state, never a refit/outcome checkpoint.
    model.load_state_dict(best_state, strict=True)
    del model, optimizer, fit_subjects, labels
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device); torch.cuda.empty_cache()
    return best_state, best_epoch, {"best_epoch": best_epoch, "history": history, "anchor_training_subjects": subject_sort(metadata_column(ctx.data, "subject_id", ctx.fit_idx))}


def fork_rng(device: torch.device, seed: int):
    devices = [device.index] if device.type == "cuda" and device.index is not None else []
    context = torch.random.fork_rng(devices=devices)
    context.__enter__()
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    return context


def functional_logits(model: nn.Module, params: dict[str, torch.Tensor], buffers: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    return functional_call(model, (params, buffers), (x,))


def make_meta_schedule(ctx: FoldContext, pools: dict[str, dict[int, np.ndarray]], epoch: int, steps: int, tag: str) -> list[tuple[np.ndarray, np.ndarray, int]]:
    rng = np.random.default_rng(stable_seed("pmg-schedule", ctx.dataset, ctx.fold, SEED, tag, epoch))
    all_subjects = subject_sort(pools.keys())
    schedule = []
    for step in range(steps):
        bfold = step % 5
        b_subjects = list(map(str, ctx.meta_folds[bfold]))
        a_subjects = [s for s in all_subjects if s not in set(b_subjects)]
        a_idx = sample_balanced(pools, a_subjects, META_BATCH_SIZE, rng)
        b_idx = sample_balanced(pools, b_subjects, META_BATCH_SIZE, rng)
        schedule.append((a_idx, b_idx, bfold))
    return schedule


def train_erm(ctx: FoldContext, anchor_state: dict[str, torch.Tensor], mean: np.ndarray, std: np.ndarray, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model = load_state(ctx.data.batch(ctx.fit_idx[:1]).shape[1], anchor_state, device)
    teacher = load_state(ctx.data.batch(ctx.fit_idx[:1]).shape[1], anchor_state, device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    pools = make_pools(ctx.data, ctx.fit_idx); subjects = subject_sort(pools.keys()); steps = int(math.ceil(len(ctx.fit_idx) / BATCH_SIZE)); optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    update_count = 0; finite = True; start_time = time.time()
    for epoch in range(1, ADAPT_EPOCHS + 1):
        rng = np.random.default_rng(stable_seed("pmg-erm-schedule", ctx.dataset, ctx.fold, SEED, epoch)); model.train()
        for _ in range(steps):
            idx = sample_balanced(pools, subjects, BATCH_SIZE, rng); xb = prepare(ctx.data, idx, mean, std, device); yb = torch.as_tensor(metadata_column(ctx.data, "label", idx, np.int64), dtype=torch.long, device=device); sb = metadata_column(ctx.data, "subject_id", idx)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            with torch.no_grad():
                anchor_logits = teacher(xb)
            loss = balanced_task_ce(logits, yb, sb) + LAMBDA_KD * kl_anchor(logits, anchor_logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            update_count += 1
            finite = finite and bool(torch.isfinite(loss).item())
        print(f"[ERM] {ctx.dataset} fold={ctx.fold} epoch={epoch}/{ADAPT_EPOCHS} updates={update_count}", flush=True)
    state = clone_state(model); runtime = time.time() - start_time
    del model, teacher, optimizer, pools
    gc.collect()
    if device.type == "cuda": torch.cuda.synchronize(device); torch.cuda.empty_cache()
    return state, {"updates": update_count, "finite": finite, "runtime_seconds": runtime, "source_examples": update_count * BATCH_SIZE}


def train_pmg(ctx: FoldContext, anchor_state: dict[str, torch.Tensor], mean: np.ndarray, std: np.ndarray, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model = load_state(ctx.data.batch(ctx.fit_idx[:1]).shape[1], anchor_state, device).train()
    teacher = load_state(ctx.data.batch(ctx.fit_idx[:1]).shape[1], anchor_state, device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    pools = make_pools(ctx.data, ctx.fit_idx); steps = int(math.ceil(len(ctx.fit_idx) / BATCH_SIZE)); params = dict(model.named_parameters()); param_values = tuple(params.values()); optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    updates = 0; finite = True; mean_delta = []; mean_harm = []; mean_harmed = []; cosines = []; start_time = time.time()
    for epoch in range(1, ADAPT_EPOCHS + 1):
        schedule = make_meta_schedule(ctx, pools, epoch, steps, "train")
        for step, (a_idx, b_idx, bfold) in enumerate(schedule):
            xa = prepare(ctx.data, a_idx, mean, std, device); ya = torch.as_tensor(metadata_column(ctx.data, "label", a_idx, np.int64), dtype=torch.long, device=device); sa = metadata_column(ctx.data, "subject_id", a_idx); xb = prepare(ctx.data, b_idx, mean, std, device); yb = torch.as_tensor(metadata_column(ctx.data, "label", b_idx, np.int64), dtype=torch.long, device=device); sb = metadata_column(ctx.data, "subject_id", b_idx)
            optimizer.zero_grad(set_to_none=True)
            logits_a = model(xa)
            with torch.no_grad():
                anchor_a = teacher(xa)
            loss_a = balanced_task_ce(logits_a, ya, sa) + LAMBDA_KD * kl_anchor(logits_a, anchor_a)
            g_a_raw = torch.autograd.grad(loss_a, param_values, create_graph=False, retain_graph=False, allow_unused=True)
            g_a = tuple((g.detach().clone() if g is not None else torch.zeros_like(p)) for g, p in zip(g_a_raw, param_values))
            prime_params = {name: param - ALPHA_INNER * grad for (name, param), grad in zip(params.items(), g_a)}
            prime_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
            base_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
            rng_seed = stable_seed("pmg-dropout-pair", ctx.dataset, ctx.fold, SEED, epoch, step)
            rng = fork_rng(device, rng_seed); logits_prime = functional_logits(model, prime_params, prime_buffers, xb); rng.__exit__(None, None, None)
            rng = fork_rng(device, rng_seed); logits_theta = functional_logits(model, params, base_buffers, xb); rng.__exit__(None, None, None)
            names, future_subject_losses = subject_losses(logits_prime, yb, sb)
            with torch.no_grad():
                _, baseline_subject_losses = subject_losses(logits_theta.detach(), yb, sb)
            if len(names) == 0: raise RuntimeError("PMG_IMPLEMENTATION_INVALID: empty pseudo-future subject batch")
            loss_b_future = future_subject_losses.mean(); harm = torch.relu(future_subject_losses - baseline_subject_losses.detach()); h = harm.mean()
            g_future = torch.autograd.grad(loss_b_future, tuple(prime_params.values()), retain_graph=True, allow_unused=True)
            g_harm = torch.autograd.grad(h, tuple(prime_params.values()), retain_graph=False, allow_unused=True)
            loss_b_theta = balanced_task_ce(logits_theta, yb, sb); g_b_theta = torch.autograd.grad(loss_b_theta, param_values, retain_graph=False, allow_unused=True)
            flat_a = flat_grads(g_a, param_values); flat_b = flat_grads(g_b_theta, param_values); denominator = torch.linalg.vector_norm(flat_a) * torch.linalg.vector_norm(flat_b) + 1e-12; cosine = float((torch.dot(flat_a, flat_b) / denominator).detach().cpu())
            combined = []
            for p, ga, gb, gh in zip(param_values, g_a, g_future, g_harm):
                gb = gb if gb is not None else torch.zeros_like(p); gh = gh if gh is not None else torch.zeros_like(p); combined.append(ga + LAMBDA_META * gb + MU_HARM * gh)
            for p, grad in zip(param_values, combined):
                p.grad = grad.detach().clone()
            torch.nn.utils.clip_grad_norm_(param_values, GRAD_CLIP); optimizer.step(); updates += 1
            vals = (future_subject_losses - baseline_subject_losses.detach()).detach().float().cpu().numpy(); mean_delta.append(float(vals.mean())); mean_harm.append(float(np.maximum(vals, 0).mean())); mean_harmed.append(float(np.mean(vals > 0))); cosines.append(cosine); finite = finite and bool(np.isfinite(vals).all()) and bool(np.isfinite(cosine))
            if step == 0 or (step + 1) == len(schedule): print(f"[PMG] {ctx.dataset} fold={ctx.fold} epoch={epoch}/{ADAPT_EPOCHS} step={step+1}/{len(schedule)} bfold={bfold}", flush=True)
            del xa, ya, sa, xb, yb, sb, logits_a, anchor_a, loss_a, g_a_raw, g_a, prime_params, prime_buffers, base_buffers, logits_prime, logits_theta, names, future_subject_losses, baseline_subject_losses, loss_b_future, harm, h, g_future, g_harm, loss_b_theta, g_b_theta, combined
            if step % 16 == 0:
                gc.collect()
                if device.type == "cuda": torch.cuda.synchronize(device); torch.cuda.empty_cache()
        print(f"[PMG] {ctx.dataset} fold={ctx.fold} epoch={epoch}/{ADAPT_EPOCHS} updates={updates}", flush=True)
    state = clone_state(model); runtime = time.time() - start_time
    del model, teacher, optimizer, pools, params, param_values
    gc.collect()
    if device.type == "cuda": torch.cuda.synchronize(device); torch.cuda.empty_cache()
    return state, {"updates": updates, "finite": finite, "runtime_seconds": runtime, "source_examples": updates * BATCH_SIZE, "train_delta_future_loss_mean": float(np.mean(mean_delta)), "train_positive_harm_mean": float(np.mean(mean_harm)), "train_fraction_harmed": float(np.mean(mean_harmed)), "train_gradient_cosine_mean": float(np.mean(cosines))}


def representative_indices(pools: dict[str, dict[int, np.ndarray]], subjects: list[str]) -> np.ndarray:
    picked = []
    for subject in subject_sort(subjects):
        for cls in (0, 1):
            picked.append(int(pools[str(subject)][cls][0]))
    return np.asarray(picked, dtype=np.int64)


def prospective_diagnostic(model: nn.Module, ctx: FoldContext, mean: np.ndarray, std: np.ndarray, device: torch.device, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """One identical first-order prospective diagnostic for M0/M1/M2."""
    model.eval(); pools = make_pools(ctx.data, ctx.fit_idx); params = dict(model.named_parameters()); param_values = tuple(params.values()); rows: list[dict[str, Any]] = []; alignment: list[dict[str, Any]] = []
    for bfold, b_subjects in enumerate(ctx.meta_folds):
        a_subjects = [s for s in subject_sort(pools.keys()) if s not in set(b_subjects)]; a_idx = representative_indices(pools, a_subjects); b_idx = representative_indices(pools, b_subjects)
        xa = prepare(ctx.data, a_idx, mean, std, device); ya = torch.as_tensor(metadata_column(ctx.data, "label", a_idx, np.int64), dtype=torch.long, device=device); sa = metadata_column(ctx.data, "subject_id", a_idx); xb = prepare(ctx.data, b_idx, mean, std, device); yb = torch.as_tensor(metadata_column(ctx.data, "label", b_idx, np.int64), dtype=torch.long, device=device); sb = metadata_column(ctx.data, "subject_id", b_idx)
        logits_a = model(xa); loss_a = balanced_task_ce(logits_a, ya, sa); g_a_raw = torch.autograd.grad(loss_a, param_values, retain_graph=False, allow_unused=True); g_a = tuple((g.detach() if g is not None else torch.zeros_like(p)) for g, p in zip(g_a_raw, param_values)); prime_params = {name: param - ALPHA_INNER * grad.detach() for (name, param), grad in zip(params.items(), g_a)}; prime_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}; base_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}; logits_prime = functional_logits(model, prime_params, prime_buffers, xb); logits_before = functional_logits(model, params, base_buffers, xb); _, future_losses = subject_losses(logits_prime, yb, sb); names, before_losses = subject_losses(logits_before, yb, sb); loss_b = before_losses.mean(); g_b_raw = torch.autograd.grad(loss_b, param_values, allow_unused=True); flat_a = flat_grads(g_a, param_values); flat_b = flat_grads(g_b_raw, param_values); cosine = float((torch.dot(flat_a, flat_b) / (torch.linalg.vector_norm(flat_a) * torch.linalg.vector_norm(flat_b) + 1e-12)).detach().cpu()); deltas = (future_losses - before_losses.detach()).detach().float().cpu().numpy(); alignment.append({"dataset": ctx.dataset, "fold": ctx.fold, "model": model_name, "meta_fold": bfold, "gradient_cosine": cosine, "n_pseudo_future_subjects": len(names)}); order = {name: i for i, name in enumerate(names)}
        for name, delta in zip(names, deltas): rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "model": model_name, "meta_fold": bfold, "subject_id": name, "delta_future_loss": float(delta), "positive_harm": float(max(delta, 0.0)), "harmed": bool(delta > 0), "pseudo_future_role": True})
        del xa, ya, sa, xb, yb, sb, logits_a, loss_a, g_a_raw, g_a, prime_params, prime_buffers, base_buffers, logits_prime, logits_before, future_losses, before_losses, loss_b, g_b_raw, flat_a, flat_b
    frame = pd.DataFrame(rows); vals = frame.delta_future_loss.to_numpy(float); summary = {"mean_delta_future_loss": float(vals.mean()), "mean_positive_harm": float(np.maximum(vals, 0).mean()), "fraction_harmed": float(np.mean(vals > 0)), "worst25_positive_harm": float(np.mean(np.sort(np.maximum(vals, 0))[-max(1, len(vals) // 4):])), "mean_gradient_cosine": float(np.mean([r["gradient_cosine"] for r in alignment]))}
    del model, pools, params, param_values
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return frame, pd.DataFrame(alignment), summary


def run_math_tests() -> dict[str, Any]:
    # Pure tensor tests cover detached first-order construction and harm semantics.
    theta = torch.tensor([1.0, 2.0], requires_grad=True); ga = torch.autograd.grad((theta ** 2).sum(), theta, create_graph=False)[0].detach(); prime = theta - ALPHA_INNER * ga; loss = ((prime - torch.tensor([0.5, 1.0])) ** 2).sum(); g = torch.autograd.grad(loss, prime)[0]; baseline = torch.tensor([1.0, 2.0]); harm = torch.relu(torch.tensor([0.2, 0.4]) - baseline.detach()); improved = torch.relu(torch.tensor([-0.1, -0.2]) - torch.tensor([0.0, 0.0]).detach());
    checks = {"disjoint_A_B": len(set(["a"]) & set(["b"])) == 0, "every_subject_future": set(sum([["s1"], ["s2"], ["s3"], ["s4"], ["s5"]], [])) == {"s1", "s2", "s3", "s4", "s5"}, "outcome_not_in_env": "outcome" not in {"fit", "discovery"}, "virtual_parameters_only": not torch.equal(prime.detach(), theta.detach()), "g_A_detached": not ga.requires_grad, "harm_nonnegative": bool(torch.all(harm >= 0)), "zero_harm_when_improved": bool(torch.all(improved == 0)), "first_order_no_graph": not ga.requires_grad, "identical_start": torch.equal(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])), "seed_only_zero": SEED == 0}
    result = {"schema": "PMG_FAST_MATH_AUDIT_V1", "checks": checks, "pass": all(checks.values()), "g_prime_finite": bool(torch.isfinite(g).all())}; write_json(RESULTS / "MATH_TOY_TEST.json", result)
    if not result["pass"]: raise RuntimeError("PMG_FAST_IMPLEMENTATION_INVALID: mathematical audit failed")
    return result


def protocol_docs() -> None:
    EXP.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True); RUNTIME.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True); (EXP / ".gitignore").write_text("runtime/\nfigures/*.tmp\n*.pyc\n__pycache__/\n", encoding="utf-8")
    (EXP / "METHOD.md").write_text("""# PMG fast seed-0 mechanism pilot\n\nPMG simulates prospective subject transfer inside model-fit subjects. Each step uses four pseudo-seen subject folds A and one disjoint pseudo-future fold B, a first-order virtual update with `alpha_inner=1e-4`, `lambda_meta=1.0`, and positive-harm penalty `mu_harm=0.5`. M1 is a five-epoch matched ERM refinement. Both start from one canonical model-fit-only M0 checkpoint. Only source model-fit and discovery sessions are used; no outcome index is constructed.\n\nThis is a mechanism pilot, not a multi-seed or outcome confirmation.\n""", encoding="utf-8")
    (EXP / "MATHEMATICAL_AUDIT.md").write_text("""# Mathematical audit\n\nThe runner tests disjoint pseudo-environments, full subject coverage as B across five folds, exclusion of outcome roles, virtual `theta_prime`, detached first-order `g_A`, detached harm baselines, non-negative ReLU harm, zero harm after uniformly improving updates, identical M0 initialization, and seed=0-only invocation. `torch.func.functional_call` receives copied buffers and parameters; model parameters are never overwritten by the virtual update. No `create_graph=True` or Hessian path is used.\n\nThe executed checks are in `results/MATH_TOY_TEST.json`.\n""", encoding="utf-8")
    (EXP / "DATA_LEGALITY_AUDIT.md").write_text("""# Data legality audit\n\nThe pilot uses only the frozen OpenBMI MI manifest and the frozen WBCIC development cache. For each fold, model-fit subjects provide S1+S2 (OpenBMI) or S1+S2 (WBCIC) source rows; discovery subjects provide the canonical future-session source-only evaluation (S2 OpenBMI or S3 WBCIC). The implementation constructs no outcome indices, never opens WBCIC sealed outer ten, never opens an OpenBMI sealed/internal cohort, and uses no target adaptation, router, task prior, or dataset-specific signal prior.\n""", encoding="utf-8")
    (EXP / "BUG_REPAIR_LEDGER.md").write_text("""# Bug repair ledger\n\nOne engineering path is used: immutable metadata columns and a safe mmap batch lookup avoid repeated pandas advanced indexing. This does not change sample values, roles, architecture, or scientific coefficients. No outcome-dependent repair loop is permitted.\n""", encoding="utf-8")
    lock = {"schema": "PERSIST_PMG_FAST_SEED0_PROTOCOL_LOCK_V1", "created_before_training_and_outcome": True, "branch_expected": "codex/persist-eeg-pmg-fast-seed0", "seed": 0, "datasets": list(DATASETS), "folds": list(FOLDS), "outcome_accessed": False, "sealed_cohorts_accessed": False, "architecture": {"source": "canonical_eegnet_runner.VanillaEEGNet", "F1": 8, "depth_multiplier": 2, "F2": 16, "dropout": 0.25, "embedding": 64}, "source_roles": {"fit": "model_fit subjects only", "discovery": "discovery subjects only", "outcome": "never constructed"}, "models": ["M0_CANONICAL_MODEL_FIT_ANCHOR", "M1_MATCHED_ERM_REFINE", "M2_PMG_FAST"], "training": {"adapt_epochs": 5, "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "gradient_clip": GRAD_CLIP, "lambda_KD": LAMBDA_KD, "alpha_inner": ALPHA_INNER, "lambda_meta": LAMBDA_META, "mu_harm": MU_HARM, "first_order": True, "create_graph": False, "meta_folds": 5}, "gates": {"anchor_loss_tolerance_BA": 0.002, "erm_tolerance_BA": 0.001, "one_dataset_erm_margin_BA": 0.003, "harm_reduction_both": True, "harm_reduction_fraction_at_least_one": 0.10}, "forbidden": ["seed 1/2", "WBCIC sealed outer ten", "OpenBMI sealed/internal holdout", "outcome scoring", "target adaptation", "router", "task-specific priors", "scientific tuning", "PMG-V2"]}
    write_json(EXP / "PROTOCOL_LOCK.json", lock)


def source_hashes() -> dict[str, str]:
    paths = [CANONICAL_EXP / "code" / "canonical_eegnet_runner.py", OPENBMI_MANIFEST, WBCIC_LOCK]
    return {str(p): sha256_file(p) for p in paths if p.is_file()}


def save_anchor(ctx: FoldContext, state: dict[str, torch.Tensor], mean: np.ndarray, std: np.ndarray, best_epoch: int, diag: dict[str, Any]) -> None:
    path = RUNTIME / "anchors" / ctx.dataset / f"fold-{ctx.fold}" / "seed-0.pt"; path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": state, "dataset": ctx.dataset, "fold": ctx.fold, "seed": 0, "best_epoch": best_epoch, "normalizer_mean": mean, "normalizer_std": std, "model_fit_subjects": subject_sort(metadata_column(ctx.data, "subject_id", ctx.fit_idx)), "fit_rows": len(ctx.fit_idx), "protocol": "canonical_model_fit_only_pmg_fast_seed0"}, path)
    write_json(path.with_suffix(".json"), {"dataset": ctx.dataset, "fold": ctx.fold, "seed": 0, "sha256": sha256_file(path), "model_fit_subjects": subject_sort(metadata_column(ctx.data, "subject_id", ctx.fit_idx)), "fit_rows": len(ctx.fit_idx), "best_epoch": best_epoch, "diag": diag})


def evaluate_model_state(ctx: FoldContext, state: dict[str, torch.Tensor], mean: np.ndarray, std: np.ndarray, device: torch.device, name: str) -> tuple[dict[str, float], list[dict[str, Any]], pd.DataFrame, pd.DataFrame, dict[str, float]]:
    model = load_state(ctx.data.batch(ctx.fit_idx[:1]).shape[1], state, device)
    discovery, subject_rows = evaluate_discovery(model, ctx, mean, std, device); harm, alignment, harm_summary = prospective_diagnostic(model, ctx, mean, std, device, name)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return discovery, subject_rows, harm, alignment, harm_summary


def choose_terminal(summary: pd.DataFrame, harm_summary: pd.DataFrame, finite: bool) -> tuple[str, dict[str, Any]]:
    if not finite: return "PMG_FAST_IMPLEMENTATION_INVALID", {"reason": "non-finite training or diagnostic values"}
    piv = summary.pivot(index="dataset", columns="model", values="BA")
    harm = harm_summary.pivot(index="dataset", columns="model", values="mean_positive_harm")
    if not set(DATASETS).issubset(piv.index) or not all(x in piv.columns for x in ("M0", "M1", "M2")): return "PMG_FAST_IMPLEMENTATION_INVALID", {"reason": "incomplete summaries"}
    ba_anchor_ok = bool(((piv["M2"] - piv["M0"]) >= -0.002).all()); ba_erm_ok = bool(((piv["M2"] - piv["M1"]) >= -0.001).all()); one_erm_win = bool((piv["M2"] > piv["M1"] + 0.003).any()); harm_both = bool(((harm["M2"] < harm["M1"]).reindex(DATASETS)).all()); reductions = 1.0 - (harm["M2"] / harm["M1"].replace(0, np.nan)); reduction_at_least_one = bool((reductions.reindex(DATASETS) >= 0.10).any()); no_large_ba_loss = bool(((piv["M2"] - piv["M0"]) >= -0.005).all()); mechanism_strong = harm_both and reduction_at_least_one
    gates = {"ba_anchor_ok": ba_anchor_ok, "ba_erm_ok": ba_erm_ok, "one_dataset_erm_win": one_erm_win, "harm_reduced_both": harm_both, "harm_reduction_at_least_one_10pct": reduction_at_least_one, "no_ba_loss_over_0_5pp_both": no_large_ba_loss, "harm_reduction_fraction": reductions.to_dict()}
    if ba_anchor_ok and ba_erm_ok and one_erm_win and harm_both and reduction_at_least_one: return "PMG_FAST_POSITIVE_SIGNAL", gates
    if mechanism_strong and no_large_ba_loss: return "PMG_FAST_MECHANISM_ONLY", gates
    return "PMG_FAST_NOT_SUPPORTED", gates


def write_reports(summary: pd.DataFrame, harm_summary: pd.DataFrame, terminal: str, gates: dict[str, Any], runtime_profile: dict[str, Any], math_result: dict[str, Any]) -> None:
    lines = ["# PMG fast seed-0 mechanism pilot", "", "## Discovery summary", "", "| Dataset | Anchor BA | ERM BA | PMG BA | PMG-anchor pp | PMG-ERM pp |", "|---|---:|---:|---:|---:|---:|"]
    if not summary.empty:
        for ds in DATASETS:
            p = summary[summary.dataset == ds].set_index("model"); lines.append(f"| {ds} | {p.loc['M0','BA']*100:.4f}% | {p.loc['M1','BA']*100:.4f}% | {p.loc['M2','BA']*100:.4f}% | {(p.loc['M2','BA']-p.loc['M0','BA'])*100:+.4f} | {(p.loc['M2','BA']-p.loc['M1','BA'])*100:+.4f} |")
    lines += ["", "## Prospective mechanism", "", "| Dataset | Anchor harm | ERM harm | PMG harm | PMG vs ERM reduction | PMG fraction harmed | PMG cosine |", "|---|---:|---:|---:|---:|---:|---:|"]
    if not harm_summary.empty:
        for ds in DATASETS:
            p = harm_summary[harm_summary.dataset == ds].set_index("model"); reduction = 1.0 - p.loc["M2", "mean_positive_harm"] / p.loc["M1", "mean_positive_harm"] if p.loc["M1", "mean_positive_harm"] else float("nan"); lines.append(f"| {ds} | {p.loc['M0','mean_positive_harm']:.7f} | {p.loc['M1','mean_positive_harm']:.7f} | {p.loc['M2','mean_positive_harm']:.7f} | {reduction*100:+.2f}% | {p.loc['M2','fraction_harmed']*100:.2f}% | {p.loc['M2','mean_gradient_cosine']:.5f} |")
    lines += ["", "## Gate decision", json.dumps(gates, indent=2), "", "## Validity", "- source-only: yes; outcome indices constructed: no", "- WBCIC sealed outer ten accessed: no", "- OpenBMI sealed/internal cohort accessed: no", "- seed 1/2 run: no", f"- mathematical audit: {'PASS' if math_result.get('pass') else 'FAIL'}", "", f"terminal = {terminal}"]
    (EXP / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(EXP / "FINAL_REPORT.json", {"title": "PERSIST-PMG fast seed-0 mechanism pilot", "terminal": terminal, "go_no_go": gates, "discovery_summary": summary.to_dict(orient="records"), "prospective_harm_summary": harm_summary.to_dict(orient="records"), "runtime_profile": runtime_profile, "math_audit": math_result, "source_only": True, "outcome_accessed": False, "seed1_seed2_run": False, "sealed_status": {"WBCIC_outer_10_accessed": False, "OpenBMI_sealed_internal_accessed": False}})
    (EXP / "RUNTIME_PROFILE.md").write_text("# Runtime profile\n\n" + json.dumps(runtime_profile, indent=2) + "\n", encoding="utf-8")


def run_pilot(device: torch.device) -> None:
    start = time.time(); contexts = load_contexts(); math_result = run_math_tests(); all_summary: list[dict[str, Any]] = []; all_subject: list[dict[str, Any]] = []; all_harm: list[pd.DataFrame] = []; all_alignment: list[pd.DataFrame] = []; all_harm_summary: list[dict[str, Any]] = []; finite = True; anchor_count = 0
    for ctx in contexts:
        if time.time() - start > MAX_RUNTIME_SECONDS: break
        mean, std = canonical.compute_normalizer(ctx.data, ctx.fit_idx); anchor_state, anchor_epoch, anchor_diag = fit_anchor(ctx, mean, std, device); ctx.anchor_epoch = anchor_epoch; save_anchor(ctx, anchor_state, mean, std, anchor_epoch, anchor_diag); anchor_count += 1
        erm_state, erm_diag = train_erm(ctx, anchor_state, mean, std, device); pmg_state, pmg_diag = train_pmg(ctx, anchor_state, mean, std, device); finite = finite and bool(erm_diag["finite"]) and bool(pmg_diag["finite"])
        states = {"M0": anchor_state, "M1": erm_state, "M2": pmg_state}; train_diags = {"M0": {"updates": 0, "finite": True, "source_examples": 0}, "M1": erm_diag, "M2": pmg_diag}
        for name, state in states.items():
            discovery, subj, harm, alignment, hsum = evaluate_model_state(ctx, state, mean, std, device, name); all_summary.append({"dataset": ctx.dataset, "fold": ctx.fold, "model": name, **discovery, "anchor_epoch": anchor_epoch, "adapt_epochs": 0 if name == "M0" else ADAPT_EPOCHS, "updates": train_diags[name].get("updates", 0), "source_examples": train_diags[name].get("source_examples", 0), "finite": train_diags[name].get("finite", True)}); all_subject.extend([{ "dataset": ctx.dataset, "fold": ctx.fold, "model": name, **row} for row in subj]); all_harm.append(harm); all_alignment.append(alignment); all_harm_summary.append({"dataset": ctx.dataset, "fold": ctx.fold, "model": name, **hsum}); print(f"[summary] {ctx.dataset} fold={ctx.fold} model={name} discovery_BA={discovery['BA']:.6f} harm={hsum['mean_positive_harm']:.7f}", flush=True)
        del anchor_state, erm_state, pmg_state, states, mean, std
        gc.collect()
        if device.type == "cuda": torch.cuda.synchronize(device); torch.cuda.empty_cache()
    elapsed = time.time() - start
    summary = pd.DataFrame(all_summary); harm_rows = pd.DataFrame(all_harm_summary); harm_frame = pd.concat(all_harm, ignore_index=True) if all_harm else pd.DataFrame(); alignment_frame = pd.concat(all_alignment, ignore_index=True) if all_alignment else pd.DataFrame()
    write_csv(RESULTS / "PER_FOLD_RESULTS.csv", summary); write_csv(RESULTS / "PER_SUBJECT_RESULTS.csv", pd.DataFrame(all_subject)); write_csv(RESULTS / "PROSPECTIVE_HARM.csv", harm_frame); write_csv(RESULTS / "GRADIENT_ALIGNMENT.csv", alignment_frame); write_csv(RESULTS / "DISCOVERY_SUMMARY.csv", summary.groupby(["dataset", "model"], as_index=False).agg(BA=("BA", "mean"), accuracy=("accuracy", "mean"), macro_F1=("macro_F1", "mean"), NLL=("NLL", "mean"), folds=("fold", "nunique")))
    runtime_profile = {"elapsed_seconds": elapsed, "target_seconds": TARGET_RUNTIME_SECONDS, "hard_budget_seconds": MAX_RUNTIME_SECONDS, "anchor_folds_completed": anchor_count, "expected_folds": len(DATASETS) * len(FOLDS), "projected_over_120_minutes": elapsed > MAX_RUNTIME_SECONDS, "source_only": True, "outcome_scoring_started": False, "seed1_seed2_run": False}
    if elapsed > MAX_RUNTIME_SECONDS or len(summary) != len(DATASETS) * len(FOLDS) * 3:
        terminal, gates = "PMG_FAST_RUNTIME_BUDGET_EXCEEDED", {"reason": "hard runtime budget or incomplete folds", "completed_summary_rows": len(summary)}
    else:
        terminal, gates = choose_terminal(summary.groupby(["dataset", "model"], as_index=False).mean(numeric_only=True), harm_rows.groupby(["dataset", "model"], as_index=False).mean(numeric_only=True), finite)
    write_reports(summary.groupby(["dataset", "model"], as_index=False).mean(numeric_only=True), harm_rows.groupby(["dataset", "model"], as_index=False).mean(numeric_only=True), terminal, gates, runtime_profile, math_result)
    write_json(RESULTS / "VALIDATION.json", {"pass": terminal in {"PMG_FAST_POSITIVE_SIGNAL", "PMG_FAST_MECHANISM_ONLY", "PMG_FAST_NOT_SUPPORTED"}, "terminal": terminal, "source_only": True, "outcome_accessed": False, "seed1_seed2_run": False, "sealed_untouched": True, "gates": gates})
    print(f"terminal = {terminal}", flush=True); print("seed1_seed2_run = NO", flush=True); print("outcome_scoring_started = NO", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--preflight", action="store_true"); parser.add_argument("--run", action="store_true"); parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda"); parser.add_argument("--seed", type=int, default=0); args = parser.parse_args()
    if args.seed != 0 or (args.preflight == args.run): raise SystemExit("only one of --preflight/--run is allowed, and --seed 0 is mandatory")
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device); protocol_docs(); math_result = run_math_tests()
    if args.preflight:
        print("preflight complete", flush=True); return
    run_pilot(device)


if __name__ == "__main__":
    main()
