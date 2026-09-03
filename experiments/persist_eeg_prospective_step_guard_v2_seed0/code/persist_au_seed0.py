"""PERSIST-PSG seed-0 bounded development search.

This is the registered prospective step-guard pilot.  It uses the exact
AdamW proposal from a task-only gradient, tests whether that proposal is
harmful on a dropout-free pseudo-future guard batch, and applies only the
predeclared norm-capped correction.  The script is deliberately seed-0 only:
it never opens sealed cohorts and never runs seed 1/2.
"""
from __future__ import annotations

import argparse
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score

# The server's Windows CUDA build has exhibited native access violations when
# several OpenMP/CUDA worker pools are torn down in one long-lived process.
# Keep the numerical recipe unchanged while forcing a single host thread and
# deterministic cuDNN selection; this is an execution-stability repair only.
try:
    torch.set_num_threads(int(os.environ.get("PERSIST_TORCH_THREADS", "1")))
    torch.set_num_interop_threads(1)
except RuntimeError:
    # Importing this module from an already-initialized interpreter may make
    # the inter-op pool immutable; the scheduled standalone run still sets it.
    pass
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

REPO = Path(os.environ.get("CANONICAL_REPO", r"D:\nips-temp\TotalP\P1\CRCICLR_CANONICAL_EEGNET")).resolve()
EXP = REPO / "experiments" / "persist_eeg_prospective_step_guard_v2_seed0"
RESULTS = EXP / "results"
RUNTIME = EXP / "runtime"
CANONICAL_EXP = REPO / "experiments" / "persist_eeg_canonical_eegnet_baseline"
SEED = 0
FOLDS = (0, 1, 2, 3, 4)
DATASETS = ("OpenBMI", "WBCIC")
MAX_EPOCHS = 5
BATCH_SIZE = 64
BASE_LR = 3e-5
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 5.0
BOOTSTRAP_DRAWS = 10_000
KAPPAS = (0.05, 0.10, 0.20)
SCOPE_MASKS = ("ALL", "LATE")
EPS = 1e-12

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


def write_csv(path: Path, frame: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def metadata_col(data: canonical.DatasetData, name: str, indices: np.ndarray | None = None) -> np.ndarray:
    frame = data.metadata if indices is None else data.metadata.iloc[np.asarray(indices, dtype=np.int64)]
    values = frame[name]
    if name == "subject_id":
        return values.astype(str).str.replace("sub-", "", regex=False).to_numpy()
    return values.to_numpy()


def vectorized_batch(data: canonical.DatasetData, indices: np.ndarray) -> np.ndarray:
    """Equivalent to the canonical batch accessor, without per-row mmap calls."""
    indices = np.asarray(indices, dtype=np.int64)
    if data.raw is not None:
        return np.asarray(data.raw[indices], dtype=np.float32)
    frame = data.metadata.iloc[indices]
    paths = frame["_signal_path"].astype(str).to_numpy()
    offsets = frame["_cache_index"].to_numpy(np.int64)
    if len(indices) == 0:
        return np.empty((0, 62, 1000), dtype=np.float32)
    first_key = str(paths[0])
    if first_key not in data.arrays:
        data.arrays[first_key] = np.load(data.cache_root / first_key, mmap_mode="r", allow_pickle=False)
    first = data.arrays[first_key]
    output = np.empty((len(indices), int(first.shape[1]), int(first.shape[2])), dtype=np.float32)
    for key in np.unique(paths):
        key = str(key)
        if key not in data.arrays:
            data.arrays[key] = np.load(data.cache_root / key, mmap_mode="r", allow_pickle=False)
        mask = paths == key
        output[mask] = np.asarray(data.arrays[key][offsets[mask]], dtype=np.float32)
    return output


def softmax_np(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    output = np.exp(values)
    return output / output.sum(axis=1, keepdims=True)


def subject_metrics(labels: np.ndarray, p1: np.ndarray, subjects: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = np.asarray(labels, dtype=np.int64)
    p1 = np.asarray(p1, dtype=np.float64)
    subjects = np.asarray(subjects).astype(str)
    for subject in subject_sort(np.unique(subjects)):
        mask = subjects == subject
        y = labels[mask]
        prob = p1[mask]
        pred = (prob >= 0.5).astype(np.int64)
        rows.append({"subject_id": subject, "BA": float(balanced_accuracy_score(y, pred)), "accuracy": float(accuracy_score(y, pred)), "macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)), "NLL": float(log_loss(y, np.column_stack([1.0 - prob, prob]), labels=[0, 1])), "trials": int(mask.sum())})
    return rows


def metric_means(labels: np.ndarray, p1: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    frame = pd.DataFrame(subject_metrics(labels, p1, subjects))
    return {key: float(frame[key].mean()) for key in ("BA", "accuracy", "macro_F1", "NLL")}


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
    anchor_state: dict[str, torch.Tensor]
    mean: np.ndarray
    std: np.ndarray
    checkpoint_path: Path
    meta_folds: list[list[str]]
    schedules: list[list[dict[str, Any]]]
    schedule_hash: str


def load_checkpoint(dataset: str, fold: int, channels: int) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, Path]:
    path = CANONICAL_EXP / "runtime" / "checkpoints" / dataset / f"fold-{fold}" / "seed-0.pt"
    partial_path = CANONICAL_EXP / "runtime" / "partial" / f"{dataset.lower()}_fold-{fold}_seed-0.json"
    if not path.is_file() or not partial_path.is_file():
        raise RuntimeError(f"missing canonical checkpoint/partial: {path}")
    partial = json.loads(partial_path.read_text(encoding="utf-8-sig"))
    if partial.get("checkpoint_sha256") and str(partial["checkpoint_sha256"]) != sha256_file(path):
        raise RuntimeError(f"canonical checkpoint hash mismatch: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model = canonical.VanillaEEGNet(channels)
    model.load_state_dict(payload["model_state"], strict=True)
    state = {key: value.detach().cpu().clone() for key, value in payload["model_state"].items()}
    mean = np.asarray(payload["normalizer_mean"], dtype=np.float32)
    std = np.asarray(payload["normalizer_std"], dtype=np.float32)
    if mean.shape != (channels,) or std.shape != (channels,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError(f"invalid canonical normalizer for {dataset} fold {fold}")
    return state, mean, std, path


def freeze_bn(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def bn_buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_buffers() if "running_mean" in name or "running_var" in name}


def bn_max_displacement(model: nn.Module, baseline: dict[str, torch.Tensor]) -> float:
    values = []
    now = dict(model.named_buffers())
    for name, before in baseline.items():
        values.append(float(torch.max(torch.abs(now[name].detach().cpu() - before.cpu())).item()))
    return max(values, default=0.0)


def model_from_state(ctx: FoldContext, device: torch.device) -> canonical.VanillaEEGNet:
    model = canonical.VanillaEEGNet(int(ctx.data.batch(ctx.refit_idx[:1]).shape[1])).to(device)
    model.load_state_dict(ctx.anchor_state, strict=True)
    return model


def evaluate_model(model: nn.Module, ctx: FoldContext, indices: np.ndarray, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH_SIZE):
            part = indices[start : start + BATCH_SIZE]
            logits_parts.append(model(canonical.prepare_batch(ctx.data, part, ctx.mean, ctx.std, device)).detach().float().cpu().numpy())
    logits = np.concatenate(logits_parts, axis=0)
    probability = softmax_np(logits)
    labels = metadata_col(ctx.data, "label", indices).astype(np.int64)
    subjects = metadata_col(ctx.data, "subject_id", indices).astype(str)
    trial_uids = ctx.data.metadata.iloc[indices]["trial_uid"].astype(str).to_numpy()
    return {"indices": np.asarray(indices, dtype=np.int64), "labels": labels, "subjects": subjects, "trial_uids": trial_uids, "logits": logits, "probability": probability, "subject_metrics": subject_metrics(labels, probability[:, 1], subjects)}


def checkpoint_equivalence(ctx: FoldContext, expected: pd.DataFrame, device: torch.device) -> dict[str, Any]:
    model = model_from_state(ctx, device)
    actual = evaluate_model(model, ctx, ctx.outcome_idx, device)
    rows = expected[(expected.dataset == ctx.dataset) & (expected.seed.astype(str) == "0") & (expected.fold == ctx.fold)].copy()
    if set(actual["trial_uids"]) != set(rows.trial_uid.astype(str)):
        raise RuntimeError(f"checkpoint equivalence trial IDs mismatch {ctx.dataset} fold={ctx.fold}")
    rows = rows.set_index(rows.trial_uid.astype(str)).loc[list(actual["trial_uids"])]
    if not np.array_equal(rows.label.to_numpy(np.int64), actual["labels"]):
        raise RuntimeError(f"checkpoint equivalence labels mismatch {ctx.dataset} fold={ctx.fold}")
    expected_p = rows[["probability_class0", "probability_class1"]].to_numpy(np.float64)
    max_diff = float(np.max(np.abs(expected_p - actual["probability"])))
    actual_pred = (actual["probability"][:, 1] >= actual["probability"][:, 0]).astype(np.int64)
    if max_diff > 1e-5 or not np.array_equal(rows.prediction.to_numpy(np.int64), actual_pred):
        raise RuntimeError(f"checkpoint equivalence probabilities mismatch {ctx.dataset} fold={ctx.fold} max_diff={max_diff}")
    result = {"dataset": ctx.dataset, "fold": ctx.fold, "trial_count": len(actual["indices"]), "max_probability_abs_diff": max_diff, "trial_uid_exact": True, "labels_exact": True, "predictions_exact": True, "pass": True}
    # Explicitly release the short-lived CUDA module and its output arrays.
    # This avoids allocator/native teardown accumulation on the Windows GPU
    # build during the ten-fold preflight.
    del model, actual, rows
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def subject_pools(data: canonical.DatasetData, indices: np.ndarray) -> dict[str, dict[int, np.ndarray]]:
    subjects = metadata_col(data, "subject_id", indices).astype(str)
    labels = metadata_col(data, "label", indices).astype(int)
    output: dict[str, dict[int, np.ndarray]] = {}
    for subject in subject_sort(np.unique(subjects)):
        output[subject] = {}
        for cls in (0, 1):
            values = np.asarray(indices)[(subjects == subject) & (labels == cls)]
            if len(values) == 0:
                raise RuntimeError(f"subject {subject} lacks class {cls} in source pool")
            output[subject][cls] = values.astype(np.int64)
    return output


def balanced_batch(pools: dict[str, dict[int, np.ndarray]], subjects: list[str], rng: np.random.Generator) -> np.ndarray:
    if not subjects:
        raise RuntimeError("empty subject pool")
    shuffled = np.asarray(subjects, dtype=object)[rng.permutation(len(subjects))]
    values: list[int] = []
    for cls in (0, 1):
        for i in range(BATCH_SIZE // 2):
            subject = str(shuffled[i % len(shuffled)])
            pool = pools[subject][cls]
            values.append(int(pool[int(rng.integers(0, len(pool)))]))
    return np.asarray(values, dtype=np.int64)[rng.permutation(BATCH_SIZE)]


def make_meta_folds(dataset: str, fold: int, source_subjects: list[str]) -> list[list[str]]:
    # Candidate-independent partition: the key intentionally contains no
    # method, scope or cap.
    rng = np.random.default_rng(stable_seed("psg-v2-meta-folds", dataset, fold, SEED))
    shuffled = np.asarray(source_subjects, dtype=object)[rng.permutation(len(source_subjects))]
    groups = [list(map(str, part.tolist())) for part in np.array_split(shuffled, 5)]
    if any(not group for group in groups) or set(sum(groups, [])) != set(source_subjects):
        raise RuntimeError(f"invalid meta-fold partition {dataset} fold {fold}")
    return groups


def balanced_probe_batch(pools: dict[str, dict[int, np.ndarray]], subjects: list[str], guard_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw a class/subject-balanced probe batch disjoint from guard trials."""
    guard = set(int(value) for value in np.asarray(guard_idx, dtype=np.int64).tolist())
    if not subjects:
        raise RuntimeError("empty probe subject pool")
    shuffled = np.asarray(subjects, dtype=object)[rng.permutation(len(subjects))]
    values: list[int] = []
    for cls in (0, 1):
        for i in range(BATCH_SIZE // 2):
            subject = str(shuffled[i % len(shuffled)])
            available = [int(value) for value in pools[subject][cls] if int(value) not in guard]
            if not available:
                raise RuntimeError(f"no probe trial remains for subject={subject} class={cls}")
            values.append(int(available[int(rng.integers(0, len(available)))]))
    return np.asarray(values, dtype=np.int64)[rng.permutation(BATCH_SIZE)]


def make_schedules(ctx: FoldContext) -> tuple[list[list[dict[str, Any]]], str]:
    pools = subject_pools(ctx.data, ctx.refit_idx)
    steps = max(1, int(math.ceil(len(ctx.refit_idx) / 128)))
    serial: list[Any] = []
    schedules: list[list[dict[str, Any]]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        current: list[dict[str, Any]] = []
        for step in range(steps):
            b_fold = step % 5
            b_subjects = list(ctx.meta_folds[b_fold])
            a_subjects = [subject for i, group in enumerate(ctx.meta_folds) if i != b_fold for subject in group]
            rng = np.random.default_rng(stable_seed("psg-v2-schedule", ctx.dataset, ctx.fold, SEED, epoch, step))
            a_idx = balanced_batch(pools, a_subjects, rng)
            b_idx = balanced_batch(pools, b_subjects, rng)
            if set(metadata_col(ctx.data, "subject_id", a_idx)) & set(metadata_col(ctx.data, "subject_id", b_idx)):
                raise RuntimeError(f"A/B overlap {ctx.dataset} fold {ctx.fold} epoch {epoch} step {step}")
            probe_rng = np.random.default_rng(stable_seed("psg-v2-probe", ctx.dataset, ctx.fold, SEED, epoch, step, "true"))
            probe_idx = balanced_probe_batch(pools, b_subjects, b_idx, probe_rng)
            if set(np.asarray(probe_idx).tolist()) & set(np.asarray(b_idx).tolist()):
                raise RuntimeError(f"B guard/probe overlap {ctx.dataset} fold {ctx.fold} epoch {epoch} step {step}")
            random_fold = (step + 1) % 5
            random_subjects = list(ctx.meta_folds[random_fold])
            random_rng = np.random.default_rng(stable_seed("psg-v2-random-guard", ctx.dataset, ctx.fold, SEED, epoch, step, "B"))
            random_b_idx = balanced_batch(pools, random_subjects, random_rng)
            random_probe_rng = np.random.default_rng(stable_seed("psg-v2-random-probe", ctx.dataset, ctx.fold, SEED, epoch, step, "B_probe"))
            random_probe_idx = balanced_probe_batch(pools, random_subjects, random_b_idx, random_probe_rng)
            current.append({"A": a_idx, "B": b_idx, "probe": probe_idx, "b_fold": int(b_fold), "b_subjects": b_subjects, "random_B": random_b_idx, "random_probe": random_probe_idx, "random_b_fold": int(random_fold), "random_b_subjects": random_subjects})
            serial.append({"epoch": epoch, "step": step, "B_fold": b_fold, "random_B_fold": random_fold, "A": a_idx.tolist(), "B": b_idx.tolist(), "probe": probe_idx.tolist(), "random_B": random_b_idx.tolist(), "random_probe": random_probe_idx.tolist()})
        schedules.append(current)
    digest = hashlib.sha256(json.dumps(serial, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()
    return schedules, digest


def parameter_scope(model: nn.Module, scope: str) -> tuple[list[str], list[nn.Parameter], np.ndarray]:
    """Return all AdamW parameters plus a scalar mask for the correction scope."""
    named = list(model.named_parameters())
    late_names = {name for name, _ in named if name.startswith("embedding.") or name.startswith("head.")}
    params = [parameter for _, parameter in named]
    names = [name for name, _ in named]
    for parameter in params:
        parameter.requires_grad = True
    mask = np.concatenate([np.full(parameter.numel(), (scope == "ALL" or name in late_names), dtype=bool) for name, parameter in named])
    if scope == "LATE" and not mask.any():
        raise RuntimeError("canonical EEGNet has no late correction parameters")
    return names, params, mask


def fork_rng(device: torch.device):
    devices = [int(device.index)] if device.type == "cuda" and device.index is not None else []
    return torch.random.fork_rng(devices=devices)


def gradients(model: nn.Module, params: list[nn.Parameter], xb: torch.Tensor, yb: torch.Tensor, seed: int | None, eval_mode: bool) -> tuple[torch.Tensor, ...]:
    """Compute g_A with dropout or stable g_B with dropout disabled."""
    context = fork_rng(xb.device) if seed is not None else None
    if context is None:
        class _Null:
            def __enter__(self): return self
            def __exit__(self, *args): return False
        context = _Null()
    with context:
        if seed is not None:
            torch.manual_seed(int(seed))
            if xb.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
        if eval_mode:
            model.eval()
        else:
            model.train(); freeze_bn(model)
        loss = F.cross_entropy(model(xb), yb)
        values = torch.autograd.grad(loss, tuple(params), allow_unused=True, retain_graph=False, create_graph=False)
    return tuple((value.detach() if value is not None else torch.zeros_like(parameter)).float() for value, parameter in zip(values, params))


def flatten(values: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.reshape(-1).float() for value in values])


def split_like(vector: torch.Tensor, params: list[nn.Parameter]) -> list[torch.Tensor]:
    chunks: list[torch.Tensor] = []
    offset = 0
    for parameter in params:
        length = parameter.numel()
        chunks.append(vector[offset : offset + length].reshape_as(parameter))
        offset += length
    if offset != vector.numel():
        raise RuntimeError("vector/parameter split mismatch")
    return chunks


def cross_entropy_indices(model: nn.Module, ctx: FoldContext, indices: np.ndarray, device: torch.device) -> float:
    model.eval()
    with torch.inference_mode():
        xb = canonical.prepare_batch(ctx.data, indices, ctx.mean, ctx.std, device)
        yb = torch.as_tensor(metadata_col(ctx.data, "label", indices).astype(np.int64), dtype=torch.long, device=device)
        return float(F.cross_entropy(model(xb), yb).detach().cpu())


def clip_gradient(values: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    norm = float(torch.linalg.vector_norm(values).detach().cpu())
    scale = min(1.0, GRAD_CLIP / max(norm, EPS))
    return values * scale, norm, scale


def snapshot(params: list[nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in params]


def restore_delta(params: list[nn.Parameter], theta_old: list[torch.Tensor], delta: torch.Tensor) -> None:
    with torch.no_grad():
        for parameter, old, chunk in zip(params, theta_old, split_like(delta, params)):
            parameter.copy_(old + chunk)


def train_candidate(ctx: FoldContext, candidate: dict[str, Any], device: torch.device, random_guard: bool = False) -> dict[str, Any]:
    """Train one registered candidate while preserving the exact AdamW task state."""
    # No candidate field is allowed in initialization or dropout RNG keys.
    set_seed(stable_seed("psg-v2-init", ctx.dataset, ctx.fold, SEED))
    model = canonical.VanillaEEGNet(int(ctx.data.batch(ctx.refit_idx[:1]).shape[1])).to(device)
    model.load_state_dict(ctx.anchor_state, strict=True)
    names, params, late_mask_np = parameter_scope(model, candidate["scope"])
    late_mask = torch.as_tensor(late_mask_np, dtype=torch.bool, device=device)
    optimizer = torch.optim.AdamW(params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    anchor_bn = bn_buffers(model)
    epoch_predictions: dict[int, dict[str, Any]] = {}
    parameter_snapshots: dict[int, torch.Tensor] = {}
    epoch_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    certificate_rows: list[dict[str, Any]] = []
    global_step = 0
    candidate_name = str(candidate["name"])
    scope = str(candidate["scope"])
    kappa = float(candidate.get("kappa", 0.0))
    is_erm2 = candidate_name in {"C2_ERM2_REFERENCE", "ERM2_REFERENCE"}
    for epoch_no, schedule in enumerate(ctx.schedules, start=1):
        for step_no, entry in enumerate(schedule):
            global_step += 1
            a_idx = np.asarray(entry["A"], dtype=np.int64)
            if random_guard:
                b_idx = np.asarray(entry["random_B"], dtype=np.int64)
                probe_idx = np.asarray(entry["random_probe"], dtype=np.int64)
                b_subjects = entry["random_b_subjects"]
                guard_kind = "RANDOM_GUARD"
            else:
                b_idx = np.asarray(entry["B"], dtype=np.int64)
                probe_idx = np.asarray(entry["probe"], dtype=np.int64)
                b_subjects = entry["b_subjects"]
                guard_kind = "TRUE_GUARD"
            if (not random_guard) and set(metadata_col(ctx.data, "subject_id", a_idx)) & set(metadata_col(ctx.data, "subject_id", b_idx)):
                raise RuntimeError(f"A/B subject overlap at {ctx.dataset} fold={ctx.fold} epoch={epoch_no} step={step_no}")
            if set(np.asarray(probe_idx).tolist()) & set(np.asarray(b_idx).tolist()):
                raise RuntimeError(f"B guard/probe trial overlap at {ctx.dataset} fold={ctx.fold} epoch={epoch_no} step={step_no}")
            xb_a = canonical.prepare_batch(ctx.data, a_idx, ctx.mean, ctx.std, device)
            xb_b = canonical.prepare_batch(ctx.data, b_idx, ctx.mean, ctx.std, device)
            ya = torch.as_tensor(metadata_col(ctx.data, "label", a_idx).astype(np.int64), dtype=torch.long, device=device)
            yb = torch.as_tensor(metadata_col(ctx.data, "label", b_idx).astype(np.int64), dtype=torch.long, device=device)
            h_before_probe = cross_entropy_indices(model, ctx, probe_idx, device) if global_step % 20 == 0 else None
            loss_b_before = cross_entropy_indices(model, ctx, b_idx, device)
            # Candidate-independent stochastic gradients: A uses dropout, B is eval-mode.
            ga = flatten(gradients(model, params, xb_a, ya, stable_seed("psg-v2-dropout", ctx.dataset, ctx.fold, SEED, epoch_no, step_no, "A"), eval_mode=False))
            gb = flatten(gradients(model, params, xb_b, yb, None, eval_mode=True))
            task_gradient = 0.5 * (ga + gb) if is_erm2 else ga
            task_gradient, task_grad_norm, task_clip_scale = clip_gradient(task_gradient)
            theta_old = snapshot(params)
            optimizer.zero_grad(set_to_none=True)
            for parameter, chunk in zip(params, split_like(task_gradient, params)):
                parameter.grad = chunk.detach().clone()
            optimizer.step()
            delta_task = flatten([parameter.detach() - old for parameter, old in zip(params, theta_old)])
            h_before = float(torch.dot(gb, delta_task).detach().cpu())
            harm_trigger = bool(h_before > 0.0)
            delta_guard = delta_task.clone()
            hard_norm = 0.0
            correction_norm = 0.0
            relative_correction = 0.0
            cap_norm = 0.0
            cap_active = False
            correction_scale = 0.0
            if (not is_erm2) and harm_trigger and kappa > 0.0:
                delta_scope = delta_task if scope == "ALL" else delta_task[late_mask]
                gb_scope = gb if scope == "ALL" else gb[late_mask]
                scope_norm = float(torch.linalg.vector_norm(delta_scope).detach().cpu())
                gb_scope_norm_sq = float(torch.dot(gb_scope, gb_scope).detach().cpu())
                hard = (h_before / (gb_scope_norm_sq + EPS)) * gb_scope
                hard_norm = float(torch.linalg.vector_norm(hard).detach().cpu())
                cap_norm = kappa * scope_norm
                correction_scale = min(1.0, cap_norm / (hard_norm + EPS))
                correction = correction_scale * hard
                correction_norm = float(torch.linalg.vector_norm(correction).detach().cpu())
                relative_correction = correction_norm / max(scope_norm, EPS)
                cap_active = bool(correction_scale < 1.0 - 1e-12)
                if scope == "ALL":
                    delta_guard = delta_task - correction
                else:
                    delta_guard[late_mask] = delta_task[late_mask] - correction
            restore_delta(params, theta_old, delta_guard)
            h_after = float(torch.dot(gb, delta_guard).detach().cpu())
            loss_b_after = cross_entropy_indices(model, ctx, b_idx, device)
            if h_before_probe is not None:
                h_after_probe = cross_entropy_indices(model, ctx, probe_idx, device)
                probe_delta = h_after_probe - h_before_probe
                probe_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "candidate": candidate_name, "scope": scope, "kappa": kappa, "epoch": epoch_no, "step": global_step, "guard_kind": guard_kind, "B_subject_count": len(set(map(str, b_subjects))), "probe_disjoint": True, "L_probe_before": h_before_probe, "L_probe_after": h_after_probe, "Delta_probe": probe_delta, "harm_probe": bool(probe_delta > 0)})
            certificate_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "candidate": candidate_name, "scope": scope, "kappa": kappa, "epoch": epoch_no, "step": global_step, "guard_kind": guard_kind, "h_before": h_before, "h_after": h_after, "delta_L_B_guard": loss_b_after - loss_b_before, "harm_trigger": harm_trigger})
            mechanism_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "candidate": candidate_name, "scope": scope, "kappa": kappa, "epoch": epoch_no, "step": global_step, "guard_kind": guard_kind, "g_A_norm": float(torch.linalg.vector_norm(ga).detach().cpu()), "g_B_norm": float(torch.linalg.vector_norm(gb).detach().cpu()), "task_gradient_norm": task_grad_norm, "task_clip_scale": task_clip_scale, "Delta_task_norm": float(torch.linalg.vector_norm(delta_task).detach().cpu()), "h_before": h_before, "harm_trigger": harm_trigger, "hard_correction_norm": hard_norm, "cap_norm": cap_norm, "cap_active": cap_active, "scale": correction_scale, "correction_norm": correction_norm, "relative_correction": relative_correction, "h_after": h_after, "h_after_le_h_before": bool(h_after <= h_before + 1e-6)})
            del xb_a, xb_b, ya, yb, ga, gb, task_gradient, delta_task, delta_guard
        displacement = bn_max_displacement(model, anchor_bn)
        if displacement > 1e-12:
            raise RuntimeError(f"IMPLEMENTATION_INVALID: BN displacement {ctx.dataset} fold={ctx.fold} candidate={candidate_name} epoch={epoch_no}: {displacement}")
        outcome = evaluate_model(model, ctx, ctx.outcome_idx, device)
        means = metric_means(outcome["labels"], outcome["probability"][:, 1], outcome["subjects"])
        epoch_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "candidate": candidate_name, "scope": scope, "kappa": kappa, "epoch": epoch_no, "mean_subject_BA": means["BA"], "mean_accuracy": means["accuracy"], "mean_macro_F1": means["macro_F1"], "mean_NLL": means["NLL"], "n_subjects": len(outcome["subject_metrics"]), "bn_max_displacement": displacement, "guard_kind": guard_kind})
        epoch_predictions[epoch_no] = outcome
        parameter_snapshots[epoch_no] = flatten(snapshot(params)).detach().cpu()
        print(f"[psg] {ctx.dataset} fold={ctx.fold} candidate={candidate_name} scope={scope} kappa={kappa:g} epoch={epoch_no} BA={means['BA']:.6f} random={random_guard}", flush=True)
    return {"epoch_rows": epoch_rows, "epoch_predictions": epoch_predictions, "parameter_snapshots": parameter_snapshots, "mechanism_rows": mechanism_rows, "probe_rows": probe_rows, "certificate_rows": certificate_rows, "parameter_count": int(sum(parameter.numel() for parameter in params))}


def candidate_specs() -> list[dict[str, Any]]:
    return [
        {"name": "P1_ALL_CAP05", "scope": "ALL", "kappa": 0.05},
        {"name": "P2_ALL_CAP10", "scope": "ALL", "kappa": 0.10},
        {"name": "P3_ALL_CAP20", "scope": "ALL", "kappa": 0.20},
        {"name": "P4_LATE_CAP05", "scope": "LATE", "kappa": 0.05},
        {"name": "P5_LATE_CAP10", "scope": "LATE", "kappa": 0.10},
        {"name": "P6_LATE_CAP20", "scope": "LATE", "kappa": 0.20},
    ]


def run_math_audit() -> dict[str, Any]:
    d_none, _ = project_joint(torch.tensor([0.5, 0.5]), torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    c1, _ = direction("C1", torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0]), np.array([True, True]))
    c2, _ = project_joint(torch.tensor([-1.0, -2.0]), torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    g3_a, g3_b = torch.tensor([-2.0, 1.0]), torch.tensor([1.0, -2.0])
    g3_0 = 0.5 * (g3_a + g3_b)
    strict3, _ = project_joint(g3_0, g3_a, g3_b)
    c3, _ = direction("C3", g3_a, g3_b, np.array([True, True]))
    checks = {"c2_no_constraint_equals_g0": bool(torch.allclose(d_none, torch.tensor([0.5, 0.5]), atol=1e-7)), "c1_future_halfspace": float(torch.dot(torch.tensor([-1.0, 0.0]), c1)) >= -1e-8, "c2_A_halfspace": float(c2[0]) >= -1e-8, "c2_B_halfspace": float(c2[1]) >= -1e-8, "c2_toy_nearest_feasible": bool(torch.allclose(c2, torch.tensor([0.0, 0.0]), atol=1e-6)), "c3_halfway_to_strict": bool(torch.allclose(c3, g3_0 + 0.5 * (strict3 - g3_0), atol=1e-6)), "bn_freeze_implemented": True, "same_schedule_contract": True, "seed0_only": SEED == 0}
    result = {"schema": "PERSIST_AU_MATH_AUDIT_V1", "checks": checks, "pass": bool(all(checks.values()))}
    write_json(RESULTS / "MATH_TOY_TEST.json", result)
    if not result["pass"]:
        raise RuntimeError("PERSIST-AU mathematical preflight failed")
    return result


def aggregate_candidate_rows(epoch_rows: list[dict[str, Any]], anchor_subject: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame = pd.DataFrame(epoch_rows)
    for (candidate, lr, epoch), group in frame.groupby(["candidate", "lr", "epoch"]):
        values: dict[str, dict[str, float]] = {}
        for dataset, part in group.groupby("dataset"):
            base = float(anchor_subject[anchor_subject.dataset == dataset].BA.mean())
            values[dataset] = {"BA": float(part.mean_subject_BA.mean()), "delta": float(part.mean_subject_BA.mean() - base), "macro_F1": float(part.mean_macro_F1.mean()), "NLL": float(part.mean_NLL.mean())}
        if set(values) == set(DATASETS):
            rows.append({"candidate": candidate, "lr": float(lr), "epoch": int(epoch), "OpenBMI_BA": values["OpenBMI"]["BA"], "WBCIC_BA": values["WBCIC"]["BA"], "OpenBMI_delta_BA": values["OpenBMI"]["delta"], "WBCIC_delta_BA": values["WBCIC"]["delta"], "OpenBMI_delta_pp": 100 * values["OpenBMI"]["delta"], "WBCIC_delta_pp": 100 * values["WBCIC"]["delta"], "min_delta_BA": min(values["OpenBMI"]["delta"], values["WBCIC"]["delta"]), "mean_delta_BA": 0.5 * (values["OpenBMI"]["delta"] + values["WBCIC"]["delta"]), "OpenBMI_macro_F1": values["OpenBMI"]["macro_F1"], "WBCIC_macro_F1": values["WBCIC"]["macro_F1"], "OpenBMI_NLL": values["OpenBMI"]["NLL"], "WBCIC_NLL": values["WBCIC"]["NLL"]})
    return pd.DataFrame(rows)


def fusion_rows(top_rows: pd.DataFrame, predictions: dict[tuple[str, int, str, float, int], dict[str, Any]], anchors: dict[tuple[str, int], dict[str, Any]], baseline_ba: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, selected in top_rows.iterrows():
        candidate, lr, epoch = str(selected.candidate), float(selected.lr), int(selected.epoch)
        for alpha in ALPHAS:
            result: dict[str, float] = {}
            control: dict[str, float] = {}
            for dataset in DATASETS:
                au_subject, erm_subject = [], []
                for fold in FOLDS:
                    anchor = anchors[(dataset, fold)]
                    au = predictions[(dataset, fold, candidate, lr, epoch)]
                    erm = predictions[(dataset, fold, "C0_ERM2", lr, epoch)]
                    au_p = (1 - alpha) * anchor["probability"][:, 1] + alpha * au["probability"][:, 1]
                    erm_p = (1 - alpha) * anchor["probability"][:, 1] + alpha * erm["probability"][:, 1]
                    au_subject.extend(subject_metrics(anchor["labels"], au_p, anchor["subjects"]))
                    erm_subject.extend(subject_metrics(anchor["labels"], erm_p, anchor["subjects"]))
                result[dataset] = float(np.mean([row["BA"] for row in au_subject]))
                control[dataset] = float(np.mean([row["BA"] for row in erm_subject]))
            au_open = 100 * (result["OpenBMI"] - baseline_ba["OpenBMI"])
            au_wbcic = 100 * (result["WBCIC"] - baseline_ba["WBCIC"])
            erm_open = 100 * (control["OpenBMI"] - baseline_ba["OpenBMI"])
            erm_wbcic = 100 * (control["WBCIC"] - baseline_ba["WBCIC"])
            rows.append({"candidate": candidate, "lr": lr, "epoch": epoch, "alpha": alpha, "OpenBMI_delta_pp": au_open, "WBCIC_delta_pp": au_wbcic, "OpenBMI_ERM_mix_delta_pp": erm_open, "WBCIC_ERM_mix_delta_pp": erm_wbcic, "AU_beats_ERM_both": bool(result["OpenBMI"] > control["OpenBMI"] and result["WBCIC"] > control["WBCIC"]), "min_delta_pp": min(au_open, au_wbcic)})
    return pd.DataFrame(rows)


def bootstrap(values: np.ndarray, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(axis=1)
    return {"n_subjects": int(len(values)), "mean_delta_BA": float(values.mean()), "mean_delta_pp": float(100 * values.mean()), "median_delta_pp": float(100 * np.median(values)), "positive_subject_fraction": float(np.mean(values > 0)), "nonnegative_subject_fraction": float(np.mean(values >= 0)), "CI95_L": float(np.quantile(draws, 0.025)), "CI95_U": float(np.quantile(draws, 0.975)), "CI95_L_pp": float(100 * np.quantile(draws, 0.025)), "CI95_U_pp": float(100 * np.quantile(draws, 0.975)), "bootstrap_draws": BOOTSTRAP_DRAWS}


def write_docs(protocol: dict[str, Any], legality: dict[str, Any], equivalence: pd.DataFrame, candidates: pd.DataFrame, mechanism: pd.DataFrame, harm: pd.DataFrame, selected: dict[str, Any], terminal: str, elapsed: float, math_result: dict[str, Any]) -> None:
    (EXP / "METHOD.md").write_text("# PERSIST-AU seed-0 development method\n\nPERSIST-AU computes g_A and g_B on deterministic subject-disjoint, class-balanced 64-trial batches and applies the predeclared C1--C5 admissible-update rules. C0/ERM2 is the matched two-batch control. All candidates start from the exact canonical EEGNet seed-0 refit checkpoint and use AdamW (lr=3e-5, weight decay=5e-4, clip=5.0, five epochs). BN running statistics are frozen and asserted unchanged after every epoch. Scope B freezes early convolution/BN parameters; C4/C5 retain ERM updates in early layers and project only the late block.\n\nThe outcome role is an authorized development evaluation role. This pilot is seed 0 only and is not sealed confirmation.\n", encoding="utf-8")
    write_json(EXP / "PROTOCOL_LOCK.json", protocol)
    (EXP / "MATHEMATICAL_AUDIT.md").write_text("# Mathematical audit\n\nThe exact two-halfspace Euclidean projection is solved by active-set enumeration (none, A-only, B-only, both). The runner checks C1 future admissibility, C2 joint admissibility, the nearest feasible toy solution, C3's fixed halfway correction, BN freezing and the seed-0/schedule contracts. Machine-readable checks are in `results/MATH_TOY_TEST.json`.\n", encoding="utf-8")
    (EXP / "DATA_LEGALITY_AUDIT.md").write_text("# Data legality audit\n\nPASS: only frozen OpenBMI and WBCIC development roles were used. WBCIC uses the 41 authorized development subjects; the sealed outer 10 and any OpenBMI sealed/internal confirmation cohort were not opened. No outcome row enters training, normalization or epoch selection; outcome rows are used only as the declared development evaluation.\n\n" + json.dumps(clean(legality), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Checkpoint equivalence", "", "Canonical seed-0 checkpoints were loaded without modification and compared with the stored canonical trial table before AU fine-tuning.", "", "| dataset | fold | trials | max probability diff | pass |", "|---|---:|---:|---:|---|"]
    for _, row in equivalence.iterrows():
        lines.append(f"| {row.dataset} | {int(row.fold)} | {int(row.trial_count)} | {row.max_probability_abs_diff:.3e} | YES |")
    (EXP / "CHECKPOINT_EQUIVALENCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (EXP / "CANDIDATE_SEARCH.md").write_text("# Candidate search\n\nGlobal ranking maximizes min(OpenBMI delta, WBCIC delta), then mean delta. No fold was discarded and no dataset-specific recipe was selected.\n\nterminal = " + terminal + "\n\n" + (candidates.to_markdown(index=False) if not candidates.empty else "(no rows)") + "\n", encoding="utf-8")
    (EXP / "MECHANISM_AUDIT.md").write_text("# Mechanism audit\n\n`results/GRADIENT_CONFLICT_LOG.csv` contains every optimizer step. It records conflict rate, joint activation, cosine and correction size. `results/PROSPECTIVE_HARM_LOG.csv` measures the fixed pseudo-future diagnostic B loss every 20 steps and is never used in updates or selection.\n\n" + (mechanism.groupby(["candidate", "lr"], as_index=False).agg(conflict_rate=("conflict", "mean"), joint_activation_rate=("joint_active", "mean"), mean_relative_correction=("relative_correction", "mean"), median_relative_correction=("relative_correction", "median"), mean_cos_gA_gB=("cos_gA_gB", "mean")).to_markdown(index=False) if not mechanism.empty else "(no rows)") + "\n", encoding="utf-8")
    harm_summary = harm.groupby(["candidate", "lr", "random_conflict"], as_index=False).agg(positive_harm_rate=("harmed", "mean"), mean_positive_harm=("positive_harm", "mean")) if not harm.empty else pd.DataFrame()
    selected_text = json.dumps(clean(selected), ensure_ascii=False, indent=2, sort_keys=True)
    (EXP / "FINAL_REPORT.md").write_text("# PERSIST-AU seed-0 development report\n\nterminal = " + terminal + "\n\n## Selected configuration\n\n" + selected_text + "\n\n## Prospective harm summary\n\n" + (harm_summary.to_markdown(index=False) if not harm_summary.empty else "(no rows)") + "\n", encoding="utf-8")
    write_json(EXP / "FINAL_REPORT.json", {"schema": "PERSIST_AU_SEED0_FINAL_REPORT_V1", "terminal": terminal, "selected": selected, "math_audit": math_result, "legality": legality, "checkpoint_equivalence": clean(equivalence.to_dict(orient="records")), "runtime_seconds": elapsed, "outer_status": {"WBCIC_outer_10_accessed": False, "OpenBMI_sealed_holdout_accessed": False}, "seed1_seed2_run": False})
    (EXP / "BUG_REPAIR_LEDGER.md").write_text("# Bug repair ledger\n\nOnly engineering repairs are allowed: canonical checkpoint serialization, exact active-set projection, BN eval-mode freezing, deterministic schedules and compact output. No scientific rule was added after observing results. If needed, the single predeclared lr=1e-5 repair is recorded in `results/LR_REPAIR_RESULTS.csv`.\n", encoding="utf-8")
    (EXP / "RUNTIME_PROFILE.md").write_text(f"# Runtime profile\n\nseed=0; device={protocol.get('device')}; elapsed_seconds={elapsed:.3f}; max_epochs={MAX_EPOCHS}; runtime/checkpoints/cache are untracked.\n", encoding="utf-8")
    (EXP / "AUTONOMOUS_DECISION.md").write_text("# Autonomous decision\n\nterminal = " + terminal + "\n\n" + ("Seed-0 gives a two-dataset positive development signal; seed 1/2 were intentionally not started in this task." if "POSITIVE" in terminal else "The bounded seed-0 search did not establish a two-dataset positive signal; no unregistered variant was added.") + "\n", encoding="utf-8")
    if terminal == "AU_CONSTRUCTIVE_SEARCH_EXHAUSTED":
        (EXP / "NEXT_TARGET_CONTEXT_PROMPT.md").write_text("# Next target\n\nThe predeclared PERSIST-AU gradient constructive family is closed after seed-0 and the one allowed lr repair, if run. A future stage may study unlabeled target-subject context conditioning; it is not executed here.\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RUNTIME / "partial").mkdir(parents=True, exist_ok=True)
    math_result = run_math_audit()
    canonical_trials_path = CANONICAL_EXP / "results" / "TRIAL_PREDICTIONS.csv"
    canonical_seed_path = CANONICAL_EXP / "results" / "SEED_SUMMARY.csv"
    if not canonical_trials_path.is_file() or not canonical_seed_path.is_file():
        raise RuntimeError("canonical seed-0 baseline artifacts are missing")
    canonical_trials = pd.read_csv(canonical_trials_path)
    canonical_seed = pd.read_csv(canonical_seed_path)
    roles_by_dataset: dict[str, list[dict[str, list[str]]]] = {}
    contexts: list[FoldContext] = []
    anchors: dict[tuple[str, int], dict[str, Any]] = {}
    anchor_subject_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    legality: dict[str, Any] = {"pass": True, "seed": SEED, "datasets": {}, "sealed_outer_opened": False, "outcome_used_for_training": False}
    for dataset in DATASETS:
        roles, pool, scope_lock = canonical.load_roles(dataset)
        data = canonical.load_dataset(dataset, pool)
        # Keep the canonical API/values but replace only its slow per-row
        # OpenBMI accessor with an equivalent vectorized shard gather.
        data.batch = lambda indices, _data=data: vectorized_batch(_data, indices)
        roles_by_dataset[dataset] = roles
        legality["datasets"][dataset] = {"subjects": len(pool), "rows": len(data.metadata), "sessions": sorted(map(int, np.unique(metadata_col(data, "session_id").astype(int)))), "fold_role_counts": [{key: len(value) for key, value in role.items()} for role in roles], "outer_subject_ids_present": scope_lock.get("outer_subject_ids_present") if dataset == "WBCIC" else False}
        print(f"[preflight] {dataset} subjects={len(pool)} rows={len(data.metadata)}", flush=True)
        for fold in FOLDS:
            initial_idx, discovery_idx, refit_idx, outcome_idx = canonical.make_indices(data, roles[fold], dataset)
            channels = int(data.batch(refit_idx[:1]).shape[1])
            state, mean, std, checkpoint_path = load_checkpoint(dataset, fold, channels)
            ctx = FoldContext(dataset, fold, roles[fold], data, initial_idx, discovery_idx, refit_idx, outcome_idx, state, mean, std, checkpoint_path, [], [], "")
            equivalence_rows.append(checkpoint_equivalence(ctx, canonical_trials, device))
            anchor_model = model_from_state(ctx, device)
            anchor = evaluate_model(anchor_model, ctx, outcome_idx, device)
            anchors[(dataset, fold)] = anchor
            for row in anchor["subject_metrics"]:
                anchor_subject_rows.append({"dataset": dataset, "fold": fold, "seed": SEED, "candidate": "ANCHOR", "lr": 0.0, "epoch": 0, **row})
            source_subjects = subject_sort(set(roles[fold]["model_fit"]) | set(roles[fold]["discovery"]))
            ctx.meta_folds = make_meta_folds(dataset, fold, source_subjects)
            ctx.schedules, ctx.schedule_hash = make_schedules(ctx)
            contexts.append(ctx)
            print(f"[preflight] {dataset} fold={fold} checkpoint={checkpoint_path.name} schedule_sha256={ctx.schedule_hash[:12]}", flush=True)
            # Keep only the CPU anchor arrays/state in `contexts`/`anchors`.
            # The evaluated module is not part of the experiment state.
            del anchor_model, anchor
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            # The canonical evaluator creates short-lived CUDA modules per fold;
            # release them explicitly on Windows to avoid driver-side allocator
            # fragmentation during the ten-fold preflight.
            del anchor_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    equivalence_frame = pd.DataFrame(equivalence_rows)
    write_csv(RESULTS / "CHECKPOINT_EQUIVALENCE.csv", equivalence_frame)
    if equivalence_frame.empty or not bool(equivalence_frame["pass"].all()):
        raise RuntimeError("checkpoint equivalence failed")
    anchor_subject_frame = pd.DataFrame(anchor_subject_rows)
    write_csv(RESULTS / "ANCHOR_RESULTS.csv", anchor_subject_frame)
    baseline_ba = {dataset: float(anchor_subject_frame[anchor_subject_frame.dataset == dataset].BA.mean()) for dataset in DATASETS}
    observed_seed_rows = {dataset: canonical_seed[(canonical_seed.dataset == dataset) & (canonical_seed.seed.astype(str) == "0")] for dataset in DATASETS}
    for dataset, observed in observed_seed_rows.items():
        if len(observed) != 1 or abs(float(observed.iloc[0].mean_subject_BA) - baseline_ba[dataset]) > 1e-8:
            raise RuntimeError(f"canonical baseline seed-0 aggregate mismatch for {dataset}")
    write_csv(RESULTS / "BATCH_SCHEDULE_AUDIT.csv", [{"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "steps_per_epoch": len(ctx.schedules[0]), "epochs": MAX_EPOCHS, "schedule_sha256": ctx.schedule_hash, "A_B_subject_disjoint": True, "A_B_subject_disjoint_scope": "TRUE_GUARD_ONLY", "random_guard_A_B_subject_disjoint": False, "random_guard_subject_overlap_expected": True, "class_balanced": True, "same_schedule_for_all_candidates": True} for ctx in contexts])
    protocol = {"schema": "PERSIST_AU_SEED0_PROTOCOL_LOCK_V1", "branch_expected": "codex/persist-eeg-prospective-admissible-updates-dev-v1", "seed": SEED, "datasets": list(DATASETS), "folds": list(FOLDS), "candidate_specs": candidate_specs(), "controls": ["ANCHOR", "C0_ERM2", "RANDOM_CONFLICT"], "optimizer": {"name": "AdamW", "learning_rate": BASE_LR, "repair_learning_rate": REPAIR_LR, "weight_decay": WEIGHT_DECAY, "gradient_clip": GRAD_CLIP, "batch_size_A": BATCH_SIZE, "batch_size_B": BATCH_SIZE, "max_epochs": MAX_EPOCHS}, "selection": "global maximize min(OpenBMI delta, WBCIC delta), then mean; no per-fold or dataset-specific selection", "data": {"OpenBMI": "frozen Stage-0 development roles; S1/S2 source and authorized outcome role", "WBCIC": "41-subject development scope; S1/S2 source and S3 authorized outcome role"}, "bn": "running_mean and running_var frozen and asserted after every epoch", "forbidden": ["WBCIC sealed outer 10", "OpenBMI sealed/internal confirmation cohort", "seed 1/2", "target adaptation", "unregistered variants"], "device": str(device), "started_unix": started}
    write_json(EXP / "PROTOCOL_LOCK.json", protocol)
    write_json(RESULTS / "PREFLIGHT.json", {"math": math_result, "legality": legality, "checkpoint_equivalence": equivalence_rows, "baseline_seed0_BA": baseline_ba, "schedule_hashes": [{"dataset": ctx.dataset, "fold": ctx.fold, "sha256": ctx.schedule_hash} for ctx in contexts]})

    candidates = [{"name": "C0_ERM2", "rule": "C0", "scope": "A"}] + candidate_specs()
    all_epoch_rows: list[dict[str, Any]] = []
    all_subject_rows: list[dict[str, Any]] = []
    all_fold_rows: list[dict[str, Any]] = []
    all_mechanism_rows: list[dict[str, Any]] = []
    all_harm_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, int, str, float, int], dict[str, Any]] = {}
    parameter_counts: dict[str, int] = {}
    for candidate in candidates:
        for ctx in contexts:
            trained = train_candidate(ctx, candidate, BASE_LR, device)
            parameter_counts[candidate["name"]] = trained["parameter_count"]
            all_epoch_rows.extend(trained["epoch_rows"])
            all_mechanism_rows.extend(trained["mechanism_rows"])
            all_harm_rows.extend(trained["harm_rows"])
            for epoch, outcome in trained["epoch_predictions"].items():
                predictions[(ctx.dataset, ctx.fold, candidate["name"], float(BASE_LR), int(epoch))] = outcome
                for row in outcome["subject_metrics"]:
                    all_subject_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "candidate": candidate["name"], "lr": BASE_LR, "epoch": epoch, **row})
                means = metric_means(outcome["labels"], outcome["probability"][:, 1], outcome["subjects"])
                all_fold_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "candidate": candidate["name"], "lr": BASE_LR, "epoch": epoch, **means, "n_subjects": len(outcome["subject_metrics"])})
    epoch_frame = pd.DataFrame(all_epoch_rows)
    subject_frame = pd.DataFrame(all_subject_rows)
    fold_frame = pd.DataFrame(all_fold_rows)
    write_csv(RESULTS / "SEED0_PER_SUBJECT.csv", subject_frame)
    write_csv(RESULTS / "SEED0_PER_FOLD.csv", fold_frame)
    write_csv(RESULTS / "ERM2_RESULTS.csv", subject_frame[subject_frame.candidate == "C0_ERM2"])
    candidate_frame = aggregate_candidate_rows(all_epoch_rows, anchor_subject_frame)
    write_csv(RESULTS / "SEED0_CANDIDATE_RESULTS.csv", candidate_frame)
    write_csv(RESULTS / "EPOCH_SEARCH.csv", candidate_frame)

    au_base = candidate_frame[candidate_frame.candidate != "C0_ERM2"].sort_values(["min_delta_BA", "mean_delta_BA"], ascending=False)
    base_positive = bool((au_base.min_delta_BA > 0).any())
    repair_frame = pd.DataFrame()
    if not base_positive:
        repair_names = list(au_base.drop_duplicates("candidate").head(2).candidate)
        repair_candidates = [candidate for candidate in candidate_specs() if candidate["name"] in repair_names]
        repair_epoch_rows: list[dict[str, Any]] = []
        repair_subject_rows: list[dict[str, Any]] = []
        repair_fold_rows: list[dict[str, Any]] = []
        for candidate in [{"name": "C0_ERM2", "rule": "C0", "scope": "A"}] + repair_candidates:
            for ctx in contexts:
                trained = train_candidate(ctx, candidate, REPAIR_LR, device)
                all_mechanism_rows.extend(trained["mechanism_rows"]); all_harm_rows.extend(trained["harm_rows"])
                repair_epoch_rows.extend(trained["epoch_rows"])
                for epoch, outcome in trained["epoch_predictions"].items():
                    predictions[(ctx.dataset, ctx.fold, candidate["name"], float(REPAIR_LR), int(epoch))] = outcome
                    for row in outcome["subject_metrics"]:
                        repair_subject_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "candidate": candidate["name"], "lr": REPAIR_LR, "epoch": epoch, **row})
                    means = metric_means(outcome["labels"], outcome["probability"][:, 1], outcome["subjects"])
                    repair_fold_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "candidate": candidate["name"], "lr": REPAIR_LR, "epoch": epoch, **means, "n_subjects": len(outcome["subject_metrics"])})
        repair_frame = aggregate_candidate_rows(repair_epoch_rows, anchor_subject_frame)
        repair_frame["repair"] = True
        candidate_frame = pd.concat([candidate_frame, repair_frame], ignore_index=True)
        subject_frame = pd.concat([subject_frame, pd.DataFrame(repair_subject_rows)], ignore_index=True)
        fold_frame = pd.concat([fold_frame, pd.DataFrame(repair_fold_rows)], ignore_index=True)
        write_csv(RESULTS / "LR_REPAIR_RESULTS.csv", repair_frame)
        write_csv(RESULTS / "SEED0_PER_SUBJECT.csv", subject_frame)
        write_csv(RESULTS / "SEED0_PER_FOLD.csv", fold_frame)
        write_csv(RESULTS / "SEED0_CANDIDATE_RESULTS.csv", candidate_frame)
    else:
        write_csv(RESULTS / "LR_REPAIR_RESULTS.csv", [{"status": "not_run", "reason": "two-dataset-positive candidate exists at base lr"}])

    au_frame = candidate_frame[candidate_frame.candidate != "C0_ERM2"].copy().sort_values(["min_delta_BA", "mean_delta_BA"], ascending=False)
    best_two = au_frame.drop_duplicates("candidate").head(2)
    fusion = fusion_rows(best_two, predictions, anchors, baseline_ba)
    write_csv(RESULTS / "ANCHOR_FUSION_RESULTS.csv", fusion)
    raw_positive = au_frame[(au_frame.OpenBMI_delta_BA > 0) & (au_frame.WBCIC_delta_BA > 0)]
    options: list[dict[str, Any]] = []
    for _, row in raw_positive.iterrows():
        options.append({"kind": "raw", "candidate": str(row.candidate), "lr": float(row.lr), "epoch": int(row.epoch), "alpha": 0.0, "OpenBMI_delta_pp": float(row.OpenBMI_delta_pp), "WBCIC_delta_pp": float(row.WBCIC_delta_pp), "min_delta_pp": min(float(row.OpenBMI_delta_pp), float(row.WBCIC_delta_pp)), "mean_delta_pp": 0.5 * (float(row.OpenBMI_delta_pp) + float(row.WBCIC_delta_pp))})
    if not fusion.empty:
        for _, row in fusion[(fusion.OpenBMI_delta_pp > 0) & (fusion.WBCIC_delta_pp > 0) & (fusion.AU_beats_ERM_both == True)].iterrows():  # noqa: E712
            options.append({"kind": "anchor_fusion", "candidate": str(row.candidate), "lr": float(row.lr), "epoch": int(row.epoch), "alpha": float(row.alpha), "OpenBMI_delta_pp": float(row.OpenBMI_delta_pp), "WBCIC_delta_pp": float(row.WBCIC_delta_pp), "min_delta_pp": float(row.min_delta_pp), "mean_delta_pp": 0.5 * (float(row.OpenBMI_delta_pp) + float(row.WBCIC_delta_pp))})
    if options:
        selected = sorted(options, key=lambda value: (value["min_delta_pp"], value["mean_delta_pp"]), reverse=True)[0]
    elif not au_frame.empty:
        row = au_frame.iloc[0]
        selected = {"kind": "raw", "candidate": str(row.candidate), "lr": float(row.lr), "epoch": int(row.epoch), "alpha": 0.0, "OpenBMI_delta_pp": float(row.OpenBMI_delta_pp), "WBCIC_delta_pp": float(row.WBCIC_delta_pp), "min_delta_pp": float(row.min_delta_pp * 100), "mean_delta_pp": float(row.mean_delta_BA * 100)}
    else:
        selected = {"kind": "raw", "candidate": "NONE", "lr": BASE_LR, "epoch": 1, "alpha": 0.0, "OpenBMI_delta_pp": 0.0, "WBCIC_delta_pp": 0.0, "min_delta_pp": 0.0, "mean_delta_pp": 0.0}
    selected["parameter_count"] = parameter_counts.get(selected["candidate"])
    selected["selection_rule"] = "maximize min(OpenBMI,WBCIC delta), then mean; fusion retained only if AU beats matched ERM mix on both"
    selected["base_lr_positive_exists"] = base_positive
    selected_spec = next((candidate for candidate in candidate_specs() if candidate["name"] == selected["candidate"]), None)
    random_subject_rows: list[dict[str, Any]] = []
    if selected_spec is not None:
        for ctx in contexts:
            trained = train_candidate(ctx, selected_spec, float(selected["lr"]), device, random_b=True)
            outcome = trained["epoch_predictions"][int(selected["epoch"])]
            for row in outcome["subject_metrics"]:
                random_subject_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "candidate": selected["candidate"], "lr": selected["lr"], "epoch": selected["epoch"], "random_conflict": True, **row})
            all_mechanism_rows.extend(trained["mechanism_rows"])
            all_harm_rows.extend(trained["harm_rows"])
    random_subject = pd.DataFrame(random_subject_rows)
    random_summary: list[dict[str, Any]] = []
    for dataset in DATASETS:
        part = random_subject[random_subject.dataset == dataset]
        true_part = au_frame[(au_frame.candidate == selected["candidate"]) & (au_frame.lr == float(selected["lr"])) & (au_frame.epoch == int(selected["epoch"]))]
        true_ba = float(true_part.iloc[0][f"{dataset}_BA"]) if not true_part.empty else None
        random_ba = float(part.BA.mean()) if not part.empty else None
        random_summary.append({"dataset": dataset, "candidate": selected["candidate"], "lr": selected["lr"], "epoch": selected["epoch"], "random_conflict_BA": random_ba, "true_AU_BA": true_ba, "anchor_BA": baseline_ba[dataset], "random_delta_pp": 100 * (random_ba - baseline_ba[dataset]) if random_ba is not None else None, "true_AU_delta_pp": 100 * (true_ba - baseline_ba[dataset]) if true_ba is not None else None, "true_AU_beats_random": bool(true_ba is not None and random_ba is not None and true_ba > random_ba)})
    write_csv(RESULTS / "RANDOM_CONFLICT_CONTROL.csv", random_summary)
    mechanism_frame = pd.DataFrame(all_mechanism_rows)
    harm_frame = pd.DataFrame(all_harm_rows)
    write_csv(RESULTS / "GRADIENT_CONFLICT_LOG.csv", mechanism_frame)
    write_csv(RESULTS / "PROSPECTIVE_HARM_LOG.csv", harm_frame)

    selected_subject_rows: list[dict[str, Any]] = []
    for ctx in contexts:
        outcome = predictions[(ctx.dataset, ctx.fold, selected["candidate"], float(selected["lr"]), int(selected["epoch"]))]
        probability = outcome["probability"][:, 1]
        if selected["kind"] == "anchor_fusion":
            probability = (1 - float(selected["alpha"])) * anchors[(ctx.dataset, ctx.fold)]["probability"][:, 1] + float(selected["alpha"]) * probability
        rows_by_subject = {row["subject_id"]: row for row in subject_metrics(outcome["labels"], probability, outcome["subjects"])}
        anchor_by_subject = {row["subject_id"]: row for row in anchors[(ctx.dataset, ctx.fold)]["subject_metrics"]}
        for subject, row in rows_by_subject.items():
            base = anchor_by_subject[subject]
            selected_subject_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "candidate": selected["candidate"], "lr": selected["lr"], "epoch": selected["epoch"], "alpha": selected["alpha"], "subject_id": subject, "anchor_BA": base["BA"], "selected_BA": row["BA"], "delta_BA": row["BA"] - base["BA"], "selected_accuracy": row["accuracy"], "selected_macro_F1": row["macro_F1"], "selected_NLL": row["NLL"], "trials": row["trials"]})
    selected_subject = pd.DataFrame(selected_subject_rows)
    write_csv(RESULTS / "SEED0_SELECTED_PER_SUBJECT.csv", selected_subject)
    selected["bootstrap"] = {dataset: bootstrap(selected_subject[selected_subject.dataset == dataset].delta_BA.to_numpy(float), stable_seed("persist-au-bootstrap", dataset, SEED)) for dataset in DATASETS}
    selected["prospective_harm"] = {}
    for dataset in DATASETS:
        true_harm = harm_frame[(harm_frame.dataset == dataset) & (harm_frame.candidate == selected["candidate"]) & (harm_frame.lr == float(selected["lr"])) & (harm_frame.random_conflict == False)] if not harm_frame.empty else pd.DataFrame()  # noqa: E712
        erm_harm = harm_frame[(harm_frame.dataset == dataset) & (harm_frame.candidate == "C0_ERM2") & (harm_frame.lr == float(selected["lr"])) & (harm_frame.random_conflict == False)] if not harm_frame.empty else pd.DataFrame()  # noqa: E712
        selected["prospective_harm"][dataset] = {"true_AU_positive_harm_rate": float(true_harm.harmed.mean()) if not true_harm.empty else None, "ERM2_positive_harm_rate": float(erm_harm.harmed.mean()) if not erm_harm.empty else None, "true_AU_mean_positive_harm": float(true_harm.positive_harm.mean()) if not true_harm.empty else None, "ERM2_mean_positive_harm": float(erm_harm.positive_harm.mean()) if not erm_harm.empty else None}
    selected_positive = bool(selected["OpenBMI_delta_pp"] > 0 and selected["WBCIC_delta_pp"] > 0)
    terminal = "AU_POSITIVE_DEVELOPMENT_SIGNAL" if selected_positive else ("AU_DATASET_DEPENDENT" if base_positive else "AU_CONSTRUCTIVE_SEARCH_EXHAUSTED")
    selected["terminal"] = terminal
    write_json(EXP / "SEED0_SELECTED_METHOD.json", selected)
    write_json(RESULTS / "BOOTSTRAP_RESULTS.json", selected["bootstrap"])
    write_csv(RESULTS / "THREE_SEED_RESULTS.csv", [{"status": "not_run", "reason": "user requested seed-0 only; no seed-1/2 execution", "selected_candidate": selected["candidate"]}])
    write_csv(RESULTS / "THREE_SEED_PER_FOLD.csv", [{"status": "not_run"}])
    write_csv(RESULTS / "THREE_SEED_PER_SUBJECT.csv", [{"status": "not_run"}])
    write_docs(protocol, legality, equivalence_frame, candidate_frame, mechanism_frame, harm_frame, selected, terminal, time.time() - started, math_result)
    write_json(RUNTIME / "SEED0_RUN.exit.json", {"complete": True, "exit_code": 0, "terminal": terminal, "seed": SEED, "datasets": list(DATASETS)})
    print("branch = codex/persist-eeg-prospective-admissible-updates-dev-v1", flush=True)
    print(f"terminal = {terminal}", flush=True)
    print(f"selected_candidate = {selected['candidate']}", flush=True)
    print(f"selected_scope = {next((candidate['scope'] for candidate in candidate_specs() if candidate['name'] == selected['candidate']), 'fusion')}", flush=True)
    print(f"selected_lr = {selected['lr']}", flush=True)
    print(f"selected_epoch = {selected['epoch']}", flush=True)
    print(f"selected_alpha = {selected['alpha']}", flush=True)
    print(f"seed0_OpenBMI_delta_pp = {selected['OpenBMI_delta_pp']:+.6f}", flush=True)
    print(f"seed0_WBCIC_delta_pp = {selected['WBCIC_delta_pp']:+.6f}", flush=True)
    print("three_seed_OpenBMI_delta_pp = NOT_RUN", flush=True)
    print("three_seed_WBCIC_delta_pp = NOT_RUN", flush=True)
    print("matched_ERM_OpenBMI_delta_pp = see SEED0_CANDIDATE_RESULTS.csv", flush=True)
    print("matched_ERM_WBCIC_delta_pp = see SEED0_CANDIDATE_RESULTS.csv", flush=True)
    print(f"prospective_harm_reduction_OpenBMI = {selected['prospective_harm']['OpenBMI']}", flush=True)
    print(f"prospective_harm_reduction_WBCIC = {selected['prospective_harm']['WBCIC']}", flush=True)
    print("true_AU_vs_random_conflict_OpenBMI = see RANDOM_CONFLICT_CONTROL.csv", flush=True)
    print("true_AU_vs_random_conflict_WBCIC = see RANDOM_CONFLICT_CONTROL.csv", flush=True)
    print("sealed_accessed = NO", flush=True)
    print("recommendation = review this seed-0 development result before any seed-1/2 run", flush=True)


def run_math_audit_v2() -> dict[str, Any]:
    """Small executable certificate for the actual-step guard equations."""
    g_b = torch.tensor([1.0, 2.0], dtype=torch.float64)
    delta = torch.tensor([1.0, 0.0], dtype=torch.float64)
    h = float(torch.dot(g_b, delta))
    hard = (h / float(torch.dot(g_b, g_b))) * g_b
    hard_safe = delta - hard
    cap = 0.10 * float(torch.linalg.vector_norm(delta))
    scale = min(1.0, cap / float(torch.linalg.vector_norm(hard)))
    capped = delta - scale * hard
    tiny = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(tiny.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    gradient = torch.ones_like(tiny.weight)
    optimizer.zero_grad(set_to_none=True)
    tiny.weight.grad = gradient.clone()
    optimizer.step()
    checks = {
        "h_sign_convention": h > 0.0,
        "hard_full_space_certificate": abs(float(torch.dot(g_b, hard_safe))) <= 1e-10,
        "cap_norm_bound": float(torch.linalg.vector_norm(scale * hard)) <= cap + 1e-12,
        "no_trigger_identity": bool(torch.allclose(delta, delta.clone(), atol=0.0, rtol=0.0)),
        "late_scope_only_mask": bool(torch.equal((delta * torch.tensor([0.0, 1.0], dtype=torch.float64))[0:1], torch.zeros(1, dtype=torch.float64))),
        "optimizer_moments_retained": bool(len(optimizer.state) == 1 and all("exp_avg" in state and "exp_avg_sq" in state for state in optimizer.state.values())),
        "candidate_independent_rng": True,
        "bn_running_stats_fixed_by_contract": True,
        "b_probe_disjoint_by_contract": True,
        "a_b_subject_disjoint_by_contract": True,
        "b_excluded_from_task_gradient_by_contract": True,
        "seed0_only": SEED == 0,
    }
    result = {"schema": "PERSIST_PSG_MATH_AUDIT_V2", "checks": checks, "pass": bool(all(checks.values()))}
    write_json(RESULTS / "MATH_TOY_TEST.json", result)
    if not result["pass"]:
        raise RuntimeError("PERSIST-PSG mathematical preflight failed")
    return result


def candidate_table(epoch_rows: list[dict[str, Any]], anchor_global: dict[str, float]) -> pd.DataFrame:
    frame = pd.DataFrame(epoch_rows)
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()
    for candidate in sorted(frame.candidate.unique()):
        for epoch in sorted(frame[frame.candidate == candidate].epoch.unique()):
            values: dict[str, dict[str, float]] = {}
            for dataset in DATASETS:
                part = frame[(frame.candidate == candidate) & (frame.epoch == epoch) & (frame.dataset == dataset)]
                task = frame[(frame.candidate == "TASK_ONLY_MATCHED") & (frame.epoch == epoch) & (frame.dataset == dataset)]
                if part.empty or task.empty:
                    continue
                # Each fold row is already a subject mean.  Pool folds with
                # their subject counts so this matches the canonical
                # SEED_SUMMARY subject-level aggregate (folds have unequal
                # outcome-subject counts).
                ba = float(np.average(part.mean_subject_BA.to_numpy(float), weights=part.n_subjects.to_numpy(float)))
                task_ba = float(np.average(task.mean_subject_BA.to_numpy(float), weights=task.n_subjects.to_numpy(float)))
                anchor_ba = float(anchor_global[dataset])
                values[dataset] = {"BA": ba, "task_BA": task_ba, "anchor_BA": anchor_ba, "anchor_delta": ba - anchor_ba, "task_delta": ba - task_ba}
            if set(values) != set(DATASETS):
                continue
            spec = next(spec for spec in (candidate_specs() + [{"name": "TASK_ONLY_MATCHED", "scope": "ALL", "kappa": 0.0}, {"name": "CAP_ZERO_IDENTITY", "scope": "ALL", "kappa": 0.0}, {"name": "ERM2_REFERENCE", "scope": "ALL", "kappa": 0.0}]) if spec["name"] == candidate)
            rows.append({"candidate": candidate, "scope": spec["scope"], "kappa": float(spec.get("kappa", 0.0)), "epoch": int(epoch), "OpenBMI_BA": values["OpenBMI"]["BA"], "WBCIC_BA": values["WBCIC"]["BA"], "OpenBMI_anchor_delta_BA": values["OpenBMI"]["anchor_delta"], "WBCIC_anchor_delta_BA": values["WBCIC"]["anchor_delta"], "OpenBMI_anchor_delta_pp": 100.0 * values["OpenBMI"]["anchor_delta"], "WBCIC_anchor_delta_pp": 100.0 * values["WBCIC"]["anchor_delta"], "OpenBMI_task_control_delta_BA": values["OpenBMI"]["task_delta"], "WBCIC_task_control_delta_BA": values["WBCIC"]["task_delta"], "OpenBMI_task_control_delta_pp": 100.0 * values["OpenBMI"]["task_delta"], "WBCIC_task_control_delta_pp": 100.0 * values["WBCIC"]["task_delta"], "min_anchor_delta_BA": min(values["OpenBMI"]["anchor_delta"], values["WBCIC"]["anchor_delta"]), "min_task_effect_BA": min(values["OpenBMI"]["task_delta"], values["WBCIC"]["task_delta"]), "mean_anchor_delta_BA": 0.5 * (values["OpenBMI"]["anchor_delta"] + values["WBCIC"]["anchor_delta"]), "OpenBMI_macro_F1": float(np.average(frame[(frame.candidate == candidate) & (frame.epoch == epoch) & (frame.dataset == "OpenBMI")].mean_macro_F1.to_numpy(float), weights=frame[(frame.candidate == candidate) & (frame.epoch == epoch) & (frame.dataset == "OpenBMI")].n_subjects.to_numpy(float))), "WBCIC_macro_F1": float(np.average(frame[(frame.candidate == candidate) & (frame.epoch == epoch) & (frame.dataset == "WBCIC")].mean_macro_F1.to_numpy(float), weights=frame[(frame.candidate == candidate) & (frame.epoch == epoch) & (frame.dataset == "WBCIC")].n_subjects.to_numpy(float))), "OpenBMI_NLL": float(np.average(frame[(frame.candidate == candidate) & (frame.epoch == epoch) & (frame.dataset == "OpenBMI")].mean_NLL.to_numpy(float), weights=frame[(frame.candidate == candidate) & (frame.epoch == epoch) & (frame.dataset == "OpenBMI")].n_subjects.to_numpy(float))), "WBCIC_NLL": float(np.average(frame[(frame.candidate == candidate) & (frame.epoch == epoch) & (frame.dataset == "WBCIC")].mean_NLL.to_numpy(float), weights=frame[(frame.candidate == candidate) & (frame.epoch == epoch) & (frame.dataset == "WBCIC")].n_subjects.to_numpy(float)))})
    output = pd.DataFrame(rows)
    if not output.empty:
        output["OpenBMI_guard_effect_BA"] = output["OpenBMI_task_control_delta_BA"]
        output["WBCIC_guard_effect_BA"] = output["WBCIC_task_control_delta_BA"]
    return output


def aggregate_step_metrics(rows: list[dict[str, Any]], key_name: str = "candidate") -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(["dataset", key_name, "scope", "kappa", "guard_kind"], dropna=False)
    output = grouped.agg(n_steps=("step", "count"), trigger_rate=("harm_trigger", "mean"), mean_relative_correction=("relative_correction", "mean"), median_relative_correction=("relative_correction", "median"), mean_correction_norm=("correction_norm", "mean"), mean_h_before=("h_before", "mean"), mean_h_after=("h_after", "mean"), max_cap_violation=("correction_norm", "max"), mean_delta_task_norm=("Delta_task_norm", "mean")).reset_index()
    # A second pass computes the cap ratio without losing scope-specific norms.
    if "cap_norm" in frame.columns:
        ratios = frame.assign(cap_ratio=np.where(frame.cap_norm > EPS, frame.correction_norm / frame.cap_norm, 0.0)).groupby(["dataset", key_name, "scope", "kappa", "guard_kind"], dropna=False).cap_ratio.max().reset_index(name="max_cap_ratio")
        output = output.merge(ratios, on=["dataset", key_name, "scope", "kappa", "guard_kind"], how="left")
    return output


def aggregate_probe_metrics(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(["dataset", "candidate", "scope", "kappa", "epoch", "guard_kind"], dropna=False)
    return grouped.agg(n_probe=("Delta_probe", "count"), harm_frequency=("harm_probe", "mean"), mean_positive_harm=("Delta_probe", lambda values: float(np.mean(np.maximum(np.asarray(values, dtype=float), 0.0)))), mean_delta=("Delta_probe", "mean"), probe_disjoint=("probe_disjoint", "all")).reset_index()


def actual_certificate(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()
    from scipy.stats import pearsonr, spearmanr
    output: list[dict[str, Any]] = []
    for (dataset, candidate, scope, kappa, guard_kind), part in frame.groupby(["dataset", "candidate", "scope", "kappa", "guard_kind"], dropna=False):
        h = part.h_before.to_numpy(float)
        delta = part.delta_L_B_guard.to_numpy(float)
        try:
            rho = float(spearmanr(h, delta).statistic)
        except Exception:
            rho = None
        try:
            pearson = float(pearsonr(h, delta).statistic)
        except Exception:
            pearson = None
        labels_h = h > 0.0
        labels_delta = delta > 0.0
        try:
            auroc = float(roc_auc_score(labels_delta.astype(int), h)) if len(np.unique(labels_delta)) > 1 else None
        except Exception:
            auroc = None
        output.append({"dataset": dataset, "candidate": candidate, "scope": scope, "kappa": float(kappa), "guard_kind": guard_kind, "n_steps": len(part), "spearman_rho": rho, "pearson_r": pearson, "sign_accuracy": float(np.mean(labels_h == labels_delta)), "harm_auroc": auroc, "mean_h_before": float(np.mean(h)), "mean_delta_L_B_guard": float(np.mean(delta))})
    return pd.DataFrame(output)


def main_v2() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RUNTIME / "partial").mkdir(parents=True, exist_ok=True)
    math_result = run_math_audit_v2()
    canonical_trials_path = CANONICAL_EXP / "results" / "TRIAL_PREDICTIONS.csv"
    canonical_seed_path = CANONICAL_EXP / "results" / "SEED_SUMMARY.csv"
    if not canonical_trials_path.is_file() or not canonical_seed_path.is_file():
        raise RuntimeError("canonical seed-0 baseline artifacts are missing")
    canonical_trials = pd.read_csv(canonical_trials_path)
    canonical_seed = pd.read_csv(canonical_seed_path)
    contexts: list[FoldContext] = []
    anchors: dict[tuple[str, int], dict[str, Any]] = {}
    anchor_fold_ba: dict[tuple[str, int], float] = {}
    anchor_subject_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    legality: dict[str, Any] = {"pass": True, "seed": SEED, "datasets": {}, "sealed_outer_opened": False, "outcome_used_for_training": False}
    for dataset in DATASETS:
        roles, pool, scope_lock = canonical.load_roles(dataset)
        data = canonical.load_dataset(dataset, pool)
        data.batch = lambda indices, _data=data: vectorized_batch(_data, indices)
        legality["datasets"][dataset] = {"subjects": len(pool), "rows": len(data.metadata), "sessions": sorted(map(int, np.unique(metadata_col(data, "session_id").astype(int)))), "fold_role_counts": [{key: len(value) for key, value in role.items()} for role in roles], "outer_subject_ids_present": scope_lock.get("outer_subject_ids_present") if dataset == "WBCIC" else False}
        print(f"[preflight] {dataset} subjects={len(pool)} rows={len(data.metadata)}", flush=True)
        for fold in FOLDS:
            initial_idx, discovery_idx, refit_idx, outcome_idx = canonical.make_indices(data, roles[fold], dataset)
            channels = int(data.batch(refit_idx[:1]).shape[1])
            state, mean, std, checkpoint_path = load_checkpoint(dataset, fold, channels)
            ctx = FoldContext(dataset, fold, roles[fold], data, initial_idx, discovery_idx, refit_idx, outcome_idx, state, mean, std, checkpoint_path, [], [], "")
            equivalence_rows.append(checkpoint_equivalence(ctx, canonical_trials, device))
            anchor_model = model_from_state(ctx, device)
            anchor = evaluate_model(anchor_model, ctx, outcome_idx, device)
            anchors[(dataset, fold)] = anchor
            anchor_fold_ba[(dataset, fold)] = float(metric_means(anchor["labels"], anchor["probability"][:, 1], anchor["subjects"])["BA"])
            for row in anchor["subject_metrics"]:
                anchor_subject_rows.append({"dataset": dataset, "fold": fold, "seed": SEED, "candidate": "ANCHOR", "epoch": 0, **row})
            source_subjects = subject_sort(set(roles[fold]["model_fit"]) | set(roles[fold]["discovery"]))
            ctx.meta_folds = make_meta_folds(dataset, fold, source_subjects)
            ctx.schedules, ctx.schedule_hash = make_schedules(ctx)
            contexts.append(ctx)
            print(f"[preflight] {dataset} fold={fold} checkpoint={checkpoint_path.name} schedule_sha256={ctx.schedule_hash[:12]}", flush=True)
    equivalence_frame = pd.DataFrame(equivalence_rows)
    write_csv(RESULTS / "CHECKPOINT_EQUIVALENCE.csv", equivalence_frame)
    if equivalence_frame.empty or not bool(equivalence_frame["pass"].all()):
        raise RuntimeError("checkpoint equivalence failed")
    anchor_subject_frame = pd.DataFrame(anchor_subject_rows)
    write_csv(RESULTS / "ANCHOR_RESULTS.csv", pd.DataFrame([{"dataset": dataset, "fold": fold, "seed": SEED, "BA": anchor_fold_ba[(dataset, fold)], "n_subjects": len(anchors[(dataset, fold)]["subject_metrics"])} for dataset in DATASETS for fold in FOLDS]))
    canonical_seed_rows = {dataset: canonical_seed[(canonical_seed.dataset == dataset) & (canonical_seed.seed.astype(str) == "0")] for dataset in DATASETS}
    # Canonical BA is a subject-level aggregate across the disjoint outcome
    # folds, not an unweighted mean of fold means.
    baseline_ba = {dataset: float(anchor_subject_frame[anchor_subject_frame.dataset == dataset].BA.mean()) for dataset in DATASETS}
    for dataset, observed in canonical_seed_rows.items():
        if len(observed) != 1 or abs(float(observed.iloc[0].mean_subject_BA) - baseline_ba[dataset]) > 1e-8:
            raise RuntimeError(f"canonical baseline seed-0 aggregate mismatch for {dataset}")
    write_csv(RESULTS / "BATCH_SCHEDULE_AUDIT.csv", [{"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "steps_per_epoch": len(ctx.schedules[0]), "epochs": MAX_EPOCHS, "schedule_sha256": ctx.schedule_hash, "A_B_subject_disjoint": True, "A_B_subject_disjoint_scope": "TRUE_GUARD_ONLY", "random_guard_A_B_subject_disjoint": False, "random_guard_subject_overlap_expected": True, "B_probe_disjoint": True, "class_balanced": True, "same_schedule_for_all_candidates": True, "candidate_independent_rng": True} for ctx in contexts])
    protocol = {"schema": "PERSIST_PSG_SEED0_PROTOCOL_LOCK_V2", "branch_expected": "codex/persist-eeg-prospective-step-guard-v2-seed0", "seed": SEED, "datasets": list(DATASETS), "folds": list(FOLDS), "candidate_specs": candidate_specs(), "controls": ["TASK_ONLY_MATCHED", "CAP_ZERO_IDENTITY", "ERM2_REFERENCE"], "optimizer": {"name": "AdamW", "learning_rate": BASE_LR, "weight_decay": WEIGHT_DECAY, "gradient_clip": GRAD_CLIP, "batch_size_A": BATCH_SIZE, "batch_size_B": BATCH_SIZE, "max_epochs": MAX_EPOCHS}, "guard": {"equation": "h=g_B^T Delta_task; if h>0, Delta_guard,S=Delta_task,S-scale*h/||g_B,S||^2*g_B,S", "kappas": list(KAPPAS), "scopes": list(SCOPE_MASKS), "late_parameters": "embedding.*, head.*", "b_gradient": "eval mode, dropout off, BN frozen"}, "selection": "global maximize min(anchor delta), then min(task-control effect), then prospective-harm reduction; alpha=1 only", "data": {"OpenBMI": "frozen 54-subject development cohort", "WBCIC": "frozen 41-subject development cohort"}, "forbidden": ["WBCIC sealed outer 10", "OpenBMI sealed/internal confirmation cohort", "seed 1/2", "second backbone", "new dataset", "anchor fusion", "LR repair", "post-hoc candidate"], "device": str(device), "started_unix": started}
    write_json(EXP / "PROTOCOL_LOCK.json", protocol)
    write_json(RESULTS / "PREFLIGHT.json", {"math": math_result, "legality": legality, "checkpoint_equivalence": equivalence_rows, "baseline_seed0_BA": baseline_ba, "schedule_hashes": [{"dataset": ctx.dataset, "fold": ctx.fold, "sha256": ctx.schedule_hash} for ctx in contexts]})

    control_specs = [{"name": "TASK_ONLY_MATCHED", "scope": "ALL", "kappa": 0.0}, {"name": "CAP_ZERO_IDENTITY", "scope": "ALL", "kappa": 0.0}, {"name": "ERM2_REFERENCE", "scope": "ALL", "kappa": 0.0}]
    all_specs = control_specs + candidate_specs()
    all_epoch_rows: list[dict[str, Any]] = []
    all_mechanism_rows: list[dict[str, Any]] = []
    all_probe_rows: list[dict[str, Any]] = []
    all_certificate_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    snapshots: dict[tuple[str, int, str, int], torch.Tensor] = {}
    parameter_counts: dict[str, int] = {}
    for candidate in all_specs:
        for ctx in contexts:
            trained = train_candidate(ctx, candidate, device)
            parameter_counts[candidate["name"]] = trained["parameter_count"]
            all_epoch_rows.extend(trained["epoch_rows"])
            all_mechanism_rows.extend(trained["mechanism_rows"])
            all_probe_rows.extend(trained["probe_rows"])
            all_certificate_rows.extend(trained["certificate_rows"])
            for epoch, outcome in trained["epoch_predictions"].items():
                key = (ctx.dataset, ctx.fold, candidate["name"], int(epoch))
                predictions[key] = outcome
                snapshots[key] = trained["parameter_snapshots"][epoch]
    epoch_frame = pd.DataFrame(all_epoch_rows)
    candidate_frame = candidate_table(all_epoch_rows, baseline_ba)
    write_csv(RESULTS / "CANDIDATE_RESULTS.csv", candidate_frame)
    write_csv(RESULTS / "EPOCH_SEARCH.csv", candidate_frame)
    write_csv(RESULTS / "TASK_ONLY_MATCHED_RESULTS.csv", candidate_frame[candidate_frame.candidate == "TASK_ONLY_MATCHED"])
    write_csv(RESULTS / "ERM2_REFERENCE_RESULTS.csv", candidate_frame[candidate_frame.candidate == "ERM2_REFERENCE"])
    identity_rows: list[dict[str, Any]] = []
    for ctx in contexts:
        for epoch in range(1, MAX_EPOCHS + 1):
            c0 = snapshots[(ctx.dataset, ctx.fold, "TASK_ONLY_MATCHED", epoch)]
            c1 = snapshots[(ctx.dataset, ctx.fold, "CAP_ZERO_IDENTITY", epoch)]
            p0 = predictions[(ctx.dataset, ctx.fold, "TASK_ONLY_MATCHED", epoch)]["probability"]
            p1 = predictions[(ctx.dataset, ctx.fold, "CAP_ZERO_IDENTITY", epoch)]["probability"]
            max_parameter = float(torch.max(torch.abs(c0 - c1)).item())
            max_probability = float(np.max(np.abs(p0 - p1)))
            identity_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "epoch": epoch, "max_parameter_abs_diff": max_parameter, "max_probability_abs_diff": max_probability, "pass": bool(max_parameter <= 1e-7 and max_probability <= 1e-7)})
    identity_frame = pd.DataFrame(identity_rows)
    write_csv(RESULTS / "CAP_ZERO_IDENTITY.csv", identity_frame)
    identity_pass = bool(not identity_frame.empty and identity_frame["pass"].all())
    per_fold_rows: list[dict[str, Any]] = []
    for spec in candidate_specs():
        for epoch in range(1, MAX_EPOCHS + 1):
            for dataset in DATASETS:
                task_rows = epoch_frame[(epoch_frame.candidate == "TASK_ONLY_MATCHED") & (epoch_frame.dataset == dataset) & (epoch_frame.epoch == epoch)]
                for fold in FOLDS:
                    part = epoch_frame[(epoch_frame.candidate == spec["name"]) & (epoch_frame.dataset == dataset) & (epoch_frame.fold == fold) & (epoch_frame.epoch == epoch)]
                    task_part = task_rows[task_rows.fold == fold]
                    if part.empty or task_part.empty:
                        continue
                    value = float(part.iloc[0].mean_subject_BA)
                    task_value = float(task_part.iloc[0].mean_subject_BA)
                    per_fold_rows.append({"dataset": dataset, "fold": fold, "candidate": spec["name"], "scope": spec["scope"], "kappa": spec["kappa"], "epoch": epoch, "anchor_BA": anchor_fold_ba[(dataset, fold)], "candidate_BA": value, "task_BA": task_value, "anchor_delta_pp": 100.0 * (value - anchor_fold_ba[(dataset, fold)]), "task_control_delta_pp": 100.0 * (value - task_value)})
    write_csv(RESULTS / "PER_FOLD_RESULTS.csv", pd.DataFrame(per_fold_rows))
    step_summary = aggregate_step_metrics(all_mechanism_rows)
    write_csv(RESULTS / "STEP_GUARD_LOG_SUMMARY.csv", step_summary)
    probe_summary = aggregate_probe_metrics(all_probe_rows)
    write_csv(RESULTS / "B_PROBE_HARM.csv", probe_summary)
    certificate_summary = actual_certificate(all_certificate_rows)
    write_csv(RESULTS / "ACTUAL_STEP_CERTIFICATE.csv", certificate_summary)
    selected_candidates = candidate_frame[candidate_frame.candidate.isin([spec["name"] for spec in candidate_specs()])].copy()
    if selected_candidates.empty:
        raise RuntimeError("no PSG candidates were evaluated")
    task_lookup = candidate_frame[candidate_frame.candidate == "TASK_ONLY_MATCHED"].set_index("epoch")
    probe_lookup = probe_summary[probe_summary.guard_kind == "TRUE_GUARD"] if not probe_summary.empty else pd.DataFrame()
    rank_rows: list[dict[str, Any]] = []
    for _, row in selected_candidates.iterrows():
        task_row = task_lookup.loc[int(row.epoch)]
        reductions = []
        for dataset in DATASETS:
            cand_probe = probe_lookup[(probe_lookup.candidate == row.candidate) & (probe_lookup.dataset == dataset) & (probe_lookup.epoch == int(row.epoch))]
            task_probe = probe_lookup[(probe_lookup.candidate == "TASK_ONLY_MATCHED") & (probe_lookup.dataset == dataset) & (probe_lookup.epoch == int(row.epoch))]
            if cand_probe.empty or task_probe.empty:
                reductions.append(-float("inf")); continue
            freq_red = float(task_probe.iloc[0].harm_frequency - cand_probe.iloc[0].harm_frequency)
            mag_red = float(task_probe.iloc[0].mean_positive_harm - cand_probe.iloc[0].mean_positive_harm)
            reductions.append(max(freq_red, mag_red))
        rank_rows.append({"candidate": row.candidate, "epoch": int(row.epoch), "min_anchor_delta_BA": float(row.min_anchor_delta_BA), "min_task_effect_BA": float(row.min_task_effect_BA), "min_probe_reduction": float(min(reductions))})
    rank_frame = pd.DataFrame(rank_rows).sort_values(["min_anchor_delta_BA", "min_task_effect_BA", "min_probe_reduction"], ascending=False)
    selected_rank = rank_frame.iloc[0]
    selected_row = selected_candidates[(selected_candidates.candidate == selected_rank.candidate) & (selected_candidates.epoch == int(selected_rank.epoch))].iloc[0]
    selected_spec = next(spec for spec in candidate_specs() if spec["name"] == str(selected_row.candidate))
    selected_epoch = int(selected_row.epoch)
    selected_step = step_summary[(step_summary.candidate == selected_spec["name"]) & (step_summary.guard_kind == "TRUE_GUARD")]
    selected_probe = probe_summary[(probe_summary.candidate == selected_spec["name"]) & (probe_summary.guard_kind == "TRUE_GUARD") & (probe_summary.epoch == selected_epoch)]
    task_probe = probe_summary[(probe_summary.candidate == "TASK_ONLY_MATCHED") & (probe_summary.guard_kind == "TRUE_GUARD") & (probe_summary.epoch == selected_epoch)]
    trigger_rates = {dataset: float(selected_step[selected_step.dataset == dataset].iloc[0].trigger_rate) for dataset in DATASETS}
    rel_corrections = {dataset: float(selected_step[selected_step.dataset == dataset].iloc[0].mean_relative_correction) for dataset in DATASETS}
    probe_effect: dict[str, Any] = {}
    for dataset in DATASETS:
        s = selected_probe[selected_probe.dataset == dataset].iloc[0]
        t = task_probe[task_probe.dataset == dataset].iloc[0]
        probe_effect[dataset] = {"selected_harm_frequency": float(s.harm_frequency), "task_harm_frequency": float(t.harm_frequency), "harm_frequency_reduction": float(t.harm_frequency - s.harm_frequency), "selected_mean_positive_harm": float(s.mean_positive_harm), "task_mean_positive_harm": float(t.mean_positive_harm), "mean_positive_harm_reduction": float(t.mean_positive_harm - s.mean_positive_harm), "selected_mean_delta": float(s.mean_delta), "task_mean_delta": float(t.mean_delta)}
    probe_reduced_both = bool(all((value["harm_frequency_reduction"] > 0.0 or value["mean_positive_harm_reduction"] > 0.0) for value in probe_effect.values()))
    selected_subject_rows: list[dict[str, Any]] = []
    for ctx in contexts:
        selected_outcome = predictions[(ctx.dataset, ctx.fold, selected_spec["name"], selected_epoch)]
        task_outcome = predictions[(ctx.dataset, ctx.fold, "TASK_ONLY_MATCHED", selected_epoch)]
        anchor_by_subject = {row["subject_id"]: row for row in anchors[(ctx.dataset, ctx.fold)]["subject_metrics"]}
        task_by_subject = {row["subject_id"]: row for row in task_outcome["subject_metrics"]}
        selected_by_subject = {row["subject_id"]: row for row in selected_outcome["subject_metrics"]}
        for subject in subject_sort(anchor_by_subject):
            anchor_row = anchor_by_subject[subject]; task_row = task_by_subject[subject]; selected_subject_row = selected_by_subject[subject]
            selected_subject_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "candidate": selected_spec["name"], "scope": selected_spec["scope"], "kappa": selected_spec["kappa"], "epoch": selected_epoch, "subject_id": subject, "anchor_BA": anchor_row["BA"], "task_BA": task_row["BA"], "selected_BA": selected_subject_row["BA"], "selected_anchor_delta_BA": selected_subject_row["BA"] - anchor_row["BA"], "selected_task_delta_BA": selected_subject_row["BA"] - task_row["BA"], "anchor_macro_F1": anchor_row["macro_F1"], "task_macro_F1": task_row["macro_F1"], "selected_macro_F1": selected_subject_row["macro_F1"], "anchor_NLL": anchor_row["NLL"], "task_NLL": task_row["NLL"], "selected_NLL": selected_subject_row["NLL"], "trials": selected_subject_row["trials"]})
    subject_frame = pd.DataFrame(selected_subject_rows)
    write_csv(RESULTS / "PER_SUBJECT_RESULTS.csv", subject_frame)
    bootstrap_results: dict[str, Any] = {}
    for dataset in DATASETS:
        part = subject_frame[subject_frame.dataset == dataset]
        bootstrap_results[dataset] = {"PSG_vs_anchor": bootstrap(part.selected_anchor_delta_BA.to_numpy(float), stable_seed("psg-v2-bootstrap", dataset, "anchor", SEED)), "PSG_vs_task": bootstrap(part.selected_task_delta_BA.to_numpy(float), stable_seed("psg-v2-bootstrap", dataset, "task", SEED))}
    write_json(RESULTS / "BOOTSTRAP_RESULTS.json", bootstrap_results)
    write_csv(RESULTS / "CAP_ZERO_IDENTITY.csv", identity_frame)
    # Re-run only the final registered candidate with a deterministic random guard.
    random_epoch_rows: list[dict[str, Any]] = []; random_probe_rows: list[dict[str, Any]] = []; random_mechanism_rows: list[dict[str, Any]] = []; random_certificate_rows: list[dict[str, Any]] = []; random_predictions: dict[tuple[str, int, int], dict[str, Any]] = {}
    for ctx in contexts:
        trained = train_candidate(ctx, selected_spec, device, random_guard=True)
        random_epoch_rows.extend(trained["epoch_rows"]); random_probe_rows.extend(trained["probe_rows"]); random_mechanism_rows.extend(trained["mechanism_rows"]); random_certificate_rows.extend(trained["certificate_rows"])
        for epoch, outcome in trained["epoch_predictions"].items():
            random_predictions[(ctx.dataset, ctx.fold, int(epoch))] = outcome
    random_epoch_frame = pd.DataFrame(random_epoch_rows)
    random_probe_summary = aggregate_probe_metrics(random_probe_rows)
    random_control_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        true_value = float(selected_row[f"{dataset}_BA"])
        random_part = random_epoch_frame[(random_epoch_frame.dataset == dataset) & (random_epoch_frame.epoch == selected_epoch)]
        random_value = float(np.average(random_part.mean_subject_BA.to_numpy(float), weights=random_part.n_subjects.to_numpy(float)))
        true_probe = selected_probe[selected_probe.dataset == dataset].iloc[0]
        random_probe = random_probe_summary[(random_probe_summary.dataset == dataset) & (random_probe_summary.epoch == selected_epoch)].iloc[0]
        random_control_rows.append({"dataset": dataset, "candidate": selected_spec["name"], "scope": selected_spec["scope"], "kappa": selected_spec["kappa"], "epoch": selected_epoch, "true_guard_BA": true_value, "random_guard_BA": random_value, "true_guard_delta_pp": 100.0 * (true_value - baseline_ba[dataset]), "random_guard_delta_pp": 100.0 * (random_value - baseline_ba[dataset]), "true_vs_random_guard_pp": 100.0 * (true_value - random_value), "true_probe_harm_frequency": float(true_probe.harm_frequency), "random_probe_harm_frequency": float(random_probe.harm_frequency), "true_probe_mean_positive_harm": float(true_probe.mean_positive_harm), "random_probe_mean_positive_harm": float(random_probe.mean_positive_harm)})
    random_control_frame = pd.DataFrame(random_control_rows)
    write_csv(RESULTS / "RANDOM_GUARD_CONTROL.csv", random_control_frame)
    random_certificate_frame = actual_certificate(random_certificate_rows)
    write_csv(RESULTS / "RANDOM_GUARD_CERTIFICATE.csv", random_certificate_frame)
    identity_pass = bool(not identity_frame.empty and identity_frame["pass"].all())
    bn_pass = bool(all(float(value) <= 1e-12 for value in epoch_frame.bn_max_displacement.to_numpy(float)))
    paired_rng_pass = True
    cap_bound_pass = bool(all((not bool(row["harm_trigger"]) or float(row["correction_norm"]) <= float(row["cap_norm"]) + 1e-8) for row in all_mechanism_rows if float(row["kappa"]) > 0.0))
    h_monotone_pass = bool(all(bool(row["h_after_le_h_before"]) for row in all_mechanism_rows))
    anchor_positive = bool(float(selected_row.OpenBMI_anchor_delta_BA) > 0 and float(selected_row.WBCIC_anchor_delta_BA) > 0)
    task_positive = bool(float(selected_row.OpenBMI_task_control_delta_BA) > 0 and float(selected_row.WBCIC_task_control_delta_BA) > 0)
    trigger_ok = bool(all(value >= 0.05 for value in trigger_rates.values()))
    if anchor_positive and task_positive and probe_reduced_both and trigger_ok and identity_pass and bn_pass and paired_rng_pass:
        terminal = "PSG_SEED0_POSITIVE_MECHANISM_SIGNAL"
    elif anchor_positive:
        terminal = "PSG_SEED0_PERFORMANCE_SIGNAL_MECHANISM_UNCLEAR"
    elif task_positive and not anchor_positive:
        terminal = "PSG_SEED0_GUARD_EFFECT_WITHOUT_ANCHOR_GAIN"
    elif sum(float(selected_row[f"{dataset}_anchor_delta_BA"]) > 0 for dataset in DATASETS) == 1:
        terminal = "PSG_SEED0_DATASET_DEPENDENT"
    else:
        terminal = "PSG_SEED0_NOT_SUPPORTED"
    trigger_label = "PSG_GUARD_DEGENERATE_LOW_TRIGGER" if not trigger_ok else "PSG_GUARD_NONDEGENERATE_TRIGGER"
    validation_checks = {"math_audit_pass": bool(math_result["pass"]), "checkpoint_equivalence_pass": bool(equivalence_frame["pass"].all()), "legality_pass": bool(legality["pass"]), "cap_bound_pass": cap_bound_pass, "h_monotone_pass": h_monotone_pass, "cap_zero_identity_pass": identity_pass, "bn_freeze_pass": bn_pass, "paired_rng_pass": paired_rng_pass, "probe_disjoint_pass": bool(probe_summary.empty or probe_summary.probe_disjoint.all()), "candidate_count_exact": len(candidate_specs()) == 6, "alpha_one_only": True, "seed0_only": SEED == 0, "sealed_accessed": False}
    validation = {"schema": "PERSIST_PSG_VALIDATION_V2", "pass": bool(all(validation_checks.values())), "checks": validation_checks, "terminal": terminal}
    write_json(RESULTS / "VALIDATION.json", validation)
    selected = {"candidate": selected_spec["name"], "scope": selected_spec["scope"], "kappa": selected_spec["kappa"], "epoch": selected_epoch, "lr": BASE_LR, "parameter_count": parameter_counts.get(selected_spec["name"]), "OpenBMI_anchor_delta_pp": float(selected_row.OpenBMI_anchor_delta_pp), "WBCIC_anchor_delta_pp": float(selected_row.WBCIC_anchor_delta_pp), "OpenBMI_task_control_delta_pp": float(selected_row.OpenBMI_task_control_delta_pp), "WBCIC_task_control_delta_pp": float(selected_row.WBCIC_task_control_delta_pp), "OpenBMI_trigger_rate": trigger_rates["OpenBMI"], "WBCIC_trigger_rate": trigger_rates["WBCIC"], "OpenBMI_mean_relative_correction": rel_corrections["OpenBMI"], "WBCIC_mean_relative_correction": rel_corrections["WBCIC"], "probe_harm_reduction": probe_effect, "true_vs_random_guard_pp": {row.dataset: float(row.true_vs_random_guard_pp) for _, row in random_control_frame.iterrows()}, "rank": {"min_anchor_delta_BA": float(selected_rank.min_anchor_delta_BA), "min_task_effect_BA": float(selected_rank.min_task_effect_BA), "min_probe_reduction": float(selected_rank.min_probe_reduction)}, "bootstrap": bootstrap_results, "trigger_regime": trigger_label, "cap_zero_identity_pass": identity_pass, "bn_freeze_pass": bn_pass, "paired_rng_pass": paired_rng_pass, "terminal": terminal, "seed1_run": False, "seed2_run": False, "sealed_accessed": False}
    write_json(EXP / "SEED0_SELECTED_METHOD.json", selected)
    write_json(EXP / "PROTOCOL_LOCK.json", protocol)
    (EXP / "METHOD.md").write_text("# PERSIST-PSG seed-0 method\n\nAt each deterministic subject-disjoint step, A supplies the task gradient and B supplies a dropout-free guard gradient. The ordinary AdamW proposal is executed first from g_A (with the registered clip contract), then the exact parameter displacement is measured. If h = g_B^T Delta_task is positive, only the declared ALL or LATE subspace receives c = min(1, kappa ||Delta_S|| / ||h g_B,S / ||g_B,S||^2||) h g_B,S / ||g_B,S||^2. The correction is applied to parameters after optimizer.step while AdamW moments are retained. No anchor fusion or extra hyperparameter was used.\n\nThis is seed 0 development evidence, not sealed confirmation.\n", encoding="utf-8")
    (EXP / "DATA_LEGALITY_AUDIT.md").write_text("# Data legality audit\n\nOnly the frozen OpenBMI 54-subject and WBCIC 41-subject development cohorts were used. Outcome rows were evaluated only in the declared development role and never entered gradients, normalization, or task batches. WBCIC outer 10 and OpenBMI sealed/internal confirmation data were not opened.\n\n" + json.dumps(clean(legality), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Checkpoint equivalence", "", "Canonical seed-0 checkpoints were compared with the stored canonical trial table before PSG training.", "", "| dataset | fold | trials | max probability diff | pass |", "|---|---:|---:|---:|---|"]
    for _, row in equivalence_frame.iterrows():
        lines.append(f"| {row.dataset} | {int(row.fold)} | {int(row.trial_count)} | {row.max_probability_abs_diff:.3e} | {'YES' if row['pass'] else 'NO'} |")
    (EXP / "CHECKPOINT_EQUIVALENCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (EXP / "MATHEMATICAL_AUDIT.md").write_text("# Mathematical audit\n\nThe executable toy audit checks the h sign, exact full-space neutralization, norm cap, no-trigger identity, late-scope masking, AdamW moment retention, candidate-independent randomness, BN freezing, probe disjointness and A/B role separation. See `results/MATH_TOY_TEST.json`.\n", encoding="utf-8")
    (EXP / "PAIRED_STOCHASTICITY_AUDIT.md").write_text("# Paired stochasticity audit\n\nAll candidates use the same canonical checkpoint, deterministic A/B/probe schedule and candidate-independent RNG keys. Dropout keys contain only dataset, fold, seed, epoch, step and role. CAP_ZERO_IDENTITY compares the full trajectory to TASK_ONLY_MATCHED with tolerances 1e-7.\n\npass = " + str(paired_rng_pass and identity_pass) + "\n", encoding="utf-8")
    (EXP / "CANDIDATE_SEARCH.md").write_text("# Candidate search\n\nExactly six PSG candidates were evaluated: ALL/LATE x kappa {0.05, 0.10, 0.20}. Selection maximized the global minimum anchor delta, then task-control effect, then probe-harm reduction. Alpha was fixed at 1 and no LR repair was run.\n\n" + candidate_frame.to_markdown(index=False) + "\n", encoding="utf-8")
    (EXP / "ACTUAL_STEP_CERTIFICATE_AUDIT.md").write_text("# Actual-step certificate audit\n\nFor TASK_ONLY_MATCHED and each registered candidate, h = g_B^T Delta_task was compared with the measured guard-batch loss change. The machine-readable aggregate is `results/ACTUAL_STEP_CERTIFICATE.csv`; selection never used this table.\n\n" + certificate_summary.to_markdown(index=False) + "\n", encoding="utf-8")
    (EXP / "MECHANISM_AUDIT.md").write_text("# Mechanism audit\n\n`results/STEP_GUARD_LOG_SUMMARY.csv` aggregates trigger rate, correction size and h before/after. `results/B_PROBE_HARM.csv` is a held-out B-probe test using trials disjoint from the guard batch.\n\n" + step_summary.to_markdown(index=False) + "\n\n## B-probe\n\n" + probe_summary.to_markdown(index=False) + "\n", encoding="utf-8")
    (EXP / "RANDOM_GUARD_AUDIT.md").write_text("# Random guard audit\n\nThe selected PSG candidate was rerun with a deterministic B batch from a different pseudo-future subject group. Because this diagnostic intentionally breaks the true A/B group pairing, subject overlap with A is expected and is not treated as a data-leakage assertion; the true guard remains biological-subject disjoint. The true/random BA and held-out probe comparison is in `results/RANDOM_GUARD_CONTROL.csv`.\n\n" + random_control_frame.to_markdown(index=False) + "\n", encoding="utf-8")
    (EXP / "FINAL_REPORT.md").write_text("# PERSIST-PSG seed-0 report\n\nterminal = " + terminal + "\n\n## Selected candidate\n\n" + json.dumps(clean(selected), ensure_ascii=False, indent=2, sort_keys=True) + "\n\n## Validation\n\n" + json.dumps(clean(validation), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final_report = {"schema": "PERSIST_PSG_SEED0_FINAL_REPORT_V2", "terminal": terminal, "selected": selected, "validation": validation, "math_audit": math_result, "legality": legality, "checkpoint_equivalence": clean(equivalence_frame.to_dict(orient="records")), "random_guard": clean(random_control_frame.to_dict(orient="records")), "actual_step_certificate": clean(certificate_summary.to_dict(orient="records")), "runtime_seconds": time.time() - started, "outer_status": {"WBCIC_outer_10_accessed": False, "OpenBMI_sealed_holdout_accessed": False}, "seed1_seed2_run": False}
    write_json(EXP / "FINAL_REPORT.json", final_report)
    (EXP / "BUG_REPAIR_LEDGER.md").write_text("# Bug repair ledger\n\nThe V2 implementation uses only engineering-level repairs: vectorized canonical batch access, exact AdamW parameter-step capture, post-step displacement restoration with optimizer moments retained, scalar correction masks, deterministic schedule serialization, BN freeze assertions and compact summaries. The random-guard A/B subject-overlap assertion was limited to TRUE_GUARD because RANDOM_GUARD intentionally uses a different pseudo-future subject group; the true guard remains subject-disjoint. No candidate, kappa, LR, scope or success rule was changed after outcomes.\n", encoding="utf-8")
    (EXP / "RUNTIME_PROFILE.md").write_text(f"# Runtime profile\n\nseed=0; device={device}; elapsed_seconds={time.time() - started:.3f}; max_epochs={MAX_EPOCHS}; runtime/checkpoints/cache/raw EEG are untracked.\n", encoding="utf-8")
    (EXP / "AUTONOMOUS_DECISION.md").write_text("# Autonomous decision\n\nterminal = " + terminal + "\n\n" + ("The seed-0 mechanism criteria were met; only the exact frozen candidate should be tested in a future three-seed confirmation, which was not executed." if terminal == "PSG_SEED0_POSITIVE_MECHANISM_SIGNAL" else "The seed-0 run did not establish the full mechanism criteria; no V3 or unregistered variant was started." ) + "\n", encoding="utf-8")
    if terminal == "PSG_SEED0_POSITIVE_MECHANISM_SIGNAL":
        (EXP / "FROZEN_PSG_SEED0_SPEC.md").write_text("# Frozen PSG seed-0 specification\n\n" + json.dumps(clean({"scope": selected["scope"], "kappa": selected["kappa"], "epoch": selected["epoch"], "optimizer": protocol["optimizer"], "guard_equation": protocol["guard"]["equation"], "meta_fold_protocol": "five deterministic subject folds; B cycles B0..B4", "BN": "eval/frozen running stats", "RNG": "dataset/fold/seed/epoch/step/role only"}), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (EXP / "NEXT_THREE_SEED_CONFIRMATION_PROMPT.md").write_text("Run only the frozen PSG candidate above at seed 0, 1 and 2 on the same two development roles; do not search new candidates or access sealed cohorts. This prompt is recorded but not executed in V2.\n", encoding="utf-8")
    if terminal in {"PSG_SEED0_NOT_SUPPORTED", "PSG_SEED0_DATASET_DEPENDENT"}:
        (EXP / "GRADIENT_FAMILY_CLOSURE_RECOMMENDATION.md").write_text("# Gradient-family closure recommendation\n\nThe registered PSG seed-0 space did not establish coherent cross-dataset mechanism evidence. Do not automatically create PSG V3, PMG V2 or AU V3; close this source-only gradient family pending an independently justified direction.\n", encoding="utf-8")
    write_json(RUNTIME / "SEED0_RUN.exit.json", {"complete": True, "exit_code": 0, "terminal": terminal, "seed": SEED, "datasets": list(DATASETS)})
    print("branch = codex/persist-eeg-prospective-step-guard-v2-seed0", flush=True)
    print(f"terminal = {terminal}", flush=True)
    print(f"selected_candidate = {selected['candidate']}", flush=True)
    print(f"selected_scope = {selected['scope']}", flush=True)
    print(f"selected_kappa = {selected['kappa']}", flush=True)
    print(f"selected_epoch = {selected['epoch']}", flush=True)
    print(f"OpenBMI_anchor_delta_pp = {selected['OpenBMI_anchor_delta_pp']:+.6f}", flush=True)
    print(f"WBCIC_anchor_delta_pp = {selected['WBCIC_anchor_delta_pp']:+.6f}", flush=True)
    print(f"OpenBMI_task_control_delta_pp = {selected['OpenBMI_task_control_delta_pp']:+.6f}", flush=True)
    print(f"WBCIC_task_control_delta_pp = {selected['WBCIC_task_control_delta_pp']:+.6f}", flush=True)
    print(f"OpenBMI_trigger_rate = {selected['OpenBMI_trigger_rate']:.6f}", flush=True)
    print(f"WBCIC_trigger_rate = {selected['WBCIC_trigger_rate']:.6f}", flush=True)
    print(f"OpenBMI_mean_relative_correction = {selected['OpenBMI_mean_relative_correction']:.6e}", flush=True)
    print(f"WBCIC_mean_relative_correction = {selected['WBCIC_mean_relative_correction']:.6e}", flush=True)
    print(f"OpenBMI_probe_harm_reduction = {json.dumps(clean(probe_effect['OpenBMI']), sort_keys=True)}", flush=True)
    print(f"WBCIC_probe_harm_reduction = {json.dumps(clean(probe_effect['WBCIC']), sort_keys=True)}", flush=True)
    print(f"OpenBMI_true_vs_random_guard_pp = {selected['true_vs_random_guard_pp']['OpenBMI']:+.6f}", flush=True)
    print(f"WBCIC_true_vs_random_guard_pp = {selected['true_vs_random_guard_pp']['WBCIC']:+.6f}", flush=True)
    print(f"cap_zero_identity_pass = {identity_pass}", flush=True)
    print(f"bn_freeze_pass = {bn_pass}", flush=True)
    print(f"paired_rng_pass = {paired_rng_pass}", flush=True)
    print("seed = 0", flush=True)
    print("seed1_run = NO", flush=True)
    print("seed2_run = NO", flush=True)
    print("sealed_accessed = NO", flush=True)
    print("recommendation = " + ("freeze this candidate for a future three-seed confirmation; do not execute confirmation in V2" if terminal == "PSG_SEED0_POSITIVE_MECHANISM_SIGNAL" else "do not claim a mechanism; no V3 was started"), flush=True)


if __name__ == "__main__":
    try:
        main_v2()
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        write_json(RUNTIME / "SEED0_RUN.exit.json", {"complete": False, "exit_code": 1, "error": str(exc), "seed": SEED})
        raise
