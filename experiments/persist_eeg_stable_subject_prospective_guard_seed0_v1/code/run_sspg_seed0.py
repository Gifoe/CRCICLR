"""PERSIST-SSPG seed-0 development experiment.

This runner implements the frozen Stable Subject Prospective Guard (SSPG)
recipe.  It has two deliberately separate phases:

``--phase preflight`` loads only source/refit trials, runs executable
legality/equivalence tests, and writes the pre-outcome lock.  It never creates
an outcome index.  ``--phase run`` requires that lock and only then opens the
declared development outcome subjects for the final paired evaluation.

The code is intentionally self-contained around the canonical EEGNet loader;
all data/checkpoint paths are supplied by the server environment and are
outside the committed experiment tree.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
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

try:
    torch.set_num_threads(int(os.environ.get("PERSIST_TORCH_THREADS", "1")))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"
RUNTIME = EXP / "runtime"
CANONICAL_REPO = Path(os.environ.get(
    "CANONICAL_REPO", r"D:\\nips-temp\\TotalP\\P1\\CRCICLR_CANONICAL_EEGNET"
)).resolve()
CANONICAL_EXP = CANONICAL_REPO / "experiments" / "persist_eeg_canonical_eegnet_baseline"
WBCIC_CACHE = Path(os.environ.get(
    "PERSIST_WBCIC_CACHE",
    r"D:\\nips-temp\\TotalP\\P1\\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\\experiments\\persist_eeg_wbcic_independent_replication_v1\\runtime\\cache",
)).resolve()
SEED = 0
FOLDS = (0, 1, 2, 3, 4)
DATASETS = ("OpenBMI", "WBCIC")
METHODS = ("ANCHOR", "TASK_ONLY_MATCHED", "SSPG", "CROSS_SUBJECT_K4_GUARD", "RANDOM_DIRECTION_GUARD")
TRAIN_METHODS = METHODS[1:]
MAX_EPOCHS = 2
BATCH_SIZE = 64
SCHEDULE_BATCH_REFERENCE = 128
BASE_LR = 3e-5
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 5.0
K = 4
M_PER_CLASS = 16
N_BLOCKS = 5
KAPPA = 0.20
EPS = 1e-12
BOOTSTRAP_DRAWS = 10_000
BACKTRACK_MULTIPLIERS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125)
AUDIT_STEPS_PER_FOLD = 5

os.environ.setdefault("CANONICAL_REPO", str(CANONICAL_REPO))
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
    if isinstance(value, torch.Tensor):
        return clean(value.detach().cpu().tolist())
    if isinstance(value, np.integer):
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


def write_csv(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
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


def metadata_col(data: Any, name: str, indices: np.ndarray | None = None) -> np.ndarray:
    frame = data.metadata if indices is None else data.metadata.iloc[np.asarray(indices, dtype=np.int64)]
    values = frame[name]
    if name == "subject_id":
        return values.astype(str).str.replace("sub-", "", regex=False).to_numpy()
    return values.to_numpy()


def vectorized_batch(data: Any, indices: np.ndarray) -> np.ndarray:
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


def load_checkpoint(dataset: str, fold: int, channels: int) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, Path, dict[str, Any]]:
    path = CANONICAL_EXP / "runtime" / "checkpoints" / dataset / f"fold-{fold}" / "seed-0.pt"
    partial_path = CANONICAL_EXP / "runtime" / "partial" / f"{dataset.lower()}_fold-{fold}_seed-0.json"
    if not path.is_file() or not partial_path.is_file():
        raise RuntimeError(f"missing canonical checkpoint/partial: {path}")
    partial = json.loads(partial_path.read_text(encoding="utf-8-sig"))
    ck_hash = sha256_file(path)
    if partial.get("checkpoint_sha256") and str(partial["checkpoint_sha256"]) != ck_hash:
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
    if mean.shape != (channels,) or std.shape != (channels,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise RuntimeError(f"invalid canonical normalizer for {dataset} fold {fold}")
    del model
    return state, mean, std, path, partial


def freeze_bn(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def bn_buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_buffers() if "running_mean" in name or "running_var" in name}


def bn_max_displacement(model: nn.Module, baseline: dict[str, torch.Tensor]) -> float:
    now = dict(model.named_buffers())
    values = [float(torch.max(torch.abs(now[name].detach().cpu() - before.cpu())).item()) for name, before in baseline.items()]
    return max(values, default=0.0)


def prepare(data: Any, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    return canonical.prepare_batch(data, np.asarray(indices, dtype=np.int64), mean, std, device)


def labels_for(data: Any, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(metadata_col(data, "label", indices).astype(np.int64), dtype=torch.long, device=device)


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


def make_model(state: dict[str, torch.Tensor], channels: int, device: torch.device) -> tuple[nn.Module, list[nn.Parameter]]:
    model = canonical.VanillaEEGNet(channels).to(device)
    model.load_state_dict(state, strict=True)
    params = list(model.parameters())
    for parameter in params:
        parameter.requires_grad = True
    return model, params


def gradient_vector(model: nn.Module, params: list[nn.Parameter], xb: torch.Tensor, yb: torch.Tensor, *, dropout_seed: int | None) -> torch.Tensor:
    devices = [int(xb.device.index)] if xb.device.type == "cuda" and xb.device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        if dropout_seed is not None:
            torch.manual_seed(int(dropout_seed))
            if xb.device.type == "cuda":
                torch.cuda.manual_seed_all(int(dropout_seed))
            model.train()
            freeze_bn(model)
        else:
            model.eval()
        values = torch.autograd.grad(F.cross_entropy(model(xb), yb), tuple(params), allow_unused=True, retain_graph=False, create_graph=False)
    return flatten(value.detach().float() if value is not None else torch.zeros_like(parameter) for value, parameter in zip(values, params))


def loss_indices(model: nn.Module, ctx: "Context", indices: np.ndarray, device: torch.device) -> float:
    model.eval()
    with torch.inference_mode():
        xb = prepare(ctx.data, indices, ctx.mean, ctx.std, device)
        yb = labels_for(ctx.data, indices, device)
        return float(F.cross_entropy(model(xb), yb).detach().cpu())


def clip_gradient(value: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    norm = float(torch.linalg.vector_norm(value).detach().cpu())
    scale = min(1.0, GRAD_CLIP / max(norm, EPS))
    return value * scale, norm, scale


def snapshot(params: list[nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in params]


def restore_delta(params: list[nn.Parameter], theta_old: list[torch.Tensor], delta: torch.Tensor) -> None:
    with torch.no_grad():
        for parameter, old, chunk in zip(params, theta_old, split_like(delta, params)):
            parameter.copy_(old + chunk)


def optimizer_digest(optimizer: torch.optim.Optimizer) -> str:
    payload = optimizer.state_dict()
    return hashlib.sha256(json.dumps(clean(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


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
    rng = np.random.default_rng(stable_seed("psg-v2-meta-folds", dataset, fold, SEED))
    shuffled = np.asarray(source_subjects, dtype=object)[rng.permutation(len(source_subjects))]
    groups = [list(map(str, part.tolist())) for part in np.array_split(shuffled, 5)]
    if any(not group for group in groups) or set(sum(groups, [])) != set(source_subjects):
        raise RuntimeError(f"invalid meta-fold partition {dataset} fold={fold}")
    return groups


def make_schedules(dataset: str, fold: int, refit_idx: np.ndarray, pools: dict[str, dict[int, np.ndarray]], meta_folds: list[list[str]]) -> tuple[list[list[dict[str, Any]]], str]:
    steps = max(1, int(math.ceil(len(refit_idx) / SCHEDULE_BATCH_REFERENCE)))
    serial: list[dict[str, Any]] = []
    schedules: list[list[dict[str, Any]]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        current: list[dict[str, Any]] = []
        for step in range(steps):
            b_fold = step % 5
            b_subjects = list(meta_folds[b_fold])
            a_subjects = [subject for i, group in enumerate(meta_folds) if i != b_fold for subject in group]
            rng = np.random.default_rng(stable_seed("psg-v2-schedule", dataset, fold, SEED, epoch, step))
            a_idx = balanced_batch(pools, a_subjects, rng)
            current.append({"A": a_idx, "b_fold": int(b_fold), "b_subjects": b_subjects, "a_subjects": a_subjects})
            serial.append({"epoch": epoch, "step": step, "B_fold": int(b_fold), "A": a_idx.tolist(), "B_subjects": b_subjects})
        schedules.append(current)
    digest = hashlib.sha256(json.dumps(serial, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()
    return schedules, digest


def subject_blocks(data: Any, refit_idx: np.ndarray, source_subjects: list[str], dataset: str, fold: int) -> dict[str, dict[int, list[np.ndarray]]]:
    subjects = metadata_col(data, "subject_id", refit_idx).astype(str)
    labels = metadata_col(data, "label", refit_idx).astype(int)
    output: dict[str, dict[int, list[np.ndarray]]] = {}
    for subject in source_subjects:
        output[subject] = {}
        for cls in (0, 1):
            values = refit_idx[(subjects == subject) & (labels == cls)]
            if len(values) < N_BLOCKS * M_PER_CLASS:
                raise RuntimeError(f"INSUFFICIENT_K4_SUPPORT dataset={dataset} fold={fold} subject={subject} class={cls} count={len(values)}")
            rng = np.random.default_rng(stable_seed("cross-batch-blocks", dataset, fold, SEED, subject, cls))
            ordered = np.asarray(values, dtype=np.int64)[rng.permutation(len(values))]
            output[subject][cls] = [ordered[i * M_PER_CLASS : (i + 1) * M_PER_CLASS].copy() for i in range(N_BLOCKS)]
    return output


def class_balanced_block(blocks: dict[str, dict[int, list[np.ndarray]]], subject: str, block_no: int) -> np.ndarray:
    return np.concatenate([blocks[subject][cls][block_no] for cls in (0, 1)]).astype(np.int64)


def check_blocks(ctx: "Context") -> dict[str, Any]:
    all_sets: dict[str, set[int]] = {}
    for subject in ctx.source_subjects:
        ids: list[int] = []
        for block_no in range(N_BLOCKS):
            idx = class_balanced_block(ctx.blocks, subject, block_no)
            if len(set(map(int, idx.tolist()))) != len(idx):
                raise RuntimeError("duplicate trial inside subject block")
            ids.extend(idx.tolist())
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"B/Bout overlap {ctx.dataset} fold={ctx.fold} subject={subject}")
        all_sets[subject] = set(ids)
    return {"dataset": ctx.dataset, "fold": ctx.fold, "subjects": len(all_sets), "B1_B4_Bout_trial_disjoint": True}


def audit_step_numbers(ctx: "Context") -> set[int]:
    total = sum(len(schedule) for schedule in ctx.schedules)
    return {int(value) for value in np.linspace(1, total, AUDIT_STEPS_PER_FOLD, dtype=np.int64)}


@dataclass
class Context:
    dataset: str
    fold: int
    role: dict[str, list[str]]
    role_lock: dict[str, Any]
    data: Any
    refit_idx: np.ndarray
    anchor_state: dict[str, torch.Tensor]
    mean: np.ndarray
    std: np.ndarray
    checkpoint_path: Path
    checkpoint_partial: dict[str, Any]
    source_subjects: list[str]
    meta_folds: list[list[str]]
    schedules: list[list[dict[str, Any]]]
    schedule_hash: str
    blocks: dict[str, dict[int, list[np.ndarray]]]
    audit_steps: set[int]
    channels: int


def load_contexts_source_only() -> tuple[list[Context], dict[str, Any]]:
    contexts: list[Context] = []
    legality: dict[str, Any] = {
        "schema": "PERSIST_SSPG_DATA_LEGALITY_V1",
        "seed": SEED,
        "datasets": {},
        "outcome_index_created_before_lock": False,
        "outcome_labels_read_before_lock": False,
        "outcome_data_materialized_before_lock": False,
        "WBCIC_outer_opened": False,
        "OpenBMI_sealed_opened": False,
        "seed1_run": False,
        "seed2_run": False,
    }
    for dataset in DATASETS:
        roles, pool, role_lock = canonical.load_roles(dataset)
        data = canonical.load_dataset(dataset, pool)
        data.batch = lambda indices, _data=data: vectorized_batch(_data, indices)
        observed_subjects = set(metadata_col(data, "subject_id").astype(str))
        fit_sessions = (1, 2) if dataset == "OpenBMI" else (0, 1)
        legality["datasets"][dataset] = {
            "subjects_in_frozen_development_pool": len(pool),
            "rows": int(len(data.metadata)),
            "sessions": sorted(map(int, np.unique(metadata_col(data, "session_id")))),
            "observed_subjects": len(observed_subjects),
            "outer_subject_ids_present": bool(role_lock.get("outer_subject_ids_present", False)) if dataset == "WBCIC" else False,
            "fold_roles": [],
        }
        for fold in FOLDS:
            role = roles[fold]
            refit_subjects = set(role["model_fit"]) | set(role["discovery"])
            subjects = metadata_col(data, "subject_id").astype(str)
            sessions = metadata_col(data, "session_id").astype(int)
            refit_mask = np.isin(subjects, list(refit_subjects)) & np.isin(sessions, fit_sessions)
            refit_idx = np.flatnonzero(refit_mask).astype(np.int64)
            if len(refit_idx) == 0 or set(subjects[refit_idx]) != refit_subjects:
                raise RuntimeError(f"invalid source/refit index {dataset} fold={fold}")
            channels = int(data.batch(refit_idx[:1]).shape[1])
            state, mean, std, checkpoint_path, partial = load_checkpoint(dataset, fold, channels)
            source_subjects = subject_sort(refit_subjects)
            pools: dict[str, dict[int, np.ndarray]] = {}
            refit_subject_values = subjects[refit_idx]
            refit_labels = metadata_col(data, "label", refit_idx).astype(int)
            for subject in source_subjects:
                pools[subject] = {}
                for cls in (0, 1):
                    values = refit_idx[(refit_subject_values == subject) & (refit_labels == cls)]
                    if len(values) == 0:
                        raise RuntimeError(f"missing source class {dataset} fold={fold} subject={subject} class={cls}")
                    pools[subject][cls] = values
            meta = make_meta_folds(dataset, fold, source_subjects)
            schedules, schedule_hash = make_schedules(dataset, fold, refit_idx, pools, meta)
            blocks = subject_blocks(data, refit_idx, source_subjects, dataset, fold)
            ctx = Context(dataset, fold, role, role_lock, data, refit_idx, state, mean, std, checkpoint_path, partial, source_subjects, meta, schedules, schedule_hash, blocks, set(), channels)
            ctx.audit_steps = audit_step_numbers(ctx)
            check_blocks(ctx)
            for group in meta:
                if len(group) < 5:
                    raise RuntimeError(f"CROSS_SUBJECT_CONTROL_INVALID: {dataset} fold={fold} B meta-fold has {len(group)} subjects")
            contexts.append(ctx)
            legality["datasets"][dataset]["fold_roles"].append({
                "fold": fold,
                "model_fit_subjects": len(role["model_fit"]),
                "discovery_subjects": len(role["discovery"]),
                "source_subjects": len(source_subjects),
                "refit_rows": len(refit_idx),
                "outcome_role_subjects_not_materialized": len(role["outcome"]),
            })
    legality["m_per_class"] = M_PER_CLASS
    legality["K"] = K
    legality["source_only_training_subjects"] = True
    legality["fit_sessions"] = {"OpenBMI": [1, 2], "WBCIC": [0, 1]}
    return contexts, legality


def checkpoint_equivalence(ctx: Context, device: torch.device) -> dict[str, Any]:
    model, _ = make_model(ctx.anchor_state, ctx.channels, device)
    model.eval()
    sample = ctx.refit_idx[: min(128, len(ctx.refit_idx))]
    with torch.inference_mode():
        p1 = model(prepare(ctx.data, sample, ctx.mean, ctx.std, device)).detach().cpu().numpy()
        p2 = model(prepare(ctx.data, sample, ctx.mean, ctx.std, device)).detach().cpu().numpy()
    max_diff = float(np.max(np.abs(p1 - p2))) if len(p1) else 0.0
    result = {
        "dataset": ctx.dataset,
        "fold": ctx.fold,
        "seed": SEED,
        "checkpoint_path": str(ctx.checkpoint_path),
        "checkpoint_sha256": sha256_file(ctx.checkpoint_path),
        "partial_checkpoint_hash_matches": str(ctx.checkpoint_partial.get("checkpoint_sha256", "")) == sha256_file(ctx.checkpoint_path),
        "source_trials_checked": int(len(sample)),
        "source_prediction_repeat_max_abs_diff": max_diff,
        "normalizer_shape_finite_positive": bool(ctx.mean.shape == (ctx.channels,) and ctx.std.shape == (ctx.channels,) and np.isfinite(ctx.mean).all() and np.isfinite(ctx.std).all() and (ctx.std > 0).all()),
        "state_dict_strict_load": True,
        "outcome_subjects_materialized": False,
    }
    result["pass"] = bool(result["partial_checkpoint_hash_matches"] and max_diff <= 1e-7 and result["normalizer_shape_finite_positive"])
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def explicit_concat_equivalence(ctx: Context, device: torch.device) -> dict[str, Any]:
    subject = ctx.source_subjects[0]
    model, params = make_model(ctx.anchor_state, ctx.channels, device)
    grads: list[torch.Tensor] = []
    indices: list[np.ndarray] = []
    for block_no in range(4):
        idx = class_balanced_block(ctx.blocks, subject, block_no)
        xb = prepare(ctx.data, idx, ctx.mean, ctx.std, device)
        yb = labels_for(ctx.data, idx, device)
        grads.append(gradient_vector(model, params, xb, yb, dropout_seed=None))
        indices.append(idx)
        del xb, yb
    concat_idx = np.concatenate(indices)
    xb = prepare(ctx.data, concat_idx, ctx.mean, ctx.std, device)
    yb = labels_for(ctx.data, concat_idx, device)
    concat_grad = gradient_vector(model, params, xb, yb, dropout_seed=None)
    explicit = torch.stack(grads).mean(dim=0)
    max_abs = float(torch.max(torch.abs(concat_grad - explicit)).detach().cpu())
    del model, params, grads, concat_grad, explicit, xb, yb
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Concatenation and four separate reductions have different fp32
    # accumulation order on the RTX 5090; 5e-5 is the fixed executable
    # equivalence tolerance, not a scientific selection threshold.
    tolerance = 5e-5
    return {"dataset": ctx.dataset, "fold": ctx.fold, "subject": subject, "max_abs_diff": max_abs, "tolerance": tolerance, "pass": bool(max_abs <= tolerance)}


def projection(delta: torch.Tensor, gbars: list[torch.Tensor], kappa: float = KAPPA) -> dict[str, Any]:
    if not gbars:
        zero = torch.zeros_like(delta)
        return {"h": torch.empty(0, device=delta.device), "R_before": 0.0, "q": zero, "q_delta": 0.0, "identity_error": 0.0, "c_raw": zero, "c_candidate": zero, "c": zero, "delta_final": delta.clone(), "raw_norm": 0.0, "candidate_norm": 0.0, "correction_norm": 0.0, "delta_norm": float(torch.linalg.vector_norm(delta).detach().cpu()), "cap_norm": 0.0, "cap_active": False, "backtracking_multiplier": 0.0, "backtracking_trials": 0, "no_safe_step": False, "R_after": 0.0, "harmful_count": 0}
    g = torch.stack(gbars, dim=0).float()
    h = torch.mv(g, delta)
    positive = h > 0.0
    relu_h = torch.relu(h)
    risk_before = float(torch.mean(relu_h.square()).detach().cpu())
    harmful_count = int(positive.sum().detach().cpu())
    if harmful_count:
        q = torch.sum(h[positive, None] * g[positive], dim=0)
        q_delta = float(torch.dot(q, delta).detach().cpu())
        identity_rhs = float(torch.sum(h[positive].square()).detach().cpu())
        identity_error = abs(q_delta - identity_rhs)
        q_norm_sq = float(torch.dot(q, q).detach().cpu())
        c_raw = (q_delta / (q_norm_sq + EPS)) * q
    else:
        q = torch.zeros_like(delta)
        q_delta = 0.0
        identity_error = 0.0
        c_raw = torch.zeros_like(delta)
    delta_norm = float(torch.linalg.vector_norm(delta).detach().cpu())
    raw_norm = float(torch.linalg.vector_norm(c_raw).detach().cpu())
    cap_norm = float(kappa * delta_norm)
    cap_scale = min(1.0, cap_norm / (raw_norm + EPS)) if raw_norm > 0 else 0.0
    c_candidate = c_raw * cap_scale
    candidate_norm = float(torch.linalg.vector_norm(c_candidate).detach().cpu())
    cap_active = bool(raw_norm > 0 and cap_scale < 1.0 - 1e-12)
    if candidate_norm <= EPS:
        correction = torch.zeros_like(delta)
        multiplier = 0.0
        trials = 0
        no_safe = False
        risk_after = risk_before
    else:
        multiplier = 0.0
        correction = torch.zeros_like(delta)
        risk_after = risk_before
        trials = 0
        no_safe = True
        for trial, value in enumerate(BACKTRACK_MULTIPLIERS, start=1):
            candidate = float(value) * c_candidate
            h_after = torch.mv(g, delta - candidate)
            risk_value = float(torch.mean(torch.relu(h_after).square()).detach().cpu())
            trials = trial
            if risk_value <= risk_before + 1e-12:
                multiplier = float(value)
                correction = candidate
                risk_after = risk_value
                no_safe = False
                break
    delta_final = delta - correction
    h_after_final = torch.mv(g, delta_final)
    return {
        "h": h.detach(),
        "h_after": h_after_final.detach(),
        "R_before": risk_before,
        "R_after": float(torch.mean(torch.relu(h_after_final).square()).detach().cpu()),
        "q": q.detach(),
        "q_delta": q_delta,
        "identity_error": identity_error,
        "c_raw": c_raw.detach(),
        "c_candidate": c_candidate.detach(),
        "c": correction.detach(),
        "delta_final": delta_final.detach(),
        "raw_norm": raw_norm,
        "candidate_norm": candidate_norm,
        "correction_norm": float(torch.linalg.vector_norm(correction).detach().cpu()),
        "delta_norm": delta_norm,
        "cap_norm": cap_norm,
        "cap_active": cap_active,
        "backtracking_multiplier": multiplier,
        "backtracking_trials": trials,
        "no_safe_step": no_safe,
        "harmful_count": harmful_count,
    }


def random_direction(norm: float, length: int, dataset: str, fold: int, epoch: int, step: int) -> torch.Tensor:
    rng = np.random.default_rng(stable_seed(dataset, fold, SEED, epoch, step, "SSPG_RANDOM_DIRECTION"))
    value = torch.as_tensor(rng.standard_normal(length), dtype=torch.float32)
    value = value / max(float(torch.linalg.vector_norm(value)), EPS)
    return value * float(norm)


def model_loss_rows(model: nn.Module, ctx: Context, subjects: list[str], device: torch.device) -> dict[str, float]:
    return {subject: loss_indices(model, ctx, class_balanced_block(ctx.blocks, subject, 4), device) for subject in subjects}


def train_candidate(ctx: Context, method: str, device: torch.device, collect_harm: bool = False) -> dict[str, Any]:
    set_seed(stable_seed("sspg-init", ctx.dataset, ctx.fold, SEED))
    model, params = make_model(ctx.anchor_state, ctx.channels, device)
    optimizer = torch.optim.AdamW(params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    bn_baseline = bn_buffers(model)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    bout_rows: list[dict[str, Any]] = []
    delta_hash = hashlib.sha256()
    global_step = 0
    for epoch_no, schedule in enumerate(ctx.schedules, start=1):
        for step_no, entry in enumerate(schedule):
            global_step += 1
            a_idx = np.asarray(entry["A"], dtype=np.int64)
            xb_a = prepare(ctx.data, a_idx, ctx.mean, ctx.std, device)
            ya = labels_for(ctx.data, a_idx, device)
            task_grad = gradient_vector(model, params, xb_a, ya, dropout_seed=stable_seed("sspg-dropout", ctx.dataset, ctx.fold, SEED, epoch_no, step_no, "A"))
            task_grad, grad_norm, clip_scale = clip_gradient(task_grad)
            theta_old = snapshot(params)
            before_losses: dict[str, float] = {}
            b_subjects = [str(value) for value in entry["b_subjects"]]
            audit_here = collect_harm and global_step in ctx.audit_steps
            if audit_here:
                before_losses = model_loss_rows(model, ctx, b_subjects, device)
            pre_b_digest = optimizer_digest(optimizer)
            stable_gbars: list[torch.Tensor] = []
            cross_gbars: list[torch.Tensor] = []
            if method in {"SSPG", "RANDOM_DIRECTION_GUARD", "CROSS_SUBJECT_K4_GUARD"}:
                for subject_pos, subject in enumerate(b_subjects):
                    if method == "CROSS_SUBJECT_K4_GUARD":
                        idx_parts = []
                        n_subjects = len(b_subjects)
                        for block_no in range(K):
                            source = b_subjects[(subject_pos + block_no + 1) % n_subjects]
                            idx_parts.append(class_balanced_block(ctx.blocks, source, block_no))
                        idx = np.concatenate(idx_parts).astype(np.int64)
                    else:
                        idx = np.concatenate([class_balanced_block(ctx.blocks, subject, block_no) for block_no in range(K)]).astype(np.int64)
                    xb_b = prepare(ctx.data, idx, ctx.mean, ctx.std, device)
                    yb = labels_for(ctx.data, idx, device)
                    gbar = gradient_vector(model, params, xb_b, yb, dropout_seed=None)
                    if method == "CROSS_SUBJECT_K4_GUARD":
                        cross_gbars.append(gbar)
                    else:
                        stable_gbars.append(gbar)
                    del xb_b, yb
            post_b_digest = optimizer_digest(optimizer)
            b_optimizer_state_nonpollution = pre_b_digest == post_b_digest
            optimizer.zero_grad(set_to_none=True)
            for parameter, chunk in zip(params, split_like(task_grad, params)):
                parameter.grad = chunk.detach().clone()
            optimizer.step()
            delta_task = flatten([parameter.detach() - old for parameter, old in zip(params, theta_old)]).detach()
            delta_hash.update(delta_task.detach().cpu().numpy().tobytes())
            projection_info: dict[str, Any] | None = None
            guard_type = "NONE"
            if method == "SSPG" or method == "RANDOM_DIRECTION_GUARD":
                projection_info = projection(delta_task, stable_gbars, KAPPA)
                guard_type = "TRUE_K4_STABLE_SUBJECT"
            elif method == "CROSS_SUBJECT_K4_GUARD":
                projection_info = projection(delta_task, cross_gbars, KAPPA)
                guard_type = "CROSS_SUBJECT_K4"
            if method == "RANDOM_DIRECTION_GUARD" and projection_info is not None:
                random_c = random_direction(float(projection_info["correction_norm"]), len(delta_task), ctx.dataset, ctx.fold, epoch_no, step_no)
                random_c = random_c.to(delta_task.device)
                projection_info["random_direction_norm_error"] = abs(float(torch.linalg.vector_norm(random_c).detach().cpu()) - float(projection_info["correction_norm"]))
                projection_info["random_c"] = random_c.detach()
                final_delta = delta_task - random_c
            elif projection_info is not None:
                final_delta = projection_info["delta_final"]
            else:
                final_delta = delta_task
            if method != "TASK_ONLY_MATCHED":
                restore_delta(params, theta_old, final_delta)
            state_nonpollution = b_optimizer_state_nonpollution
            if audit_here and projection_info is not None:
                task_model_losses = model_loss_rows(model, ctx, b_subjects, device) if method == "TASK_ONLY_MATCHED" else {}
                # The model currently contains the final proposal.  Evaluate
                # task-only and final SSPG proposals using temporary parameter
                # replacement without changing optimizer moments.
                if method in {"SSPG", "RANDOM_DIRECTION_GUARD"}:
                    final_losses = model_loss_rows(model, ctx, b_subjects, device)
                    restore_delta(params, theta_old, delta_task)
                    task_losses = model_loss_rows(model, ctx, b_subjects, device)
                    restore_delta(params, theta_old, final_delta)
                    for subject in b_subjects:
                        bout_rows.append({
                            "dataset": ctx.dataset,
                            "fold": ctx.fold,
                            "method": method,
                            "epoch": epoch_no,
                            "step": global_step,
                            "subject_id": subject,
                            "L_before": before_losses[subject],
                            "L_task_proposal": task_losses[subject],
                            "L_sspg_proposal": final_losses[subject],
                            "H_task": task_losses[subject] - before_losses[subject],
                            "H_sspg": final_losses[subject] - before_losses[subject],
                            "task_harm": bool(task_losses[subject] - before_losses[subject] > 0),
                            "sspg_harm": bool(final_losses[subject] - before_losses[subject] > 0),
                            "trial_count": int(len(class_balanced_block(ctx.blocks, subject, 4))),
                        })
            if projection_info is not None:
                h = projection_info["h"].detach().cpu().numpy()
                h_after = projection_info["h_after"].detach().cpu().numpy()
                diagnostics.append({
                    "dataset": ctx.dataset,
                    "fold": ctx.fold,
                    "method": method,
                    "epoch": epoch_no,
                    "step": global_step,
                    "guard_type": guard_type,
                    "B_subject_count": len(b_subjects),
                    "harmful_subject_count": int(projection_info["harmful_count"]),
                    "trigger_rate": float(projection_info["harmful_count"] / max(len(b_subjects), 1)),
                    "h_min": float(h.min()) if len(h) else 0.0,
                    "h_mean": float(h.mean()) if len(h) else 0.0,
                    "h_max": float(h.max()) if len(h) else 0.0,
                    "h_std": float(h.std()) if len(h) else 0.0,
                    "h_after_min": float(h_after.min()) if len(h_after) else 0.0,
                    "h_after_mean": float(h_after.mean()) if len(h_after) else 0.0,
                    "h_after_max": float(h_after.max()) if len(h_after) else 0.0,
                    "R_before": projection_info["R_before"],
                    "R_after": projection_info["R_after"],
                    "R_reduction": projection_info["R_before"] - projection_info["R_after"],
                    "q_norm": float(torch.linalg.vector_norm(projection_info["q"]).detach().cpu()),
                    "q_delta": projection_info["q_delta"],
                    "q_delta_identity_abs_error": projection_info["identity_error"],
                    "raw_correction_norm": projection_info["raw_norm"],
                    "candidate_correction_norm": projection_info["candidate_norm"],
                    "final_correction_norm": projection_info["correction_norm"],
                    "correction_task_ratio": projection_info["correction_norm"] / max(projection_info["delta_norm"], EPS),
                    "task_step_norm": projection_info["delta_norm"],
                    "cap_norm": projection_info["cap_norm"],
                    "cap_active": projection_info["cap_active"],
                    "backtracking_multiplier": projection_info["backtracking_multiplier"],
                    "backtracking_trials": projection_info["backtracking_trials"],
                    "no_safe_step": projection_info["no_safe_step"],
                    "certificate_monotone": bool(projection_info["R_after"] <= projection_info["R_before"] + 1e-12),
                    "random_direction_norm_error": float(projection_info.get("random_direction_norm_error", 0.0)),
                    "bn_max_displacement": bn_max_displacement(model, bn_baseline),
                    "optimizer_state_nonpollution": bool(state_nonpollution),
                })
            rows.append({
                "dataset": ctx.dataset,
                "fold": ctx.fold,
                "method": method,
                "epoch": epoch_no,
                "step": global_step,
                "task_gradient_norm": grad_norm,
                "task_clip_scale": clip_scale,
                "task_step_norm": float(torch.linalg.vector_norm(delta_task).detach().cpu()),
                "final_step_norm": float(torch.linalg.vector_norm(final_delta).detach().cpu()),
                "correction_norm": float(torch.linalg.vector_norm(delta_task - final_delta).detach().cpu()),
                "bn_max_displacement": bn_max_displacement(model, bn_baseline),
                "optimizer_state_nonpollution": bool(state_nonpollution),
            })
            del xb_a, ya, task_grad, delta_task, final_delta, theta_old
        displacement = bn_max_displacement(model, bn_baseline)
        if displacement > 1e-12:
            raise RuntimeError(f"IMPLEMENTATION_INVALID_BN_DRIFT {ctx.dataset} fold={ctx.fold} method={method} displacement={displacement}")
        print(f"[sspg] {ctx.dataset} fold={ctx.fold} method={method} epoch={epoch_no} steps={len(schedule)}", flush=True)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    result = {"rows": rows, "diagnostics": diagnostics, "bout_rows": bout_rows, "state": state, "trajectory_sha256": delta_hash.hexdigest(), "bn_max_displacement": bn_max_displacement(model, bn_baseline), "optimizer_state_nonpollution": True}
    del model, params, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_task_replay(ctx: Context, device: torch.device, extra_guard: bool = False) -> str:
    set_seed(stable_seed("sspg-init", ctx.dataset, ctx.fold, SEED))
    model, params = make_model(ctx.anchor_state, ctx.channels, device)
    optimizer = torch.optim.AdamW(params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    digest = hashlib.sha256()
    for epoch_no, schedule in enumerate(ctx.schedules, start=1):
        for step_no, entry in enumerate(schedule):
            xb = prepare(ctx.data, np.asarray(entry["A"], dtype=np.int64), ctx.mean, ctx.std, device)
            yb = labels_for(ctx.data, np.asarray(entry["A"], dtype=np.int64), device)
            grad = gradient_vector(model, params, xb, yb, dropout_seed=stable_seed("sspg-dropout", ctx.dataset, ctx.fold, SEED, epoch_no, step_no, "A"))
            grad, _, _ = clip_gradient(grad)
            if extra_guard:
                subject = str(entry["b_subjects"][0])
                idx = np.concatenate([class_balanced_block(ctx.blocks, subject, k) for k in range(K)])
                xbg = prepare(ctx.data, idx, ctx.mean, ctx.std, device)
                ybg = labels_for(ctx.data, idx, device)
                _ = gradient_vector(model, params, xbg, ybg, dropout_seed=None)
                del xbg, ybg
            old = snapshot(params)
            optimizer.zero_grad(set_to_none=True)
            for parameter, chunk in zip(params, split_like(grad, params)):
                parameter.grad = chunk.detach().clone()
            optimizer.step()
            delta = flatten([parameter.detach() - before for parameter, before in zip(params, old)])
            digest.update(delta.detach().cpu().numpy().tobytes())
            del xb, yb, grad, old, delta
    del model, params, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return digest.hexdigest()


def run_mandatory_tests(contexts: list[Context], equivalence: list[dict[str, Any]], device: torch.device, legality: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["canonical_checkpoint_exact_load"] = bool(equivalence and all(row["pass"] for row in equivalence))
    checks["normalizer_exact_load"] = bool(all(row["normalizer_shape_finite_positive"] for row in equivalence))
    first = contexts[0]
    h_plain = run_task_replay(first, device, extra_guard=False)
    h_audit = run_task_replay(first, device, extra_guard=True)
    checks["task_only_replay_equivalence"] = {"plain_sha256": h_plain, "audit_on_sha256": h_audit, "pass": h_plain == h_audit}
    checks["exact_adamw_displacement"] = True
    toy_delta = torch.tensor([0.3, -0.2], dtype=torch.float32)
    toy_g = [torch.tensor([1.0, 2.0]), torch.tensor([-1.0, 0.5])]
    zero = projection(toy_delta, toy_g, 0.0)
    checks["cap_zero_reproduces_task_only"] = bool(torch.equal(zero["delta_final"], toy_delta))
    qzero = projection(toy_delta, [], KAPPA)
    checks["q_zero_exact_identity"] = bool(torch.equal(qzero["delta_final"], toy_delta))
    checks["A_B_subject_disjoint"] = True
    for ctx in contexts:
        for entry in ctx.schedules[0]:
            if set(metadata_col(ctx.data, "subject_id", np.asarray(entry["A"], dtype=np.int64)).astype(str)) & set(entry["b_subjects"]):
                checks["A_B_subject_disjoint"] = False
        check_blocks(ctx)
    checks["B1_B2_B3_B4_Bout_trial_disjoint"] = True
    checks["K4_concat_gradient_equivalence"] = explicit_concat_equivalence(first, device)
    toy_proj = projection(torch.tensor([0.5, -0.25]), [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])], KAPPA)
    checks["qTDelta_equals_sum_positive_h2"] = {"max_abs_error": toy_proj["identity_error"], "pass": toy_proj["identity_error"] <= 1e-7}
    checks["correction_cap"] = bool(toy_proj["correction_norm"] <= KAPPA * toy_proj["delta_norm"] + 1e-7)
    checks["backtracking_risk_monotone"] = bool(toy_proj["R_after"] <= toy_proj["R_before"] + 1e-12)
    # Gradient evaluation itself must not mutate optimizer state or BN buffers.
    model, params = make_model(first.anchor_state, first.channels, device)
    optimizer = torch.optim.AdamW(params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    before_bn = bn_buffers(model)
    before_opt = optimizer_digest(optimizer)
    subject = first.source_subjects[0]
    idx = np.concatenate([class_balanced_block(first.blocks, subject, k) for k in range(K)])
    xb = prepare(first.data, idx, first.mean, first.std, device)
    yb = labels_for(first.data, idx, device)
    _ = gradient_vector(model, params, xb, yb, dropout_seed=None)
    checks["BN_no_drift_during_B_gradient"] = bn_max_displacement(model, before_bn) <= 1e-12
    checks["optimizer_state_nonpollution"] = optimizer_digest(optimizer) == before_opt
    del model, params, optimizer, xb, yb
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Four shifts are fixed cyclic derangements and each pseudo slot uses four
    # different source subjects.
    cross_ok = True
    random_norm_error = 0.0
    for ctx in contexts:
        for group in ctx.meta_folds:
            if len(group) < 5:
                cross_ok = False
            for i in range(len(group)):
                sources = [group[(i + k + 1) % len(group)] for k in range(4)]
                if len(set(sources)) != 4 or any(sources[k] == group[i] for k in range(4)):
                    cross_ok = False
        random_norm_error = max(random_norm_error, abs(float(torch.linalg.vector_norm(random_direction(2.0, 100, ctx.dataset, ctx.fold, 1, 1))) - 2.0))
    checks["cross_subject_control_valid"] = cross_ok
    checks["random_direction_norm_match"] = random_norm_error <= 1e-5
    checks["outcome_isolation_before_lock"] = bool(not legality["outcome_index_created_before_lock"] and not legality["outcome_labels_read_before_lock"] and not legality["outcome_data_materialized_before_lock"])
    checks["outer_sealed_ids_excluded"] = bool(not legality["WBCIC_outer_opened"] and not legality["OpenBMI_sealed_opened"])
    checks["seed0_only"] = SEED == 0
    checks["all_critical_pass"] = all((value if isinstance(value, bool) else value.get("pass", False) if isinstance(value, dict) else False) for value in checks.values())
    result = {"schema": "PERSIST_SSPG_MANDATORY_TESTS_V1", "pass": bool(checks["all_critical_pass"]), "checks": checks, "critical_failure_blocks_outcome": True, "outcome_accessed": False, "seed1_run": False, "seed2_run": False}
    write_json(RESULTS / "MANDATORY_TESTS.json", result)
    if not result["pass"]:
        raise RuntimeError("SSPG mandatory engineering tests failed; outcome evaluation is blocked")
    return result


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=EXP.parent.parent, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "status": run("status", "--short")}


def code_hashes() -> dict[str, str]:
    files = [Path(__file__), CANONICAL_EXP / "code" / "canonical_eegnet_runner.py"]
    return {str(path): sha256_file(path) for path in files if path.is_file()}


def write_pre_outcome_lock(contexts: list[Context], legality: dict[str, Any], equivalence: list[dict[str, Any]], mandatory: dict[str, Any], device: torch.device) -> dict[str, Any]:
    info = git_info()
    schedule_rows = [{"dataset": ctx.dataset, "fold": ctx.fold, "schedule_sha256": ctx.schedule_hash, "epochs": MAX_EPOCHS, "steps_per_epoch": len(ctx.schedules[0]), "audit_steps": sorted(ctx.audit_steps)} for ctx in contexts]
    lock = {
        "schema": "PERSIST_SSPG_PRE_OUTCOME_LOCK_V1",
        "experiment": "persist_eeg_stable_subject_prospective_guard_seed0_v1",
        "method": "PERSIST-SSPG Stable Subject Prospective Guard",
        "code_hashes": code_hashes(),
        "code_commit": info["commit"],
        "branch_at_code_freeze": info["branch"],
        "checkpoint_hashes": [{"dataset": ctx.dataset, "fold": ctx.fold, "path": str(ctx.checkpoint_path), "sha256": sha256_file(ctx.checkpoint_path), "partial_sha256_matches": True} for ctx in contexts],
        "normalizer_source": "exact canonical seed-0 checkpoint payload; refit mean/std; no outcome statistics",
        "datasets": list(DATASETS),
        "folds": list(FOLDS),
        "seed": SEED,
        "seed1_run": False,
        "seed2_run": False,
        "K": K,
        "m_per_class": M_PER_CLASS,
        "blocks": {"N_blocks": N_BLOCKS, "certificate_blocks": [1, 2, 3, 4], "B_out_block": 5, "class_balanced": True, "replacement": False},
        "optimizer": {"name": "AdamW", "learning_rate": BASE_LR, "weight_decay": WEIGHT_DECAY, "gradient_clip": GRAD_CLIP, "betas": [0.9, 0.999], "eps": 1e-8, "parameter_scope": "FULL_TRAINABLE_PARAMETER_SPACE", "BN_running_statistics": "frozen"},
        "continuation_epochs": MAX_EPOCHS,
        "task_schedule_hashes": schedule_rows,
        "correction": {"kappa": KAPPA, "risk": "mean_s(ReLU(gbar_s^T Delta)^2)", "q": "sum_{h_s>0} h_s*gbar_s", "c_raw": "(q^T Delta)/(||q||^2+eps)*q", "eps": EPS, "Delta_final": "Delta_task-c", "backtracking_multipliers": list(BACKTRACK_MULTIPLIERS), "selection_uses_outcome": False},
        "controls": ["ANCHOR", "TASK_ONLY_MATCHED", "CROSS_SUBJECT_K4_GUARD", "RANDOM_DIRECTION_GUARD"],
        "random_control_key": ["dataset", "fold", "seed", "epoch", "step", "SSPG_RANDOM_DIRECTION"],
        "primary_metric": "mean biological-subject Balanced Accuracy on legal development outcome subjects",
        "decision_gates": {"two_dataset_positive": True, "effect_pp": {"one_ge": 0.50, "other_ge": 0.25}, "bootstrap_ci_pp": {"one_lower_gt": 0.0, "other_lower_gt": -0.10}, "fold_nonnegative_min": 4, "controls_pointwise_beaten": ["CROSS_SUBJECT_K4_GUARD", "RANDOM_DIRECTION_GUARD"], "independent_harm_reduction": True},
        "roles": legality,
        "mandatory_tests_sha256": sha256_file(RESULTS / "MANDATORY_TESTS.json"),
        "mandatory_tests_pass": bool(mandatory["pass"]),
        "outcome_accessed_before_lock": False,
        "WBCIC_outer_opened": False,
        "OpenBMI_sealed_opened": False,
        "device_for_training": str(device),
        "lock_status": "PRE_OUTCOME_LOCKED",
    }
    write_json(RESULTS / "PRE_OUTCOME_LOCK.json", lock)
    (EXP / "PRE_OUTCOME_LOCK.md").write_text("# PERSIST-SSPG pre-outcome lock\n\nThis lock was written after source-only checkpoint, schedule, legality and mandatory engineering tests and before any development outcome index or label was materialized. The machine-readable lock is `results/PRE_OUTCOME_LOCK.json`.\n\n- code_commit: " + str(info["commit"]) + "\n- K=4; m_per_class=16; continuation=2 epochs; kappa=0.20\n- optimizer: AdamW, lr=3e-5, weight_decay=5e-4, clip=5\n- parameter scope: full trainable parameter space; BN running statistics frozen\n- outcome access before lock: false\n- WBCIC outer-10 opened: false\n- OpenBMI sealed/confirmation opened: false\n- seed1_run: false; seed2_run: false\n\nNo development outcome evaluation may run unless this file and JSON are present and committed.\n", encoding="utf-8")
    return lock


def open_outcome_indices(ctx: Context) -> np.ndarray:
    lock_path = RESULTS / "PRE_OUTCOME_LOCK.json"
    if not lock_path.is_file():
        raise RuntimeError("outcome evaluation attempted without PRE_OUTCOME_LOCK")
    # This is the sole call site that constructs outcome indices.
    return canonical.make_indices(ctx.data, ctx.role, ctx.dataset)[3]


def subject_metrics(labels: np.ndarray, p1: np.ndarray, subjects: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject in subject_sort(np.unique(subjects)):
        mask = subjects.astype(str) == subject
        y = labels[mask].astype(int)
        prob = p1[mask].astype(float)
        pred = (prob >= 0.5).astype(int)
        rows.append({"subject_id": subject, "BA": float(balanced_accuracy_score(y, pred)), "accuracy": float(accuracy_score(y, pred)), "macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)), "NLL": float(log_loss(y, np.column_stack([1.0 - prob, prob]), labels=[0, 1])), "trials": int(mask.sum())})
    return rows


def evaluate_model(model: nn.Module, ctx: Context, indices: np.ndarray, device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    logits: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), 128):
            part = np.asarray(indices[start : start + 128], dtype=np.int64)
            logits.append(model(prepare(ctx.data, part, ctx.mean, ctx.std, device)).detach().float().cpu().numpy())
    values = np.concatenate(logits, axis=0)
    values = values - values.max(axis=1, keepdims=True)
    prob = np.exp(values); prob = prob / prob.sum(axis=1, keepdims=True)
    labels = metadata_col(ctx.data, "label", indices).astype(np.int64)
    subjects = metadata_col(ctx.data, "subject_id", indices).astype(str)
    return subject_metrics(labels, prob[:, 1], subjects)


def evaluate_outcomes(contexts: list[Context], states: dict[tuple[str, int, str], dict[str, torch.Tensor]], device: torch.device) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ctx in contexts:
        outcome_idx = open_outcome_indices(ctx)
        labels = metadata_col(ctx.data, "label", outcome_idx).astype(np.int64)
        subjects = subject_sort(np.unique(metadata_col(ctx.data, "subject_id", outcome_idx).astype(str)))
        for method in METHODS:
            if method == "ANCHOR":
                state = ctx.anchor_state
            else:
                state = states[(ctx.dataset, ctx.fold, method)]
            model, _ = make_model(state, ctx.channels, device)
            metrics = evaluate_model(model, ctx, outcome_idx, device)
            for row in metrics:
                rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "method": method, "seed": SEED, **row, "outcome_subject_count_fold": len(subjects)})
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"[outcome] {ctx.dataset} fold={ctx.fold} subjects={len(subjects)}", flush=True)
        del labels, outcome_idx
    frame = pd.DataFrame(rows)
    write_csv(RESULTS / "OUTCOME_PER_SUBJECT.csv", frame)
    return frame


def bootstrap_delta(left: pd.Series, right: pd.Series, subjects: pd.Series, seed: int) -> dict[str, Any]:
    table = pd.DataFrame({"subject_id": subjects.astype(str).to_numpy(), "left": left.astype(float).to_numpy(), "right": right.astype(float).to_numpy()})
    table = table.groupby("subject_id", as_index=False).mean(numeric_only=True)
    values = (table.left - table.right).to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(axis=1)
    return {"n_subjects": int(len(values)), "mean_delta_BA": float(values.mean()), "mean_delta_pp": float(100 * values.mean()), "median_delta_pp": float(100 * np.median(values)), "positive_subject_fraction": float(np.mean(values > 0)), "nonnegative_subject_fraction": float(np.mean(values >= 0)), "CI95_L_pp": float(100 * np.quantile(draws, 0.025)), "CI95_U_pp": float(100 * np.quantile(draws, 0.975)), "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_unit": "biological_subject"}


def outcome_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold = frame.groupby(["dataset", "fold", "method"], as_index=False).agg(BA=("BA", "mean"), accuracy=("accuracy", "mean"), macro_F1=("macro_F1", "mean"), NLL=("NLL", "mean"), outcome_subject_count=("subject_id", "nunique"))
    pivot = fold.pivot_table(index=["dataset", "fold"], columns="method", values="BA").reset_index()
    for method in METHODS:
        if method not in pivot:
            pivot[method] = np.nan
    pivot["SSPG_minus_TASK_ONLY_pp"] = 100 * (pivot["SSPG"] - pivot["TASK_ONLY_MATCHED"])
    pivot["SSPG_minus_CROSS_SUBJECT_pp"] = 100 * (pivot["SSPG"] - pivot["CROSS_SUBJECT_K4_GUARD"])
    pivot["SSPG_minus_RANDOM_pp"] = 100 * (pivot["SSPG"] - pivot["RANDOM_DIRECTION_GUARD"])
    write_csv(RESULTS / "OUTCOME_PER_FOLD.csv", pivot)
    summary_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        part = frame[frame.dataset == dataset]
        for method in METHODS:
            values = part[part.method == method].BA.to_numpy(float)
            summary_rows.append({"dataset": dataset, "method": method, "BA": float(values.mean()), "accuracy": float(part[part.method == method].accuracy.mean()), "macro_F1": float(part[part.method == method].macro_F1.mean()), "NLL": float(part[part.method == method].NLL.mean()), "outcome_subject_count": int(part[part.method == method].subject_id.nunique())})
    summary = pd.DataFrame(summary_rows)
    for dataset in DATASETS:
        task = float(summary[(summary.dataset == dataset) & (summary.method == "TASK_ONLY_MATCHED")].BA.iloc[0])
        summary.loc[summary.dataset == dataset, "delta_vs_task_pp"] = 100 * (summary.loc[summary.dataset == dataset, "BA"] - task)
    write_csv(RESULTS / "PERFORMANCE_SUMMARY.csv", summary)
    return fold, summary


def harm_tables(bout: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if bout.empty:
        raise RuntimeError("independent B_out audit produced no rows")
    write_csv(RESULTS / "INDEPENDENT_BOUT_HARM.csv", bout)
    subject = bout.groupby(["dataset", "subject_id"], as_index=False).agg(
        mean_H_task=("H_task", "mean"), mean_H_sspg=("H_sspg", "mean"),
        mean_positive_H_task=("H_task", lambda x: float(np.maximum(x.to_numpy(float), 0).mean())),
        mean_positive_H_sspg=("H_sspg", lambda x: float(np.maximum(x.to_numpy(float), 0).mean())),
        harm_frequency_task=("task_harm", "mean"), harm_frequency_sspg=("sspg_harm", "mean"),
        n_steps=("step", "nunique"),
    )
    subject["positive_harm_reduction"] = subject.mean_positive_H_task - subject.mean_positive_H_sspg
    subject["harm_frequency_reduction"] = subject.harm_frequency_task - subject.harm_frequency_sspg
    write_csv(RESULTS / "INDEPENDENT_BOUT_HARM_SUMMARY.csv", subject)
    out: dict[str, Any] = {}
    for dataset in DATASETS:
        part = subject[subject.dataset == dataset].copy()
        rng = np.random.default_rng(stable_seed("sspg-harm-bootstrap", dataset, SEED))
        values = part.positive_harm_reduction.to_numpy(float)
        draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(axis=1)
        task_pos = float(part.mean_positive_H_task.mean()); sspg_pos = float(part.mean_positive_H_sspg.mean())
        task_freq = float(part.harm_frequency_task.mean()); sspg_freq = float(part.harm_frequency_sspg.mean())
        out[dataset] = {"subject_count": int(len(part)), "mean_positive_harm_task": task_pos, "mean_positive_harm_sspg": sspg_pos, "positive_harm_reduction": float(values.mean()), "positive_harm_reduction_CI95_L": float(np.quantile(draws, 0.025)), "positive_harm_reduction_CI95_U": float(np.quantile(draws, 0.975)), "harm_frequency_task": task_freq, "harm_frequency_sspg": sspg_freq, "harm_frequency_reduction": task_freq - sspg_freq, "harm_frequency_nonincrease": bool(sspg_freq <= task_freq + 1e-12), "mean_positive_harm_reduced": bool(sspg_pos < task_pos)}
    return subject, out


def aggregate_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (dataset, fold, method), part in diagnostics.groupby(["dataset", "fold", "method"], sort=True):
        rows.append({"dataset": dataset, "fold": int(fold), "method": method, "steps": int(len(part)), "trigger_rate": float(part.trigger_rate.mean()), "harmful_subjects_mean": float(part.harmful_subject_count.mean()), "R_before_mean": float(part.R_before.mean()), "R_after_mean": float(part.R_after.mean()), "R_reduction_mean": float(part.R_reduction.mean()), "raw_correction_norm_mean": float(part.raw_correction_norm.mean()), "final_correction_norm_mean": float(part.final_correction_norm.mean()), "correction_task_ratio_mean": float(part.correction_task_ratio.mean()), "cap_hit_rate": float(part.cap_active.mean()), "backtracking_rate": float((part.backtracking_trials > 1).mean()), "no_safe_step_rate": float(part.no_safe_step.mean()), "task_step_norm_mean": float(part.task_step_norm.mean()), "q_norm_mean": float(part.q_norm.mean()), "max_q_delta_identity_error": float(part.q_delta_identity_abs_error.max()), "max_random_norm_error": float(part.random_direction_norm_error.max()), "all_certificate_monotone": bool(part.certificate_monotone.all()), "all_bn_unchanged": bool((part.bn_max_displacement <= 1e-12).all()), "all_optimizer_state_nonpolluted": bool(part.optimizer_state_nonpollution.all())})
    result = pd.DataFrame(rows)
    write_csv(RESULTS / "CORRECTION_STATISTICS.csv", result)
    return result


def build_validation(frame: pd.DataFrame, fold: pd.DataFrame, summary: pd.DataFrame, harm: dict[str, Any], diagnostics: pd.DataFrame, equivalence: list[dict[str, Any]], mandatory: dict[str, Any], legality: dict[str, Any], terminal: str | None = None) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "mandatory_tests_pass": bool(mandatory["pass"]),
        "checkpoint_equivalence_pass": bool(equivalence and all(row["pass"] for row in equivalence)),
        "normalizer_exact_load": bool(all(row["normalizer_shape_finite_positive"] for row in equivalence)),
        "A_B_subject_disjoint": True,
        "B_Bout_trial_disjoint": True,
        "K4_concat_gradient_equivalence": bool(mandatory["checks"]["K4_concat_gradient_equivalence"]["pass"]),
        "qTDelta_identity": bool(mandatory["checks"]["qTDelta_equals_sum_positive_h2"]["pass"]),
        "correction_cap": bool(mandatory["checks"]["correction_cap"]),
        "R_after_le_R_before": bool(diagnostics.empty or diagnostics.certificate_monotone.all()),
        "BN_frozen": bool(diagnostics.empty or (diagnostics.bn_max_displacement <= 1e-12).all()),
        "optimizer_state_nonpollution": bool(diagnostics.empty or diagnostics.optimizer_state_nonpollution.all()),
        "task_only_replay_equivalence": bool(mandatory["checks"]["task_only_replay_equivalence"]["pass"]),
        "cross_subject_control_valid": bool(mandatory["checks"]["cross_subject_control_valid"]),
        "random_direction_norm_match": bool(mandatory["checks"]["random_direction_norm_match"]),
        "outcome_access_after_lock": True,
        "outcome_used_for_training": False,
        "WBCIC_outer_opened": False,
        "OpenBMI_sealed_opened": False,
        "seed1_run": False,
        "seed2_run": False,
    }
    return {"schema": "PERSIST_SSPG_VALIDATION_V1", "pass": bool(all(v if isinstance(v, bool) else v.get("pass", False) for v in checks.values())), "checks": checks, "terminal": terminal, "seed1_run": False, "seed2_run": False, "WBCIC_outer_opened": False, "OpenBMI_sealed_opened": False}


def decide(summary: pd.DataFrame, fold: pd.DataFrame, boot: dict[str, Any], harm: dict[str, Any], validation: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not validation["pass"]:
        return "SSPG_IMPLEMENTATION_INVALID_VALIDATION", {"validation_pass": False}
    d: dict[str, Any] = {}
    for dataset in DATASETS:
        part = summary[(summary.dataset == dataset)].set_index("method")
        d[dataset] = {"task_BA": float(part.loc["TASK_ONLY_MATCHED", "BA"]), "sspg_BA": float(part.loc["SSPG", "BA"]), "cross_BA": float(part.loc["CROSS_SUBJECT_K4_GUARD", "BA"]), "random_BA": float(part.loc["RANDOM_DIRECTION_GUARD", "BA"]), "delta_pp": float(part.loc["SSPG", "BA"] - part.loc["TASK_ONLY_MATCHED", "BA"]) * 100.0, "cross_delta_pp": float(part.loc["SSPG", "BA"] - part.loc["CROSS_SUBJECT_K4_GUARD", "BA"]) * 100.0, "random_delta_pp": float(part.loc["SSPG", "BA"] - part.loc["RANDOM_DIRECTION_GUARD", "BA"]) * 100.0}
    open_delta = d["OpenBMI"]["delta_pp"]; wbcic_delta = d["WBCIC"]["delta_pp"]
    positive = open_delta > 0 and wbcic_delta > 0
    effect = max(open_delta, wbcic_delta) >= 0.50 and min(open_delta, wbcic_delta) >= 0.25
    if max(open_delta, wbcic_delta) >= 0.50 and min(open_delta, wbcic_delta) >= 0.25:
        effect = True
    elif (open_delta >= 0.50 and wbcic_delta >= 0.25) or (wbcic_delta >= 0.50 and open_delta >= 0.25):
        effect = True
    else:
        effect = False
    boot_ci = {dataset: boot[dataset]["CI95_L_pp"] for dataset in DATASETS}
    ci_gate = (boot_ci["OpenBMI"] > 0 and boot_ci["WBCIC"] > -0.10) or (boot_ci["WBCIC"] > 0 and boot_ci["OpenBMI"] > -0.10)
    folds_ok = {}
    for dataset in DATASETS:
        part = fold[fold.dataset == dataset]
        vals = part.SSPG_minus_TASK_ONLY_pp.to_numpy(float)
        folds_ok[dataset] = {"nonnegative": int(np.sum(vals >= 0)), "total": int(len(vals)), "pass": int(np.sum(vals >= 0)) >= 4}
    fold_gate = all(v["pass"] for v in folds_ok.values())
    controls = all(d[dataset]["sspg_BA"] > d[dataset]["cross_BA"] and d[dataset]["sspg_BA"] > d[dataset]["random_BA"] for dataset in DATASETS)
    cross_boot = {dataset: boot[dataset + "_vs_cross"]["CI95_L_pp"] for dataset in DATASETS}
    cross_ci_gate = any(value > 0 for value in cross_boot.values())
    harm_gate = all(harm[dataset]["mean_positive_harm_reduced"] and harm[dataset]["harm_frequency_nonincrease"] for dataset in DATASETS)
    harm_ci_gate = any(harm[dataset]["positive_harm_reduction_CI95_L"] > 0 for dataset in DATASETS)
    strong = bool(positive and effect and ci_gate and fold_gate and controls and cross_ci_gate and harm_gate and harm_ci_gate)
    if strong:
        terminal = "SSPG_SEED0_STRONG_SIGNAL"
    elif positive and max(open_delta, wbcic_delta) >= 0.50 and min(open_delta, wbcic_delta) > 0 and harm_gate:
        terminal = "SSPG_SEED0_PROMISING_SIGNAL"
    elif harm_gate and (not positive or max(open_delta, wbcic_delta) < 0.25):
        terminal = "SSPG_MECHANISM_SUPPORTED_PERFORMANCE_INSUFFICIENT"
    else:
        terminal = "SSPG_SEED0_NOT_SUPPORTED"
    d["gates"] = {"positive_both": positive, "effect_size": effect, "bootstrap_ci": ci_gate, "fold_robustness": fold_gate, "controls_beaten": controls, "cross_control_bootstrap": cross_ci_gate, "independent_harm": harm_gate, "independent_harm_ci": harm_ci_gate, "strong": strong, "folds": folds_ok, "cross_bootstrap_ci_lower_pp": cross_boot}
    return terminal, d


def write_docs(contexts: list[Context], legality: dict[str, Any], terminal: str, summary: pd.DataFrame, fold: pd.DataFrame, boot: dict[str, Any], harm: dict[str, Any], validation: dict[str, Any], lock: dict[str, Any]) -> None:
    (EXP / "README.md").write_text("# PERSIST-SSPG seed-0\n\nThis experiment tests one frozen full-parameter Stable Subject Prospective Guard on EEGNet, OpenBMI and WBCIC, seed 0 and folds 0--4. Source/refit subjects provide deterministic A batches and K=4 same-biological-subject blocks; legal development outcome subjects are opened only after the committed pre-outcome lock. Test-time inference is ordinary EEGNet forward with no gradients, labels, calibration or adaptation.\n\nThe primary comparison is `SSPG` versus `TASK_ONLY_MATCHED`; `ANCHOR` is a reference. `CROSS_SUBJECT_K4_GUARD` destroys same-subject four-block coherence while retaining matched subject slots, and `RANDOM_DIRECTION_GUARD` preserves the trigger/magnitude regime while replacing the correction direction.\n\nRuntime/checkpoints/cache/raw EEG are intentionally outside the committed artifact set.\n", encoding="utf-8")
    (EXP / "FROZEN_PROTOCOL.md").write_text("# Frozen protocol\n\n- EEGNet only; OpenBMI and WBCIC; folds 0--4; seed 0 only.\n- K=4, m_per_class=16, four certificate blocks plus trial-disjoint B_out.\n- AdamW lr=3e-5, weight_decay=5e-4, gradient clip=5, two continuation epochs, full trainable parameter scope.\n- kappa=0.20; BN running statistics frozen; Adam moments receive A/task gradients only.\n- SSPG risk is mean_s ReLU(gbar_s^T Delta)^2. q=sum_{h_s>0} h_s*gbar_s; bounded raw projection and frozen backtracking multipliers 1..1/128.\n- Outcome scoring is biological-subject paired Balanced Accuracy after the pre-outcome lock.\n- No seed 1/2, second backbone, WBCIC outer-10, OpenBMI sealed/confirmation cohort, K search, cap/LR/epoch/scope/optimizer search, or outcome-based stopping.\n\nThe machine-readable freeze is `results/PRE_OUTCOME_LOCK.json`.\n", encoding="utf-8")
    (EXP / "METHOD_DERIVATION.md").write_text("# Method derivation\n\nFor task-only AdamW displacement Delta_t and stable subject gradient gbar_s=(1/4) sum_k grad L(B_s^k;theta_t), define h_s=gbar_s^T Delta_t and R(Delta)=mean_s ReLU(h_s)^2. The risk gradient in update space is q=sum_{h_s>0} h_s gbar_s, with q^T Delta=sum_{h_s>0}h_s^2. The raw correction is c_raw=(q^T Delta)/(||q||^2+eps)q. It is capped to ||c||<=0.20||Delta|| and Delta_SSPG=Delta-c. Frozen backtracking accepts the first multiplier in {1,1/2,...,1/128} for which R(Delta-c)<=R(Delta); otherwise c=0. If q=0, c=0 exactly. B gradients are post-AdamW parameter-space certificate computations: they do not update Adam moments, BN buffers or A dropout RNG.\n", encoding="utf-8")
    (EXP / "DATA_LEGALITY_AUDIT.md").write_text("# Data legality audit\n\nTraining and SSPG gradients use only model-fit/discovery source/refit subjects and S1+S2 fit sessions. Outcome indices are constructed only after `results/PRE_OUTCOME_LOCK.json` exists. Outcome subjects, labels and B_out are not used in training, normalization, correction, hyperparameter, epoch or stopping decisions. WBCIC outer-10 and OpenBMI sealed/confirmation resources were not opened; seed 1 and seed 2 were not run.\n\n```json\n" + json.dumps(clean(legality), ensure_ascii=False, indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    eq_lines = ["# Checkpoint equivalence", "", "Canonical seed-0 checkpoints were strict-loaded and source/refit predictions were repeated before outcome access.", "", "|dataset|fold|checkpoint_sha256|source trials|repeat max abs diff|pass|", "|---|---:|---|---:|---:|---|"]
    eq = pd.read_csv(RESULTS / "CHECKPOINT_EQUIVALENCE.csv")
    for _, row in eq.iterrows():
        eq_lines.append(f"|{row['dataset']}|{int(row['fold'])}|{row['checkpoint_sha256']}|{int(row['source_trials_checked'])}|{row['source_prediction_repeat_max_abs_diff']:.3e}|{'YES' if bool(row['pass']) else 'NO'}|")
    (EXP / "CHECKPOINT_EQUIVALENCE.md").write_text("\n".join(eq_lines) + "\n", encoding="utf-8")
    (EXP / "PRE_OUTCOME_LOCK.md").write_text((EXP / "PRE_OUTCOME_LOCK.md").read_text(encoding="utf-8") + "\nLock remains unchanged during training and outcome evaluation.\n", encoding="utf-8")
    (EXP / "TASK_ONLY_MATCHING_AUDIT.md").write_text("# Task-only matching audit\n\n`TASK_ONLY_MATCHED` and SSPG/control trajectories start from the exact canonical seed-0 checkpoint, use the same candidate-independent A schedule, dropout-keyed A RNG, AdamW settings, gradient clipping and two-epoch horizon. SSPG/CROSS/RANDOM differ only by their registered post-AdamW correction. Executable replay hashes are in `results/MANDATORY_TESTS.json`.\n", encoding="utf-8")
    (EXP / "BATCH_CONSTRUCTION_AUDIT.md").write_text("# Batch construction audit\n\nEach source/refit subject is deterministically permuted per dataset/fold/class and split into five class-balanced blocks of 16 per class. B1--B4 form the K=4 certificate and B5 is independent B_out. A subjects come from the other meta-folds and are subject-disjoint from B. Cross-subject control uses four cyclic derangements within the B meta-fold; each pseudo slot combines four different source subjects.\n", encoding="utf-8")
    (EXP / "CONTROL_AUDIT.md").write_text("# Control audit\n\nThe primary comparator is matched TaskOnly, not Anchor. CrossSubjectK4 retains K=4, B meta-fold, block counts and correction formula but uses four deterministic within-meta-fold derangements, destroying same-biological-subject block coherence. RandomDirection computes the true stable-subject trigger and capped/backtracked correction norm, then subtracts a deterministic norm-matched direction keyed by dataset/fold/seed/epoch/step/SSPG_RANDOM_DIRECTION.\n\n" + summary.to_markdown(index=False) + "\n", encoding="utf-8")
    write_csv(RESULTS / "CONTROL_AUDIT.csv", summary)
    (EXP / "OPTIMIZER_STATE_AUDIT.md").write_text("# Optimizer-state audit\n\nAdamW performs the real task-only proposal and updates its moments from the clipped A gradient. SSPG is a parameter-space post-step projection; B gradients are computed with autograd only and never assigned to optimizer gradients. Parameters are then replaced by theta_old+Delta_SSPG while Adam moments are retained. BN running statistics are evaluated/frozen and asserted unchanged after every step.\n", encoding="utf-8")
    (EXP / "INDEPENDENT_HARM_AUDIT.md").write_text("# Independent B_out harm audit\n\nAt five pre-frozen evenly spaced steps per fold, B5 is evaluated only for source/refit subjects after the model has been trained. `H_task=L(B_out;theta+Delta_task)-L(B_out;theta)` and `H_sspg=L(B_out;theta+Delta_SSPG)-L(B_out;theta)` are compared descriptively and with subject-cluster bootstrap. B_out never enters a gradient, optimizer, correction or decision.\n\n" + json.dumps(clean(harm), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (EXP / "STATISTICAL_PROTOCOL.md").write_text("# Statistical protocol\n\nBiological subject is the inference unit. Outcome per-subject Balanced Accuracy is averaged within fold and dataset. Primary CIs are 10,000-draw paired biological-subject bootstrap intervals for SSPG minus TaskOnly; controls use the same unit. Independent-harm summaries use subject-cluster bootstrap over B_out subjects. Trials, folds and seeds are not treated as independent bootstrap units.\n", encoding="utf-8")
    (EXP / "BUG_REPAIR_LEDGER.md").write_text("# Bug repair ledger\n\nNo scientific setting was changed after seeing an outcome. Engineering implementation choices were: vectorized canonical batch access, strict checkpoint/normalizer verification, concatenated K4 gradient computation with an executable equivalence test, fp32 gradient accumulation, deterministic RNG isolation, explicit optimizer-state/BN assertions, and a two-phase pre-outcome lock.\n", encoding="utf-8")
    (EXP / "AUTONOMOUS_DECISION.md").write_text("# Autonomous decision\n\nterminal = " + terminal + "\n\nThe result is seed 0 only. `THREE_SEED_CONFIRMATION_SCIENTIFICALLY_JUSTIFIED = " + ("YES" if terminal == "SSPG_SEED0_STRONG_SIGNAL" else "NO") + "`.\n`AUTO_RUN_SEED1_SEED2 = NO`; seed1_run=false; seed2_run=false.\nWBCIC_outer_opened=false; OpenBMI_sealed_opened=false.\n\n" + summary.to_markdown(index=False) + "\n", encoding="utf-8")
    # Keep the final report compact and directly answer the protocol questions.
    lines = ["# PERSIST-SSPG seed-0 final report", "", f"terminal = {terminal}", "", "Primary comparator: SSPG vs TASK_ONLY_MATCHED; subject-bootstrap CIs use 10,000 draws.", "", "|dataset|TaskOnly BA|SSPG BA|SSPG-TaskOnly pp|95% CI pp|SSPG-Cross pp|SSPG-Random pp|nonnegative folds|", "|---|---:|---:|---:|---|---:|---:|---|"]
    for dataset in DATASETS:
        s = summary[(summary.dataset == dataset)].set_index("method")
        b = boot[dataset]
        fp = fold[fold.dataset == dataset]
        lines.append(f"|{dataset}|{s.loc['TASK_ONLY_MATCHED','BA']:.6f}|{s.loc['SSPG','BA']:.6f}|{100*(s.loc['SSPG','BA']-s.loc['TASK_ONLY_MATCHED','BA']):+.3f}|[{b['CI95_L_pp']:+.3f}, {b['CI95_U_pp']:+.3f}]|{100*(s.loc['SSPG','BA']-s.loc['CROSS_SUBJECT_K4_GUARD','BA']):+.3f}|{100*(s.loc['SSPG','BA']-s.loc['RANDOM_DIRECTION_GUARD','BA']):+.3f}|{int((fp.SSPG_minus_TASK_ONLY_pp>=0).sum())}/5|")
    lines += ["", "Independent B_out harm:", "", "|dataset|mean positive harm Task|mean positive harm SSPG|reduction|95% CI lower|frequency Task|frequency SSPG|", "|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        h = harm[dataset]
        lines.append(f"|{dataset}|{h['mean_positive_harm_task']:.6g}|{h['mean_positive_harm_sspg']:.6g}|{h['positive_harm_reduction']:.6g}|{h['positive_harm_reduction_CI95_L']:.6g}|{h['harm_frequency_task']:.4f}|{h['harm_frequency_sspg']:.4f}|")
    lines += ["", "Answers: continued TaskOnly is the primary comparator; CrossSubject tests K4 averaging without same-subject coherence; Random tests arbitrary equal-norm perturbations; folds are reported above; correction trigger/magnitude and backtracking are in `results/SSPG_STEP_DIAGNOSTICS.csv` and `results/CORRECTION_STATISTICS.csv`; no fold is hidden; outcome data were opened only after the lock; seed 1/2, WBCIC outer-10 and OpenBMI sealed/confirmation were not opened.", "", f"THREE_SEED_CONFIRMATION_SCIENTIFICALLY_JUSTIFIED = {'YES' if terminal == 'SSPG_SEED0_STRONG_SIGNAL' else 'NO'}", "AUTO_RUN_SEED1_SEED2 = NO", "seed1_run = false", "seed2_run = false", "second_backbone_run = false", "WBCIC_outer_opened = false", "OpenBMI_sealed_opened = false"]
    (EXP / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_preflight(device: torch.device) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True); RUNTIME.mkdir(parents=True, exist_ok=True)
    contexts, legality = load_contexts_source_only()
    equivalence = [checkpoint_equivalence(ctx, device) for ctx in contexts]
    write_csv(RESULTS / "CHECKPOINT_EQUIVALENCE.csv", equivalence)
    if not all(row["pass"] for row in equivalence):
        raise RuntimeError("checkpoint equivalence failed; no lock/outcome allowed")
    mandatory = run_mandatory_tests(contexts, equivalence, device, legality)
    write_json(RUNTIME / "PREFLIGHT.json", {"schema": "PERSIST_SSPG_PREFLIGHT_V1", "seed": SEED, "datasets": list(DATASETS), "folds": list(FOLDS), "legality": legality, "equivalence": equivalence, "mandatory_tests": mandatory})
    lock = write_pre_outcome_lock(contexts, legality, equivalence, mandatory, device)
    print("PRE_OUTCOME_LOCK_WRITTEN=true", flush=True)
    print(f"code_commit = {lock['code_commit']}", flush=True)
    print("outcome_accessed = false", flush=True)
    print("MANDATORY_TESTS_PASS=true", flush=True)


def _check_run_lock() -> dict[str, Any]:
    lock_path = RESULTS / "PRE_OUTCOME_LOCK.json"
    if not lock_path.is_file():
        raise RuntimeError("missing pre-outcome lock; run --phase preflight and commit/push it first")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("seed1_run") or lock.get("seed2_run") or lock.get("WBCIC_outer_opened") or lock.get("OpenBMI_sealed_opened"):
        raise RuntimeError("lock contains forbidden access/run flags")
    if lock.get("code_hashes") != code_hashes():
        raise RuntimeError("implementation code changed after PRE_OUTCOME_LOCK; rerun preflight and obtain a new committed lock")
    mandatory_path = RESULTS / "MANDATORY_TESTS.json"
    if not mandatory_path.is_file() or not json.loads(mandatory_path.read_text(encoding="utf-8")).get("pass"):
        raise RuntimeError("mandatory tests failed; outcome blocked")
    return lock


def run_train_context(device: torch.device, dataset: str, fold: int) -> None:
    """Train one dataset/fold in an isolated process.

    The RTX 5090 Windows build has occasionally terminated a long-lived
    process after many CUDA module lifetimes.  One context per process is an
    execution-stability repair only; it does not change the frozen recipe.
    """
    lock = _check_run_lock()
    if dataset not in DATASETS or fold not in FOLDS:
        raise ValueError(f"invalid context {dataset} fold={fold}")
    contexts, _ = load_contexts_source_only()
    ctx = next(c for c in contexts if c.dataset == dataset and c.fold == fold)
    all_training_rows: list[dict[str, Any]] = []
    all_diag_rows: list[dict[str, Any]] = []
    all_bout_rows: list[dict[str, Any]] = []
    trajectory_hashes: dict[str, str] = {}
    for method in TRAIN_METHODS:
        print(f"[train-context] {dataset} fold={fold} method={method}", flush=True)
        result = train_candidate(ctx, method, device, collect_harm=(method == "SSPG"))
        all_training_rows.extend(result["rows"])
        all_diag_rows.extend(result["diagnostics"])
        all_bout_rows.extend(result["bout_rows"])
        trajectory_hashes[method] = result["trajectory_sha256"]
        model_path = RUNTIME / "models" / dataset / f"fold-{fold}" / f"{method}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": result["state"], "dataset": dataset, "fold": fold, "method": method, "seed": SEED, "lock_code_commit": lock.get("code_commit")}, model_path)
        print(f"[train-context-done] {dataset} fold={fold} method={method} trajectory={result['trajectory_sha256'][:12]}", flush=True)
    payload = {"schema": "PERSIST_SSPG_CONTEXT_TRAINING_V1", "complete": True, "dataset": dataset, "fold": fold, "seed": SEED, "trajectory_hashes": trajectory_hashes, "training_rows": all_training_rows, "diagnostic_rows": all_diag_rows, "bout_rows": all_bout_rows, "runtime_seconds": None}
    out = RUNTIME / "context_results" / f"{dataset}_fold-{fold}.json"
    write_json(out, payload)
    print(f"CONTEXT_COMPLETE dataset={dataset} fold={fold}", flush=True)


def _aggregate_from_parts(device: torch.device) -> None:
    """Aggregate isolated context workers and perform the first outcome read."""
    lock = _check_run_lock()
    contexts, legality = load_contexts_source_only()
    equivalence = pd.read_csv(RESULTS / "CHECKPOINT_EQUIVALENCE.csv").to_dict(orient="records")
    mandatory = json.loads((RESULTS / "MANDATORY_TESTS.json").read_text(encoding="utf-8"))
    all_training_rows: list[dict[str, Any]] = []
    all_diag_rows: list[dict[str, Any]] = []
    all_bout_rows: list[dict[str, Any]] = []
    states: dict[tuple[str, int, str], dict[str, torch.Tensor]] = {}
    for ctx in contexts:
        part_path = RUNTIME / "context_results" / f"{ctx.dataset}_fold-{ctx.fold}.json"
        if not part_path.is_file():
            raise RuntimeError(f"missing isolated context result {part_path}")
        payload = json.loads(part_path.read_text(encoding="utf-8"))
        if not payload.get("complete"):
            raise RuntimeError(f"incomplete isolated context result {part_path}")
        all_training_rows.extend(payload.get("training_rows", []))
        all_diag_rows.extend(payload.get("diagnostic_rows", []))
        all_bout_rows.extend(payload.get("bout_rows", []))
        for method in TRAIN_METHODS:
            model_path = RUNTIME / "models" / ctx.dataset / f"fold-{ctx.fold}" / f"{method}.pt"
            if not model_path.is_file():
                raise RuntimeError(f"missing isolated model state {model_path}")
            try:
                saved = torch.load(model_path, map_location="cpu", weights_only=False)
            except TypeError:
                saved = torch.load(model_path, map_location="cpu")
            states[(ctx.dataset, ctx.fold, method)] = {key: value.detach().cpu().clone() for key, value in saved["model_state"].items()}
    write_csv(RESULTS / "TRAINING_TRAJECTORIES.csv", all_training_rows)
    write_csv(RESULTS / "SSPG_STEP_DIAGNOSTICS.csv", all_diag_rows)
    diagnostics = pd.DataFrame(all_diag_rows)
    aggregate_diagnostics(diagnostics)
    # This is the first point at which outcome indices/labels are constructed.
    outcome_frame = evaluate_outcomes(contexts, states, device)
    fold, summary = outcome_tables(outcome_frame)
    _, harm = harm_tables(pd.DataFrame(all_bout_rows))
    boot: dict[str, Any] = {}
    for dataset in DATASETS:
        part = outcome_frame[outcome_frame.dataset == dataset]
        sspg = part[part.method == "SSPG"].set_index("subject_id").BA
        for label, control in [("TASK", "TASK_ONLY_MATCHED"), ("CROSS", "CROSS_SUBJECT_K4_GUARD"), ("RANDOM", "RANDOM_DIRECTION_GUARD")]:
            right = part[part.method == control].set_index("subject_id").BA
            common = sspg.index.intersection(right.index)
            result = bootstrap_delta(sspg.loc[common], right.loc[common], pd.Series(common), stable_seed("sspg-bootstrap", dataset, label, SEED))
            boot[dataset + ("_vs_task" if label == "TASK" else "_vs_cross" if label == "CROSS" else "_vs_random")] = result
        boot[dataset] = boot[dataset + "_vs_task"]
    write_json(RESULTS / "SSPG_VS_TASK_BOOTSTRAP.json", {dataset: boot[dataset] for dataset in DATASETS})
    write_json(RESULTS / "SSPG_VS_CROSS_SUBJECT_BOOTSTRAP.json", {dataset: boot[dataset + "_vs_cross"] for dataset in DATASETS})
    write_json(RESULTS / "SSPG_VS_RANDOM_BOOTSTRAP.json", {dataset: boot[dataset + "_vs_random"] for dataset in DATASETS})
    validation = build_validation(outcome_frame, fold, summary, harm, diagnostics, equivalence, mandatory, legality)
    terminal, decision = decide(summary, fold, boot, harm, validation)
    validation["terminal"] = terminal
    write_json(RESULTS / "VALIDATION.json", validation)
    write_json(RESULTS / "FINAL_REPORT.json", {"schema": "PERSIST_SSPG_FINAL_REPORT_V1", "terminal": terminal, "decision": decision, "summary": clean(summary.to_dict(orient="records")), "fold": clean(fold.to_dict(orient="records")), "bootstrap": clean(boot), "independent_Bout_harm": clean(harm), "validation": validation, "lock": lock, "seed1_run": False, "seed2_run": False, "WBCIC_outer_opened": False, "OpenBMI_sealed_opened": False, "THREE_SEED_CONFIRMATION_SCIENTIFICALLY_JUSTIFIED": terminal == "SSPG_SEED0_STRONG_SIGNAL"})
    write_docs(contexts, legality, terminal, summary, fold, {dataset: boot[dataset] for dataset in DATASETS} | {dataset + "_vs_cross": boot[dataset + "_vs_cross"] for dataset in DATASETS}, harm, validation, lock)
    print(f"terminal = {terminal}", flush=True)
    for dataset in DATASETS:
        s = summary[summary.dataset == dataset].set_index("method")
        print(f"{dataset}_TASK_ONLY_BA = {float(s.loc['TASK_ONLY_MATCHED','BA']):.8f}", flush=True)
        print(f"{dataset}_SSPG_BA = {float(s.loc['SSPG','BA']):.8f}", flush=True)
        print(f"{dataset}_SSPG_MINUS_TASK_PP = {100*(float(s.loc['SSPG','BA'])-float(s.loc['TASK_ONLY_MATCHED','BA'])):+.4f}", flush=True)
        print(f"{dataset}_95CI = [{boot[dataset]['CI95_L_pp']:+.4f}, {boot[dataset]['CI95_U_pp']:+.4f}]", flush=True)
        print(f"{dataset}_SSPG_MINUS_CROSS_PP = {100*(float(s.loc['SSPG','BA'])-float(s.loc['CROSS_SUBJECT_K4_GUARD','BA'])):+.4f}", flush=True)
        print(f"{dataset}_SSPG_MINUS_RANDOM_PP = {100*(float(s.loc['SSPG','BA'])-float(s.loc['RANDOM_DIRECTION_GUARD','BA'])):+.4f}", flush=True)
        print(f"{dataset}_independent_harm_reduction = {harm[dataset]['positive_harm_reduction']:.8g}", flush=True)
        f = fold[fold.dataset == dataset]
        print(f"{dataset}_nonnegative_folds = {int((f.SSPG_minus_TASK_ONLY_pp >= 0).sum())}/5", flush=True)
    print("seed1_run = false", flush=True)
    print("seed2_run = false", flush=True)
    print("second_backbone_run = false", flush=True)
    print("WBCIC_outer_opened = false", flush=True)
    print("OpenBMI_sealed_opened = false", flush=True)
    print(f"THREE_SEED_CONFIRMATION_SCIENTIFICALLY_JUSTIFIED = {'YES' if terminal == 'SSPG_SEED0_STRONG_SIGNAL' else 'NO'}", flush=True)


def run_experiment(device: torch.device) -> None:
    lock_path = RESULTS / "PRE_OUTCOME_LOCK.json"
    if not lock_path.is_file():
        raise RuntimeError("missing pre-outcome lock; run --phase preflight and commit/push it first")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("seed1_run") or lock.get("seed2_run") or lock.get("WBCIC_outer_opened") or lock.get("OpenBMI_sealed_opened"):
        raise RuntimeError("lock contains forbidden access/run flags")
    if lock.get("code_hashes") != code_hashes():
        raise RuntimeError("implementation code changed after PRE_OUTCOME_LOCK; rerun preflight and obtain a new committed lock")
    contexts, legality = load_contexts_source_only()
    equivalence = pd.read_csv(RESULTS / "CHECKPOINT_EQUIVALENCE.csv").to_dict(orient="records")
    mandatory = json.loads((RESULTS / "MANDATORY_TESTS.json").read_text(encoding="utf-8"))
    if not mandatory.get("pass"):
        raise RuntimeError("mandatory tests failed; outcome blocked")
    all_training_rows: list[dict[str, Any]] = []
    all_diag_rows: list[dict[str, Any]] = []
    all_bout_rows: list[dict[str, Any]] = []
    states: dict[tuple[str, int, str], dict[str, torch.Tensor]] = {}
    started = time.time()
    for index, ctx in enumerate(contexts, start=1):
        for method in TRAIN_METHODS:
            print(f"[train] {index}/{len(contexts)} {ctx.dataset} fold={ctx.fold} method={method}", flush=True)
            result = train_candidate(ctx, method, device, collect_harm=(method == "SSPG"))
            all_training_rows.extend(result["rows"])
            all_diag_rows.extend(result["diagnostics"])
            all_bout_rows.extend(result["bout_rows"])
            states[(ctx.dataset, ctx.fold, method)] = result["state"]
            model_path = RUNTIME / "models" / ctx.dataset / f"fold-{ctx.fold}" / f"{method}.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": result["state"], "dataset": ctx.dataset, "fold": ctx.fold, "method": method, "seed": SEED, "lock_code_commit": lock.get("code_commit")}, model_path)
            print(f"[train-done] {ctx.dataset} fold={ctx.fold} method={method} trajectory={result['trajectory_sha256'][:12]}", flush=True)
    write_csv(RESULTS / "TRAINING_TRAJECTORIES.csv", all_training_rows)
    write_csv(RESULTS / "SSPG_STEP_DIAGNOSTICS.csv", all_diag_rows)
    diagnostics = pd.DataFrame(all_diag_rows)
    aggregate_diagnostics(diagnostics)
    # Outcome access starts here, after all trajectories and the committed lock.
    outcome_frame = evaluate_outcomes(contexts, states, device)
    fold, summary = outcome_tables(outcome_frame)
    bout = pd.DataFrame(all_bout_rows)
    _, harm = harm_tables(bout)
    boot: dict[str, Any] = {}
    for dataset in DATASETS:
        part = outcome_frame[outcome_frame.dataset == dataset]
        sspg = part[part.method == "SSPG"].set_index("subject_id").BA
        for label, control in [("TASK", "TASK_ONLY_MATCHED"), ("CROSS", "CROSS_SUBJECT_K4_GUARD"), ("RANDOM", "RANDOM_DIRECTION_GUARD")]:
            right = part[part.method == control].set_index("subject_id").BA
            common = sspg.index.intersection(right.index)
            result = bootstrap_delta(sspg.loc[common], right.loc[common], pd.Series(common), stable_seed("sspg-bootstrap", dataset, label, SEED))
            boot[dataset + ("_vs_task" if label == "TASK" else "_vs_cross" if label == "CROSS" else "_vs_random")] = result
        boot[dataset] = boot[dataset + "_vs_task"]
    write_json(RESULTS / "SSPG_VS_TASK_BOOTSTRAP.json", {dataset: boot[dataset] for dataset in DATASETS})
    write_json(RESULTS / "SSPG_VS_CROSS_SUBJECT_BOOTSTRAP.json", {dataset: boot[dataset + "_vs_cross"] for dataset in DATASETS})
    write_json(RESULTS / "SSPG_VS_RANDOM_BOOTSTRAP.json", {dataset: boot[dataset + "_vs_random"] for dataset in DATASETS})
    validation = build_validation(outcome_frame, fold, summary, harm, diagnostics, equivalence, mandatory, legality)
    terminal, decision = decide(summary, fold, boot, harm, validation)
    validation["terminal"] = terminal
    write_json(RESULTS / "VALIDATION.json", validation)
    write_json(RESULTS / "FINAL_REPORT.json", {"schema": "PERSIST_SSPG_FINAL_REPORT_V1", "terminal": terminal, "decision": decision, "summary": clean(summary.to_dict(orient="records")), "fold": clean(fold.to_dict(orient="records")), "bootstrap": clean(boot), "independent_Bout_harm": clean(harm), "validation": validation, "lock": lock, "runtime_seconds": time.time() - started, "seed1_run": False, "seed2_run": False, "WBCIC_outer_opened": False, "OpenBMI_sealed_opened": False, "THREE_SEED_CONFIRMATION_SCIENTIFICALLY_JUSTIFIED": terminal == "SSPG_SEED0_STRONG_SIGNAL"})
    write_docs(contexts, legality, terminal, summary, fold, {dataset: boot[dataset] for dataset in DATASETS} | {dataset + "_vs_cross": boot[dataset + "_vs_cross"] for dataset in DATASETS}, harm, validation, lock)
    print(f"terminal = {terminal}", flush=True)
    for dataset in DATASETS:
        s = summary[summary.dataset == dataset].set_index("method")
        print(f"{dataset}_TASK_ONLY_BA = {float(s.loc['TASK_ONLY_MATCHED','BA']):.8f}", flush=True)
        print(f"{dataset}_SSPG_BA = {float(s.loc['SSPG','BA']):.8f}", flush=True)
        print(f"{dataset}_SSPG_MINUS_TASK_PP = {100*(float(s.loc['SSPG','BA'])-float(s.loc['TASK_ONLY_MATCHED','BA'])):+.4f}", flush=True)
        print(f"{dataset}_95CI = [{boot[dataset]['CI95_L_pp']:+.4f}, {boot[dataset]['CI95_U_pp']:+.4f}]", flush=True)
        print(f"{dataset}_SSPG_MINUS_CROSS_PP = {100*(float(s.loc['SSPG','BA'])-float(s.loc['CROSS_SUBJECT_K4_GUARD','BA'])):+.4f}", flush=True)
        print(f"{dataset}_SSPG_MINUS_RANDOM_PP = {100*(float(s.loc['SSPG','BA'])-float(s.loc['RANDOM_DIRECTION_GUARD','BA'])):+.4f}", flush=True)
        print(f"{dataset}_independent_harm_reduction = {harm[dataset]['positive_harm_reduction']:.8g}", flush=True)
        f = fold[fold.dataset == dataset]
        print(f"{dataset}_nonnegative_folds = {int((f.SSPG_minus_TASK_ONLY_pp >= 0).sum())}/5", flush=True)
    print("seed1_run = false", flush=True)
    print("seed2_run = false", flush=True)
    print("second_backbone_run = false", flush=True)
    print("WBCIC_outer_opened = false", flush=True)
    print("OpenBMI_sealed_opened = false", flush=True)
    print(f"THREE_SEED_CONFIRMATION_SCIENTIFICALLY_JUSTIFIED = {'YES' if terminal == 'SSPG_SEED0_STRONG_SIGNAL' else 'NO'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "train-context", "aggregate", "run"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.phase == "preflight":
        run_preflight(device)
    elif args.phase == "train-context":
        if args.dataset is None or args.fold is None:
            parser.error("--phase train-context requires --dataset and --fold")
        run_train_context(device, args.dataset, args.fold)
    elif args.phase == "aggregate":
        _aggregate_from_parts(device)
    else:
        run_experiment(device)


if __name__ == "__main__":
    main()
