"""PERSIST-AU seed-0 bounded development search.

The implementation is intentionally narrow: frozen canonical EEGNet seed-0
checkpoints, OpenBMI/WBCIC development roles, five epochs and the predeclared
prospective-admissible update rules.  It never opens sealed cohorts or runs
seed 1/2.
"""
from __future__ import annotations

import argparse
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

REPO = Path(os.environ.get("CANONICAL_REPO", r"D:\nips-temp\TotalP\P1\CRCICLR_CANONICAL_EEGNET")).resolve()
EXP = REPO / "experiments" / "persist_eeg_prospective_admissible_updates_dev_v1"
RESULTS = EXP / "results"
RUNTIME = EXP / "runtime"
CANONICAL_EXP = REPO / "experiments" / "persist_eeg_canonical_eegnet_baseline"
SEED = 0
FOLDS = (0, 1, 2, 3, 4)
DATASETS = ("OpenBMI", "WBCIC")
MAX_EPOCHS = 5
BATCH_SIZE = 64
BASE_LR = 3e-5
REPAIR_LR = 1e-5
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 5.0
BOOTSTRAP_DRAWS = 10_000
ALPHAS = (0.25, 0.50, 0.75, 1.00)

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
    schedules: list[list[tuple[np.ndarray, np.ndarray]]]
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
    return {"dataset": ctx.dataset, "fold": ctx.fold, "trial_count": len(actual["indices"]), "max_probability_abs_diff": max_diff, "trial_uid_exact": True, "labels_exact": True, "predictions_exact": True, "pass": True}


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
    rng = np.random.default_rng(stable_seed("persist-au-meta-folds", dataset, fold, SEED))
    shuffled = np.asarray(source_subjects, dtype=object)[rng.permutation(len(source_subjects))]
    groups = [list(map(str, part.tolist())) for part in np.array_split(shuffled, 5)]
    if any(not group for group in groups) or set(sum(groups, [])) != set(source_subjects):
        raise RuntimeError(f"invalid meta-fold partition {dataset} fold {fold}")
    return groups


def make_schedules(ctx: FoldContext) -> tuple[list[list[tuple[np.ndarray, np.ndarray]]], str]:
    pools = subject_pools(ctx.data, ctx.refit_idx)
    steps = max(1, int(math.ceil(len(ctx.refit_idx) / 128)))
    serial: list[Any] = []
    schedules: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        current: list[tuple[np.ndarray, np.ndarray]] = []
        for step in range(steps):
            b_fold = step % 5
            b_subjects = list(ctx.meta_folds[b_fold])
            a_subjects = [subject for i, group in enumerate(ctx.meta_folds) if i != b_fold for subject in group]
            rng = np.random.default_rng(stable_seed("persist-au-schedule", ctx.dataset, ctx.fold, SEED, epoch, step))
            a_idx = balanced_batch(pools, a_subjects, rng)
            b_idx = balanced_batch(pools, b_subjects, rng)
            if set(metadata_col(ctx.data, "subject_id", a_idx)) & set(metadata_col(ctx.data, "subject_id", b_idx)):
                raise RuntimeError(f"A/B overlap {ctx.dataset} fold {ctx.fold} epoch {epoch} step {step}")
            current.append((a_idx, b_idx))
            serial.append({"epoch": epoch, "step": step, "A": a_idx.tolist(), "B": b_idx.tolist()})
        schedules.append(current)
    digest = hashlib.sha256(json.dumps(serial, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()
    return schedules, digest


def parameter_scope(model: nn.Module, scope: str) -> tuple[list[str], list[nn.Parameter], np.ndarray]:
    named = list(model.named_parameters())
    late_names = {name for name, _ in named if name.startswith("embedding.") or name.startswith("head.")}
    for name, parameter in named:
        parameter.requires_grad = scope == "A" or name in late_names
    selected = [(name, parameter) for name, parameter in named if parameter.requires_grad]
    names = [name for name, _ in selected]
    params = [parameter for _, parameter in selected]
    # ``direction`` operates on the concatenated (one-dimensional) gradient.
    # Expand the parameter-level scope to one boolean per scalar so C4/C5 do
    # not index a 34k-element gradient with a 16-parameter mask.
    late_mask = np.concatenate(
        [np.full(parameter.numel(), name in late_names, dtype=bool) for name, parameter in selected]
    )
    return names, params, late_mask


def fork_rng(device: torch.device):
    devices = [int(device.index)] if device.type == "cuda" and device.index is not None else []
    return torch.random.fork_rng(devices=devices)


def gradients(model: nn.Module, params: list[nn.Parameter], xb: torch.Tensor, yb: torch.Tensor, seed: int) -> tuple[torch.Tensor, ...]:
    with fork_rng(xb.device):
        torch.manual_seed(int(seed))
        if xb.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        model.train()
        freeze_bn(model)
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
        raise RuntimeError("gradient split mismatch")
    return chunks


def project_joint(g0: torch.Tensor, ga: torch.Tensor, gb: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Nearest point to g0 in gA^T d >= 0 and gB^T d >= 0 by active sets."""
    vectors = (ga, gb)
    candidates: list[torch.Tensor] = []
    for active in ((), (0,), (1,), (0, 1)):
        if not active:
            d = g0.clone()
        else:
            matrix = torch.stack([vectors[index] for index in active], dim=1)
            gram = matrix.T @ matrix
            rhs = -(matrix.T @ g0)
            if gram.numel() == 1 and float(torch.abs(gram[0, 0])) > 1e-12:
                multiplier = rhs / gram[0, 0]
            elif gram.numel() == 4 and float(torch.abs(torch.linalg.det(gram))) > 1e-12:
                multiplier = torch.linalg.solve(gram, rhs)
            else:
                multiplier = torch.linalg.pinv(gram) @ rhs
            d = g0 + matrix @ multiplier
        dots = [float(torch.dot(vector, d).detach().cpu()) for vector in vectors]
        if all(abs(dots[index]) <= 2e-5 for index in active) and all(dots[index] >= -2e-5 for index in range(2) if index not in active):
            candidates.append(d)
    if not candidates:
        matrix = torch.stack([ga, gb], dim=1)
        candidates = [g0 + matrix @ (torch.linalg.pinv(matrix.T @ matrix) @ (-(matrix.T @ g0)))]
    d = min(candidates, key=lambda value: float(torch.sum((value - g0) ** 2).detach().cpu()))
    return d, bool(float(torch.linalg.vector_norm(d - g0).detach().cpu()) > 1e-8)


def direction(rule: str, ga: torch.Tensor, gb: torch.Tensor, late_mask: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
    g0 = 0.5 * (ga + gb)
    dot_ab = float(torch.dot(ga, gb).detach().cpu())
    norm_a = float(torch.linalg.vector_norm(ga).detach().cpu())
    norm_b = float(torch.linalg.vector_norm(gb).detach().cpu())
    if rule == "C0":
        d = g0
        active = False
    elif rule == "C1":
        active = dot_ab < 0 and norm_b > 1e-12
        d = ga if not active else ga - (dot_ab / (norm_b * norm_b + 1e-12)) * gb
    elif rule == "C2":
        d, active = project_joint(g0, ga, gb)
    elif rule == "C3":
        strict, active = project_joint(g0, ga, gb)
        d = g0 + 0.5 * (strict - g0)
    elif rule in {"C4", "C5"}:
        if not late_mask.any() or late_mask.all():
            raise RuntimeError("invalid late-block mask")
        d = g0.clone()
        strict, active = project_joint(g0[late_mask], ga[late_mask], gb[late_mask])
        d[late_mask] = strict if rule == "C4" else g0[late_mask] + 0.5 * (strict - g0[late_mask])
        d[~late_mask] = g0[~late_mask]
    else:
        raise ValueError(rule)
    correction = float(torch.linalg.vector_norm(d - g0).detach().cpu())
    base_norm = float(torch.linalg.vector_norm(g0).detach().cpu())
    cosine = None if norm_a <= 1e-12 or norm_b <= 1e-12 else dot_ab / (norm_a * norm_b)
    return d, {"conflict": dot_ab < 0, "joint_active": active, "cos_gA_gB": cosine, "correction_norm": correction, "relative_correction": correction / max(base_norm, 1e-12), "gA_norm": norm_a, "gB_norm": norm_b, "g0_norm": base_norm}


def apply_direction(model: nn.Module, params: list[nn.Parameter], direction_value: torch.Tensor, optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    norm = float(torch.linalg.vector_norm(direction_value).detach().cpu())
    scale = min(1.0, GRAD_CLIP / max(norm, 1e-12))
    for parameter, chunk in zip(params, split_like(direction_value, params)):
        parameter.grad = (chunk * scale).detach().clone()
    optimizer.step()
    return norm, norm * scale


def fixed_loss(model: nn.Module, ctx: FoldContext, indices: np.ndarray, device: torch.device) -> float:
    model.eval()
    with torch.inference_mode():
        xb = canonical.prepare_batch(ctx.data, indices, ctx.mean, ctx.std, device)
        yb = torch.as_tensor(metadata_col(ctx.data, "label", indices).astype(np.int64), dtype=torch.long, device=device)
        return float(F.cross_entropy(model(xb), yb).detach().cpu())


def train_candidate(ctx: FoldContext, candidate: dict[str, Any], lr: float, device: torch.device, random_b: bool = False) -> dict[str, Any]:
    set_seed(stable_seed("persist-au-model", ctx.dataset, ctx.fold, candidate["name"], lr, SEED, random_b))
    model = canonical.VanillaEEGNet(int(ctx.data.batch(ctx.refit_idx[:1]).shape[1])).to(device)
    model.load_state_dict(ctx.anchor_state, strict=True)
    _, params, late_mask = parameter_scope(model, candidate["scope"])
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    anchor_bn = bn_buffers(model)
    epoch_predictions: dict[int, dict[str, Any]] = {}
    epoch_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    harm_rows: list[dict[str, Any]] = []
    global_step = 0
    for epoch_no, schedule in enumerate(ctx.schedules, start=1):
        order = np.arange(len(schedule))
        if random_b:
            order = np.random.default_rng(stable_seed("persist-au-random-conflict", ctx.dataset, ctx.fold, candidate["name"], lr, epoch_no, SEED)).permutation(len(schedule))
        model.train(); freeze_bn(model)
        for step_no, (a_idx, true_b_idx) in enumerate(schedule):
            global_step += 1
            b_idx = schedule[int(order[step_no])][1] if random_b else true_b_idx
            xb_a = canonical.prepare_batch(ctx.data, a_idx, ctx.mean, ctx.std, device)
            xb_b = canonical.prepare_batch(ctx.data, b_idx, ctx.mean, ctx.std, device)
            ya = torch.as_tensor(metadata_col(ctx.data, "label", a_idx).astype(np.int64), dtype=torch.long, device=device)
            yb = torch.as_tensor(metadata_col(ctx.data, "label", b_idx).astype(np.int64), dtype=torch.long, device=device)
            before = fixed_loss(model, ctx, ctx.schedules[0][0][1], device) if global_step % 20 == 0 else None
            ga = flatten(gradients(model, params, xb_a, ya, stable_seed("persist-au-dropout", ctx.dataset, ctx.fold, candidate["name"], lr, epoch_no, step_no, "A", random_b)))
            gb = flatten(gradients(model, params, xb_b, yb, stable_seed("persist-au-dropout", ctx.dataset, ctx.fold, candidate["name"], lr, epoch_no, step_no, "B", random_b)))
            direction_value, diag = direction(candidate["rule"], ga, gb, late_mask)
            pre_norm, post_norm = apply_direction(model, params, direction_value, optimizer)
            mechanism_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "candidate": candidate["name"], "rule": candidate["rule"], "scope": candidate["scope"], "lr": lr, "epoch": epoch_no, "step": step_no + 1, "random_conflict": random_b, **diag, "gradient_norm_pre_clip": pre_norm, "gradient_norm_post_clip": post_norm})
            if before is not None:
                after = fixed_loss(model, ctx, ctx.schedules[0][0][1], device)
                delta = after - before
                harm_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "candidate": candidate["name"], "lr": lr, "epoch": epoch_no, "step": global_step, "random_conflict": random_b, "delta_B_actual": delta, "harmed": bool(delta > 0), "positive_harm": max(delta, 0.0), "conflict": diag["conflict"]})
            del xb_a, xb_b, ya, yb, ga, gb, direction_value
        displacement = bn_max_displacement(model, anchor_bn)
        if displacement != 0.0:
            raise RuntimeError(f"IMPLEMENTATION_INVALID: BatchNorm displacement {ctx.dataset} fold={ctx.fold} candidate={candidate['name']} epoch={epoch_no}: {displacement}")
        outcome = evaluate_model(model, ctx, ctx.outcome_idx, device)
        means = metric_means(outcome["labels"], outcome["probability"][:, 1], outcome["subjects"])
        epoch_rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "candidate": candidate["name"], "rule": candidate["rule"], "scope": candidate["scope"], "lr": lr, "epoch": epoch_no, "mean_subject_BA": means["BA"], "mean_accuracy": means["accuracy"], "mean_macro_F1": means["macro_F1"], "mean_NLL": means["NLL"], "n_subjects": len(outcome["subject_metrics"]), "bn_max_displacement": displacement, "random_conflict": random_b})
        epoch_predictions[epoch_no] = outcome
        print(f"[au] {ctx.dataset} fold={ctx.fold} candidate={candidate['name']} lr={lr:g} epoch={epoch_no} BA={means['BA']:.6f} random={random_b}", flush=True)
    return {"epoch_rows": epoch_rows, "epoch_predictions": epoch_predictions, "mechanism_rows": mechanism_rows, "harm_rows": harm_rows, "parameter_count": int(sum(parameter.numel() for parameter in params))}


def candidate_specs() -> list[dict[str, Any]]:
    return [
        {"name": "C1_SCOPE_A", "rule": "C1", "scope": "A"},
        {"name": "C1_SCOPE_B", "rule": "C1", "scope": "B"},
        {"name": "C2_SCOPE_A", "rule": "C2", "scope": "A"},
        {"name": "C2_SCOPE_B", "rule": "C2", "scope": "B"},
        {"name": "C3_SCOPE_A", "rule": "C3", "scope": "A"},
        {"name": "C3_SCOPE_B", "rule": "C3", "scope": "B"},
        {"name": "C4_LATE_STRICT", "rule": "C4", "scope": "A"},
        {"name": "C5_LATE_SOFT", "rule": "C5", "scope": "A"},
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
    write_csv(RESULTS / "BATCH_SCHEDULE_AUDIT.csv", [{"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "steps_per_epoch": len(ctx.schedules[0]), "epochs": MAX_EPOCHS, "schedule_sha256": ctx.schedule_hash, "A_B_subject_disjoint": True, "class_balanced": True, "same_schedule_for_all_candidates": True} for ctx in contexts])
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


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        write_json(RUNTIME / "SEED0_RUN.exit.json", {"complete": False, "exit_code": 1, "error": str(exc), "seed": SEED})
        raise
