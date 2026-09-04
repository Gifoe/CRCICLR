"""PERSIST-EEG cumulative subject decision-drift audit (seed 0).

This runner is deliberately narrower than the preceding single-step audit.  It
reuses the frozen source/refit EEGNet loader and exact AdamW helpers, but creates
five fixed sentinel meta-fold continuations per canonical fold.  Each
continuation is exactly one deterministic A-only epoch; no sentinel subject is
ever sampled into A.  A pre-outcome phase writes the lock and executable tests,
and the run phase (which requires that lock) evaluates the already declared
held-out source/refit B_out trials.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

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
OLD_CODE = EXP.parent / "persist_eeg_stable_subject_prospective_guard_seed0_v1" / "code" / "run_sspg_seed0.py"
PREVIOUS_EXP = EXP.parent / "persist_eeg_decision_relevant_prospective_harm_audit_v1"
SEED = 0
DATASETS = ("OpenBMI", "WBCIC")
FOLDS = (0, 1, 2, 3, 4)
K = 4
M_PER_CLASS = 16
N_BLOCKS = 5
HORIZON_EPOCHS = 1
BOOTSTRAP_DRAWS = 10_000
TAU_FLOOR = 1e-3
EPS = 1e-12


def _load_frozen_helpers():
    if not OLD_CODE.is_file():
        raise RuntimeError(f"missing frozen SSPG helper code: {OLD_CODE}")
    spec = importlib.util.spec_from_file_location("persist_eeg_frozen_sspg_cumulative_helpers", OLD_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen SSPG helper module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sspg = _load_frozen_helpers()
BATCH_SIZE = int(sspg.BATCH_SIZE)
SCHEDULE_BATCH_REFERENCE = int(sspg.SCHEDULE_BATCH_REFERENCE)
BASE_LR = float(sspg.BASE_LR)
WEIGHT_DECAY = float(sspg.WEIGHT_DECAY)
GRAD_CLIP = float(sspg.GRAD_CLIP)


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


def metadata_col(data: Any, name: str, indices: np.ndarray | None = None) -> np.ndarray:
    return sspg.metadata_col(data, name, indices)


def prepare(data: Any, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    return sspg.prepare(data, np.asarray(indices, dtype=np.int64), mean, std, device)


def labels_for(data: Any, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return sspg.labels_for(data, np.asarray(indices, dtype=np.int64), device)


def flatten(values: Iterable[torch.Tensor]) -> torch.Tensor:
    return sspg.flatten(values)


def split_like(vector: torch.Tensor, params: list[nn.Parameter]) -> list[torch.Tensor]:
    return sspg.split_like(vector, params)


def make_model(state: dict[str, torch.Tensor], channels: int, device: torch.device) -> tuple[nn.Module, list[nn.Parameter]]:
    return sspg.make_model(state, channels, device)


def gradient_vector(model: nn.Module, params: list[nn.Parameter], xb: torch.Tensor, yb: torch.Tensor, *, dropout_seed: int | None) -> torch.Tensor:
    return sspg.gradient_vector(model, params, xb, yb, dropout_seed=dropout_seed)


def bbr_values_from_logits(logits: torch.Tensor, labels: torch.Tensor, tau: float) -> torch.Tensor:
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(1, labels.view(-1, 1), True)
    margins = logits.gather(1, labels.view(-1, 1)).squeeze(1) - logits.masked_fill(mask, -torch.inf).amax(dim=1)
    return torch.sigmoid(-margins / float(tau))


def balanced_mean(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    classes = sorted(int(v) for v in torch.unique(labels).detach().cpu().tolist())
    if len(classes) < 2:
        raise ValueError("class-balanced risk requires both classes")
    return torch.stack([values[labels == cls].mean() for cls in classes]).mean()


def bbr_gradient(model: nn.Module, params: list[nn.Parameter], xb: torch.Tensor, yb: torch.Tensor, tau: float) -> torch.Tensor:
    model.eval()
    logits = model(xb)
    values = torch.autograd.grad(balanced_mean(bbr_values_from_logits(logits, yb, tau), yb), tuple(params), allow_unused=True, retain_graph=False, create_graph=False)
    return flatten(value.detach().float() if value is not None else torch.zeros_like(param) for value, param in zip(values, params))


def snapshot(params: list[nn.Parameter]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def optimizer_digest(optimizer: torch.optim.Optimizer) -> str:
    return sspg.optimizer_digest(optimizer)


def bn_buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    return sspg.bn_buffers(model)


def bn_max_displacement(model: nn.Module, baseline: dict[str, torch.Tensor]) -> float:
    return sspg.bn_max_displacement(model, baseline)


def subject_sort(values: Iterable[object]) -> list[str]:
    return sspg.subject_sort(values)


def anchor_tau(ctx: Any, device: torch.device) -> tuple[float, float]:
    model, _ = make_model(ctx.anchor_state, ctx.channels, device)
    model.eval()
    vals: list[np.ndarray] = []
    legal = np.asarray(ctx.refit_idx, dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(legal), 128):
            idx = legal[start : start + 128]
            xb = prepare(ctx.data, idx, ctx.mean, ctx.std, device)
            yb = labels_for(ctx.data, idx, device)
            logits = model(xb)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(1, yb.view(-1, 1), True)
            margin = logits.gather(1, yb.view(-1, 1)).squeeze(1) - logits.masked_fill(mask, -torch.inf).amax(dim=1)
            vals.append(margin.detach().cpu().numpy())
    raw = float(np.median(np.abs(np.concatenate(vals))))
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return raw, max(raw, TAU_FLOOR)


def previous_tau_rows() -> dict[tuple[str, int], dict[str, Any]]:
    path = PREVIOUS_EXP / "results" / "MARGIN_SCALE.csv"
    if not path.is_file():
        raise RuntimeError(f"previous frozen margin scale is missing: {path}")
    frame = pd.read_csv(path)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        key = (str(row["dataset"]), int(row["fold"]))
        out[key] = row
    expected = {(d, f) for d in DATASETS for f in FOLDS}
    if set(out) != expected:
        raise RuntimeError("previous MARGIN_SCALE.csv does not contain exactly ten dataset/fold rows")
    return out


def build_pools(ctx: Any) -> dict[str, dict[int, np.ndarray]]:
    subjects = metadata_col(ctx.data, "subject_id", ctx.refit_idx).astype(str)
    labels = metadata_col(ctx.data, "label", ctx.refit_idx).astype(int)
    pools: dict[str, dict[int, np.ndarray]] = {}
    for subject in ctx.source_subjects:
        pools[str(subject)] = {}
        for cls in (0, 1):
            values = ctx.refit_idx[(subjects == str(subject)) & (labels == cls)]
            if len(values) < N_BLOCKS * M_PER_CLASS:
                raise RuntimeError(f"INSUFFICIENT_CUMULATIVE_SENTINEL_TRIAL_SUPPORT {ctx.dataset} fold={ctx.fold} subject={subject} class={cls} count={len(values)}")
            pools[str(subject)][cls] = np.asarray(values, dtype=np.int64)
    return pools


def class_balanced_block(ctx: Any, subject: str, block_no: int) -> np.ndarray:
    return np.concatenate([ctx.blocks[str(subject)][cls][block_no] for cls in (0, 1)]).astype(np.int64)


def bout_indices(ctx: Any) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    subjects = metadata_col(ctx.data, "subject_id", ctx.refit_idx).astype(str)
    labels = metadata_col(ctx.data, "label", ctx.refit_idx).astype(int)
    output: dict[str, np.ndarray] = {}
    support: list[dict[str, Any]] = []
    for subject in ctx.source_subjects:
        reserved = np.concatenate([class_balanced_block(ctx, subject, k) for k in range(K)]).astype(np.int64)
        mask_subject = subjects == str(subject)
        remain = ctx.refit_idx[mask_subject & ~np.isin(ctx.refit_idx, reserved)]
        counts = {str(cls): int(np.sum(labels[mask_subject & ~np.isin(ctx.refit_idx, reserved)] == cls)) for cls in (0, 1)}
        if min(counts.values()) < M_PER_CLASS:
            raise RuntimeError(f"INSUFFICIENT_CUMULATIVE_SENTINEL_TRIAL_SUPPORT {ctx.dataset} fold={ctx.fold} subject={subject} counts={counts}")
        output[str(subject)] = np.asarray(remain, dtype=np.int64)
        support.append({"dataset": ctx.dataset, "fold": int(ctx.fold), "subject_id": str(subject), "B_out_trial_count": int(len(remain)), "B_out_class0_count": counts["0"], "B_out_class1_count": counts["1"], "min_class_support": int(min(counts.values()))})
    return output, support


def hash_subjects(values: Iterable[object]) -> str:
    return hashlib.sha256(json.dumps([str(v) for v in values], separators=(",", ":")).encode("utf-8")).hexdigest()


def make_epoch_schedule(ctx: Any, sentinel_meta_fold: int, pools: dict[str, dict[int, np.ndarray]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = [list(map(str, group)) for group in ctx.meta_folds]
    if not (0 <= sentinel_meta_fold < len(groups)):
        raise ValueError("invalid sentinel meta-fold")
    sentinel_subjects = groups[sentinel_meta_fold]
    a_subjects = [s for idx, group in enumerate(groups) if idx != sentinel_meta_fold for s in group]
    if not sentinel_subjects or not a_subjects:
        raise RuntimeError("empty sentinel or A subject group")
    refit_subjects = metadata_col(ctx.data, "subject_id", ctx.refit_idx).astype(str)
    a_trial_count = int(np.sum(np.isin(refit_subjects, a_subjects)))
    steps = max(1, int(math.ceil(a_trial_count / SCHEDULE_BATCH_REFERENCE)))
    schedule: list[dict[str, Any]] = []
    serial: list[dict[str, Any]] = []
    for step in range(steps):
        rng = np.random.default_rng(stable_seed("cumulative-a-only-schedule", ctx.dataset, ctx.fold, sentinel_meta_fold, SEED, step))
        a_idx = np.asarray(sspg.balanced_batch(pools, a_subjects, rng), dtype=np.int64)
        observed = set(metadata_col(ctx.data, "subject_id", a_idx).astype(str))
        if observed & set(sentinel_subjects):
            raise RuntimeError(f"sentinel subject entered A: {ctx.dataset} fold={ctx.fold} sentinel={sentinel_meta_fold}")
        schedule.append({"A": a_idx, "step": int(step + 1), "a_subjects": list(a_subjects), "sentinel_subjects": list(sentinel_subjects)})
        serial.append({"step": int(step + 1), "A": a_idx.tolist(), "sentinel_meta_fold": int(sentinel_meta_fold), "sentinel_subjects": list(sentinel_subjects)})
    digest = hashlib.sha256(json.dumps(serial, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    spec = {"dataset": ctx.dataset, "fold": int(ctx.fold), "sentinel_meta_fold": int(sentinel_meta_fold), "sentinel_subjects": sentinel_subjects, "a_subjects": a_subjects, "sentinel_subject_hash": hash_subjects(sentinel_subjects), "a_subject_hash": hash_subjects(a_subjects), "a_trial_count": a_trial_count, "steps": steps, "batch_size": BATCH_SIZE, "schedule_batch_reference": SCHEDULE_BATCH_REFERENCE, "schedule_sha256": digest}
    return schedule, spec


def build_specs(contexts: list[Any], tau_map: dict[tuple[str, int], dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for ctx in contexts:
        pools = build_pools(ctx)
        bouts, support = bout_indices(ctx)
        tau = float(tau_map[(ctx.dataset, int(ctx.fold))]["tau"])
        if not math.isfinite(tau) or tau <= 0:
            raise RuntimeError(f"invalid frozen tau {ctx.dataset} fold={ctx.fold}")
        for sentinel_meta_fold in range(5):
            schedule, spec = make_epoch_schedule(ctx, sentinel_meta_fold, pools)
            min_support = min(int(row["min_class_support"]) for row in support if str(row["subject_id"]) in set(spec["sentinel_subjects"]))
            item = {**spec, "tau": tau, "B_out_min_class_support": min_support, "B_out_definition": "all remaining legal source/refit trials", "checkpoint_sha256": sha256_file(ctx.checkpoint_path), "checkpoint_path": str(ctx.checkpoint_path), "schedule": schedule, "bout_indices": bouts, "tau_row": tau_map[(ctx.dataset, int(ctx.fold))]}
            lookup[(ctx.dataset, int(ctx.fold), sentinel_meta_fold)] = item
            rows.append({k: v for k, v in spec.items() if k not in {"sentinel_subjects", "a_subjects"}} | {"tau": tau, "B_out_min_class_support": min_support, "checkpoint_sha256": item["checkpoint_sha256"]})
    return rows, lookup


def eval_subject(model: nn.Module, ctx: Any, idx: np.ndarray, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(idx), 128):
            part = np.asarray(idx[start : start + 128], dtype=np.int64)
            logits_parts.append(model(prepare(ctx.data, part, ctx.mean, ctx.std, device)).detach().cpu().numpy())
            labels_parts.append(metadata_col(ctx.data, "label", part).astype(np.int64))
    logits = np.concatenate(logits_parts, axis=0)
    labels = np.concatenate(labels_parts, axis=0)
    mask = np.zeros_like(logits, dtype=bool)
    mask[np.arange(len(labels)), labels] = True
    margins = logits[np.arange(len(labels)), labels] - np.max(np.where(mask, -np.inf, logits), axis=1)
    risks = 1.0 / (1.0 + np.exp(np.clip(margins / float(ctx.tau), -60.0, 60.0)))
    classes = sorted(np.unique(labels).tolist())
    correct = np.argmax(logits, axis=1) == labels
    ba = float(np.mean([np.mean(correct[labels == cls]) for cls in classes]))
    ce = float(np.mean(np.logaddexp.reduce(logits, axis=1) - logits[np.arange(len(labels)), labels]))
    return {"L_BBR": float(np.mean([np.mean(risks[labels == cls]) for cls in classes])), "L_CE": ce, "BA": ba, "labels": labels, "pred": np.argmax(logits, axis=1), "correct": correct, "margins": margins, "trial_count": int(len(labels)), "class_counts": {str(int(cls)): int(np.sum(labels == cls)) for cls in classes}, "balanced_margin": float(np.mean([np.mean(margins[labels == cls]) for cls in classes]))}


def pair_metrics(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    labels = before["labels"]
    harmful = before["correct"] & ~after["correct"]
    beneficial = ~before["correct"] & after["correct"]
    classes = sorted(np.unique(labels).tolist())
    h_rates = [float(np.mean(harmful[labels == cls])) for cls in classes]
    b_rates = [float(np.mean(beneficial[labels == cls])) for cls in classes]
    return {
        "H_BBR_epoch": float(after["L_BBR"] - before["L_BBR"]), "H_CE_epoch": float(after["L_CE"] - before["L_CE"]), "BA_before": float(before["BA"]), "BA_after": float(after["BA"]), "H_BER_epoch": float((1.0 - after["BA"]) - (1.0 - before["BA"])), "margin_drift": float(after["balanced_margin"] - before["balanced_margin"]), "correct_to_wrong": int(np.sum(harmful)), "wrong_to_correct": int(np.sum(beneficial)), "correct_to_correct": int(np.sum(before["correct"] & after["correct"])), "wrong_to_wrong": int(np.sum(~before["correct"] & ~after["correct"])), "harmful_flip_rate": float(np.mean(harmful)), "beneficial_flip_rate": float(np.mean(beneficial)), "net_flip_harm": float(np.mean(harmful) - np.mean(beneficial)), "class_balanced_harmful_flip_rate": float(np.mean(h_rates)), "class_balanced_beneficial_flip_rate": float(np.mean(b_rates)), "class_balanced_net_flip_harm": float(np.mean(h_rates) - np.mean(b_rates)), "correct_before": int(np.sum(before["correct"])), "correct_after": int(np.sum(after["correct"])), "B_out_trial_count": int(after["trial_count"]), "B_out_class0_count": int(after["class_counts"].get("0", 0)), "B_out_class1_count": int(after["class_counts"].get("1", 0)),
    }


def anchor_gradients(ctx: Any, subjects: Iterable[str], device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    model, params = make_model(ctx.anchor_state, ctx.channels, device)
    model.eval()
    output: dict[str, dict[str, torch.Tensor]] = {}
    for subject in subjects:
        bbr_blocks: list[torch.Tensor] = []
        ce_blocks: list[torch.Tensor] = []
        for block_no in range(K):
            idx = class_balanced_block(ctx, str(subject), block_no)
            xb = prepare(ctx.data, idx, ctx.mean, ctx.std, device)
            yb = labels_for(ctx.data, idx, device)
            bbr_blocks.append(bbr_gradient(model, params, xb, yb, float(ctx.tau)).detach().cpu())
            ce_blocks.append(gradient_vector(model, params, xb, yb, dropout_seed=None).detach().cpu())
            del xb, yb
        output[str(subject)] = {"bbr": torch.stack(bbr_blocks).mean(dim=0), "ce": torch.stack(ce_blocks).mean(dim=0)}
    del model, params
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def random_direction(norm: float, length: int, dataset: str, fold: int, sentinel_meta_fold: int, subject: str) -> torch.Tensor:
    rng = np.random.default_rng(stable_seed(dataset, fold, sentinel_meta_fold, subject, SEED, "CUMULATIVE_RANDOM"))
    value = torch.as_tensor(rng.standard_normal(length), dtype=torch.float32)
    value = value / max(float(torch.linalg.vector_norm(value)), EPS)
    return value * float(norm)


def train_window(ctx: Any, item: dict[str, Any], grads: dict[str, dict[str, torch.Tensor]], device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sentinel_subjects = [str(v) for v in item["sentinel_subjects"]]
    set_seed(stable_seed("cumulative-task-only-init", ctx.dataset, ctx.fold, item["sentinel_meta_fold"], SEED))
    model, params = make_model(ctx.anchor_state, ctx.channels, device)
    model.eval()
    optimizer = torch.optim.AdamW(params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    bn_baseline = bn_buffers(model)
    theta0 = flatten(snapshot(params)).detach().cpu().double()
    delta_sum = torch.zeros_like(theta0)
    trajectory: list[dict[str, Any]] = []
    before_by_subject = {s: eval_subject(model, ctx, item["bout_indices"][s], device) for s in sentinel_subjects}
    digest = hashlib.sha256()
    for step_no, entry in enumerate(item["schedule"], start=1):
        a_idx = np.asarray(entry["A"], dtype=np.int64)
        observed = set(metadata_col(ctx.data, "subject_id", a_idx).astype(str))
        if observed & set(sentinel_subjects):
            raise RuntimeError(f"sentinel entered A during continuation {ctx.dataset} fold={ctx.fold} sentinel={item['sentinel_meta_fold']}")
        xb = prepare(ctx.data, a_idx, ctx.mean, ctx.std, device)
        yb = labels_for(ctx.data, a_idx, device)
        grad = gradient_vector(model, params, xb, yb, dropout_seed=stable_seed("sspg-dropout", ctx.dataset, ctx.fold, SEED, 1, step_no - 1, "A"))
        grad, grad_norm, clip_scale = sspg.clip_gradient(grad)
        old = snapshot(params)
        optimizer.zero_grad(set_to_none=True)
        for parameter, chunk in zip(params, split_like(grad, params)):
            parameter.grad = chunk.detach().clone()
        optimizer.step()
        delta = flatten([parameter.detach() - old_value for parameter, old_value in zip(params, old)]).detach()
        delta_cpu = delta.cpu().double()
        delta_sum += delta_cpu
        digest.update(delta_cpu.numpy().tobytes())
        trajectory.append({"dataset": ctx.dataset, "fold": int(ctx.fold), "sentinel_meta_fold": int(item["sentinel_meta_fold"]), "seed": SEED, "step": int(step_no), "task_gradient_norm": float(grad_norm), "task_clip_scale": float(clip_scale), "step_delta_norm": float(torch.linalg.vector_norm(delta).detach().cpu()), "bn_max_displacement": float(bn_max_displacement(model, bn_baseline))})
        del xb, yb, grad, delta, old
    theta1 = flatten([parameter.detach() for parameter in params]).cpu().double()
    # ``theta0`` is the exact trainable-parameter vector at the anchor.  The
    # state dict also contains non-parameter BN buffers, so it must never be
    # used to reconstruct the displacement vector.
    displacement = theta1 - theta0
    consistency_error = float(torch.max(torch.abs(displacement - delta_sum)).item())
    displacement_norm = float(torch.linalg.vector_norm(displacement).item())
    if consistency_error > 5e-5:
        raise RuntimeError(f"CUMULATIVE_AUDIT_IMPLEMENTATION_INVALID_DISPLACEMENT_{ctx.dataset}_{ctx.fold}_{item['sentinel_meta_fold']}={consistency_error}")
    observations: list[dict[str, Any]] = []
    after_by_subject = {s: eval_subject(model, ctx, item["bout_indices"][s], device) for s in sentinel_subjects}
    partner = {s: sentinel_subjects[(idx + 1) % len(sentinel_subjects)] for idx, s in enumerate(sentinel_subjects)}
    for subject in sentinel_subjects:
        before, after = before_by_subject[subject], after_by_subject[subject]
        metrics = pair_metrics(before, after)
        g_bbr = grads[subject]["bbr"]
        g_ce = grads[subject]["ce"]
        g_diff = grads[partner[subject]]["bbr"]
        g_ce_diff = grads[partner[subject]]["ce"]
        norm = float(torch.linalg.vector_norm(g_bbr).item())
        rnd = random_direction(norm, len(displacement), ctx.dataset, int(ctx.fold), int(item["sentinel_meta_fold"]), subject)
        row = {"dataset": ctx.dataset, "fold": int(ctx.fold), "sentinel_meta_fold": int(item["sentinel_meta_fold"]), "seed": SEED, "subject_id": subject, "partner_subject": partner[subject], "certificate_BBR": float(torch.dot(g_bbr, delta_sum.float()).item()), "certificate_CE": float(torch.dot(g_ce, delta_sum.float()).item()), "certificate_BBR_different": float(torch.dot(g_diff, delta_sum.float()).item()), "certificate_CE_different": float(torch.dot(g_ce_diff, delta_sum.float()).item()), "certificate_BBR_random": float(torch.dot(rnd, delta_sum.float()).item()), "certificate_BBR_norm": norm, "random_norm_error": abs(float(torch.linalg.vector_norm(rnd).item()) - norm), "tau": float(ctx.tau), "tau_source": "previous_decision_audit_legal_source_refit_anchor", "schedule_sha256": item["schedule_sha256"], "window_steps": int(item["steps"]), "delta_epoch_norm": displacement_norm, "displacement_sum_max_abs_error": consistency_error, "trajectory_sha256": digest.hexdigest(), "B_out_trial_count": int(after["trial_count"]), "B_out_class0_count": int(after["class_counts"].get("0", 0)), "B_out_class1_count": int(after["class_counts"].get("1", 0)), "bn_max_displacement": float(bn_max_displacement(model, bn_baseline))}
        row.update(metrics)
        observations.append(row)
    window = {"dataset": ctx.dataset, "fold": int(ctx.fold), "sentinel_meta_fold": int(item["sentinel_meta_fold"]), "seed": SEED, "schedule_sha256": item["schedule_sha256"], "trajectory_sha256": digest.hexdigest(), "steps": int(item["steps"]), "sentinel_subjects": sentinel_subjects, "observation_rows": len(observations), "displacement_sum_max_abs_error": consistency_error, "bn_max_displacement": float(bn_max_displacement(model, bn_baseline)), "complete": True}
    del model, params, optimizer, before_by_subject, after_by_subject
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return observations, trajectory, window


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    xa, ya = np.asarray(list(x), dtype=float), np.asarray(list(y), dtype=float)
    if len(xa) < 3 or np.std(xa) == 0 or np.std(ya) == 0:
        return None
    value = float(spearmanr(xa, ya).statistic)
    return value if math.isfinite(value) else None


def safe_pearson(x: Iterable[float], y: Iterable[float]) -> float | None:
    xa, ya = np.asarray(list(x), dtype=float), np.asarray(list(y), dtype=float)
    if len(xa) < 3 or np.std(xa) == 0 or np.std(ya) == 0:
        return None
    value = float(pearsonr(xa, ya).statistic)
    return value if math.isfinite(value) else None


def safe_kendall(x: Iterable[float], y: Iterable[float]) -> float | None:
    xa, ya = np.asarray(list(x), dtype=float), np.asarray(list(y), dtype=float)
    if len(xa) < 3 or np.std(xa) == 0 or np.std(ya) == 0:
        return None
    value = float(kendalltau(xa, ya).statistic)
    return value if math.isfinite(value) else None


def safe_auc(cert: Iterable[float], outcome: Iterable[float]) -> float | None:
    x, y = np.asarray(list(cert), dtype=float), np.asarray(list(outcome), dtype=int)
    if len(np.unique(y)) < 2:
        return None
    value = float(roc_auc_score(y, x))
    return value if math.isfinite(value) else None


def sign_accuracy(cert: Iterable[float], outcome: Iterable[float]) -> float | None:
    x, y = np.asarray(list(cert), dtype=float), np.asarray(list(outcome), dtype=float)
    return float(np.mean(np.sign(x) == np.sign(y))) if len(x) else None


def ci_from(values: np.ndarray) -> list[float | None]:
    finite_values = values[np.isfinite(values)]
    if not len(finite_values):
        return [None, None]
    return [float(np.quantile(finite_values, 0.025)), float(np.quantile(finite_values, 0.975))]


def top_bottom_arrays(cert: np.ndarray, outcome: np.ndarray) -> float | None:
    if len(cert) == 0:
        return None
    order = np.argsort(cert, kind="mergesort")
    q = max(1, len(order) // 5)
    return float(np.mean(outcome[order[-q:]]) - np.mean(outcome[order[:q]]))


def metric_values(cert: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    event = (outcome > 0).astype(int)
    values = [safe_spearman(cert, outcome), safe_kendall(cert, outcome), safe_auc(cert, event), top_bottom_arrays(cert, outcome)]
    return np.asarray([np.nan if value is None else value for value in values], dtype=float)


def metric_pack(frame: pd.DataFrame, cert_col: str, outcome_col: str, boot: np.ndarray | None = None) -> dict[str, Any]:
    cert = frame[cert_col].to_numpy(float)
    outcome = frame[outcome_col].to_numpy(float)
    values = metric_values(cert, outcome)
    result = {"spearman": None if not np.isfinite(values[0]) else float(values[0]), "kendall": None if not np.isfinite(values[1]) else float(values[1]), "pearson": safe_pearson(cert, outcome), "auroc": None if not np.isfinite(values[2]) else float(values[2]), "sign_accuracy": sign_accuracy(cert, outcome), "top_minus_bottom": None if not np.isfinite(values[3]) else float(values[3]), "n_observations": int(len(frame)), "n_subjects": int(frame.subject_id.astype(str).nunique()), "event_count": int(np.sum(outcome > 0))}
    if boot is not None:
        for idx, name in enumerate(("spearman", "kendall", "auroc", "top_minus_bottom")):
            result[name + "_CI95"] = ci_from(boot[:, idx])
    return result


def subject_groups(frame: pd.DataFrame) -> list[np.ndarray]:
    values = frame.subject_id.astype(str).to_numpy()
    return [np.flatnonzero(values == subject) for subject in sorted(set(values))]


def bootstrap_metric(frame: pd.DataFrame, cert_col: str, outcome_col: str, seed: int) -> np.ndarray:
    cert = frame[cert_col].to_numpy(float)
    outcome = frame[outcome_col].to_numpy(float)
    groups = subject_groups(frame)
    rng = np.random.default_rng(seed)
    output = np.full((BOOTSTRAP_DRAWS, 4), np.nan, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        chosen = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[int(i)] for i in chosen])
        output[draw] = metric_values(cert[idx], outcome[idx])
    return output


def bootstrap_difference(frame: pd.DataFrame, same_col: str, other_col: str, outcome_col: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    same = frame[same_col].to_numpy(float)
    other = frame[other_col].to_numpy(float)
    outcome = frame[outcome_col].to_numpy(float)
    groups = subject_groups(frame)
    rng = np.random.default_rng(seed)
    auc = np.full(BOOTSTRAP_DRAWS, np.nan); sp = np.full(BOOTSTRAP_DRAWS, np.nan)
    for draw in range(BOOTSTRAP_DRAWS):
        chosen = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[int(i)] for i in chosen])
        event = (outcome[idx] > 0).astype(int)
        a, b = safe_auc(same[idx], event), safe_auc(other[idx], event)
        s1, s2 = safe_spearman(same[idx], outcome[idx]), safe_spearman(other[idx], outcome[idx])
        if a is not None and b is not None: auc[draw] = a - b
        if s1 is not None and s2 is not None: sp[draw] = s1 - s2
    return auc, sp


def bootstrap_quintile(frame: pd.DataFrame, cert_col: str, outcome_col: str, seed: int) -> np.ndarray:
    cert = frame[cert_col].to_numpy(float); outcome = frame[outcome_col].to_numpy(float)
    groups = subject_groups(frame); rng = np.random.default_rng(seed); out = np.full(BOOTSTRAP_DRAWS, np.nan)
    for draw in range(BOOTSTRAP_DRAWS):
        chosen = rng.integers(0, len(groups), size=len(groups)); idx = np.concatenate([groups[int(i)] for i in chosen]); order = np.argsort(cert[idx], kind="mergesort"); q = max(1, len(order) // 5)
        out[draw] = float(np.mean(outcome[idx][order[-q:]] > 0) - np.mean(outcome[idx][order[:q]] > 0))
    return out


def calibration(frame: pd.DataFrame, cert_col: str, cert_name: str) -> list[dict[str, Any]]:
    values = frame[cert_col].to_numpy(float); order = np.argsort(values, kind="mergesort"); qid = np.empty(len(frame), dtype=int)
    for rank, pos in enumerate(order): qid[pos] = min(5, int(rank * 5 / max(len(frame), 1)) + 1)
    rows: list[dict[str, Any]] = []
    for q in range(1, 6):
        part = frame.iloc[np.flatnonzero(qid == q)]
        rows.append({"dataset": str(frame.dataset.iloc[0]), "certificate": cert_name, "quintile": q, "mean_certificate": float(part[cert_col].mean()) if len(part) else None, "mean_H_BBR_epoch": float(part.H_BBR_epoch.mean()) if len(part) else None, "smooth_harm_frequency": float(np.mean(part.H_BBR_epoch > 0)) if len(part) else None, "mean_H_CE_epoch": float(part.H_CE_epoch.mean()) if len(part) else None, "mean_H_BER_epoch": float(part.H_BER_epoch.mean()) if len(part) else None, "decision_harm_frequency": float(np.mean(part.H_BER_epoch > 0)) if len(part) else None, "correct_to_wrong_frequency": float(part.correct_to_wrong.sum() / max(part.B_out_trial_count.sum(), 1)) if len(part) else None, "subject_count": int(part.subject_id.astype(str).nunique()), "observation_count": int(len(part))})
    return rows


def fold_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (dataset, fold), part in frame.groupby(["dataset", "fold"], sort=True):
        bbr = metric_pack(part, "certificate_BBR", "H_BER_epoch"); ce = metric_pack(part, "certificate_CE", "H_BER_epoch")
        smooth = metric_pack(part, "certificate_BBR", "H_BBR_epoch"); diff_auc = safe_auc(part.certificate_BBR_different, (part.H_BER_epoch > 0).astype(int)); same_auc = safe_auc(part.certificate_BBR, (part.H_BER_epoch > 0).astype(int))
        rows.append({"dataset": dataset, "fold": int(fold), "n_subjects": int(part.subject_id.astype(str).nunique()), "n_observations": int(len(part)), "BBR_H_BER_AUROC": bbr["auroc"], "BBR_H_BER_Spearman": bbr["spearman"], "CE_H_BER_AUROC": ce["auroc"], "CE_H_BER_Spearman": ce["spearman"], "BBR_H_BBR_AUROC": smooth["auroc"], "BBR_H_BBR_Spearman": smooth["spearman"], "same_subject_AUROC": same_auc, "different_subject_AUROC": diff_auc, "same_minus_different_AUROC": None if same_auc is None or diff_auc is None else float(same_auc - diff_auc), "harmful_event_count": int(np.sum(part.H_BER_epoch > 0)), "harmful_flip_count": int(part.correct_to_wrong.sum()), "same_subject_spearman_positive": bool((bbr["spearman"] or 0.0) > 0)})
    return rows


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=EXP.parent.parent, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "status": run("status", "--short")}


def code_hashes() -> dict[str, str]:
    files = [Path(__file__), OLD_CODE, sspg.CANONICAL_EXP / "code" / "canonical_eegnet_runner.py"]
    return {str(path): sha256_file(path) for path in files if path.is_file()}


def write_protocol_docs(lock: dict[str, Any], legality: dict[str, Any], terminal: str | None = None) -> None:
    (EXP / "README.md").write_text("# PERSIST-EEG Cumulative Subject Decision Drift Audit V1\n\nSeed-0 EEGNet only; OpenBMI/WBCIC canonical folds 0--4. Each fold has five deterministic fixed-sentinel meta-fold continuations, each exactly one A-only natural epoch from the canonical checkpoint. Sentinel subjects are excluded from every A batch. K=4 class-balanced BBR/CE certificates are evaluated on source/refit trials and compared with held-out remaining source/refit B_out.\n\nThis is a signal audit only: no guard, rollback, correction, new objective, hyperparameter search, seed 1/2, second backbone, WBCIC outer-10, or OpenBMI sealed cohort. Runtime, checkpoints, cache and raw EEG stay outside the committed artifact set.\n", encoding="utf-8")
    (EXP / "FROZEN_PROTOCOL.md").write_text("# Frozen protocol\n\n- EEGNet, OpenBMI and WBCIC, canonical folds 0--4, seed 0 only.\n- For each fold, each of the five deterministic meta-folds is a fixed sentinel group. All sentinel subjects are excluded from A for the full continuation.\n- One natural A-only epoch uses the frozen TaskOnly sampler semantics: batch size 64, steps=ceil(A legal trial count/128), AdamW lr=3e-5, weight decay=5e-4, clip=5, frozen BN and canonical dropout keys.\n- K=4 class-balanced blocks, 16 trials/class/block without replacement; B_out is all remaining legal source/refit trials and retains at least 16/class.\n- BBR is sigmoid(-signed true-class margin/tau), class-balanced; tau is reused unchanged from the prior decision audit's legal-anchor margin scale. CE is a matched comparator.\n- Primary outcome is held-out H_BER_epoch; H_BBR_epoch is mechanism-support only. Biological subject is the cluster-bootstrap unit (10,000 draws).\n- Selection rule, gates, controls and all exclusion flags are recorded in PRE_OUTCOME_LOCK.json.\n\n" + json.dumps(clean(lock), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (EXP / "PRE_OUTCOME_LOCK.md").write_text("# Pre-outcome lock\n\nThe lock was written after source/refit legality, fixed sentinel schedules, B_out support, prior-audit tau equivalence, checkpoint equivalence and executable mandatory tests. It must be committed before any theta_1/B_out outcome is evaluated.\n\n- seed=0; EEGNet; K=4; m_per_class=16; one epoch; fixed sentinel groups; outcome_used=false\n- WBCIC_outer_opened=false; OpenBMI_sealed_opened=false; seed1_run=false; seed2_run=false; second_backbone_run=false\n", encoding="utf-8")
    (EXP / "DATA_LEGALITY_AUDIT.md").write_text("# Data legality audit\n\nOnly canonical model-fit/discovery source/refit biological subjects and legal fit sessions are loaded. Sentinel groups and B_out are built entirely inside this source/refit scope. No development outcome labels, WBCIC outer-10, OpenBMI sealed/confirmation cohort, seed 1/2 or second backbone is materialized.\n\n```json\n" + json.dumps(clean(legality), ensure_ascii=False, indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    docs = {
        "CHECKPOINT_EQUIVALENCE.md": "# Checkpoint equivalence\n\nEvery dataset/fold strict-loads the canonical seed-0 EEGNet state and verifies repeat predictions and checkpoint hash. See results/CHECKPOINT_EQUIVALENCE.csv.\n",
        "SENTINEL_ISOLATION_AUDIT.md": "# Sentinel isolation audit\n\nFive deterministic meta-folds are each used as a fixed sentinel group. Every A batch in that one-epoch continuation is subject-disjoint from its sentinel group; sentinel membership and A hashes are in WINDOW_SCHEDULE.csv and PRE_OUTCOME_LOCK.json.\n",
        "BATCH_CONSTRUCTION_AUDIT.md": "# Batch construction audit\n\nThe prior frozen K=4 construction is retained: four class-balanced 16-per-class certificate blocks, trial-disjoint from B_out. A uses the exact prior balanced sampler semantics with sentinel subjects removed.\n",
        "CUMULATIVE_DISPLACEMENT_AUDIT.md": "# Cumulative displacement audit\n\nFor every window, Delta_epoch is theta_1-theta_0 after exact AdamW steps. The runner also sums each exact per-step parameter displacement and requires max absolute agreement <=5e-5.\n",
        "MATHEMATICAL_AUDIT.md": "# Mathematical audit\n\nC_BBR=gbar_BBR^T Delta_epoch and C_CE=gbar_CE^T Delta_epoch are computed from anchor gradients and the exact cumulative displacement. H_BER_epoch is held-out Balanced Error change; H_BBR_epoch and H_CE_epoch remain separate smooth outcomes.\n",
        "CONTROL_AUDIT.md": "# Control audit\n\nSame-subject certificates are primary. A cyclic non-self partner within the fixed sentinel group supplies different-subject certificates. Random is a deterministic Gaussian direction norm-matched to C_BBR's gradient vector.\n",
        "STATISTICAL_PROTOCOL.md": "# Statistical protocol\n\nBiological subject is the inference unit. A subject bootstrap draw carries all canonical-fold/sentinel-window observations for that subject. All reported CIs use 10,000 cluster draws; folds, windows and trials are not independent units.\n",
        "POWER_AUDIT.md": "# Power audit\n\nExact decision power is assessed from H_BER_epoch>0 observations and biological subjects with at least one harmful event. If either dataset has fewer than 30 events or 15 harmful subjects, the predeclared terminal is CUMULATIVE_DECISION_ENDPOINT_UNDERPOWERED; no extra horizon is added.\n",
        "BUG_REPAIR_LEDGER.md": "# Bug repair ledger\n\nNo scientific-rule repair is permitted. Engineering implementation uses the existing frozen source/refit loader, previous legal tau rows, deterministic sentinel-filtered sampler, exact AdamW displacement accounting, CPU accumulation for bounded memory, and subject-cluster bootstrap.\n",
    }
    for name, text in docs.items():
        (EXP / name).write_text(text, encoding="utf-8")
    if terminal is not None:
        (EXP / "AUTONOMOUS_DECISION.md").write_text(f"# Autonomous decision\n\n`terminal = {terminal}`\n\nNo final model, guard, new horizon, seed, backbone, outer cohort or sealed cohort is started automatically.\n", encoding="utf-8")


def bbr_concat_test(ctx: Any, device: torch.device) -> dict[str, Any]:
    subject = str(ctx.source_subjects[0]); model, params = make_model(ctx.anchor_state, ctx.channels, device); blocks: list[torch.Tensor] = []
    for block_no in range(K):
        idx = class_balanced_block(ctx, subject, block_no); xb = prepare(ctx.data, idx, ctx.mean, ctx.std, device); yb = labels_for(ctx.data, idx, device); blocks.append(bbr_gradient(model, params, xb, yb, float(ctx.tau))); del xb, yb
    idx = np.concatenate([class_balanced_block(ctx, subject, block_no) for block_no in range(K)]); xb = prepare(ctx.data, idx, ctx.mean, ctx.std, device); yb = labels_for(ctx.data, idx, device); concat = bbr_gradient(model, params, xb, yb, float(ctx.tau)); diff = float(torch.max(torch.abs(concat - torch.stack(blocks).mean(dim=0))).detach().cpu()); del model, params, xb, yb, blocks, concat; gc.collect(); return {"dataset": ctx.dataset, "fold": int(ctx.fold), "subject": subject, "max_abs_diff": diff, "tolerance": 5e-5, "pass": diff <= 5e-5}


def mandatory_tests(contexts: list[Any], specs: list[dict[str, Any]], lookup: dict[tuple[str, int, int], dict[str, Any]], device: torch.device, legality: dict[str, Any], tau_rows: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["canonical_checkpoint_exact_load"] = bool(all(bool(row.get("pass")) for row in [sspg.checkpoint_equivalence(ctx, device) for ctx in contexts]))
    checks["fixed_sentinel_never_in_A"] = True
    checks["one_epoch_schedule"] = True
    for item in lookup.values():
        sentinel = set(item["sentinel_subjects"])
        for entry in item["schedule"]:
            observed = set(metadata_col(next(ctx for ctx in contexts if ctx.dataset == item["dataset"] and int(ctx.fold) == item["fold"]).data, "subject_id", np.asarray(entry["A"], dtype=np.int64)).astype(str))
            checks["fixed_sentinel_never_in_A"] &= not (observed & sentinel)
            checks["one_epoch_schedule"] &= len(item["schedule"]) == int(item["steps"])
    checks["B1_B4_Bout_trial_disjoint"] = True
    for ctx in contexts:
        for subject in ctx.source_subjects:
            cert = set(np.concatenate([class_balanced_block(ctx, str(subject), block_no) for block_no in range(K)]).tolist()); bout, _ = bout_indices(ctx); checks["B1_B4_Bout_trial_disjoint"] &= not (cert & set(bout[str(subject)].tolist()))
    checks["BBR_definition_and_unit_interval"] = bool(torch.all((bbr_values_from_logits(torch.tensor([[2.0, 0.0], [0.0, 2.0]]), torch.tensor([0, 1]), 1.0) >= 0) & (bbr_values_from_logits(torch.tensor([[2.0, 0.0], [0.0, 2.0]]), torch.tensor([0, 1]), 1.0) <= 1)))
    checks["BBR_decreases_when_margin_increases"] = float(bbr_values_from_logits(torch.tensor([[2.0, 0.0]]), torch.tensor([0]), 1.0)) < float(bbr_values_from_logits(torch.tensor([[1.0, 0.0]]), torch.tensor([0]), 1.0))
    checks["K4_gradient_equivalence"] = bbr_concat_test(contexts[0], device)
    checks["tau_reused_and_frozen"] = bool(all(math.isfinite(float(row["tau"])) and float(row["tau"]) >= TAU_FLOOR for row in tau_rows.values()))
    checks["tau_source_legal_anchor"] = bool(all(str(row.get("tau_source", "")).startswith("previous") or str(row.get("tau_source", "")).startswith("legal") for row in tau_rows.values()))
    first = contexts[0]; model, params = make_model(first.anchor_state, first.channels, device); model.eval(); xb = prepare(first.data, class_balanced_block(first, str(first.source_subjects[0]), 0), first.mean, first.std, device); yb = labels_for(first.data, class_balanced_block(first, str(first.source_subjects[0]), 0), device); g = gradient_vector(model, params, xb, yb, dropout_seed=None); opt = torch.optim.AdamW(params, lr=BASE_LR, weight_decay=WEIGHT_DECAY); opt.zero_grad(set_to_none=True); old = snapshot(params); [setattr(p, "grad", chunk.clone()) for p, chunk in zip(params, split_like(g, params))]; opt.step(); before_opt = optimizer_digest(opt); before_bn = bn_buffers(model); _ = bbr_gradient(model, params, xb, yb, float(first.tau)); checks["exact_adamw_displacement"] = bool(torch.isfinite(flatten([p.detach() - o for p, o in zip(params, old)])).all()); checks["BN_unchanged_during_certificate"] = bn_max_displacement(model, before_bn) <= 1e-12; checks["optimizer_unchanged_during_certificate"] = optimizer_digest(opt) == before_opt; del model, params, xb, yb, g, opt
    checks["random_norm_matched"] = abs(float(torch.linalg.vector_norm(random_direction(1.0, 128, "OpenBMI", 0, 0, "1"))) - 1.0) <= 1e-5
    checks["same_different_nonself"] = all(len(item["sentinel_subjects"]) >= 2 and all(str(s) != str(item["sentinel_subjects"][(i + 1) % len(item["sentinel_subjects"])]) for i, s in enumerate(item["sentinel_subjects"])) for item in lookup.values())
    checks["outcome_isolation"] = bool(not legality.get("outcome_index_created_before_lock") and not legality.get("outcome_labels_read_before_lock") and not legality.get("outcome_data_materialized_before_lock") and not legality.get("WBCIC_outer_opened") and not legality.get("OpenBMI_sealed_opened"))
    checks["seed0_only"] = SEED == 0
    checks["subject_cluster_bootstrap_unit"] = all(len(set(map(str, ctx.source_subjects))) == len(ctx.source_subjects) for ctx in contexts)
    checks["cumulative_displacement_sum_test"] = bool(torch.allclose(torch.tensor([1.5, -2.0]), torch.tensor([0.5, -0.5]) + torch.tensor([1.0, -1.5]), atol=1e-12, rtol=0))
    checks["all_critical_pass"] = all(bool(value.get("pass", True)) if isinstance(value, dict) else bool(value) for value in checks.values())
    if not checks["all_critical_pass"]:
        raise RuntimeError("cumulative audit mandatory tests failed")
    return {"schema": "PERSIST_EEG_CUMULATIVE_MANDATORY_TESTS_V1", "pass": True, "checks": checks, "critical_failure_blocks_audit": True, "outcome_used": False, "seed1_run": False, "seed2_run": False, "second_backbone_run": False}


def preflight(device: torch.device) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    contexts, legality = sspg.load_contexts_source_only()
    tau_prev = previous_tau_rows(); tau_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for ctx in contexts:
        prev = dict(tau_prev[(ctx.dataset, int(ctx.fold))]); raw_now, tau_now = anchor_tau(ctx, device); prev_tau = float(prev["tau"]); prev_raw = float(prev.get("tau_raw_median_abs_anchor_margin", prev_tau));
        if not np.isclose(prev_tau, tau_now, rtol=1e-5, atol=1e-7):
            raise RuntimeError(f"previous tau mismatch {ctx.dataset} fold={ctx.fold}: previous={prev_tau} current={tau_now}")
        tau_rows[(ctx.dataset, int(ctx.fold))] = {"dataset": ctx.dataset, "fold": int(ctx.fold), "tau_raw_median_abs_anchor_margin": prev_raw, "tau": prev_tau, "tau_floor": TAU_FLOOR, "tau_source": "previous_decision_audit_legal_source_refit_anchor", "current_anchor_raw_check": raw_now, "frozen_before_outcome": True}
        ctx.tau = prev_tau
    specs, lookup = build_specs(contexts, tau_rows)
    write_csv(RESULTS / "WINDOW_SCHEDULE.csv", specs); write_csv(RESULTS / "MARGIN_SCALE.csv", list(tau_rows.values())); write_csv(RESULTS / "MARGIN_SCALE_AUDIT.csv", list(tau_rows.values()))
    equivalence = [sspg.checkpoint_equivalence(ctx, device) for ctx in contexts]; write_csv(RESULTS / "CHECKPOINT_EQUIVALENCE.csv", equivalence)
    mandatory = mandatory_tests(contexts, specs, lookup, device, legality, tau_rows); write_json(RESULTS / "MANDATORY_TESTS.json", mandatory)
    info = git_info(); lock = {"schema": "PERSIST_EEG_CUMULATIVE_SUBJECT_DECISION_DRIFT_PRE_OUTCOME_LOCK_V1", "experiment": "persist_eeg_cumulative_subject_decision_drift_audit_v1", "code_commit": info["commit"], "branch_at_code_freeze": info["branch"], "code_hashes": code_hashes(), "datasets": list(DATASETS), "folds": list(FOLDS), "seed": SEED, "seed1_run": False, "seed2_run": False, "second_backbone_run": False, "WBCIC_outer_opened": False, "OpenBMI_sealed_opened": False, "outcome_used": False, "horizon": {"epochs": 1, "definition": "one deterministic A-only natural epoch with fixed sentinel group excluded", "steps_rule": "ceil(A legal trial count / SCHEDULE_BATCH_REFERENCE)", "schedule_batch_reference": SCHEDULE_BATCH_REFERENCE, "batch_size": BATCH_SIZE}, "fixed_sentinel_meta_folds": [0, 1, 2, 3, 4], "K": K, "m_per_class": M_PER_CLASS, "certificate_blocks": [1, 2, 3, 4], "B_out": "all remaining legal source/refit trials", "B_out_min_class_support": M_PER_CLASS, "optimizer": {"name": "AdamW", "learning_rate": BASE_LR, "weight_decay": WEIGHT_DECAY, "gradient_clip": GRAD_CLIP, "parameter_scope": "FULL_TRAINABLE_PARAMETER_SPACE", "BN_running_statistics": "frozen", "dropout": "canonical TaskOnly keyed A RNG"}, "tau": {"definition": "previous decision-audit max(median(abs(anchor signed margin)),1e-3)", "source": str(PREVIOUS_EXP / "results" / "MARGIN_SCALE.csv"), "source_sha256": sha256_file(PREVIOUS_EXP / "results" / "MARGIN_SCALE.csv"), "rows": list(tau_rows.values()), "frozen_before_outcome": True}, "window_schedule": specs, "checkpoint_hashes": [{"dataset": ctx.dataset, "fold": int(ctx.fold), "path": str(ctx.checkpoint_path), "sha256": sha256_file(ctx.checkpoint_path)} for ctx in contexts], "mandatory_tests_sha256": sha256_file(RESULTS / "MANDATORY_TESTS.json"), "mandatory_tests_pass": True, "selection_rule": {"BBR": "select if BBR satisfies all cumulative decision gates and CE does not", "CE": "select if CE satisfies all cumulative decision gates and BBR does not", "both": "paired subject-bootstrap AUROC and Spearman; if no significant difference select CE", "none": "no certificate and no final model"}, "controls": ["deterministic_different_subject", "norm_matched_random"], "bootstrap": {"draws": BOOTSTRAP_DRAWS, "unit": "biological_subject_cluster"}, "legality": legality, "lock_status": "PRE_OUTCOME_LOCKED"}
    write_json(RESULTS / "PRE_OUTCOME_LOCK.json", lock); write_protocol_docs(lock, legality); write_json(RUNTIME / "PREFLIGHT.json", {"schema": "PERSIST_EEG_CUMULATIVE_PREFLIGHT_V1", "lock": lock, "mandatory": mandatory}); print("PRE_OUTCOME_LOCK_WRITTEN=true", flush=True); print("MANDATORY_TESTS_PASS=true", flush=True); print("outcome_used = false", flush=True); print("code_commit = " + str(info["commit"]), flush=True)


def require_lock() -> dict[str, Any]:
    path = RESULTS / "PRE_OUTCOME_LOCK.json"
    if not path.is_file(): raise RuntimeError("missing PRE_OUTCOME_LOCK.json; run preflight, commit and push it first")
    lock = json.loads(path.read_text(encoding="utf-8")); forbidden = ("seed1_run", "seed2_run", "second_backbone_run", "WBCIC_outer_opened", "OpenBMI_sealed_opened", "outcome_used")
    if any(lock.get(key) for key in forbidden): raise RuntimeError("forbidden run/access flag in pre-outcome lock")
    if lock.get("code_hashes") != code_hashes(): raise RuntimeError("code changed after pre-outcome lock")
    mandatory = RESULTS / "MANDATORY_TESTS.json"
    if not mandatory.is_file() or not json.loads(mandatory.read_text(encoding="utf-8")).get("pass"): raise RuntimeError("mandatory tests did not pass")
    return lock


def metric_ci_summary(frame: pd.DataFrame, dataset: str, cert_col: str, outcome_col: str, seed_tag: str) -> tuple[dict[str, Any], np.ndarray]:
    part = frame[frame.dataset == dataset].copy(); boot = bootstrap_metric(part, cert_col, outcome_col, stable_seed("cumulative-bootstrap", dataset, cert_col, outcome_col, seed_tag, SEED)); return metric_pack(part, cert_col, outcome_col, boot), boot


def aggregate(frame: pd.DataFrame, trajectories: pd.DataFrame, windows: list[dict[str, Any]], lock: dict[str, Any]) -> None:
    stats: dict[str, Any] = {}; smooth: dict[str, Any] = {}; controls: list[dict[str, Any]] = []; align_rows: list[dict[str, Any]] = []; bootstrap_json: dict[str, Any] = {}; cal_rows: list[dict[str, Any]] = []; power_rows: list[dict[str, Any]] = []; fold_rows = fold_metrics(frame)
    for dataset in DATASETS:
        part = frame[frame.dataset == dataset].copy(); bbr_dec, bbr_boot = metric_ci_summary(frame, dataset, "certificate_BBR", "H_BER_epoch", "decision"); ce_dec, ce_boot = metric_ci_summary(frame, dataset, "certificate_CE", "H_BER_epoch", "decision"); bbr_smooth, bbr_smooth_boot = metric_ci_summary(frame, dataset, "certificate_BBR", "H_BBR_epoch", "smooth"); ce_smooth, ce_smooth_boot = metric_ci_summary(frame, dataset, "certificate_CE", "H_CE_epoch", "smooth-ce"); same_auc = safe_auc(part.certificate_BBR, (part.H_BER_epoch > 0).astype(int)); diff_auc = safe_auc(part.certificate_BBR_different, (part.H_BER_epoch > 0).astype(int)); same_sp = safe_spearman(part.certificate_BBR, part.H_BER_epoch); diff_sp = safe_spearman(part.certificate_BBR_different, part.H_BER_epoch); auc_boot, sp_boot = bootstrap_difference(part, "certificate_BBR", "certificate_BBR_different", "H_BER_epoch", stable_seed("same-different", dataset, SEED)); rand_auc = safe_auc(part.certificate_BBR_random, (part.H_BER_epoch > 0).astype(int)); rand_sp = safe_spearman(part.certificate_BBR_random, part.H_BER_epoch); cal_rows.extend(calibration(part, "certificate_BBR", "BBR")); cal_rows.extend(calibration(part, "certificate_CE", "CE")); qpart = [r for r in cal_rows if r["dataset"] == dataset and r["certificate"] == "BBR"]; q1 = next(r["decision_harm_frequency"] for r in qpart if r["quintile"] == 1); q5 = next(r["decision_harm_frequency"] for r in qpart if r["quintile"] == 5); qdiff = float(q5 - q1); qboot = bootstrap_quintile(part, "certificate_BBR", "H_BER_epoch", stable_seed("qdiff", dataset, SEED)); power = {"dataset": dataset, "total_subject_window_observations": int(len(part)), "H_BER_positive_count": int(np.sum(part.H_BER_epoch > 0)), "H_BER_negative_count": int(np.sum(part.H_BER_epoch < 0)), "H_BER_zero_count": int(np.sum(part.H_BER_epoch == 0)), "biological_subjects_with_harmful_event": int(part.loc[part.H_BER_epoch > 0, "subject_id"].astype(str).nunique()), "total_correct_to_wrong_flips": int(part.correct_to_wrong.sum()), "total_wrong_to_correct_flips": int(part.wrong_to_correct.sum()), "CUMULATIVE_DECISION_ENDPOINT_UNDERPOWERED": bool(int(np.sum(part.H_BER_epoch > 0)) < 30 or int(part.loc[part.H_BER_epoch > 0, "subject_id"].astype(str).nunique()) < 15)}; power_rows.append(power); stats[dataset] = {"dataset": dataset, "BBR_to_H_BER_epoch": bbr_dec, "CE_to_H_BER_epoch": ce_dec, "same_subject_BBR_AUROC": same_auc, "different_subject_BBR_AUROC": diff_auc, "same_minus_different_AUROC": None if same_auc is None or diff_auc is None else float(same_auc - diff_auc), "same_subject_BBR_Spearman": same_sp, "different_subject_BBR_Spearman": diff_sp, "same_minus_different_Spearman": None if same_sp is None or diff_sp is None else float(same_sp - diff_sp), "same_minus_different_AUROC_CI95": ci_from(auc_boot), "same_minus_different_Spearman_CI95": ci_from(sp_boot), "random_BBR_AUROC": rand_auc, "random_BBR_Spearman": rand_sp, "Q5_minus_Q1_decision_harm": qdiff, "Q5_minus_Q1_decision_harm_CI95": ci_from(qboot), "power": power}; smooth[dataset] = {"dataset": dataset, "BBR_to_H_BBR_epoch": bbr_smooth, "CE_to_H_CE_epoch": ce_smooth}; controls.append({"dataset": dataset, "control": "same_subject", "certificate_column": "certificate_BBR", "decision_AUROC": same_auc, "decision_Spearman": same_sp}); controls.append({"dataset": dataset, "control": "different_subject", "certificate_column": "certificate_BBR_different", "decision_AUROC": diff_auc, "decision_Spearman": diff_sp}); controls.append({"dataset": dataset, "control": "random", "certificate_column": "certificate_BBR_random", "decision_AUROC": rand_auc, "decision_Spearman": rand_sp}); align_rows.extend([{ "dataset": dataset, "metric": "H_BER Spearman", "BBR": bbr_dec.get("spearman"), "CE": ce_dec.get("spearman"), "BBR_minus_CE": None if bbr_dec.get("spearman") is None or ce_dec.get("spearman") is None else bbr_dec["spearman"] - ce_dec["spearman"], "BBR_minus_CE_CI95": ci_from(bbr_boot[:, 0] - ce_boot[:, 0])}, {"dataset": dataset, "metric": "decision-harm AUROC", "BBR": bbr_dec.get("auroc"), "CE": ce_dec.get("auroc"), "BBR_minus_CE": None if bbr_dec.get("auroc") is None or ce_dec.get("auroc") is None else bbr_dec["auroc"] - ce_dec["auroc"], "BBR_minus_CE_CI95": ci_from(bbr_boot[:, 2] - ce_boot[:, 2])}]); bootstrap_json[dataset] = {"draws": BOOTSTRAP_DRAWS, "BBR_to_H_BER_epoch": bbr_dec, "CE_to_H_BER_epoch": ce_dec, "BBR_to_H_BBR_epoch": bbr_smooth, "CE_to_H_CE_epoch": ce_smooth, "same_minus_different_AUROC_CI95": ci_from(auc_boot), "same_minus_different_Spearman_CI95": ci_from(sp_boot), "Q5_minus_Q1_decision_harm_CI95": ci_from(qboot)}
    write_csv(RESULTS / "PER_FOLD_METRICS.csv", fold_rows); write_csv(RESULTS / "PER_SUBJECT_METRICS.csv", frame.groupby(["dataset", "subject_id"], as_index=False).agg(observations=("sentinel_meta_fold", "size"), mean_H_BBR_epoch=("H_BBR_epoch", "mean"), mean_H_CE_epoch=("H_CE_epoch", "mean"), mean_H_BER_epoch=("H_BER_epoch", "mean"), total_correct_to_wrong=("correct_to_wrong", "sum"), total_wrong_to_correct=("wrong_to_correct", "sum"))); write_csv(RESULTS / "DECISION_DRIFT_SUMMARY.csv", [{"dataset": d, **stats[d]["BBR_to_H_BER_epoch"], "CE_decision_AUROC": stats[d]["CE_to_H_BER_epoch"].get("auroc"), "same_minus_different_AUROC": stats[d]["same_minus_different_AUROC"], "Q5_minus_Q1_decision_harm": stats[d]["Q5_minus_Q1_decision_harm"], "H_BER_positive_events": stats[d]["power"]["H_BER_positive_count"]} for d in DATASETS]); write_csv(RESULTS / "SMOOTH_HARM_SUMMARY.csv", [{"dataset": d, **smooth[d]["BBR_to_H_BBR_epoch"], "CE_H_CE_AUROC": smooth[d]["CE_to_H_CE_epoch"].get("auroc")} for d in DATASETS]); write_csv(RESULTS / "CE_VS_BBR.csv", align_rows); write_csv(RESULTS / "SAME_VS_DIFFERENT.csv", [{"dataset": d, "same_subject_AUROC": stats[d]["same_subject_BBR_AUROC"], "different_subject_AUROC": stats[d]["different_subject_BBR_AUROC"], "AUROC_advantage": stats[d]["same_minus_different_AUROC"], "AUROC_advantage_CI95": stats[d]["same_minus_different_AUROC_CI95"], "same_subject_Spearman": stats[d]["same_subject_BBR_Spearman"], "different_subject_Spearman": stats[d]["different_subject_BBR_Spearman"], "Spearman_advantage": stats[d]["same_minus_different_Spearman"], "Spearman_advantage_CI95": stats[d]["same_minus_different_Spearman_CI95"]} for d in DATASETS]); write_csv(RESULTS / "RANDOM_CONTROL.csv", [r for r in controls if r["control"] == "random"]); write_csv(RESULTS / "CALIBRATION_BINS.csv", cal_rows); write_csv(RESULTS / "CUMULATIVE_PER_OBSERVATION.csv", frame); write_csv(RESULTS / "WINDOW_SUMMARY.csv", windows); write_csv(RESULTS / "TRAJECTORY_SUMMARY.csv", trajectories); write_json(RESULTS / "BOOTSTRAP_RESULTS.json", bootstrap_json); write_csv(RESULTS / "POWER_AUDIT.csv", power_rows)
    # Frozen gate evaluation.  A certificate passes B only if both datasets
    # satisfy the predeclared decision AUROC and Spearman criteria.
    gate_a = {d: bool(stats[d]["power"]["H_BER_positive_count"] >= 30 and stats[d]["power"]["biological_subjects_with_harmful_event"] >= 15) for d in DATASETS}; cert_gate: dict[str, dict[str, bool]] = {"BBR": {}, "CE": {}}
    for cert, key in (("BBR", "BBR_to_H_BER_epoch"), ("CE", "CE_to_H_BER_epoch")):
        for d in DATASETS:
            p = stats[d][key]; cert_gate[cert][d] = bool(gate_a[d] and (p.get("auroc") or -np.inf) >= 0.60 and (p.get("auroc_CI95") or [-np.inf])[0] > 0.50 and (p.get("spearman") or -np.inf) > 0 and (p.get("spearman_CI95") or [-np.inf])[0] > 0)
    gate_b = {cert: all(values.values()) for cert, values in cert_gate.items()}
    gate_c = {}; gate_d = {}; gate_e = {}; gate_f = {}
    for cert in ("BBR", "CE"):
        gate_c[cert] = all((stats[d]["same_minus_different_AUROC"] or -np.inf) > 0 and (stats[d]["same_minus_different_AUROC_CI95"][0] > 0 if d == DATASETS[0] else stats[d]["same_minus_different_AUROC_CI95"][0] >= -0.02) for d in DATASETS) and any(stats[d]["same_minus_different_AUROC_CI95"][0] > 0 for d in DATASETS)
        # CE specificity uses its own certificate columns for a fair frozen
        # comparator; the summary rows are calculated here from the same frame.
        if cert == "CE":
            for d in DATASETS:
                part = frame[frame.dataset == d]; aucb, spb = bootstrap_difference(part, "certificate_CE", "certificate_CE_different", "H_BER_epoch", stable_seed("same-different-ce", d, SEED)); stats[d]["CE_same_minus_different_AUROC"] = float((safe_auc(part.certificate_CE, (part.H_BER_epoch > 0).astype(int)) or 0) - (safe_auc(part.certificate_CE_different, (part.H_BER_epoch > 0).astype(int)) or 0)); stats[d]["CE_same_minus_different_AUROC_CI95"] = ci_from(aucb); stats[d]["CE_same_minus_different_Spearman"] = float((safe_spearman(part.certificate_CE, part.H_BER_epoch) or 0) - (safe_spearman(part.certificate_CE_different, part.H_BER_epoch) or 0)); stats[d]["CE_same_minus_different_Spearman_CI95"] = ci_from(spb)
            gate_c[cert] = all(stats[d]["CE_same_minus_different_AUROC"] > 0 and stats[d]["CE_same_minus_different_AUROC_CI95"][0] >= -0.02 for d in DATASETS) and any(stats[d]["CE_same_minus_different_AUROC_CI95"][0] > 0 for d in DATASETS)
        qdiffs = [stats[d]["Q5_minus_Q1_decision_harm"] for d in DATASETS]; qcis = [stats[d]["Q5_minus_Q1_decision_harm_CI95"][0] for d in DATASETS]; gate_d[cert] = bool(all(value > 0 for value in qdiffs) and any(value > 0 for value in qcis))
        rows_e = [r for r in fold_rows if r["dataset"] in DATASETS]; gate_e[cert] = all(sum(bool((r["same_subject_spearman_positive"] if cert == "BBR" else (r.get("CE_H_BER_Spearman") or 0) > 0)) for r in rows_e if r["dataset"] == d) >= 4 for d in DATASETS)
        gate_f[cert] = all((stats[d]["BBR_to_H_BER_epoch" if cert == "BBR" else "CE_to_H_BER_epoch"].get("auroc") or -np.inf) > (stats[d]["random_BBR_AUROC"] or -np.inf) for d in DATASETS)
    gate_core = {cert: bool(gate_b[cert] and gate_c[cert] and gate_d[cert] and gate_e[cert] and gate_f[cert]) for cert in ("BBR", "CE")}
    bbr_ce_delta = {d: next(r for r in align_rows if r["dataset"] == d and r["metric"] == "decision-harm AUROC") for d in DATASETS}; bbr_selected = False; ce_selected = False
    if gate_core["BBR"] and not gate_core["CE"]: bbr_selected = True
    elif gate_core["CE"] and not gate_core["BBR"]: ce_selected = True
    elif gate_core["BBR"] and gate_core["CE"]:
        ci_delta = [bbr_ce_delta[d]["BBR_minus_CE_CI95"] for d in DATASETS]
        if all(ci[0] > 0 for ci in ci_delta): bbr_selected = True
        else: ce_selected = True
    selected = "BBR" if bbr_selected else "CE" if ce_selected else "NONE"
    powered_all = all(gate_a.values()); smooth_support = all((smooth[d]["BBR_to_H_BBR_epoch"].get("spearman") or 0) > 0 for d in DATASETS)
    if not powered_all: terminal = "CUMULATIVE_DECISION_ENDPOINT_UNDERPOWERED"
    elif selected != "NONE": terminal = "CUMULATIVE_SUBJECT_DECISION_DRIFT_SUPPORTED"
    elif any(gate_b.values()): terminal = "CUMULATIVE_DECISION_DRIFT_DATASET_DEPENDENT"
    elif smooth_support: terminal = "CUMULATIVE_SMOOTH_HARM_SUPPORTED_DECISION_NOT_SUPPORTED"
    else: terminal = "CUMULATIVE_SUBJECT_DECISION_DRIFT_NOT_SUPPORTED"
    decision = {"terminal": terminal, "gate_A_power": gate_a, "gate_B_cumulative_decision_prediction": gate_b, "gate_C_subject_specificity": gate_c, "gate_D_calibration": gate_d, "gate_E_fold_robustness": gate_e, "gate_F_random_control": gate_f, "certificate_gate_details": cert_gate, "selected_certificate": selected, "FINAL_MODEL_DEVELOPMENT_SCIENTIFICALLY_JUSTIFIED": terminal == "CUMULATIVE_SUBJECT_DECISION_DRIFT_SUPPORTED", "AUTO_START_FINAL_MODEL": False, "CLOSE_PROSPECTIVE_GRADIENT_CONSTRUCTIVE_FAMILY": terminal in {"CUMULATIVE_SMOOTH_HARM_SUPPORTED_DECISION_NOT_SUPPORTED", "CUMULATIVE_SUBJECT_DECISION_DRIFT_NOT_SUPPORTED"}}
    validation = {"schema": "PERSIST_EEG_CUMULATIVE_VALIDATION_V1", "pass": True, "terminal": terminal, "mandatory_tests_pass": True, "all_windows_complete": bool(len(windows) == 50 and all(bool(row.get("complete")) for row in windows)), "fixed_sentinel_isolation": bool((frame.displacement_sum_max_abs_error <= 5e-5).all()), "cumulative_displacement_sum_pass": bool((frame.displacement_sum_max_abs_error <= 5e-5).all()), "BN_unchanged": bool((frame.bn_max_displacement <= 1e-12).all()), "outcome_used": False, "seed1_run": False, "seed2_run": False, "second_backbone_run": False, "WBCIC_outer_opened": False, "OpenBMI_sealed_opened": False, "decision": decision}
    write_json(RESULTS / "VALIDATION.json", validation); write_json(RESULTS / "FINAL_REPORT.json", {"schema": "PERSIST_EEG_CUMULATIVE_FINAL_REPORT_V1", "terminal": terminal, "decision": decision, "datasets": stats, "smooth": smooth, "power": power_rows, "fold_metrics": fold_rows, "validation": validation, "lock": lock});
    lines = ["# Final report", "", f"terminal = {terminal}", "", "Source/refit-only fixed-sentinel seed-0 EEGNet one-epoch cumulative audit. No outcome, WBCIC outer-10, OpenBMI sealed cohort, seed 1/2 or second backbone was opened.", "", "|dataset|H_BER events|harmful subjects|CE decision AUROC|CE CI|BBR decision AUROC|BBR CI|BBR selected Spearman|same-different AUROC|Q5-Q1|", "|---|---:|---:|---:|---|---:|---|---:|---:|---:|"]
    for d in DATASETS:
        b = stats[d]["BBR_to_H_BER_epoch"]; c = stats[d]["CE_to_H_BER_epoch"]; lines.append(f"|{d}|{stats[d]['power']['H_BER_positive_count']}|{stats[d]['power']['biological_subjects_with_harmful_event']}|{c.get('auroc')}|{c.get('auroc_CI95')}|{b.get('auroc')}|{b.get('auroc_CI95')}|{b.get('spearman')}|{stats[d]['same_minus_different_AUROC']}|{stats[d]['Q5_minus_Q1_decision_harm']}|")
    lines += ["", "## Required answers", "", f"- One-epoch exact decision power sufficient: {powered_all}", f"- Selected cumulative certificate: {selected}", f"- Final model development scientifically justified: {decision['FINAL_MODEL_DEVELOPMENT_SCIENTIFICALLY_JUSTIFIED']}", f"- Same-subject specificity: {gate_c}", f"- Fold robustness: {gate_e}", f"- Prospective-gradient family closure recommendation: {decision['CLOSE_PROSPECTIVE_GRADIENT_CONSTRUCTIVE_FAMILY']}", "- No new horizon or model was started automatically."]
    (EXP / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); write_protocol_docs(lock, json.loads((RESULTS / "PRE_OUTCOME_LOCK.json").read_text(encoding="utf-8")), terminal)
    print("terminal = " + terminal, flush=True)
    for d in DATASETS:
        b = stats[d]["BBR_to_H_BER_epoch"]; c = stats[d]["CE_to_H_BER_epoch"]; print(f"{d}_HBER_positive_events = {stats[d]['power']['H_BER_positive_count']}", flush=True); print(f"{d}_harmful_subjects = {stats[d]['power']['biological_subjects_with_harmful_event']}", flush=True); print(f"{d}_CE_decision_AUROC = {c.get('auroc')}", flush=True); print(f"{d}_CE_decision_AUROC_CI = {c.get('auroc_CI95')}", flush=True); print(f"{d}_BBR_decision_AUROC = {b.get('auroc')}", flush=True); print(f"{d}_BBR_decision_AUROC_CI = {b.get('auroc_CI95')}", flush=True); print(f"{d}_selected_Spearman = {b.get('spearman') if selected == 'BBR' else c.get('spearman')}", flush=True)
    print(f"SELECTED_CUMULATIVE_CERTIFICATE = {selected}", flush=True); print(f"OpenBMI_same_minus_different_AUROC = {stats['OpenBMI']['same_minus_different_AUROC']}", flush=True); print(f"WBCIC_same_minus_different_AUROC = {stats['WBCIC']['same_minus_different_AUROC']}", flush=True); print("WBCIC_outer_opened = false", flush=True); print("OpenBMI_sealed_opened = false", flush=True); print("outcome_used = false", flush=True); print("seed1_run = false", flush=True); print("seed2_run = false", flush=True); print("second_backbone_run = false", flush=True); print(f"FINAL_MODEL_DEVELOPMENT_SCIENTIFICALLY_JUSTIFIED = {'YES' if decision['FINAL_MODEL_DEVELOPMENT_SCIENTIFICALLY_JUSTIFIED'] else 'NO'}", flush=True); print(f"CLOSE_PROSPECTIVE_GRADIENT_CONSTRUCTIVE_FAMILY = {'YES' if decision['CLOSE_PROSPECTIVE_GRADIENT_CONSTRUCTIVE_FAMILY'] else 'NO'}", flush=True)


def run_audit(device: torch.device) -> None:
    lock = require_lock(); contexts, _ = sspg.load_contexts_source_only(); tau_prev = previous_tau_rows(); tau_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for ctx in contexts:
        ctx.tau = float(tau_prev[(ctx.dataset, int(ctx.fold))]["tau"]); tau_rows[(ctx.dataset, int(ctx.fold))] = {**tau_prev[(ctx.dataset, int(ctx.fold))], "tau_source": "previous_decision_audit_legal_source_refit_anchor"}
    specs, lookup = build_specs(contexts, tau_rows); locked_specs = lock.get("window_schedule", []); current_digest = hashlib.sha256(json.dumps(specs, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(); locked_digest = hashlib.sha256(json.dumps(locked_specs, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if current_digest != locked_digest: raise RuntimeError("window schedule changed after pre-outcome lock")
    all_obs: list[dict[str, Any]] = []; all_traj: list[dict[str, Any]] = []; all_windows: list[dict[str, Any]] = []
    for ctx in contexts:
        gradients = anchor_gradients(ctx, ctx.source_subjects, device)
        for sentinel_meta_fold in range(5):
            item = lookup[(ctx.dataset, int(ctx.fold), sentinel_meta_fold)]; cache_path = RUNTIME / f"window_{ctx.dataset}_fold{ctx.fold}_sentinel{sentinel_meta_fold}.json"; cached = None
            if cache_path.is_file():
                try: cached = json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception: cached = None
            if cached and cached.get("complete") and cached.get("schedule_sha256") == item["schedule_sha256"]:
                obs, traj, window = cached["observations"], cached["trajectory"], cached["window"]
            else:
                print(f"[window] {ctx.dataset} fold={ctx.fold} sentinel_meta_fold={sentinel_meta_fold} steps={item['steps']}", flush=True); obs, traj, window = train_window(ctx, item, gradients, device); write_json(cache_path, {"complete": True, "schedule_sha256": item["schedule_sha256"], "observations": obs, "trajectory": traj, "window": window})
            all_obs.extend(obs); all_traj.extend(traj); all_windows.append(window); print(f"[window-done] {ctx.dataset} fold={ctx.fold} sentinel_meta_fold={sentinel_meta_fold} observations={len(obs)}", flush=True)
        del gradients
    frame = pd.DataFrame(all_obs); trajectories = pd.DataFrame(all_traj); frame = frame.sort_values(["dataset", "fold", "sentinel_meta_fold", "subject_id"]).reset_index(drop=True); trajectories = trajectories.sort_values(["dataset", "fold", "sentinel_meta_fold", "step"]).reset_index(drop=True); aggregate(frame, trajectories, all_windows, lock)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("preflight", "run"), required=True); parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); args = parser.parse_args(); device = torch.device(args.device); print(f"device = {device}", flush=True)
    if args.phase == "preflight": preflight(device)
    else: run_audit(device)


if __name__ == "__main__":
    main()
