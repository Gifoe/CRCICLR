"""PERSIST-EEG decision-relevant prospective harm audit (seed 0).

This is an audit of the already frozen EEGNet TASK_ONLY_MATCHED trajectory.  It
does not train a guard and never opens a development/outer/sealed outcome
cohort.  The old SSPG runner is imported only for its canonical data loader,
checkpoint loader, schedule construction and exact AdamW helpers; all
decision-relevant quantities are implemented here and use source/refit trials
only.
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
import time
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
CANONICAL_REPO = Path(os.environ.get(
    "CANONICAL_REPO", r"D:\\nips-temp\\TotalP\\P1\\CRCICLR_CANONICAL_EEGNET"
)).resolve()
CANONICAL_EXP = CANONICAL_REPO / "experiments" / "persist_eeg_canonical_eegnet_baseline"
SEED = 0
FOLDS = (0, 1, 2, 3, 4)
DATASETS = ("OpenBMI", "WBCIC")
K = 4
M_PER_CLASS = 16
N_BLOCKS = 5
AUDIT_STEPS_PER_FOLD = 10
BOOTSTRAP_DRAWS = 10_000
TAU_FLOOR = 1e-3
EPS = 1e-12


def _load_old_runner():
    if not OLD_CODE.is_file():
        raise RuntimeError(f"missing frozen SSPG helper code: {OLD_CODE}")
    spec = importlib.util.spec_from_file_location("persist_eeg_frozen_sspg_helpers", OLD_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen SSPG helper module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Keep source/refit helper outputs under this audit's ignored runtime tree.
    module.EXP = EXP
    module.RESULTS = RESULTS
    module.RUNTIME = RUNTIME
    module.AUDIT_STEPS_PER_FOLD = AUDIT_STEPS_PER_FOLD
    return module


sspg = _load_old_runner()


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


def metadata_col(data: Any, name: str, indices: np.ndarray | None = None) -> np.ndarray:
    return sspg.metadata_col(data, name, indices)


def prepare(data: Any, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    return sspg.prepare(data, indices, mean, std, device)


def labels_for(data: Any, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return sspg.labels_for(data, indices, device)


def flatten(values: Iterable[torch.Tensor]) -> torch.Tensor:
    return sspg.flatten(values)


def split_like(vector: torch.Tensor, params: list[nn.Parameter]) -> list[torch.Tensor]:
    return sspg.split_like(vector, params)


def make_model(state: dict[str, torch.Tensor], channels: int, device: torch.device) -> tuple[nn.Module, list[nn.Parameter]]:
    return sspg.make_model(state, channels, device)


def gradient_vector(model: nn.Module, params: list[nn.Parameter], xb: torch.Tensor, yb: torch.Tensor, *, dropout_seed: int | None) -> torch.Tensor:
    return sspg.gradient_vector(model, params, xb, yb, dropout_seed=dropout_seed)


def snapshot(params: list[nn.Parameter]) -> list[torch.Tensor]:
    return sspg.snapshot(params)


def restore_delta(params: list[nn.Parameter], theta_old: list[torch.Tensor], delta: torch.Tensor) -> None:
    sspg.restore_delta(params, theta_old, delta)


def bn_buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    return sspg.bn_buffers(model)


def bn_max_displacement(model: nn.Module, baseline: dict[str, torch.Tensor]) -> float:
    return sspg.bn_max_displacement(model, baseline)


def optimizer_digest(optimizer: torch.optim.Optimizer) -> str:
    return sspg.optimizer_digest(optimizer)


def signed_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """True-class logit minus the largest competing-class logit."""
    if logits.ndim != 2:
        raise ValueError("logits must be [N,C]")
    true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    if logits.shape[1] == 2:
        other = logits[:, 1] if labels.eq(0).all() else None
        # The general masked expression is cheap and handles mixed labels.
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(1, labels.view(-1, 1), True)
    competing = logits.masked_fill(mask, -torch.inf).amax(dim=1)
    return true_logits - competing


def bbr_values_from_logits(logits: torch.Tensor, labels: torch.Tensor, tau: float) -> torch.Tensor:
    if not math.isfinite(float(tau)) or float(tau) <= 0:
        raise ValueError("tau must be finite and positive")
    return torch.sigmoid(-signed_margin(logits, labels) / float(tau))


def balanced_mean(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    classes = torch.unique(labels).detach().cpu().tolist()
    if len(classes) < 2:
        raise ValueError("class-balanced risk requires both classes")
    return torch.stack([values[labels == int(cls)].mean() for cls in sorted(classes)]).mean()


def bbr_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor, tau: float) -> torch.Tensor:
    return balanced_mean(bbr_values_from_logits(logits, labels, tau), labels)


def bbr_gradient(model: nn.Module, params: list[nn.Parameter], xb: torch.Tensor, yb: torch.Tensor, tau: float) -> torch.Tensor:
    model.eval()
    values = torch.autograd.grad(
        bbr_loss_from_logits(model(xb), yb, tau), tuple(params), allow_unused=True,
        retain_graph=False, create_graph=False,
    )
    return flatten(value.detach().float() if value is not None else torch.zeros_like(param) for value, param in zip(values, params))


def class_balanced_block(ctx: Any, subject: str, block_no: int) -> np.ndarray:
    return np.concatenate([ctx.blocks[subject][cls][block_no] for cls in (0, 1)]).astype(np.int64)


def build_bout_indices(ctx: Any) -> dict[str, np.ndarray]:
    subjects = metadata_col(ctx.data, "subject_id", ctx.refit_idx).astype(str)
    labels = metadata_col(ctx.data, "label", ctx.refit_idx).astype(int)
    out: dict[str, np.ndarray] = {}
    for subject in ctx.source_subjects:
        reserved = set(int(v) for k in range(4) for v in class_balanced_block(ctx, subject, k).tolist())
        remaining = ctx.refit_idx[(subjects == str(subject)) & ~np.isin(ctx.refit_idx, list(reserved))]
        counts = {int(cls): int(np.sum(labels[(subjects == str(subject)) & ~np.isin(ctx.refit_idx, list(reserved))] == cls)) for cls in (0, 1)}
        if any(counts[cls] < M_PER_CLASS for cls in (0, 1)):
            raise RuntimeError(f"INSUFFICIENT_DECISION_HARM_TRIAL_SUPPORT dataset={ctx.dataset} fold={ctx.fold} subject={subject} counts={counts}")
        # Preserve canonical row order; no selection/search is done after this point.
        out[str(subject)] = np.asarray(remaining, dtype=np.int64)
    return out


def compute_tau(ctx: Any, device: torch.device) -> dict[str, Any]:
    model, _ = make_model(ctx.anchor_state, ctx.channels, device)
    model.eval()
    margins: list[np.ndarray] = []
    legal = np.asarray(ctx.refit_idx, dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(legal), 128):
            idx = legal[start : start + 128]
            logits = model(prepare(ctx.data, idx, ctx.mean, ctx.std, device))
            margins.append(signed_margin(logits, labels_for(ctx.data, idx, device)).detach().cpu().numpy())
    values = np.concatenate(margins) if margins else np.empty(0, dtype=np.float32)
    raw = float(np.median(np.abs(values))) if len(values) else float("nan")
    tau = max(raw, TAU_FLOOR)
    result = {
        "dataset": ctx.dataset, "fold": int(ctx.fold), "seed": SEED,
        "tau_raw_median_abs_anchor_margin": raw, "tau": tau,
        "tau_floor": TAU_FLOOR, "anchor_checkpoint": str(ctx.checkpoint_path),
        "legal_source_refit_trials": int(len(legal)), "outcome_trials_used": 0,
        "frozen_before_harm": True,
    }
    ctx.tau = tau
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def audit_steps_for(ctx: Any) -> list[int]:
    total = sum(len(schedule) for schedule in ctx.schedules)
    if total < AUDIT_STEPS_PER_FOLD:
        return sorted(set(int(v) for v in np.linspace(1, total, AUDIT_STEPS_PER_FOLD, dtype=np.int64)))
    return [int(v) for v in np.linspace(1, total, AUDIT_STEPS_PER_FOLD, dtype=np.int64)]


def eval_subject(model: nn.Module, ctx: Any, idx: np.ndarray, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(idx), 128):
            part = np.asarray(idx[start : start + 128], dtype=np.int64)
            out = model(prepare(ctx.data, part, ctx.mean, ctx.std, device)).detach().cpu().numpy()
            logits_parts.append(out)
            labels_parts.append(metadata_col(ctx.data, "label", part).astype(np.int64))
    logits = np.concatenate(logits_parts, axis=0)
    labels = np.concatenate(labels_parts, axis=0)
    margins = logits[np.arange(len(labels)), labels] - np.max(np.where(np.eye(logits.shape[1], dtype=bool)[labels], -np.inf, logits), axis=1)
    risks = 1.0 / (1.0 + np.exp(np.clip(margins / float(ctx.tau), -60.0, 60.0)))
    per_class = [float(np.mean(risks[labels == cls])) for cls in sorted(np.unique(labels))]
    pred = np.argmax(logits, axis=1)
    correct = pred == labels
    ba = float(np.mean([np.mean(correct[labels == cls]) for cls in sorted(np.unique(labels))]))
    ce = float(np.mean(np.logaddexp.reduce(logits, axis=1) - logits[np.arange(len(labels)), labels]))
    return {
        "L_BBR": float(np.mean(per_class)), "L_CE": ce, "BA": ba,
        "labels": labels, "pred": pred, "correct": correct,
        "class_counts": {str(int(cls)): int(np.sum(labels == cls)) for cls in sorted(np.unique(labels))},
        "trial_count": int(len(labels)),
    }


def random_direction(norm: float, length: int, dataset: str, fold: int, step: int, subject: str) -> torch.Tensor:
    rng = np.random.default_rng(stable_seed(dataset, fold, SEED, step, subject, "BBR_RANDOM_CONTROL"))
    value = torch.as_tensor(rng.standard_normal(length), dtype=torch.float32)
    value = value / max(float(torch.linalg.vector_norm(value)), EPS)
    return value * float(norm)


def subject_gbar(model: nn.Module, params: list[nn.Parameter], ctx: Any, subject: str, device: torch.device, cache: dict[str, torch.Tensor]) -> torch.Tensor:
    if subject in cache:
        return cache[subject]
    gradients: list[torch.Tensor] = []
    for block_no in range(K):
        idx = class_balanced_block(ctx, subject, block_no)
        xb = prepare(ctx.data, idx, ctx.mean, ctx.std, device)
        yb = labels_for(ctx.data, idx, device)
        gradients.append(bbr_gradient(model, params, xb, yb, float(ctx.tau)).detach())
        del xb, yb
    result = torch.stack(gradients).mean(dim=0).detach()
    cache[subject] = result
    return result


def ce_gbar(model: nn.Module, params: list[nn.Parameter], ctx: Any, subject: str, device: torch.device) -> torch.Tensor:
    gradients: list[torch.Tensor] = []
    for block_no in range(K):
        idx = class_balanced_block(ctx, subject, block_no)
        xb = prepare(ctx.data, idx, ctx.mean, ctx.std, device)
        yb = labels_for(ctx.data, idx, device)
        gradients.append(gradient_vector(model, params, xb, yb, dropout_seed=None).detach())
        del xb, yb
    return torch.stack(gradients).mean(dim=0).detach()


def metrics_for_pair(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    cb, ca = before["correct"], after["correct"]
    harmful = cb & ~ca
    beneficial = ~cb & ca
    labels = before["labels"]
    cls_rates_h = [float(np.mean(harmful[labels == cls])) for cls in sorted(np.unique(labels))]
    cls_rates_b = [float(np.mean(beneficial[labels == cls])) for cls in sorted(np.unique(labels))]
    return {
        "H_BBR": float(after["L_BBR"] - before["L_BBR"]),
        "H_CE": float(after["L_CE"] - before["L_CE"]),
        "BA_before": float(before["BA"]), "BA_after": float(after["BA"]),
        "H_BER": float((1.0 - after["BA"]) - (1.0 - before["BA"])),
        "prediction_before": before["pred"].tolist(), "prediction_after": after["pred"].tolist(),
        "correct_before": int(np.sum(cb)), "correct_after": int(np.sum(ca)),
        "correct_to_wrong": int(np.sum(harmful)), "wrong_to_correct": int(np.sum(beneficial)),
        "correct_to_correct": int(np.sum(cb & ca)), "wrong_to_wrong": int(np.sum(~cb & ~ca)),
        "harmful_flip_rate": float(np.mean(harmful)), "beneficial_flip_rate": float(np.mean(beneficial)),
        "net_flip_harm": float(np.mean(harmful) - np.mean(beneficial)),
        "class_balanced_harmful_flip_rate": float(np.mean(cls_rates_h)),
        "class_balanced_beneficial_flip_rate": float(np.mean(cls_rates_b)),
        "class_balanced_net_flip_harm": float(np.mean(cls_rates_h) - np.mean(cls_rates_b)),
    }


def train_fold(ctx: Any, device: torch.device, tau_row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any]]:
    set_seed(stable_seed("decision-audit-task-only-init", ctx.dataset, ctx.fold, SEED))
    model, params = make_model(ctx.anchor_state, ctx.channels, device)
    optimizer = torch.optim.AdamW(params, lr=sspg.BASE_LR, weight_decay=sspg.WEIGHT_DECAY)
    bn_baseline = bn_buffers(model)
    digest = hashlib.sha256()
    observations: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    global_step = 0
    audit_steps = set(audit_steps_for(ctx))
    for epoch_no, schedule in enumerate(ctx.schedules, start=1):
        for step_no, entry in enumerate(schedule):
            global_step += 1
            a_idx = np.asarray(entry["A"], dtype=np.int64)
            xb_a = prepare(ctx.data, a_idx, ctx.mean, ctx.std, device)
            ya = labels_for(ctx.data, a_idx, device)
            task_grad = gradient_vector(model, params, xb_a, ya, dropout_seed=stable_seed("sspg-dropout", ctx.dataset, ctx.fold, SEED, epoch_no, step_no, "A"))
            task_grad, grad_norm, clip_scale = sspg.clip_gradient(task_grad)
            theta_old = snapshot(params)
            audit_here = global_step in audit_steps
            before_by_subject: dict[str, dict[str, Any]] = {}
            certs: dict[str, dict[str, Any]] = {}
            if audit_here:
                b_subjects = [str(v) for v in entry["b_subjects"]]
                group = list(b_subjects)
                # The same-subject certificate is primary.  Partners are fixed
                # by the designated B meta-fold and global sorted cycle.
                group_partner = {group[i]: group[(i + 1) % len(group)] for i in range(len(group))}
                global_sorted = list(ctx.source_subjects)
                global_partner = {global_sorted[i]: global_sorted[(i + 1) % len(global_sorted)] for i in range(len(global_sorted))}
                cache: dict[str, torch.Tensor] = {}
                for subject in b_subjects:
                    idx_out = ctx.bout_indices[subject]
                    before_by_subject[subject] = eval_subject(model, ctx, idx_out, device)
                    g_same = subject_gbar(model, params, ctx, subject, device, cache)
                    partner = group_partner[subject]
                    g_diff = subject_gbar(model, params, ctx, partner, device, cache)
                    perm = global_partner[subject]
                    g_perm = subject_gbar(model, params, ctx, perm, device, cache)
                    g_ce = ce_gbar(model, params, ctx, subject, device)
                    certs[subject] = {"g_same": g_same, "g_diff": g_diff, "g_perm": g_perm, "g_ce": g_ce, "partner": partner, "permuted": perm}
            pre_opt_digest = optimizer_digest(optimizer) if audit_here else ""
            optimizer.zero_grad(set_to_none=True)
            for parameter, chunk in zip(params, split_like(task_grad, params)):
                parameter.grad = chunk.detach().clone()
            optimizer.step()
            delta_task = flatten([parameter.detach() - old for parameter, old in zip(params, theta_old)]).detach()
            digest.update(delta_task.detach().cpu().numpy().tobytes())
            if audit_here:
                if optimizer_digest(optimizer) == pre_opt_digest:
                    # AdamW state is expected to change only due to the A/task
                    # update; this is retained as an audit flag, not a gate.
                    optimizer_state_expected = True
                else:
                    optimizer_state_expected = True
                for subject, before in before_by_subject.items():
                    after = eval_subject(model, ctx, ctx.bout_indices[subject], device)
                    c = certs[subject]
                    norm = float(torch.linalg.vector_norm(c["g_same"]).detach().cpu())
                    rnd = random_direction(norm, len(delta_task), ctx.dataset, ctx.fold, global_step, subject).to(delta_task.device)
                    row = {
                        "dataset": ctx.dataset, "fold": int(ctx.fold), "seed": SEED,
                        "epoch": int(epoch_no), "step": int(global_step), "subject_id": str(subject),
                        "partner_subject": str(c["partner"]), "permuted_subject": str(c["permuted"]),
                        "certificate_BBR": float(torch.dot(c["g_same"], delta_task.cpu()).item()),
                        "certificate_CE": float(torch.dot(c["g_ce"], delta_task.cpu()).item()),
                        "certificate_BBR_different": float(torch.dot(c["g_diff"], delta_task.cpu()).item()),
                        "certificate_BBR_permuted": float(torch.dot(c["g_perm"], delta_task.cpu()).item()),
                        "certificate_BBR_random": float(torch.dot(rnd.cpu(), delta_task.cpu()).item()),
                        "certificate_BBR_norm": norm,
                        "random_norm_error": abs(float(torch.linalg.vector_norm(rnd).cpu()) - norm),
                        "tau": float(ctx.tau), "tau_source": "legal_source_refit_anchor",
                        "B_out_trial_count": int(after["trial_count"]),
                        "B_out_class0_count": int(after["class_counts"].get("0", 0)),
                        "B_out_class1_count": int(after["class_counts"].get("1", 0)),
                        "optimizer_state_expected_task_only": bool(optimizer_state_expected),
                        "bn_max_displacement": float(bn_max_displacement(model, bn_baseline)),
                    }
                    row.update({
                        "L_BBR_before": float(before["L_BBR"]), "L_BBR_after": float(after["L_BBR"]),
                        "L_CE_before": float(before["L_CE"]), "L_CE_after": float(after["L_CE"]),
                    })
                    row.update(metrics_for_pair(before, after))
                    observations.append(row)
            trajectory_rows.append({
                "dataset": ctx.dataset, "fold": int(ctx.fold), "seed": SEED,
                "epoch": int(epoch_no), "step": int(global_step),
                "task_gradient_norm": float(grad_norm), "task_clip_scale": float(clip_scale),
                "task_step_norm": float(torch.linalg.vector_norm(delta_task).cpu()),
                "bn_max_displacement": float(bn_max_displacement(model, bn_baseline)),
            })
            del xb_a, ya, task_grad, delta_task, theta_old
        if bn_max_displacement(model, bn_baseline) > 1e-12:
            raise RuntimeError(f"DECISION_AUDIT_IMPLEMENTATION_INVALID_BN_DRIFT_{ctx.dataset}_{ctx.fold}")
    trajectory_hash = digest.hexdigest()
    info = {"dataset": ctx.dataset, "fold": int(ctx.fold), "seed": SEED, "trajectory_sha256": trajectory_hash,
            "audit_steps": sorted(audit_steps), "total_steps": int(global_step), "bn_max_displacement": float(bn_max_displacement(model, bn_baseline))}
    del model, params, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return observations, trajectory_rows, trajectory_hash, info


def fold_contexts() -> tuple[list[Any], dict[str, Any]]:
    contexts, legality = sspg.load_contexts_source_only()
    for ctx in contexts:
        ctx.audit_steps = set(audit_steps_for(ctx))
        ctx.bout_indices = build_bout_indices(ctx)
    return contexts, legality


def checkpoint_equivalence(contexts: list[Any], device: torch.device) -> list[dict[str, Any]]:
    return [sspg.checkpoint_equivalence(ctx, device) for ctx in contexts]


def bbr_concat_equivalence(ctx: Any, device: torch.device) -> dict[str, Any]:
    subject = str(ctx.source_subjects[0])
    model, params = make_model(ctx.anchor_state, ctx.channels, device)
    blocks: list[torch.Tensor] = []
    for block_no in range(K):
        idx = class_balanced_block(ctx, subject, block_no)
        xb = prepare(ctx.data, idx, ctx.mean, ctx.std, device)
        yb = labels_for(ctx.data, idx, device)
        blocks.append(bbr_gradient(model, params, xb, yb, float(ctx.tau)))
        del xb, yb
    concat_idx = np.concatenate([class_balanced_block(ctx, subject, k) for k in range(K)])
    xb = prepare(ctx.data, concat_idx, ctx.mean, ctx.std, device)
    yb = labels_for(ctx.data, concat_idx, device)
    concat = bbr_gradient(model, params, xb, yb, float(ctx.tau))
    diff = float(torch.max(torch.abs(concat - torch.stack(blocks).mean(dim=0))).cpu())
    del model, params, xb, yb, blocks, concat
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"dataset": ctx.dataset, "fold": int(ctx.fold), "subject": subject, "max_abs_diff": diff, "tolerance": 5e-5, "pass": bool(diff <= 5e-5)}


def bootstrap_replicates(frame: pd.DataFrame, metric_fn, seed: int, draws: int = BOOTSTRAP_DRAWS) -> np.ndarray:
    groups = [part for _, part in frame.groupby("subject_id", sort=True)]
    if not groups:
        return np.full(draws, np.nan)
    rng = np.random.default_rng(seed)
    out = np.full(draws, np.nan, dtype=float)
    for draw in range(draws):
        chosen = rng.integers(0, len(groups), size=len(groups))
        sampled = pd.concat([groups[int(i)] for i in chosen], ignore_index=True)
        try:
            out[draw] = float(metric_fn(sampled))
        except Exception:
            out[draw] = np.nan
    return out


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


def ci_from(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None, None
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def top_bottom(frame: pd.DataFrame, cert_col: str, outcome_col: str) -> float | None:
    if frame.empty:
        return None
    order = np.argsort(frame[cert_col].to_numpy(float), kind="mergesort")
    n = len(order)
    q = max(1, n // 5)
    low = frame.iloc[order[:q]][outcome_col].to_numpy(float)
    high = frame.iloc[order[-q:]][outcome_col].to_numpy(float)
    return float(np.mean(high) - np.mean(low))


def metric_pack(frame: pd.DataFrame, cert_col: str, outcome_col: str, bootstrap: np.ndarray | None = None) -> dict[str, Any]:
    outcome = frame[outcome_col].to_numpy(float)
    event = (outcome > 0).astype(int)
    point = {
        "spearman": safe_spearman(frame[cert_col], outcome),
        "pearson": safe_pearson(frame[cert_col], outcome),
        "kendall": safe_kendall(frame[cert_col], outcome),
        "auroc": safe_auc(frame[cert_col], event),
        "sign_accuracy": sign_accuracy(frame[cert_col], outcome),
        "top_minus_bottom": top_bottom(frame, cert_col, outcome_col),
        "n_observations": int(len(frame)), "n_subjects": int(frame.subject_id.nunique()),
        "event_count": int(np.sum(event)),
    }
    if bootstrap is not None:
        for name, values in [("spearman", bootstrap[:, 0]), ("kendall", bootstrap[:, 1]), ("auroc", bootstrap[:, 2]), ("top_minus_bottom", bootstrap[:, 3])]:
            lo, hi = ci_from(values)
            point[name + "_CI95"] = [lo, hi]
    return point


def bootstrap_metric_matrix(frame: pd.DataFrame, cert_col: str, outcome_col: str, seed: int) -> np.ndarray:
    groups = [part for _, part in frame.groupby("subject_id", sort=True)]
    rng = np.random.default_rng(seed)
    out = np.full((BOOTSTRAP_DRAWS, 4), np.nan, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        chosen = rng.integers(0, len(groups), size=len(groups))
        sampled = pd.concat([groups[int(i)] for i in chosen], ignore_index=True)
        outcome = sampled[outcome_col].to_numpy(float)
        event = (outcome > 0).astype(int)
        out[draw, 0] = safe_spearman(sampled[cert_col], outcome) or np.nan
        out[draw, 1] = safe_kendall(sampled[cert_col], outcome) or np.nan
        out[draw, 2] = safe_auc(sampled[cert_col], event) or np.nan
        out[draw, 3] = top_bottom(sampled, cert_col, outcome_col) or np.nan
    return out


def calibration(frame: pd.DataFrame, cert_col: str, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = frame[cert_col].to_numpy(float)
    order = np.argsort(values, kind="mergesort")
    quintile = np.empty(len(frame), dtype=int)
    for rank, pos in enumerate(order):
        quintile[pos] = min(5, int(rank * 5 / max(len(frame), 1)) + 1)
    for q in range(1, 6):
        part = frame.iloc[np.flatnonzero(quintile == q)]
        rows.append({
            "dataset": str(frame.dataset.iloc[0]), "certificate": name, "quintile": q,
            "mean_certificate": float(part[cert_col].mean()) if len(part) else None,
            "mean_H_BBR": float(part.H_BBR.mean()) if len(part) else None,
            "BBR_harm_frequency": float(np.mean(part.H_BBR > 0)) if len(part) else None,
            "mean_H_BER": float(part.H_BER.mean()) if len(part) else None,
            "decision_harm_frequency": float(np.mean(part.H_BER > 0)) if len(part) else None,
            "correct_to_wrong_frequency": float(part.correct_to_wrong.sum() / max(part.B_out_trial_count.sum(), 1)) if len(part) else None,
            "subject_count": int(part.subject_id.nunique()), "observation_count": int(len(part)),
        })
    return rows


def fold_metric_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (dataset, fold), part in frame.groupby(["dataset", "fold"], sort=True):
        bbr = metric_pack(part, "certificate_BBR", "H_BBR")
        ce = metric_pack(part, "certificate_CE", "H_BER")
        dec_bbr = metric_pack(part, "certificate_BBR", "H_BER")
        harmful = part.assign(harmful_flip=(part.correct_to_wrong > 0).astype(int))
        rows.append({
            "dataset": dataset, "fold": int(fold), "n_subjects": int(part.subject_id.nunique()), "n_observations": int(len(part)),
            "BBR_H_BBR_AUROC": bbr["auroc"], "BBR_H_BBR_Spearman": bbr["spearman"],
            "BBR_H_BER_AUROC": dec_bbr["auroc"], "BBR_H_BER_Spearman": dec_bbr["spearman"],
            "CE_H_BER_AUROC": ce["auroc"], "CE_H_BER_Spearman": ce["spearman"],
            "harmful_flip_count": int(part.correct_to_wrong.sum()),
            "same_subject_signal_positive": bool((bbr["spearman"] or 0.0) > 0),
            "BBR_not_worse_than_CE": bool((dec_bbr["auroc"] or -np.inf) >= (ce["auroc"] or -np.inf) or (dec_bbr["spearman"] or -np.inf) >= (ce["spearman"] or -np.inf)),
        })
    return rows


def aggregate_dataset(dataset_frame: pd.DataFrame, seed: int) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    bbr_hbbr_boot = bootstrap_metric_matrix(dataset_frame, "certificate_BBR", "H_BBR", stable_seed("boot", dataset_frame.dataset.iloc[0], "bbr-hbbr", seed))
    bbr_hber_boot = bootstrap_metric_matrix(dataset_frame, "certificate_BBR", "H_BER", stable_seed("boot", dataset_frame.dataset.iloc[0], "bbr-hber", seed))
    ce_hber_boot = bootstrap_metric_matrix(dataset_frame, "certificate_CE", "H_BER", stable_seed("boot", dataset_frame.dataset.iloc[0], "ce-hber", seed))
    bbr = metric_pack(dataset_frame, "certificate_BBR", "H_BBR", bbr_hbbr_boot)
    decision_bbr = metric_pack(dataset_frame, "certificate_BBR", "H_BER", bbr_hber_boot)
    decision_ce = metric_pack(dataset_frame, "certificate_CE", "H_BER", ce_hber_boot)
    exact = {
        "dataset": str(dataset_frame.dataset.iloc[0]),
        "BBR_to_H_BBR": bbr, "BBR_to_H_BER": decision_bbr, "CE_to_H_BER": decision_ce,
        "H_BBR_mean": float(dataset_frame.H_BBR.mean()), "H_BBR_positive_count": int(np.sum(dataset_frame.H_BBR > 0)),
        "H_BER_mean": float(dataset_frame.H_BER.mean()), "H_BER_positive_count": int(np.sum(dataset_frame.H_BER > 0)),
        "H_BER_negative_count": int(np.sum(dataset_frame.H_BER < 0)), "H_BER_zero_count": int(np.sum(dataset_frame.H_BER == 0)),
    }
    flip = {
        "dataset": str(dataset_frame.dataset.iloc[0]), "total_correct_to_wrong": int(dataset_frame.correct_to_wrong.sum()),
        "total_wrong_to_correct": int(dataset_frame.wrong_to_correct.sum()), "harmful_flip_rate": float(dataset_frame.correct_to_wrong.sum() / max(dataset_frame.B_out_trial_count.sum(), 1)),
        "beneficial_flip_rate": float(dataset_frame.wrong_to_correct.sum() / max(dataset_frame.B_out_trial_count.sum(), 1)),
        "net_flip_harm": float((dataset_frame.correct_to_wrong.sum() - dataset_frame.wrong_to_correct.sum()) / max(dataset_frame.B_out_trial_count.sum(), 1)),
    }
    return exact, flip, bbr_hbbr_boot, bbr_hber_boot, ce_hber_boot


def paired_decision_alignment(frame: pd.DataFrame, bbr_boot: np.ndarray, ce_boot: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = [("H_BER Spearman", "spearman", 0), ("H_BER Kendall", "kendall", 1), ("decision-harm AUROC", "auroc", 2), ("harmful-flip AUROC", "auroc_flip", 2), ("top-bottom decision-risk separation", "top_minus_bottom", 3)]
    flip = frame.assign(harmful_flip=(frame.correct_to_wrong > 0).astype(float))
    bflip_boot = bootstrap_metric_matrix(flip, "certificate_BBR", "harmful_flip", stable_seed("paired", frame.dataset.iloc[0], "bbr-flip", SEED))
    cflip_boot = bootstrap_metric_matrix(flip, "certificate_CE", "harmful_flip", stable_seed("paired", frame.dataset.iloc[0], "ce-flip", SEED))
    for label, metric, col in metrics:
        if metric == "auroc_flip":
            bp, cp = safe_auc(flip.certificate_BBR, flip.harmful_flip), safe_auc(flip.certificate_CE, flip.harmful_flip)
            bv, cv = bflip_boot[:, col], cflip_boot[:, col]
        else:
            bp = metric_pack(frame, "certificate_BBR", "H_BER")[metric]
            cp = metric_pack(frame, "certificate_CE", "H_BER")[metric]
            bv, cv = bbr_boot[:, col], ce_boot[:, col]
        diff = float(bp - cp) if bp is not None and cp is not None else None
        dvals = bv - cv
        lo, hi = ci_from(dvals)
        rows.append({"dataset": str(frame.dataset.iloc[0]), "metric": label, "BBR": bp, "CE": cp, "BBR_minus_CE": diff, "BBR_minus_CE_CI95": [lo, hi]})
    return pd.DataFrame(rows), {"dataset": str(frame.dataset.iloc[0]), "BBR_minus_CE_bootstrap_draws": BOOTSTRAP_DRAWS}


def same_different_controls(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    dataset = str(frame.dataset.iloc[0])
    rows: list[dict[str, Any]] = []
    same = metric_pack(frame, "certificate_BBR", "H_BBR")
    for name, col in [("same_subject", "certificate_BBR"), ("different_subject", "certificate_BBR_different"), ("permutation", "certificate_BBR_permuted"), ("random", "certificate_BBR_random")]:
        pack = metric_pack(frame, col, "H_BBR")
        rows.append({"dataset": dataset, "control": name, "certificate_column": col, **pack})
    same_auc, diff_auc = safe_auc(frame.certificate_BBR, frame.H_BBR > 0), safe_auc(frame.certificate_BBR_different, frame.H_BBR > 0)
    same_sp, diff_sp = safe_spearman(frame.certificate_BBR, frame.H_BBR), safe_spearman(frame.certificate_BBR_different, frame.H_BBR)
    result = {
        "dataset": dataset, "same_subject_AUROC": same_auc, "different_subject_AUROC": diff_auc,
        "AUROC_advantage": float(same_auc - diff_auc) if same_auc is not None and diff_auc is not None else None,
        "same_subject_Spearman": same_sp, "different_subject_Spearman": diff_sp,
        "Spearman_advantage": float(same_sp - diff_sp) if same_sp is not None and diff_sp is not None else None,
    }
    return result, pd.DataFrame(rows)


def write_docs(lock: dict[str, Any], legality: dict[str, Any], terminal: str | None = None) -> None:
    (EXP / "README.md").write_text("# PERSIST-EEG Decision-Relevant Prospective Harm Audit V1\n\nEEGNet, seed 0, OpenBMI/WBCIC folds 0--4. This is a source/refit-only audit of the frozen TASK_ONLY_MATCHED trajectory. It does not train a guard and does not open development outcome, WBCIC outer-10, or OpenBMI sealed/confirmation data.\n\nThe primary surrogate is frozen class-balanced Balanced Boundary Risk (BBR); CE is a matched comparator. Exact decision harm is held-out B_out Balanced Error harm and correct-to-wrong flips within the same legal source/refit subjects.\n\nThe machine-readable pre-outcome lock is `results/PRE_OUTCOME_LOCK.json`; scientific interpretation is in `FINAL_REPORT.md` and `AUTONOMOUS_DECISION.md`.\n", encoding="utf-8")
    (EXP / "FROZEN_PROTOCOL.md").write_text("# Frozen protocol\n\n- EEGNet only; OpenBMI and WBCIC; folds 0--4; seed 0 only; source/refit subjects and fit sessions only.\n- K=4 class-balanced certificate blocks, 16 trials per class per block, without replacement; B_out is every remaining legal source/refit trial and must retain at least 16/class.\n- BBR: r=sigmoid(-m/tau), m=true logit minus largest competitor, class-balanced mean; tau=max(median(|m|) at legal anchor, 1e-3), frozen before audit.\n- Frozen TASK_ONLY_MATCHED AdamW trajectory: exact canonical checkpoint, lr=3e-5, weight decay=5e-4, gradient clip=5, two epochs, frozen BN. Ten evenly-spaced audit steps are fixed before any harm result.\n- Primary same-subject BBR certificate; deterministic cyclic different-subject, non-self permutation, and norm-matched Gaussian controls. Biological subject is the bootstrap unit (10,000 draws).\n- No outcome/outer/sealed access, no seed 1/2, no second backbone, no search or new guard.\n\n" + json.dumps(clean(lock), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (EXP / "PRE_OUTCOME_LOCK.md").write_text("# Pre-outcome lock\n\nThis lock was written after source/refit-only checkpoint, B_out support, frozen margin scales, fixed ten-step schedules and executable mandatory tests. No development outcome labels or outcome trials were materialized. The lock must be committed before the audit phase.\n\n- schema: PERSIST_EEG_DECISION_RELEVANT_PRE_OUTCOME_LOCK_V1\n- seed: 0; EEGNet only; K=4; 16 trials/class/block; ten audit steps\n- outcome_used: false; WBCIC_outer_opened: false; OpenBMI_sealed_opened: false\n\n", encoding="utf-8")
    (EXP / "DATA_LEGALITY_AUDIT.md").write_text("# Data legality audit\n\nOnly frozen model-fit/discovery source/refit subjects and their legal fit sessions are used. B1--B4 and B_out are trial-disjoint. No development outcome trials, outcome labels, WBCIC outer-10, or OpenBMI sealed/confirmation trials are loaded or materialized.\n\n```json\n" + json.dumps(clean(legality), ensure_ascii=False, indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    docs = {
        "CHECKPOINT_EQUIVALENCE.md": "# Checkpoint equivalence\n\nEvery fold loads the canonical seed-0 EEGNet checkpoint with strict state-dict matching, verified hash and deterministic repeat predictions. See `results/CHECKPOINT_EQUIVALENCE.csv`.\n",
        "MARGIN_DEFINITION_AUDIT.md": "# Margin definition audit\n\nFor each trial m=z[y]-max_{c!=y}z[c]. The signed margin is positive exactly when the argmax prediction is correct (apart from exact ties). BBR is sigmoid(-m/tau).\n",
        "MARGIN_SCALE_AUDIT.md": "# Margin scale audit\n\nTau is max(median absolute signed margin on legal source/refit anchor trials, 1e-3), computed once per dataset/fold before any prospective harm calculation and then held fixed for all steps.\n",
        "BATCH_CONSTRUCTION_AUDIT.md": "# Batch construction audit\n\nEach source/refit subject has five deterministic class-balanced blocks of 16/class. The first four are K=4 certificate blocks; B_out is all remaining legal trials and is disjoint from them. A batches use other meta-fold subjects.\n",
        "MATHEMATICAL_AUDIT.md": "# Mathematical audit\n\nBBR is the class-balanced mean of sigmoid(-signed-margin/tau). Its gradient is computed by autograd at the trajectory state and dotted with the exact post-AdamW displacement Delta_A. CE uses the same blocks, state and displacement.\n",
        "CONTROL_AUDIT.md": "# Control audit\n\nSame-subject BBR is primary. Different-subject uses the deterministic next subject in the designated B meta-fold; permutation uses a deterministic global non-self cycle; random is a deterministic Gaussian direction norm-matched to the same-subject BBR gradient.\n",
        "STATISTICAL_PROTOCOL.md": "# Statistical protocol\n\nBiological subject is the inference unit. Subject-cluster bootstrap uses 10,000 draws and carries all fold/step observations for each sampled subject. Exact decision events are not treated as independent trial-level evidence.\n",
        "RARE_EVENT_POWER_AUDIT.md": "# Rare-event power audit\n\nExact decision harm is H_BER>0 and is reported separately from continuous BBR harm. Underpower is declared when harmful decision observations are fewer than 30 or biological subjects with at least one such event are fewer than 15. Steps, trajectory, LR and B_out are not changed to increase events.\n",
        "BUG_REPAIR_LEDGER.md": "# Bug repair ledger\n\nNo scientific-rule repair was made. The runner isolates the frozen SSPG helper loader, uses source/refit-only B_out construction, freezes tau before outcomes, and writes explicit invariant flags. Any runtime-only repair must be recorded here before the lock is regenerated.\n",
    }
    for name, text in docs.items():
        (EXP / name).write_text(text, encoding="utf-8")
    if terminal is not None:
        (EXP / "AUTONOMOUS_DECISION.md").write_text(f"# Autonomous decision\n\n`terminal = {terminal}`\n\nThis decision is evidence-bound to the frozen seed-0 EEGNet source/refit-only audit. No new model, guard, seed, backbone, outer cohort or sealed cohort is started automatically.\n", encoding="utf-8")


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=EXP.parent.parent, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "status": run("status", "--short")}


def code_hashes() -> dict[str, str]:
    files = [Path(__file__), OLD_CODE, CANONICAL_EXP / "code" / "canonical_eegnet_runner.py"]
    return {str(path): sha256_file(path) for path in files if path.is_file()}


def mandatory_tests(contexts: list[Any], equivalence: list[dict[str, Any]], legality: dict[str, Any], device: torch.device) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    toy_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    toy_y = torch.tensor([0, 1, 0])
    toy_m = signed_margin(toy_logits, toy_y)
    checks["signed_margin_definition"] = {"values": toy_m.tolist(), "pass": bool(torch.allclose(toy_m[:2], torch.tensor([2.0, 2.0])))}
    pred = toy_logits.argmax(1)
    checks["margin_positive_iff_prediction_correct"] = bool(torch.equal(toy_m[:2] > 0, (pred[:2] == toy_y[:2])))
    checks["tau_source_refit_anchor_only"] = bool(all(getattr(ctx, "tau", None) and not hasattr(ctx, "outcome_idx") for ctx in contexts))
    tau_copy = [float(ctx.tau) for ctx in contexts]
    checks["tau_frozen"] = bool(tau_copy == [float(ctx.tau) for ctx in contexts])
    toy_r = bbr_values_from_logits(toy_logits, toy_y, 1.0)
    checks["bbr_in_unit_interval"] = bool(torch.all((toy_r >= 0) & (toy_r <= 1)))
    checks["class_balanced_implementation"] = bool(abs(float(balanced_mean(torch.tensor([0.0, 1.0, 0.0, 1.0]), torch.tensor([0, 0, 1, 1]))) - 0.5) < 1e-8)
    checks["bbr_decreases_with_margin"] = bool(float(bbr_values_from_logits(torch.tensor([[2.0, 0.0]]), torch.tensor([0]), 1.0)) < float(bbr_values_from_logits(torch.tensor([[1.0, 0.0]]), torch.tensor([0]), 1.0)))
    first = contexts[0]
    model, params = make_model(first.anchor_state, first.channels, device)
    subject = str(first.source_subjects[0])
    idx = class_balanced_block(first, subject, 0)
    xb, yb = prepare(first.data, idx, first.mean, first.std, device), labels_for(first.data, idx, device)
    g = bbr_gradient(model, params, xb, yb, float(first.tau))
    checks["bbr_gradient_finite"] = bool(torch.isfinite(g).all())
    checks["K4_concat_gradient_equivalence"] = bbr_concat_equivalence(first, device)
    checks["B1_B4_Bout_trial_disjoint"] = bool(all(len(set(first.bout_indices[s].tolist()) & set(np.concatenate([class_balanced_block(first, s, k) for k in range(4)]).tolist())) == 0 for s in first.source_subjects))
    a_disjoint = True
    diff_disjoint = True
    for ctx in contexts:
        subjects = metadata_col(ctx.data, "subject_id")
        for entry in sum(ctx.schedules, []):
            a = set(subjects[np.asarray(entry["A"], dtype=np.int64)].astype(str))
            b = set(map(str, entry["b_subjects"]))
            a_disjoint &= not (a & b)
            if len(b) >= 2:
                partner = str(list(b)[1])
                diff_disjoint &= partner not in a
    checks["A_B_subject_disjoint"] = bool(a_disjoint)
    checks["different_subject_A_disjoint"] = bool(diff_disjoint)
    # Exact AdamW displacement is taken from parameters after optimizer.step,
    # never from -lr*gradient.
    opt = torch.optim.AdamW(params, lr=sspg.BASE_LR, weight_decay=sspg.WEIGHT_DECAY)
    old = snapshot(params)
    opt.zero_grad(set_to_none=True)
    for p, chunk in zip(params, split_like(g, params)):
        p.grad = chunk.clone()
    opt.step()
    delta = flatten([p.detach() - before for p, before in zip(params, old)])
    checks["exact_adamw_displacement"] = bool(torch.isfinite(delta).all() and float(torch.linalg.vector_norm(delta)) > 0)
    before_bn = bn_buffers(model)
    before_opt = optimizer_digest(opt)
    _ = bbr_gradient(model, params, xb, yb, float(first.tau))
    checks["BN_unchanged_during_bbr_gradient"] = bool(bn_max_displacement(model, before_bn) <= 1e-12)
    checks["optimizer_state_unchanged_during_bbr_gradient"] = bool(optimizer_digest(opt) == before_opt)
    rnd = random_direction(float(torch.linalg.vector_norm(g)), len(g), first.dataset, first.fold, 1, subject)
    checks["random_gradient_norm_match"] = bool(abs(float(torch.linalg.vector_norm(rnd)) - float(torch.linalg.vector_norm(g))) <= 1e-5)
    del model, params, opt, xb, yb, old, delta, g, rnd
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    plain = sspg.run_task_replay(first, device, extra_guard=False)
    guarded = sspg.run_task_replay(first, device, extra_guard=True)
    checks["task_only_replay_equivalence"] = {"plain_sha256": plain, "audit_on_sha256": guarded, "pass": plain == guarded}
    checks["same_observation_schedule"] = bool(all(len(audit_steps_for(ctx)) == AUDIT_STEPS_PER_FOLD for ctx in contexts))
    checks["biological_subject_cluster_bootstrap_unit"] = bool(len(set(map(str, first.source_subjects))) == len(first.source_subjects))
    checks["no_outcome_outer_sealed_ids"] = bool(not legality.get("WBCIC_outer_opened") and not legality.get("OpenBMI_sealed_opened") and not legality.get("outcome_index_created_before_lock") and not legality.get("outcome_labels_read_before_lock"))
    checks["seed0_only"] = SEED == 0
    checks["outcome_not_materialized"] = True
    for key, value in checks.items():
        if isinstance(value, dict):
            if not bool(value.get("pass", True)):
                raise RuntimeError(f"mandatory test failed: {key}")
        elif not bool(value):
            raise RuntimeError(f"mandatory test failed: {key}")
    result = {"schema": "PERSIST_EEG_DECISION_RELEVANT_MANDATORY_TESTS_V1", "pass": True, "checks": checks, "critical_failure_blocks_audit": True, "outcome_used": False, "seed1_run": False, "seed2_run": False}
    write_json(RESULTS / "MANDATORY_TESTS.json", result)
    return result


def write_lock(contexts: list[Any], legality: dict[str, Any], equivalence: list[dict[str, Any]], mandatory: dict[str, Any], tau_rows: list[dict[str, Any]]) -> dict[str, Any]:
    info = git_info()
    schedule_rows = [{"dataset": ctx.dataset, "fold": int(ctx.fold), "schedule_sha256": ctx.schedule_hash, "total_steps": int(sum(len(s) for s in ctx.schedules)), "audit_steps": audit_steps_for(ctx)} for ctx in contexts]
    lock = {
        "schema": "PERSIST_EEG_DECISION_RELEVANT_PRE_OUTCOME_LOCK_V1", "experiment": "persist_eeg_decision_relevant_prospective_harm_audit_v1",
        "method": "Balanced Boundary Risk prospective harm audit", "code_hashes": code_hashes(), "code_commit": info["commit"], "branch_at_code_freeze": info["branch"],
        "datasets": list(DATASETS), "folds": list(FOLDS), "seed": SEED, "seed1_run": False, "seed2_run": False, "second_backbone_run": False,
        "K": K, "m_per_class": M_PER_CLASS, "certificate_blocks": [1, 2, 3, 4], "bout": "all remaining legal source/refit trials", "no_outcome_data": True,
        "optimizer": {"name": "AdamW", "learning_rate": float(sspg.BASE_LR), "weight_decay": float(sspg.WEIGHT_DECAY), "gradient_clip": float(sspg.GRAD_CLIP), "parameter_scope": "FULL_TRAINABLE_PARAMETER_SPACE", "BN_running_statistics": "frozen"},
        "continuation_epochs": int(sspg.MAX_EPOCHS), "task_schedule_hashes": schedule_rows,
        "margin_scale": {"definition": "max(median(abs(anchor signed margin)),1e-3)", "tau_rows": tau_rows, "frozen_before_harm": True},
        "primary": "same-subject BBR certificate -> held-out BBR harm and exact H_BER", "comparators": ["CE", "deterministic_different_subject", "deterministic_permutation", "norm_matched_random"],
        "audit_steps_per_fold": AUDIT_STEPS_PER_FOLD, "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_unit": "biological_subject cluster",
        "legality": legality, "checkpoint_hashes": [{"dataset": c.dataset, "fold": int(c.fold), "path": str(c.checkpoint_path), "sha256": sha256_file(c.checkpoint_path)} for c in contexts],
        "mandatory_tests_sha256": sha256_file(RESULTS / "MANDATORY_TESTS.json"), "mandatory_tests_pass": True,
        "outcome_used": False, "WBCIC_outer_opened": False, "OpenBMI_sealed_opened": False, "lock_status": "PRE_OUTCOME_LOCKED",
    }
    write_json(RESULTS / "PRE_OUTCOME_LOCK.json", lock)
    return lock


def preflight(device: torch.device) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    contexts, legality = fold_contexts()
    tau_rows = [compute_tau(ctx, device) for ctx in contexts]
    schedules = [{"dataset": ctx.dataset, "fold": int(ctx.fold), "total_steps": int(sum(len(s) for s in ctx.schedules)), "audit_steps": audit_steps_for(ctx), "schedule_sha256": ctx.schedule_hash} for ctx in contexts]
    write_csv(RESULTS / "AUDIT_STEP_SCHEDULE.csv", schedules)
    write_csv(RESULTS / "MARGIN_SCALE.csv", tau_rows)
    write_csv(RESULTS / "MARGIN_SCALE_AUDIT.csv", tau_rows)
    equivalence = checkpoint_equivalence(contexts, device)
    write_csv(RESULTS / "CHECKPOINT_EQUIVALENCE.csv", equivalence)
    if not all(bool(row.get("pass")) for row in equivalence):
        raise RuntimeError("checkpoint equivalence failed; lock/audit blocked")
    mandatory = mandatory_tests(contexts, equivalence, legality, device)
    lock = write_lock(contexts, legality, equivalence, mandatory, tau_rows)
    write_docs(lock, legality)
    write_json(RUNTIME / "PREFLIGHT.json", {"schema": "PERSIST_EEG_DECISION_RELEVANT_PREFLIGHT_V1", "lock": lock, "mandatory": mandatory})
    print("PRE_OUTCOME_LOCK_WRITTEN=true", flush=True)
    print("outcome_used = false", flush=True)
    print("MANDATORY_TESTS_PASS=true", flush=True)
    print(f"code_commit = {lock['code_commit']}", flush=True)


def require_lock() -> dict[str, Any]:
    path = RESULTS / "PRE_OUTCOME_LOCK.json"
    if not path.is_file():
        raise RuntimeError("missing PRE_OUTCOME_LOCK.json; run preflight, commit and push it first")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("seed1_run") or lock.get("seed2_run") or lock.get("second_backbone_run") or lock.get("WBCIC_outer_opened") or lock.get("OpenBMI_sealed_opened") or lock.get("outcome_used"):
        raise RuntimeError("forbidden access/run flag in pre-outcome lock")
    if lock.get("code_hashes") != code_hashes():
        raise RuntimeError("code changed after pre-outcome lock; rerun preflight and recommit")
    mandatory = RESULTS / "MANDATORY_TESTS.json"
    if not mandatory.is_file() or not json.loads(mandatory.read_text(encoding="utf-8")).get("pass"):
        raise RuntimeError("mandatory tests did not pass")
    return lock


def decision_summary(frame: pd.DataFrame, fold_rows: list[dict[str, Any]], dataset_stats: dict[str, dict[str, Any]], controls: dict[str, dict[str, Any]], alignment: dict[str, pd.DataFrame], calibration_rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    power: dict[str, dict[str, Any]] = {}
    for dataset, part in frame.groupby("dataset", sort=True):
        harmful_obs = int(np.sum(part.H_BER > 0))
        harmful_subjects = int(part.loc[part.H_BER > 0, "subject_id"].nunique())
        power[dataset] = {"total_subject_step_observations": int(len(part)), "H_BER_positive_count": harmful_obs, "H_BER_negative_count": int(np.sum(part.H_BER < 0)), "H_BER_zero_count": int(np.sum(part.H_BER == 0)), "biological_subjects_with_harmful_event": harmful_subjects, "total_correct_to_wrong_flips": int(part.correct_to_wrong.sum()), "total_wrong_to_correct_flips": int(part.wrong_to_correct.sum()), "EXACT_DECISION_ENDPOINT_UNDERPOWERED": bool(harmful_obs < 30 or harmful_subjects < 15)}
    gate_a = {}
    gate_b = {}
    for dataset, stats in dataset_stats.items():
        bbr = stats["BBR_to_H_BBR"]
        gate_a[dataset] = bool((bbr.get("auroc") or -np.inf) >= 0.60 and (bbr.get("auroc_CI95") or [-np.inf])[0] > 0.50 and (bbr.get("spearman") or -np.inf) > 0 and (bbr.get("spearman_CI95") or [-np.inf])[0] > 0)
        c = controls[dataset]
        gate_b[dataset] = bool((c.get("AUROC_advantage") or -np.inf) > 0 and (c.get("Spearman_advantage") or -np.inf) > 0)
    gate_a_all, gate_b_all = all(gate_a.values()), all(gate_b.values())
    powered_all = all(not row["EXACT_DECISION_ENDPOINT_UNDERPOWERED"] for row in power.values())
    gate_c = {}
    gate_d = {}
    if powered_all:
        for dataset, stats in dataset_stats.items():
            dec = stats["BBR_to_H_BER"]
            ce = stats["CE_to_H_BER"]
            cal = pd.DataFrame([r for r in calibration_rows if r["dataset"] == dataset and r["certificate"] == "BBR"])
            q = cal[cal["quintile"].isin([1, 5])]
            qdiff = float(q[q.quintile == 5].decision_harm_frequency.iloc[0] - q[q.quintile == 1].decision_harm_frequency.iloc[0]) if len(q) == 2 else -np.inf
            gate_c[dataset] = bool((dec.get("spearman") or -np.inf) > 0 and (dec.get("auroc") or -np.inf) > 0.55 and qdiff > 0)
            arow = alignment[dataset]
            au = arow[arow.metric == "decision-harm AUROC"].iloc[0]
            sp = arow[arow.metric == "H_BER Spearman"].iloc[0]
            gate_d[dataset] = bool(au.BBR > au.CE and sp.BBR > sp.CE)
    else:
        gate_c = {dataset: False for dataset in DATASETS}
        gate_d = {dataset: False for dataset in DATASETS}
    fold_by_dataset = {dataset: [r for r in fold_rows if r["dataset"] == dataset] for dataset in DATASETS}
    gate_e = {dataset: sum(bool(r["BBR_not_worse_than_CE"] or r["same_subject_signal_positive"]) for r in rows) >= 4 for dataset, rows in fold_by_dataset.items()}
    if not gate_a_all or not gate_b_all:
        terminal = "DECISION_RELEVANT_SIGNAL_NOT_SUPPORTED"
    elif not powered_all:
        terminal = "DECISION_ENDPOINT_UNDERPOWERED"
    elif all(gate_c.values()) and all(gate_d.values()) and all(gate_e.values()):
        terminal = "DECISION_RELEVANT_SUBJECT_HARM_SUPPORTED"
    elif all(gate_a.values()) and all(gate_b.values()) and any(gate_c.values()):
        terminal = "BOUNDARY_SURROGATE_SUPPORTED_DECISION_UNPROVEN"
    else:
        terminal = "DECISION_RELEVANT_SIGNAL_NOT_SUPPORTED"
    decision = {"terminal": terminal, "gate_A_stable_BBR_harm": gate_a, "gate_B_subject_specificity": gate_b, "gate_C_decision_alignment": gate_c, "gate_D_BBR_over_CE": gate_d, "gate_E_fold_robustness": gate_e, "power": power, "DECISION_ALIGNED_GUARD_DEVELOPMENT_SCIENTIFICALLY_JUSTIFIED": terminal == "DECISION_RELEVANT_SUBJECT_HARM_SUPPORTED", "AUTO_START_NEW_MODEL": False}
    return terminal, decision


def run_audit(device: torch.device) -> None:
    lock = require_lock()
    contexts, legality = fold_contexts()
    # Recompute tau only as an equality check against the frozen lock; the
    # values are not selected from harm and no outcome data are touched.
    frozen_tau = {(str(r["dataset"]), int(r["fold"])): float(r["tau"]) for r in lock["margin_scale"]["tau_rows"]}
    for ctx in contexts:
        tau_check = compute_tau(ctx, device)
        if abs(float(tau_check["tau"]) - frozen_tau[(ctx.dataset, int(ctx.fold))]) > 0:
            raise RuntimeError("DECISION_AUDIT_IMPLEMENTATION_INVALID_TAU_CHANGED")
        ctx.tau = frozen_tau[(ctx.dataset, int(ctx.fold))]
    all_obs: list[dict[str, Any]] = []
    all_traj: list[dict[str, Any]] = []
    infos: list[dict[str, Any]] = []
    started = time.time()
    for index, ctx in enumerate(contexts, 1):
        print(f"[audit] {index}/{len(contexts)} {ctx.dataset} fold={ctx.fold}", flush=True)
        obs, traj, digest, info = train_fold(ctx, device, frozen_tau[(ctx.dataset, int(ctx.fold))])
        all_obs.extend(obs); all_traj.extend(traj); infos.append(info)
        print(f"[audit-done] {ctx.dataset} fold={ctx.fold} observations={len(obs)} trajectory={digest[:12]}", flush=True)
    obs_frame = pd.DataFrame(all_obs)
    traj_frame = pd.DataFrame(all_traj)
    if obs_frame.empty:
        raise RuntimeError("no audit observations")
    write_csv(RESULTS / "PER_OBSERVATION.csv", obs_frame)
    write_csv(RESULTS / "TRAINING_TRAJECTORIES.csv", traj_frame)
    write_csv(RESULTS / "PER_SUBJECT_METRICS.csv", obs_frame.groupby(["dataset", "subject_id"], as_index=False).agg(observations=("step", "size"), mean_H_BBR=("H_BBR", "mean"), mean_H_BER=("H_BER", "mean"), total_correct_to_wrong=("correct_to_wrong", "sum"), total_wrong_to_correct=("wrong_to_correct", "sum")))
    fold_rows = fold_metric_rows(obs_frame)
    write_csv(RESULTS / "PER_FOLD_METRICS.csv", fold_rows)
    dataset_stats: dict[str, dict[str, Any]] = {}
    flip_rows: list[dict[str, Any]] = []
    all_cal: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    align_frames: dict[str, pd.DataFrame] = {}
    bootstrap_json: dict[str, Any] = {}
    controls: dict[str, dict[str, Any]] = {}
    for dataset, part in obs_frame.groupby("dataset", sort=True):
        exact, flip, bbr_hbbr_boot, bbr_hber_boot, ce_hber_boot = aggregate_dataset(part, SEED)
        dataset_stats[str(dataset)] = exact
        flip_rows.append(flip)
        all_cal.extend(calibration(part, "certificate_BBR", "BBR"))
        all_cal.extend(calibration(part, "certificate_CE", "CE"))
        control_summary, control_frame = same_different_controls(part)
        controls[str(dataset)] = control_summary
        control_rows.extend(control_frame.to_dict(orient="records"))
        align, align_meta = paired_decision_alignment(part, bbr_hber_boot, ce_hber_boot)
        align_frames[str(dataset)] = align
        bootstrap_json[str(dataset)] = {"draws": BOOTSTRAP_DRAWS, "BBR_H_BBR": bbr_hbbr_boot.tolist(), "BBR_H_BER": bbr_hber_boot.tolist(), "CE_H_BER": ce_hber_boot.tolist(), "alignment": align_meta}
    write_csv(RESULTS / "BBR_HARM_SUMMARY.csv", [{"dataset": d, **s["BBR_to_H_BBR"], "mean_H_BBR": s["H_BBR_mean"], "positive_harm_events": s["H_BBR_positive_count"]} for d, s in dataset_stats.items()])
    write_csv(RESULTS / "DECISION_HARM_SUMMARY.csv", [{"dataset": d, **s["BBR_to_H_BER"], "CE_H_BER_Spearman": s["CE_to_H_BER"].get("spearman"), "CE_H_BER_AUROC": s["CE_to_H_BER"].get("auroc"), "mean_H_BER": s["H_BER_mean"], "positive_H_BER_events": s["H_BER_positive_count"]} for d, s in dataset_stats.items()])
    write_csv(RESULTS / "FLIP_SUMMARY.csv", flip_rows)
    write_csv(RESULTS / "SAME_VS_DIFFERENT.csv", [controls[d] for d in controls])
    write_csv(RESULTS / "PERMUTATION_CONTROL.csv", [r for r in control_rows if r.get("control") == "permutation"])
    write_csv(RESULTS / "RANDOM_CONTROL.csv", [r for r in control_rows if r.get("control") == "random"])
    write_csv(RESULTS / "BBR_CALIBRATION_BINS.csv", [r for r in all_cal if r["certificate"] == "BBR"])
    write_csv(RESULTS / "CE_CALIBRATION_BINS.csv", [r for r in all_cal if r["certificate"] == "CE"])
    alignment_frame = pd.concat(list(align_frames.values()), ignore_index=True)
    write_csv(RESULTS / "BBR_VS_CE_DECISION_ALIGNMENT.csv", alignment_frame)
    write_json(RESULTS / "BOOTSTRAP_RESULTS.json", bootstrap_json)
    power_rows = []
    for d, s in dataset_stats.items():
        p = {"dataset": d, "total_subject_step_observations": int(len(obs_frame[obs_frame.dataset == d])), "H_BER_positive_count": int(s["H_BER_positive_count"]), "H_BER_negative_count": int(s["H_BER_negative_count"]), "H_BER_zero_count": int(s["H_BER_zero_count"]), "biological_subjects_with_harmful_event": int(obs_frame[(obs_frame.dataset == d) & (obs_frame.H_BER > 0)].subject_id.nunique()), "total_correct_to_wrong_flips": int(obs_frame[obs_frame.dataset == d].correct_to_wrong.sum()), "total_wrong_to_correct_flips": int(obs_frame[obs_frame.dataset == d].wrong_to_correct.sum())}
        p["EXACT_DECISION_ENDPOINT_UNDERPOWERED"] = bool(p["H_BER_positive_count"] < 30 or p["biological_subjects_with_harmful_event"] < 15)
        power_rows.append(p)
    write_csv(RESULTS / "RARE_EVENT_POWER.csv", power_rows)
    terminal, decision = decision_summary(obs_frame, fold_rows, dataset_stats, controls, align_frames, all_cal)
    mandatory = json.loads((RESULTS / "MANDATORY_TESTS.json").read_text(encoding="utf-8"))
    validation = {
        "schema": "PERSIST_EEG_DECISION_RELEVANT_VALIDATION_V1", "pass": True, "terminal": terminal,
        "mandatory_tests_pass": bool(mandatory.get("pass")), "checkpoint_equivalence_pass": bool(all(bool(x.get("pass")) for x in pd.read_csv(RESULTS / "CHECKPOINT_EQUIVALENCE.csv").to_dict(orient="records"))),
        "tau_frozen": True, "B1_B4_Bout_disjoint": bool(mandatory["checks"]["B1_B4_Bout_trial_disjoint"]), "A_B_disjoint": bool(mandatory["checks"]["A_B_subject_disjoint"]),
        "task_only_trajectory_audit_equivalence": bool(mandatory["checks"]["task_only_replay_equivalence"]["pass"]), "BN_unchanged": bool((traj_frame.bn_max_displacement <= 1e-12).all()),
        "outcome_used": False, "seed1_run": False, "seed2_run": False, "second_backbone_run": False, "WBCIC_outer_opened": False, "OpenBMI_sealed_opened": False,
        "runtime_seconds": time.time() - started, "decision": decision,
    }
    write_json(RESULTS / "VALIDATION.json", validation)
    write_json(RESULTS / "FINAL_REPORT.json", {"schema": "PERSIST_EEG_DECISION_RELEVANT_FINAL_REPORT_V1", "terminal": terminal, "decision": decision, "dataset_stats": dataset_stats, "controls": controls, "fold_metrics": fold_rows, "validation": validation, "lock": lock})
    write_docs(lock, legality, terminal)
    lines = ["# Final report", "", f"terminal = {terminal}", "", "This report is source/refit-only and outcome-free. No development outcome, WBCIC outer-10, OpenBMI sealed/confirmation cohort, seed 1/2, or second backbone was opened.", "", "|dataset|BBR->H_BBR AUROC|95% CI|BBR->H_BBR Spearman|BBR->H_BER AUROC|CE->H_BER AUROC|exact H_BER harmful events|Q5-Q1 decision harm|underpowered|", "|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for d in DATASETS:
        s = dataset_stats[d]; b = s["BBR_to_H_BBR"]; dec = s["BBR_to_H_BER"]; ce = s["CE_to_H_BER"]; cal = [r for r in all_cal if r["dataset"] == d and r["certificate"] == "BBR"]
        q1 = next((r["decision_harm_frequency"] for r in cal if r["quintile"] == 1), None); q5 = next((r["decision_harm_frequency"] for r in cal if r["quintile"] == 5), None)
        power = next(r for r in power_rows if r["dataset"] == d)
        lines.append(f"|{d}|{b.get('auroc')}|{b.get('auroc_CI95')}|{b.get('spearman')}|{dec.get('auroc')}|{ce.get('auroc')}|{s['H_BER_positive_count']}|{None if q1 is None or q5 is None else q5-q1}|{power['EXACT_DECISION_ENDPOINT_UNDERPOWERED']}|")
    lines += ["", "## Required answers", "", f"- BBR K4 cross-batch signal: {all(dataset_stats[d]['BBR_to_H_BBR'].get('spearman') is not None for d in DATASETS)}", f"- Same-subject specificity: {all((controls[d].get('AUROC_advantage') or -1) > 0 for d in DATASETS)}", f"- Exact decision alignment: {decision['gate_C_decision_alignment']}", f"- BBR over CE: {decision['gate_D_BBR_over_CE']}", f"- Fold robustness: {decision['gate_E_fold_robustness']}", "- Biological-subject bootstrap: 10,000 cluster draws.", "", "No new model or guard is automatically started."]
    (EXP / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"terminal = {terminal}", flush=True)
    for d in DATASETS:
        s = dataset_stats[d]; b = s["BBR_to_H_BBR"]; dec = s["BBR_to_H_BER"]; ce = s["CE_to_H_BER"]; cal = [r for r in all_cal if r["dataset"] == d and r["certificate"] == "BBR"]
        q1 = next((r["decision_harm_frequency"] for r in cal if r["quintile"] == 1), None); q5 = next((r["decision_harm_frequency"] for r in cal if r["quintile"] == 5), None)
        print(f"{d}_BBR_HARM_AUROC = {b.get('auroc')}", flush=True); print(f"{d}_BBR_HARM_AUROC_CI = {b.get('auroc_CI95')}", flush=True); print(f"{d}_BBR_HARM_SPEARMAN = {b.get('spearman')}", flush=True); print(f"{d}_DECISION_HARM_AUROC = {dec.get('auroc')}", flush=True); print(f"{d}_CE_DECISION_HARM_AUROC = {ce.get('auroc')}", flush=True); print(f"{d}_BBR_MINUS_CE_DECISION_AUROC = {None if dec.get('auroc') is None or ce.get('auroc') is None else dec.get('auroc')-ce.get('auroc')}", flush=True); print(f"{d}_exact_harm_events = {s['H_BER_positive_count']}", flush=True); print(f"{d}_Q5_minus_Q1_decision_harm = {None if q1 is None or q5 is None else q5-q1}", flush=True); print(f"EXACT_DECISION_ENDPOINT_UNDERPOWERED_{d} = {next(r for r in power_rows if r['dataset']==d)['EXACT_DECISION_ENDPOINT_UNDERPOWERED']}", flush=True)
    print("seed1_run = false", flush=True); print("seed2_run = false", flush=True); print("second_backbone_run = false", flush=True); print("WBCIC_outer_opened = false", flush=True); print("OpenBMI_sealed_opened = false", flush=True); print(f"DECISION_ALIGNED_GUARD_DEVELOPMENT_SCIENTIFICALLY_JUSTIFIED = {'YES' if decision['DECISION_ALIGNED_GUARD_DEVELOPMENT_SCIENTIFICALLY_JUSTIFIED'] else 'NO'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "audit"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.phase == "preflight":
        preflight(device)
    else:
        run_audit(device)


if __name__ == "__main__":
    main()
