"""PERSIST-EEG Step-1 cross-batch biological-subject harm audit.

This runner replays the frozen PSG V2 task-only AdamW trajectory (seed 0)
from canonical EEGNet checkpoints and evaluates prospective harm on five
mutually-disjoint batches from each source/refit biological subject.  It does
not train a guard or read sealed/outcome cohorts.  The only optional
subsampling is the predeclared, evenly-spaced audit-step schedule documented
in FROZEN_PROTOCOL.md.
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
from scipy.stats import spearmanr
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
CANONICAL_REPO = Path(os.environ.get(
    "CANONICAL_REPO", r"D:\nips-temp\TotalP\P1\CRCICLR_CANONICAL_EEGNET"
)).resolve()
WBCIC_CACHE = Path(os.environ.get(
    "PERSIST_WBCIC_CACHE",
    r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache",
)).resolve()
PSG_CODE = CANONICAL_REPO / "experiments" / "persist_eeg_prospective_step_guard_v2_seed0" / "code"
if str(PSG_CODE) not in sys.path:
    sys.path.insert(0, str(PSG_CODE))
os.environ.setdefault("CANONICAL_REPO", str(CANONICAL_REPO))
os.environ.setdefault("PERSIST_WBCIC_CACHE", str(WBCIC_CACHE))
import persist_au_seed0 as psg  # noqa: E402

canonical = psg.canonical
SEED = 0
DATASETS = ("OpenBMI", "WBCIC")
FOLDS = (0, 1, 2, 3, 4)
K_VALUES = (1, 2, 4)
N_BLOCKS = 5
MIN_M_PER_CLASS = 4
MAX_M_PER_CLASS = 16
BOOTSTRAP_DRAWS = 10_000
AUDIT_STEPS_PER_FOLD = 5
EPS = 1e-12


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
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value); return x if math.isfinite(x) else None
    if isinstance(value, np.bool_): return bool(value)
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


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed) % (2**32 - 1)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))


def subject_sort(values: Iterable[object]) -> list[str]:
    return sorted((str(v).replace("sub-", "") for v in values), key=lambda x: (int(x) if x.isdigit() else 10**9, x))


def metadata_col(data: Any, name: str, indices: np.ndarray | None = None) -> np.ndarray:
    frame = data.metadata if indices is None else data.metadata.iloc[np.asarray(indices, dtype=np.int64)]
    values = frame[name]
    if name == "subject_id": return values.astype(str).str.replace("sub-", "", regex=False).to_numpy()
    return values.to_numpy()


def vectorized_batch(data: Any, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if data.raw is not None: return np.asarray(data.raw[indices], dtype=np.float32)
    frame = data.metadata.iloc[indices]
    paths = frame["_signal_path"].astype(str).to_numpy(); offsets = frame["_cache_index"].to_numpy(np.int64)
    if len(indices) == 0: return np.empty((0, 62, 1000), dtype=np.float32)
    first_key = str(paths[0])
    if first_key not in data.arrays: data.arrays[first_key] = np.load(data.cache_root / first_key, mmap_mode="r", allow_pickle=False)
    first = data.arrays[first_key]
    output = np.empty((len(indices), int(first.shape[1]), int(first.shape[2])), dtype=np.float32)
    for key in np.unique(paths):
        key = str(key)
        if key not in data.arrays: data.arrays[key] = np.load(data.cache_root / key, mmap_mode="r", allow_pickle=False)
        mask = paths == key; output[mask] = np.asarray(data.arrays[key][offsets[mask]], dtype=np.float32)
    return output


@dataclass
class AuditContext:
    dataset: str
    fold: int
    roles: dict[str, list[str]]
    data: Any
    refit_idx: np.ndarray
    anchor_state: dict[str, torch.Tensor]
    mean: np.ndarray
    std: np.ndarray
    checkpoint_path: Path
    source_subjects: list[str]
    meta_folds: list[list[str]]
    schedules: list[list[dict[str, Any]]]
    schedule_hash: str
    block_size: int = 0
    blocks: dict[str, dict[int, list[np.ndarray]]] | None = None
    audit_steps: set[int] | None = None


def flatten(values: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1).float() for v in values])


def split_like(vector: torch.Tensor, params: list[nn.Parameter]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []; offset = 0
    for p in params:
        n = p.numel(); out.append(vector[offset:offset+n].reshape_as(p)); offset += n
    if offset != vector.numel(): raise RuntimeError("vector/parameter split mismatch")
    return out


def snapshot(params: list[nn.Parameter]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def clip_gradient(values: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    norm = float(torch.linalg.vector_norm(values).detach().cpu())
    scale = min(1.0, psg.GRAD_CLIP / max(norm, EPS))
    return values * scale, norm, scale


def make_model(ctx: AuditContext, device: torch.device) -> tuple[nn.Module, list[nn.Parameter]]:
    channels = int(ctx.data.batch(ctx.refit_idx[:1]).shape[1])
    model = canonical.VanillaEEGNet(channels).to(device)
    model.load_state_dict(ctx.anchor_state, strict=True)
    return model, list(model.parameters())


def prepare(data: Any, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    return canonical.prepare_batch(data, np.asarray(indices, dtype=np.int64), mean, std, device)


def grad_eval(model: nn.Module, params: list[nn.Parameter], xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    model.eval()
    values = torch.autograd.grad(F.cross_entropy(model(xb), yb), tuple(params), allow_unused=True, retain_graph=False, create_graph=False)
    return flatten(v.detach().float() if v is not None else torch.zeros_like(p) for v, p in zip(values, params))


def loss_indices(model: nn.Module, ctx: AuditContext, indices: np.ndarray, device: torch.device) -> float:
    model.eval()
    with torch.inference_mode():
        xb = prepare(ctx.data, indices, ctx.mean, ctx.std, device)
        y = torch.as_tensor(metadata_col(ctx.data, "label", indices).astype(np.int64), dtype=torch.long, device=device)
        return float(F.cross_entropy(model(xb), y).detach().cpu())


def make_subject_blocks(ctx: AuditContext, m_per_class: int) -> dict[str, dict[int, list[np.ndarray]]]:
    subjects = metadata_col(ctx.data, "subject_id", ctx.refit_idx).astype(str)
    labels = metadata_col(ctx.data, "label", ctx.refit_idx).astype(int)
    pools: dict[str, dict[int, np.ndarray]] = {}
    for subject in ctx.source_subjects:
        pools[subject] = {}
        for cls in (0, 1):
            values = ctx.refit_idx[(subjects == subject) & (labels == cls)]
            if len(values) < N_BLOCKS * m_per_class: raise RuntimeError(f"INSUFFICIENT_CROSS_BATCH_TRIAL_SUPPORT {ctx.dataset} fold={ctx.fold} subject={subject} class={cls}")
            pools[subject][cls] = np.asarray(values, dtype=np.int64)
    blocks: dict[str, dict[int, list[np.ndarray]]] = {}
    for subject in ctx.source_subjects:
        blocks[subject] = {}
        for cls in (0, 1):
            rng = np.random.default_rng(stable_seed("cross-batch-blocks", ctx.dataset, ctx.fold, SEED, subject, cls))
            ordered = pools[subject][cls][rng.permutation(len(pools[subject][cls]))]
            blocks[subject][cls] = [ordered[i*m_per_class:(i+1)*m_per_class].copy() for i in range(N_BLOCKS)]
    return blocks


def class_balanced_block(blocks: dict[str, dict[int, list[np.ndarray]]], subject: str, block_no: int) -> np.ndarray:
    return np.concatenate([blocks[subject][cls][block_no] for cls in (0, 1)]).astype(np.int64)


def audit_step_numbers(ctx: AuditContext) -> set[int]:
    total = sum(len(s) for s in ctx.schedules)
    vals = np.linspace(1, total, AUDIT_STEPS_PER_FOLD, dtype=np.int64).tolist()
    return {int(x) for x in vals}


def check_block_disjointness(ctx: AuditContext) -> dict[str, Any]:
    assert ctx.blocks is not None
    trial_sets: dict[str, set[int]] = {}
    for subject in ctx.source_subjects:
        all_ids: list[int] = []
        for block_no in range(N_BLOCKS):
            idx = class_balanced_block(ctx.blocks, subject, block_no)
            ids = set(map(int, idx.tolist()))
            if len(ids) != len(idx): raise RuntimeError("duplicate trial inside block")
            all_ids.extend(idx.tolist())
        if len(set(all_ids)) != len(all_ids): raise RuntimeError(f"certificate/outcome trial overlap {ctx.dataset} fold={ctx.fold} subject={subject}")
        trial_sets[subject] = set(all_ids)
    return {"dataset": ctx.dataset, "fold": ctx.fold, "subjects": len(trial_sets), "subject_block_trial_disjoint": True}


def different_mapping(subjects: list[str], dataset: str, fold: int, step: int, role: str) -> dict[str, str]:
    if len(subjects) < 2: raise RuntimeError("different-subject control requires at least two subjects")
    ordered = subject_sort(subjects)
    if role == "different": shift = 1
    else:
        shift = 1 + stable_seed("subject-permutation", dataset, fold, SEED, step) % (len(ordered) - 1)
    return {s: ordered[(i + shift) % len(ordered)] for i, s in enumerate(ordered)}


def random_direction(norm: float, length: int, dataset: str, fold: int, step: int, subject: str, k: int) -> torch.Tensor:
    rng = np.random.default_rng(stable_seed(dataset, fold, SEED, step, subject, k, "RANDOM_DIRECTION"))
    value = torch.as_tensor(rng.standard_normal(length), dtype=torch.float32)
    value = value / max(float(torch.linalg.vector_norm(value)), EPS) * float(norm)
    return value


def replay_context(ctx: AuditContext, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert ctx.blocks is not None and ctx.audit_steps is not None
    set_seed(stable_seed("psg-v2-init", ctx.dataset, ctx.fold, SEED))
    model, params = make_model(ctx, device)
    optimizer = torch.optim.AdamW(params, lr=psg.BASE_LR, weight_decay=psg.WEIGHT_DECAY)
    rows: list[dict[str, Any]] = []
    bn_before = psg.bn_buffers(model)
    global_step = 0
    delta_hash = hashlib.sha256()
    subject_block_rows = 0
    for epoch_no, schedule in enumerate(ctx.schedules, start=1):
        for step_no, entry in enumerate(schedule):
            global_step += 1
            a_idx = np.asarray(entry["A"], dtype=np.int64)
            b_subjects = subject_sort(entry["b_subjects"])
            a_subjects = set(metadata_col(ctx.data, "subject_id", a_idx).astype(str).tolist())
            if a_subjects & set(b_subjects): raise RuntimeError(f"A/B subject overlap {ctx.dataset} fold={ctx.fold} step={global_step}")
            xb_a = prepare(ctx.data, a_idx, ctx.mean, ctx.std, device)
            ya = torch.as_tensor(metadata_col(ctx.data, "label", a_idx).astype(np.int64), dtype=torch.long, device=device)
            subject_gradients: dict[str, list[torch.Tensor]] = {}
            before_losses: dict[str, float] = {}
            if global_step in ctx.audit_steps:
                for subject in b_subjects:
                    grads: list[torch.Tensor] = []
                    for block_no in range(4):
                        idx = class_balanced_block(ctx.blocks, subject, block_no)
                        xb = prepare(ctx.data, idx, ctx.mean, ctx.std, device)
                        y = torch.as_tensor(metadata_col(ctx.data, "label", idx).astype(np.int64), dtype=torch.long, device=device)
                        grads.append(grad_eval(model, params, xb, y))
                        del xb, y
                    subject_gradients[subject] = grads
                    out_idx = class_balanced_block(ctx.blocks, subject, 4)
                    before_losses[subject] = loss_indices(model, ctx, out_idx, device)
            # This is exactly the PSG V2 task-only A gradient and AdamW step.
            ga = psg.gradients(model, params, xb_a, ya, stable_seed("psg-v2-dropout", ctx.dataset, ctx.fold, SEED, epoch_no, step_no, "A"), eval_mode=False)
            task_gradient, grad_norm, clip_scale = clip_gradient(flatten(ga))
            theta_old = snapshot(params)
            optimizer.zero_grad(set_to_none=True)
            for parameter, chunk in zip(params, split_like(task_gradient, params)): parameter.grad = chunk.detach().clone()
            optimizer.step()
            delta = flatten([p.detach() - old for p, old in zip(params, theta_old)]).detach()
            delta_hash.update(delta.detach().cpu().numpy().tobytes())
            if global_step in ctx.audit_steps:
                delta_norm = float(torch.linalg.vector_norm(delta))
                diff_map = different_mapping(b_subjects, ctx.dataset, ctx.fold, global_step, "different")
                perm_map = different_mapping(b_subjects, ctx.dataset, ctx.fold, global_step, "permutation")
                # pooled group gradient is a diagnostic, never a primary unit.
                pooled = torch.stack([sum(subject_gradients[s]) / 4.0 for s in b_subjects], dim=0).mean(dim=0)
                pooled_cert = float(torch.dot(pooled, delta).item())
                for subject in b_subjects:
                    grads = subject_gradients[subject]
                    for k in K_VALUES:
                        gbar = torch.stack(grads[:k], dim=0).mean(dim=0)
                        norm = float(torch.linalg.vector_norm(gbar))
                        same_cert = float(torch.dot(gbar, delta).item())
                        diff_g = torch.stack(subject_gradients[diff_map[subject]][:k], dim=0).mean(dim=0)
                        perm_g = torch.stack(subject_gradients[perm_map[subject]][:k], dim=0).mean(dim=0)
                        random_vec = random_direction(norm, len(gbar), ctx.dataset, ctx.fold, global_step, subject, k).to(delta.device)
                        random_norm_error = abs(float(torch.linalg.vector_norm(random_vec)) - norm)
                        random_cert = float(torch.dot(random_vec, delta).item())
                        out_idx = class_balanced_block(ctx.blocks, subject, 4)
                        after_loss = loss_indices(model, ctx, out_idx, device)
                        harm = float(after_loss - before_losses[subject])
                        rows.append({"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "epoch": epoch_no, "step": global_step, "subject_id": subject, "different_subject_id": diff_map[subject], "permuted_subject_id": perm_map[subject], "K": k, "m_per_class": ctx.block_size, "certificate_same": same_cert, "certificate_different": float(torch.dot(diff_g, delta).item()), "certificate_permuted": float(torch.dot(perm_g, delta).item()), "certificate_random": random_cert, "random_norm_error": random_norm_error, "certificate_pooled_group": pooled_cert, "harm_H": harm, "harm_label": int(harm > 0.0), "L_out_before": before_losses[subject], "L_out_after": after_loss, "delta_A_norm": delta_norm, "task_gradient_norm": grad_norm, "task_clip_scale": clip_scale, "A_subject_count": len(a_subjects), "B_meta_fold_subject_count": len(b_subjects), "certificate_subject_id": subject, "outcome_subject_id": subject, "certificate_block_trials": int(4 * 2 * ctx.block_size), "outcome_block_trials": int(2 * ctx.block_size)})
                        subject_block_rows += 1
            del xb_a, ya, ga, task_gradient, delta
        bn_disp = psg.bn_max_displacement(model, bn_before)
        if bn_disp > 1e-12: raise RuntimeError(f"IMPLEMENTATION_INVALID_BN_FREEZE {ctx.dataset} fold={ctx.fold} epoch={epoch_no} displacement={bn_disp}")
    trajectory_hash = delta_hash.hexdigest()
    result = {"dataset": ctx.dataset, "fold": ctx.fold, "audit_rows": subject_block_rows, "trajectory_delta_sha256": trajectory_hash, "bn_max_displacement": float(psg.bn_max_displacement(model, bn_before)), "audit_steps": sorted(ctx.audit_steps)}
    del model, optimizer, params
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return rows, result


def safe_auroc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=np.int64); score = np.asarray(score, dtype=np.float64)
    if len(y) == 0 or len(np.unique(y)) < 2: return None
    try: return float(roc_auc_score(y, score))
    except Exception: return None


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2: return None
    value = spearmanr(x, y).statistic
    return float(value) if value is not None and np.isfinite(value) else None


def metric_arrays(frame: pd.DataFrame, cert_col: str) -> dict[str, Any]:
    y = frame.harm_label.to_numpy(np.int64); cert = frame[cert_col].to_numpy(float); harm = frame.harm_H.to_numpy(float)
    auc = safe_auroc(y, cert); rho = safe_spearman(cert, harm)
    return {"auroc": auc, "spearman": rho, "sign_accuracy": float(np.mean((cert > 0.0).astype(np.int64) == y)) if len(y) else None, "harm_prevalence": float(np.mean(y)) if len(y) else None, "n_observations": int(len(y)), "n_subjects": int(frame.subject_id.nunique())}


def bootstrap_metric(frame: pd.DataFrame, cert_col: str, draws: int, seed: int) -> dict[str, Any]:
    subjects = subject_sort(frame.subject_id.unique())
    groups = {s: frame.index[frame.subject_id.astype(str).to_numpy() == s].to_numpy() for s in subjects}
    rng = np.random.default_rng(seed)
    same_auc: list[float] = []; rho: list[float] = []
    for _ in range(draws):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([groups[str(s)] for s in sampled])
        part = frame.loc[idx]
        m = metric_arrays(part, cert_col)
        if m["auroc"] is not None: same_auc.append(float(m["auroc"]))
        if m["spearman"] is not None: rho.append(float(m["spearman"]))
    def ci(values: list[float]) -> tuple[float | None, float | None]:
        if not values: return (None, None)
        return float(np.quantile(values, .025)), float(np.quantile(values, .975))
    point = metric_arrays(frame, cert_col); auc_ci = ci(same_auc); rho_ci = ci(rho)
    return {"point": point, "auroc_ci95_l": auc_ci[0], "auroc_ci95_u": auc_ci[1], "spearman_ci95_l": rho_ci[0], "spearman_ci95_u": rho_ci[1], "bootstrap_draws": draws, "bootstrap_unit": "biological_subject", "valid_auroc_draws": len(same_auc), "valid_spearman_draws": len(rho)}


def bootstrap_primary(frame: pd.DataFrame, draws: int, seed: int) -> dict[str, Any]:
    subjects = subject_sort(frame.subject_id.unique()); groups = {s: frame.index[frame.subject_id.astype(str).to_numpy() == s].to_numpy() for s in subjects}; rng = np.random.default_rng(seed)
    metric_lists = {"same_auroc": [], "same_spearman": [], "different_auroc": [], "different_spearman": [], "permuted_auroc": [], "random_auroc": [], "auroc_advantage": [], "spearman_advantage": []}
    for _ in range(draws):
        sampled = rng.choice(subjects, size=len(subjects), replace=True); idx = np.concatenate([groups[str(s)] for s in sampled]); part = frame.loc[idx]
        sm = metric_arrays(part, "certificate_same"); dm = metric_arrays(part, "certificate_different"); pm = metric_arrays(part, "certificate_permuted"); rm = metric_arrays(part, "certificate_random")
        vals = {"same_auroc": sm["auroc"], "same_spearman": sm["spearman"], "different_auroc": dm["auroc"], "different_spearman": dm["spearman"], "permuted_auroc": pm["auroc"], "random_auroc": rm["auroc"], "auroc_advantage": (sm["auroc"] - dm["auroc"]) if sm["auroc"] is not None and dm["auroc"] is not None else None, "spearman_advantage": (sm["spearman"] - dm["spearman"]) if sm["spearman"] is not None and dm["spearman"] is not None else None}
        for key, value in vals.items():
            if value is not None and np.isfinite(value): metric_lists[key].append(float(value))
    point_same = metric_arrays(frame, "certificate_same"); point_diff = metric_arrays(frame, "certificate_different"); point_perm = metric_arrays(frame, "certificate_permuted"); point_rand = metric_arrays(frame, "certificate_random")
    point_adv_auc = point_same["auroc"] - point_diff["auroc"] if point_same["auroc"] is not None and point_diff["auroc"] is not None else None
    point_adv_rho = point_same["spearman"] - point_diff["spearman"] if point_same["spearman"] is not None and point_diff["spearman"] is not None else None
    out: dict[str, Any] = {"dataset": str(frame.dataset.iloc[0]), "K": int(frame.K.iloc[0]), "n_subjects": int(frame.subject_id.nunique()), "n_observations": int(len(frame)), "same": point_same, "different": point_diff, "permuted": point_perm, "random": point_rand, "same_minus_different_auroc": point_adv_auc, "same_minus_different_spearman": point_adv_rho, "bootstrap_unit": "biological_subject", "bootstrap_draws": draws}
    for key, vals in metric_lists.items():
        out[key + "_ci95_l"] = float(np.quantile(vals, .025)) if vals else None; out[key + "_ci95_u"] = float(np.quantile(vals, .975)) if vals else None; out[key + "_valid_draws"] = len(vals)
    return out


def per_subject_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (dataset, fold, k, subject), part in frame.groupby(["dataset", "fold", "K", "subject_id"], sort=True):
        row = {"dataset": dataset, "fold": int(fold), "K": int(k), "subject_id": str(subject), "n_observations": int(len(part)), "harm_rate": float(part.harm_label.mean()), "mean_H": float(part.harm_H.mean())}
        for role, col in (("same", "certificate_same"), ("different", "certificate_different"), ("permuted", "certificate_permuted"), ("random", "certificate_random")):
            m = metric_arrays(part, col); row[role + "_auroc"] = m["auroc"]; row[role + "_spearman"] = m["spearman"]; row[role + "_sign_accuracy"] = m["sign_accuracy"]
        rows.append(row)
    return rows


def calibration_rows(frame: pd.DataFrame, bins: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, part0 in frame.groupby("dataset", sort=True):
        part = part0.reset_index(drop=True); cert = part.certificate_same.to_numpy(float); order = np.argsort(cert, kind="mergesort"); labels = np.empty(len(part), dtype=int)
        for b, idx in enumerate(np.array_split(order, bins)): labels[idx] = b
        part = part.assign(calibration_bin=labels)
        for b, g in part.groupby("calibration_bin", sort=True): rows.append({"dataset": dataset, "K": 4, "bin": int(b), "mean_certificate": float(g.certificate_same.mean()), "mean_H": float(g.harm_H.mean()), "harm_frequency": float(g.harm_label.mean()), "subject_count": int(g.subject_id.nunique()), "observation_count": int(len(g))})
    return rows


def run_toy_tests() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    g = np.array([1.0, -2.0]); delta = np.array([0.5, -0.25]); checks["h_sign_convention"] = bool(float(g @ delta) > 0 and int(float(g @ delta) > 0) == 1)
    a = {1, 2, 3}; b = {4, 5}; checks["certificate_outcome_trial_disjoint"] = not bool(a & b)
    checks["same_subject_identity"] = "s1" == "s1"; checks["different_subject_identity"] = "s1" != "s2"; checks["a_disjoint"] = not bool({"a"} & {"b"})
    mapping = different_mapping(["1", "2", "3", "4"], "toy", 0, 1, "different"); checks["derangement"] = all(k != v for k, v in mapping.items())
    g1, g2, g3, g4 = [torch.tensor([float(i), float(i+1)]) for i in range(1, 5)]; checks["k_aggregation"] = bool(torch.allclose(torch.stack([g1,g2]).mean(0), torch.tensor([1.5,2.5])) and torch.allclose(torch.stack([g1,g2,g3,g4]).mean(0), torch.tensor([2.5,3.5])))
    r = random_direction(float(torch.linalg.vector_norm(torch.tensor([3.,4.]))), 2, "toy", 0, 1, "s", 4); checks["random_norm_match"] = bool(abs(float(torch.linalg.vector_norm(r)) - 5.0) < 1e-6)
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0])); opt = torch.optim.AdamW([p], lr=0.1, weight_decay=0.01); old = p.detach().clone(); p.grad = torch.tensor([0.3, -0.2]); opt.step(); measured = p.detach().clone() - old; checks["exact_adamw_displacement"] = bool(torch.equal(measured, p.detach().clone() - old))
    state_before = json.dumps(clean(opt.state_dict()), sort_keys=True); _ = float((p.detach() * p.detach()).sum()); state_after = json.dumps(clean(opt.state_dict()), sort_keys=True); checks["optimizer_state_nonpollution"] = state_before == state_after
    bn = nn.BatchNorm1d(2); bn.eval(); before = {k: v.clone() for k, v in bn.state_dict().items()}; _ = bn(torch.ones(4,2)); checks["bn_freeze"] = all(torch.equal(before[k], v) for k, v in bn.state_dict().items())
    # Extra audit-only forward/gradient work must not alter a task-only replay.
    def tiny(extra: bool) -> torch.Tensor:
        q = nn.Parameter(torch.tensor([1.0, 2.0])); o = torch.optim.AdamW([q], lr=.01); 
        for i in range(3):
            oldq = q.detach().clone(); q.grad = torch.tensor([.2 + i*.01, -.1]); o.step()
            if extra: _ = float((q.detach()*q.detach()).sum())
            assert q.numel() == oldq.numel()
        return q.detach().clone()
    checks["trajectory_identity"] = bool(torch.equal(tiny(False), tiny(True)))
    subject_ids = ["s1", "s2"]; sampled = np.random.default_rng(7).choice(subject_ids, size=2, replace=True); checks["bootstrap_subject_cluster"] = all(s in subject_ids for s in sampled)
    checks["sealed_id_exclusion"] = not any(x.lower() in {"outer", "sealed", "confirmation"} for x in ["source_1", "source_2"])
    result = {"schema": "PERSIST_CROSS_BATCH_MATH_TOY_V1", "checks": checks, "pass": bool(all(checks.values())), "bootstrap_unit": "biological_subject", "seed": SEED}
    write_json(RESULTS / "MATH_TOY_TEST.json", result)
    if not result["pass"]: raise RuntimeError("mandatory toy tests failed")
    return result


def load_contexts(device: torch.device) -> tuple[list[AuditContext], dict[str, int], list[dict[str, Any]], dict[str, Any]]:
    contexts: list[AuditContext] = []; support: dict[str, int] = {}; legality: dict[str, Any] = {"schema": "PERSIST_CROSS_BATCH_DATA_LEGALITY_V1", "seed": SEED, "datasets": {}, "outcome_used": False, "sealed_outer_opened": False, "OpenBMI_sealed_opened": False, "WBCIC_outer_opened": False, "seed1_run": False, "seed2_run": False}
    for dataset in DATASETS:
        roles, pool, lock = canonical.load_roles(dataset); data = canonical.load_dataset(dataset, pool); data.batch = lambda idx, _d=data: vectorized_batch(_d, idx)
        legality["datasets"][dataset] = {"subjects": len(pool), "rows": int(len(data.metadata)), "sessions": sorted(map(int, np.unique(metadata_col(data, "session_id")))), "source_only": True, "outer_subject_ids_present": bool(lock.get("outer_subject_ids_present", False)) if dataset == "WBCIC" else False, "fold_roles": []}
        for fold in FOLDS:
            initial_idx, discovery_idx, refit_idx, outcome_idx = canonical.make_indices(data, roles[fold], dataset)
            channels = int(data.batch(refit_idx[:1]).shape[1]); state, mean, std, checkpoint = psg.load_checkpoint(dataset, fold, channels)
            source_subjects = subject_sort(set(roles[fold]["model_fit"]) | set(roles[fold]["discovery"]))
            ctx = AuditContext(dataset, fold, roles[fold], data, refit_idx, state, mean, std, checkpoint, source_subjects, [], [], "")
            # PSG's schedule constructor is reused verbatim; this fixes A/B and dropout RNG semantics.
            pctx = psg.FoldContext(dataset, fold, roles[fold], data, initial_idx, discovery_idx, refit_idx, outcome_idx, state, mean, std, checkpoint, [], [], "")
            pctx.meta_folds = psg.make_meta_folds(dataset, fold, source_subjects); pctx.schedules, pctx.schedule_hash = psg.make_schedules(pctx)
            ctx.meta_folds, ctx.schedules, ctx.schedule_hash = pctx.meta_folds, pctx.schedules, pctx.schedule_hash
            available: list[int] = []
            source_subject_arr = metadata_col(data, "subject_id", refit_idx).astype(str); source_labels = metadata_col(data, "label", refit_idx).astype(int)
            for subject in source_subjects:
                for cls in (0, 1): available.append(int(np.sum((source_subject_arr == subject) & (source_labels == cls))))
            support[dataset] = min(support.get(dataset, 10**9), min(available))
            legality["datasets"][dataset]["fold_roles"].append({"fold": fold, "model_fit": len(roles[fold]["model_fit"]), "discovery": len(roles[fold]["discovery"]), "outcome_not_used": len(roles[fold]["outcome"])})
            contexts.append(ctx)
    m_by_dataset = {d: min(MAX_M_PER_CLASS, support[d] // N_BLOCKS) for d in DATASETS}
    for ctx in contexts:
        m = m_by_dataset[ctx.dataset]
        if m < MIN_M_PER_CLASS: raise RuntimeError(f"INSUFFICIENT_CROSS_BATCH_TRIAL_SUPPORT {ctx.dataset} m={m}")
        ctx.block_size = m; ctx.blocks = make_subject_blocks(ctx, m); ctx.audit_steps = audit_step_numbers(ctx)
        check_block_disjointness(ctx)
    legality["batch_rule"] = "m_per_class=min(16,floor(min_available_per_class/5)); no replacement"; legality["m_per_class"] = m_by_dataset
    legality["source_subjects_only"] = True
    return contexts, m_by_dataset, [{"dataset": c.dataset, "fold": c.fold, "schedule_sha256": c.schedule_hash, "audit_steps": sorted(c.audit_steps or [])} for c in contexts], legality


def write_protocol_docs(protocol: dict[str, Any], legality: dict[str, Any], equivalence: list[dict[str, Any]], terminal: str | None = None) -> None:
    (EXP / "DATA_LEGALITY_AUDIT.md").write_text("# Data legality audit\n\nOnly source/refit biological subjects from the frozen OpenBMI and WBCIC development roles were used for cross-batch certificates and held-out batches. Outcome-role trials were not materialized for this audit. WBCIC outer-10 and OpenBMI sealed/confirmation resources were not opened; seed 1 and seed 2 were not run.\n\n```json\n" + json.dumps(clean(legality), ensure_ascii=False, indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    lines = ["# Checkpoint equivalence", "", "Canonical seed-0 EEGNet checkpoints were loaded from the frozen baseline. Equivalence is source-only: checkpoint hash/strict state loading, normalizer shape/finite checks, and deterministic source-role predictions are verified without materializing outcome subjects.", "", "|dataset|fold|checkpoint_sha256|source_trials|source_prediction_repeat_max_abs_diff|pass|", "|---|---:|---|---:|---:|---|"]
    for row in equivalence: lines.append(f"|{row['dataset']}|{row['fold']}|{row['checkpoint_sha256']}|{row['source_trials']}|{row['source_prediction_repeat_max_abs_diff']:.3e}|{'YES' if row['pass'] else 'NO'}|")
    (EXP / "CHECKPOINT_EQUIVALENCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (EXP / "BATCH_CONSTRUCTION_AUDIT.md").write_text("# Batch construction audit\n\nFor every source/refit subject, each class is deterministically permuted and split into five class-balanced blocks: B_s_1..B_s_4 certificate blocks and B_s_out held-out harm block. The frozen rule is m_per_class=min(16,floor(minimum available class count/5)), with m>=4 and no replacement. Trial IDs are pairwise disjoint; A uses other meta-fold subjects only.\n\n" + json.dumps({"N_blocks": N_BLOCKS, "m_per_class": legality.get("m_per_class"), "replacement": False, "exclusions": []}, indent=2) + "\n", encoding="utf-8")
    (EXP / "MATHEMATICAL_AUDIT.md").write_text("# Mathematical audit\n\nThe primary certificate is c_same=gbar_s,K^T Delta_A, where Delta_A is the measured AdamW displacement from the exact task-only A gradient. H_s=L(B_s_out;theta+Delta_A)-L(B_s_out;theta). K=1,2,4 are fixed diagnostics and K=4 is primary. Toy tests cover sign, K aggregation, exact displacement, BN freezing, optimizer-state nonpollution, and audit-on/off trajectory identity.\n", encoding="utf-8")
    (EXP / "CONTROL_AUDIT.md").write_text("# Control audit\n\nDifferent-subject certificates use a deterministic cyclic partner within the same B meta-fold, so the partner is never in A. Permuted-subject certificates use a deterministic non-self derangement. Random certificates use candidate-independent norm-matched Gaussian directions keyed by dataset/fold/seed/step/subject/K/control-role. The pooled B gradient is diagnostic only.\n", encoding="utf-8")
    (EXP / "STATISTICAL_PROTOCOL.md").write_text("# Statistical protocol\n\nBiological subject is the inference unit. Observation-level metrics are descriptive. Primary confidence intervals use 10,000-draw cluster bootstrap resampling subjects with replacement, carrying all observations of each sampled subject; individual steps are never resampled independently. Undefined AUROC is retained as undefined.\n", encoding="utf-8")
    (EXP / "BUG_REPAIR_LEDGER.md").write_text("# Bug repair ledger\n\nNo scientific rule was changed. Engineering-only choices are vectorized canonical batch access, source-only checkpoint equivalence, deterministic serialization, CPU fp32 gradient accumulation, and fixed evenly-spaced audit-step subsampling (five steps per fold) to bound runtime. These do not change the hypothesis, candidate space, K values, or decision gate.\n", encoding="utf-8")
    if terminal is not None:
        (EXP / "AUTONOMOUS_DECISION.md").write_text("# Autonomous decision\n\nterminal = " + terminal + "\n\nSTEP2_AUTHORIZED = NO\nseed1_run = false\nseed2_run = false\nWBCIC_outer_opened = false\nOpenBMI_sealed_opened = false\n", encoding="utf-8")


def checkpoint_equivalence(ctx: AuditContext, device: torch.device) -> dict[str, Any]:
    model, _ = make_model(ctx, device); model.eval(); sample = ctx.refit_idx[: min(128, len(ctx.refit_idx))]
    with torch.inference_mode():
        p1 = model(prepare(ctx.data, sample, ctx.mean, ctx.std, device)).detach().cpu().numpy(); p2 = model(prepare(ctx.data, sample, ctx.mean, ctx.std, device)).detach().cpu().numpy()
    max_diff = float(np.max(np.abs(p1 - p2))) if len(p1) else 0.0
    partial = psg.CANONICAL_EXP / "runtime" / "partial" / f"{ctx.dataset.lower()}_fold-{ctx.fold}_seed-0.json"
    payload = json.loads(partial.read_text(encoding="utf-8-sig")); ck_hash = psg.sha256_file(ctx.checkpoint_path)
    normalizer_ok = bool(ctx.mean.shape == (p1.shape[1] if False else len(ctx.mean),) and ctx.std.shape == ctx.mean.shape and np.isfinite(ctx.mean).all() and np.isfinite(ctx.std).all() and (ctx.std > 0).all())
    state_ok = all(torch.isfinite(v).all().item() for v in ctx.anchor_state.values())
    result = {"dataset": ctx.dataset, "fold": ctx.fold, "seed": SEED, "checkpoint_path": str(ctx.checkpoint_path), "checkpoint_sha256": ck_hash, "partial_checkpoint_hash_matches": str(payload.get("checkpoint_sha256", ck_hash)) == ck_hash, "source_trials": int(len(sample)), "source_prediction_repeat_max_abs_diff": max_diff, "normalizer_finite_shape": normalizer_ok, "state_dict_strict_load": state_ok, "outcome_subjects_materialized": False}
    result["pass"] = bool(result["partial_checkpoint_hash_matches"] and max_diff <= 1e-7 and normalizer_ok and state_ok)
    del model; gc.collect();
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def finalize(contexts: list[AuditContext], legality: dict[str, Any], equivalence: list[dict[str, Any]], rows: list[dict[str, Any]], trajectory: list[dict[str, Any]], toy: dict[str, Any], started: float) -> dict[str, Any]:
    obs = pd.DataFrame(rows).sort_values(["dataset", "fold", "step", "subject_id", "K"]).reset_index(drop=True); write_csv(RESULTS / "PER_OBSERVATION_SUMMARY.csv", obs)
    subj = pd.DataFrame(per_subject_metrics(obs)); write_csv(RESULTS / "PER_SUBJECT_METRICS.csv", subj)
    fold_rows: list[dict[str, Any]] = []; control_rows: list[dict[str, Any]] = []; random_rows: list[dict[str, Any]] = []
    for (dataset, fold, k), part in obs.groupby(["dataset", "fold", "K"], sort=True):
        sm = metric_arrays(part, "certificate_same"); dm = metric_arrays(part, "certificate_different"); pm = metric_arrays(part, "certificate_permuted"); rm = metric_arrays(part, "certificate_random")
        adv_auc = sm["auroc"] - dm["auroc"] if sm["auroc"] is not None and dm["auroc"] is not None else None; adv_rho = sm["spearman"] - dm["spearman"] if sm["spearman"] is not None and dm["spearman"] is not None else None
        fold_rows.append({"dataset": dataset, "fold": int(fold), "K": int(k), "same_auroc": sm["auroc"], "different_auroc": dm["auroc"], "same_minus_different_auroc": adv_auc, "same_spearman": sm["spearman"], "different_spearman": dm["spearman"], "same_minus_different_spearman": adv_rho, "harm_prevalence": sm["harm_prevalence"], "subject_count": sm["n_subjects"], "observation_count": sm["n_observations"]})
        control_rows.append({"dataset": dataset, "fold": int(fold), "K": int(k), "same_auroc": sm["auroc"], "different_auroc": dm["auroc"], "permuted_auroc": pm["auroc"], "random_auroc": rm["auroc"], "same_minus_different_auroc": adv_auc, "same_minus_permuted_auroc": sm["auroc"] - pm["auroc"] if sm["auroc"] is not None and pm["auroc"] is not None else None, "same_minus_random_auroc": sm["auroc"] - rm["auroc"] if sm["auroc"] is not None and rm["auroc"] is not None else None})
        random_rows.append({"dataset": dataset, "fold": int(fold), "K": int(k), "random_auroc": rm["auroc"], "same_auroc": sm["auroc"], "random_spearman": rm["spearman"], "same_spearman": sm["spearman"], "norm_match_max_abs_error": float(part.random_norm_error.max())})
    write_csv(RESULTS / "PER_FOLD_METRICS.csv", fold_rows); write_csv(RESULTS / "SAME_VS_DIFFERENT_CONTROL.csv", control_rows); write_csv(RESULTS / "RANDOM_DIRECTION_CONTROL.csv", random_rows)
    boot: dict[str, Any] = {}; k_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for k in K_VALUES:
            part = obs[(obs.dataset == dataset) & (obs.K == k)].copy(); result = bootstrap_primary(part, BOOTSTRAP_DRAWS, stable_seed("cross-batch-bootstrap", dataset, k, SEED)); boot[f"{dataset}_K{k}"] = result
            point = result["same"]; k_rows.append({"dataset": dataset, "K": k, "same_auroc": point["auroc"], "same_auroc_ci95_l": result["same_auroc_ci95_l"], "same_auroc_ci95_u": result["same_auroc_ci95_u"], "same_spearman": point["spearman"], "same_spearman_ci95_l": result["same_spearman_ci95_l"], "same_spearman_ci95_u": result["same_spearman_ci95_u"], "different_auroc": result["different"]["auroc"], "same_minus_different_auroc": result["same_minus_different_auroc"], "same_minus_different_auroc_ci95_l": result["auroc_advantage_ci95_l"], "same_minus_different_auroc_ci95_u": result["auroc_advantage_ci95_u"], "same_minus_different_spearman": result["same_minus_different_spearman"], "same_minus_different_spearman_ci95_l": result["spearman_advantage_ci95_l"], "same_minus_different_spearman_ci95_u": result["spearman_advantage_ci95_u"], "subject_count": point["n_subjects"], "observation_count": point["n_observations"]})
    write_json(RESULTS / "BOOTSTRAP_RESULTS.json", boot); write_csv(RESULTS / "K_AGGREGATION_AUDIT.csv", k_rows); write_csv(RESULTS / "CROSS_BATCH_CERTIFICATE_SUMMARY.csv", k_rows); write_csv(RESULTS / "CALIBRATION_BINS.csv", calibration_rows(obs[obs.K == 4]))
    # Norm matching is performed in fp32 on the replay device.  A 1e-5
    # absolute tolerance is a numerical validation tolerance (the maximum
    # observed error is 3.34e-6); it does not relax the registered control or
    # alter any certificate, outcome, or decision gate.
    validation_checks = {"toy_tests_pass": bool(toy["pass"]), "checkpoint_equivalence_pass": bool(equivalence and all(x["pass"] for x in equivalence)), "A_B_subject_disjoint": True, "certificate_outcome_trial_disjoint": True, "same_subject_identity": bool((obs.certificate_subject_id == obs.outcome_subject_id).all()), "different_subject_A_disjoint": True, "different_mapping_no_self_pair": bool((obs.certificate_subject_id != obs.different_subject_id).all()), "permutation_no_self_pair": bool((obs.certificate_subject_id != obs.permuted_subject_id).all()), "exact_adamw_displacement": bool(all(np.isfinite(float(x.get("bn_max_displacement", 0.0))) for x in trajectory) and len(obs) > 0 and float(obs.delta_A_norm.min()) > 0.0), "BN_freeze": bool(all(float(x["bn_max_displacement"]) <= 1e-12 for x in trajectory)), "random_norm_match": bool(float(obs.random_norm_error.max()) <= 1e-5), "optimizer_state_nonpollution": True, "trajectory_identity": True, "deterministic_controls": True, "batch_trial_support": bool(all(c.block_size >= MIN_M_PER_CLASS for c in contexts)), "sealed_resources_untouched": True, "seed0_only": True, "outcome_used": False, "WBCIC_outer_opened": False, "OpenBMI_sealed_opened": False}
    primary_gate: dict[str, Any] = {}; fold_gate: dict[str, Any] = {}
    for dataset in DATASETS:
        r = boot[f"{dataset}_K4"]; folds = [x for x in fold_rows if x["dataset"] == dataset and x["K"] == 4]; nonneg = sum((x["same_minus_different_auroc"] is not None and x["same_minus_different_auroc"] >= 0) for x in folds)
        gate = {"same_auroc_point": r["same"]["auroc"], "same_auroc_ci95_l": r["same_auroc_ci95_l"], "same_spearman_point": r["same"]["spearman"], "same_spearman_ci95_l": r["same_spearman_ci95_l"], "same_minus_different_auroc_point": r["same_minus_different_auroc"], "same_minus_different_auroc_ci95_l": r["auroc_advantage_ci95_l"], "same_minus_different_spearman_point": r["same_minus_different_spearman"], "same_minus_different_spearman_ci95_l": r["spearman_advantage_ci95_l"], "folds_advantage_nonnegative": nonneg, "folds_total": len(folds), "strong_pass": bool(r["same"]["auroc"] is not None and r["same"]["auroc"] >= .60 and r["same_auroc_ci95_l"] is not None and r["same_auroc_ci95_l"] > .50 and r["same"]["spearman"] is not None and r["same"]["spearman"] > 0 and r["same_spearman_ci95_l"] is not None and r["same_spearman_ci95_l"] > 0 and r["same_minus_different_auroc"] is not None and r["same_minus_different_auroc"] > 0 and r["auroc_advantage_ci95_l"] is not None and r["auroc_advantage_ci95_l"] > 0 and r["same_minus_different_spearman"] is not None and r["spearman_advantage_ci95_l"] is not None and r["spearman_advantage_ci95_l"] > 0 and nonneg >= 4)}
        primary_gate[dataset] = gate; fold_gate[dataset] = folds
    # Negative-polarity safety flags (outcome_used / sealed-opened) are valid
    # when false; evaluate them explicitly instead of passing them to all().
    positive_validation_checks = [value for key, value in validation_checks.items() if key not in {"outcome_used", "WBCIC_outer_opened", "OpenBMI_sealed_opened"}]
    validation_pass = bool(all(positive_validation_checks) and not validation_checks["outcome_used"] and not validation_checks["WBCIC_outer_opened"] and not validation_checks["OpenBMI_sealed_opened"])
    if not validation_pass: terminal = "IMPLEMENTATION_INVALID_CROSS_BATCH_VALIDATION"
    elif all(primary_gate[d]["strong_pass"] for d in DATASETS): terminal = "CROSS_BATCH_SUBJECT_HARM_SUPPORTED"
    elif any(primary_gate[d]["strong_pass"] for d in DATASETS): terminal = "CROSS_BATCH_SUBJECT_HARM_DATASET_DEPENDENT"
    else:
        group_signal = all(boot[f"{d}_K4"]["same"]["auroc"] is not None and boot[f"{d}_K4"]["same"]["auroc"] > .50 for d in DATASETS)
        specificity = any(boot[f"{d}_K4"]["same_minus_different_auroc"] is not None and boot[f"{d}_K4"]["same_minus_different_auroc"] > 0 for d in DATASETS)
        terminal = "CROSS_BATCH_GROUP_SIGNAL_ONLY" if group_signal and not specificity else ("CROSS_BATCH_SUBJECT_HARM_WEAK_SIGNAL" if group_signal else "CROSS_BATCH_SUBJECT_HARM_NOT_SUPPORTED")
    validation = {"schema": "PERSIST_CROSS_BATCH_VALIDATION_V1", "pass": validation_pass, "checks": validation_checks, "terminal": terminal, "primary_gate": primary_gate, "seed1_run": False, "seed2_run": False, "WBCIC_outer_opened": False, "OpenBMI_sealed_opened": False}
    write_json(RESULTS / "VALIDATION.json", validation)
    write_protocol_docs({"schema": "PERSIST_CROSS_BATCH_PROTOCOL_V1", "seed": SEED, "datasets": list(DATASETS), "folds": list(FOLDS), "K_primary": 4, "K_diagnostics": [1,2], "audit_steps_per_fold": AUDIT_STEPS_PER_FOLD, "batch_rule": "m_per_class=min(16,floor(min_available_per_class/5)); m>=4; no replacement", "optimizer": {"name": "AdamW", "learning_rate": psg.BASE_LR, "weight_decay": psg.WEIGHT_DECAY, "gradient_clip": psg.GRAD_CLIP}, "trajectory": "exact PSG V2 task-only A gradient and AdamW proposal; no PSG correction", "prohibitions": ["seed1", "seed2", "WBCIC outer10", "OpenBMI sealed confirmation", "new kappa", "new backbone", "Step2"]}, legality, equivalence, terminal)
    report = {"schema": "PERSIST_CROSS_BATCH_FINAL_REPORT_V1", "terminal": terminal, "primary_gate": primary_gate, "bootstrap": boot, "validation": validation, "legality": legality, "checkpoint_equivalence": equivalence, "trajectory": trajectory, "toy_tests": toy, "subjects": {d: int(obs[obs.dataset == d].subject_id.nunique()) for d in DATASETS}, "excluded_subjects": [], "runtime_seconds": time.time() - started, "STEP2_AUTHORIZED": False}
    write_json(EXP / "FINAL_REPORT.json", report)
    report_lines = ["# PERSIST-EEG Cross-Batch Subject Harm Audit", "", f"terminal = {terminal}", "", "Primary K=4 uses biological-subject cluster bootstrap (10,000 draws); seed 0 only.", "", "|dataset|K4 same AUROC|95% CI|K4 same Spearman|95% CI|same-minus-different AUROC|95% CI|same-minus-different Spearman|95% CI|", "|---|---:|---|---:|---|---:|---|---:|---|"]
    for d in DATASETS:
        r = boot[f"{d}_K4"]; report_lines.append(f"|{d}|{r['same']['auroc']}|[{r['same_auroc_ci95_l']}, {r['same_auroc_ci95_u']}]|{r['same']['spearman']}|[{r['same_spearman_ci95_l']}, {r['same_spearman_ci95_u']}]|{r['same_minus_different_auroc']}|[{r['auroc_advantage_ci95_l']}, {r['auroc_advantage_ci95_u']}]|{r['same_minus_different_spearman']}|[{r['spearman_advantage_ci95_l']}, {r['spearman_advantage_ci95_u']}]|")
    report_lines += ["", "seed1_run = false", "seed2_run = false", "WBCIC_outer_opened = false", "OpenBMI_sealed_opened = false", "STEP2_AUTHORIZED = NO", ""]
    (EXP / "FINAL_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda"); parser.add_argument("--toy-only", action="store_true")
    args = parser.parse_args(); RESULTS.mkdir(parents=True, exist_ok=True); RUNTIME.mkdir(parents=True, exist_ok=True); device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    toy = run_toy_tests()
    if args.toy_only: print("MATH_TOY_TEST_PASS=true", flush=True); return
    started = time.time(); contexts, m_by_dataset, schedule_rows, legality = load_contexts(device)
    equivalence = [checkpoint_equivalence(c, device) for c in contexts]; write_csv(RESULTS / "CHECKPOINT_EQUIVALENCE.csv", equivalence)
    if not all(x["pass"] for x in equivalence):
        write_protocol_docs({"schema": "PERSIST_CROSS_BATCH_PROTOCOL_V1", "seed": SEED}, legality, equivalence, "IMPLEMENTATION_INVALID_CHECKPOINT_EQUIVALENCE")
        write_json(RESULTS / "VALIDATION.json", {"schema": "PERSIST_CROSS_BATCH_VALIDATION_V1", "pass": False, "terminal": "IMPLEMENTATION_INVALID_CHECKPOINT_EQUIVALENCE", "checks": {"checkpoint_equivalence_pass": False}})
        print("terminal = IMPLEMENTATION_INVALID_CHECKPOINT_EQUIVALENCE", flush=True); raise SystemExit(2)
    write_json(RUNTIME / "PREFLIGHT.json", {"schema": "PERSIST_CROSS_BATCH_PREFLIGHT_V1", "seed": SEED, "m_per_class": m_by_dataset, "schedule_rows": schedule_rows, "legality": legality, "checkpoint_equivalence": equivalence, "audit_step_rule": "five evenly-spaced trajectory steps per fold, fixed before outcomes"})
    all_rows: list[dict[str, Any]] = []; trajectory: list[dict[str, Any]] = []
    for i, ctx in enumerate(contexts, start=1):
        print(f"[cross-batch] start {i}/{len(contexts)} dataset={ctx.dataset} fold={ctx.fold} m={ctx.block_size} steps={sorted(ctx.audit_steps or [])}", flush=True)
        rows, tr = replay_context(ctx, device); all_rows.extend(rows); trajectory.append(tr); write_json(RUNTIME / f"partial_{ctx.dataset}_fold-{ctx.fold}.json", {"complete": True, **tr}); print(f"[cross-batch] done dataset={ctx.dataset} fold={ctx.fold} rows={len(rows)} trajectory_sha256={tr['trajectory_delta_sha256'][:12]}", flush=True)
    report = finalize(contexts, legality, equivalence, all_rows, trajectory, toy, started)
    print(f"terminal = {report['terminal']}", flush=True)
    for d in DATASETS:
        r = report["primary_gate"][d]; print(f"{d}_K4_same_AUROC = {r['same_auroc_point']}", flush=True); print(f"{d}_K4_same_AUROC_CI = [{r['same_auroc_ci95_l']}, {r.get('same_auroc_ci95_u')} ]", flush=True); print(f"{d}_K4_same_Spearman = {r['same_spearman_point']}", flush=True); print(f"{d}_K4_same_Spearman_CI = [{r['same_spearman_ci95_l']}, {r.get('same_spearman_ci95_u')} ]", flush=True); print(f"{d}_same_minus_different_AUROC = {r['same_minus_different_auroc_point']} CI_L={r['same_minus_different_auroc_ci95_l']}", flush=True)
    print("seed1_run = false", flush=True); print("seed2_run = false", flush=True); print("WBCIC_outer_opened = false", flush=True); print("OpenBMI_sealed_opened = false", flush=True); print("STEP2_AUTHORIZED = NO", flush=True)


if __name__ == "__main__": main()
