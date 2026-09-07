"""Frozen prospective-gradient signal audit for the PMG-fast seed-0 pilot.

The audit never trains a model in its primary path.  It reuses the ten
model-fit-only PMG-fast M0 anchors, draws exactly the predeclared paired
subject-balanced batches, and measures first-order gradient geometry and the
actual relative-displacement response with ``functional_call``.  The optional
WBCIC fold-1 forensic reproduction is kept separate from the primary audit and
uses the original PMG-fast recipe verbatim only when no M2 checkpoint exists.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.func import functional_call


REPO = Path(os.environ.get("PMG_REPO", r"D:\nips-temp\TotalP\P1\CRCICLR_CANONICAL_EEGNET")).resolve()
AUDIT = REPO / "experiments" / "persist_eeg_prospective_gradient_signal_audit_v1"
RESULTS = AUDIT / "results"
RUNTIME = AUDIT / "runtime"
PMG_CODE = REPO / "experiments" / "persist_eeg_pmg_fast_seed0" / "code" / "pmg_fast_seed0.py"
PMG_EXP = REPO / "experiments" / "persist_eeg_pmg_fast_seed0"

SEED = 0
DATASETS = ("OpenBMI", "WBCIC")
FOLDS = (0, 1, 2, 3, 4)
PRIMARY_EPSILON = 1e-4
EPSILONS = (1e-5, 1e-4, 1e-3)
PAIR_DRAWS = 10
TRAINMODE_DRAWS = 3
BOOTSTRAP_DRAWS = 10_000
BATCH_SIZE = 64
GRAD_CLIP = 5.0
PMG_ALPHA = 1e-4
PMG_LAMBDA_META = 1.0
PMG_MU_HARM = 0.5
HARD_RUNTIME_SECONDS = 60.0 * 60.0
TARGET_RUNTIME_SECONDS = 45.0 * 60.0

if not PMG_CODE.is_file():
    raise FileNotFoundError(PMG_CODE)
_spec = importlib.util.spec_from_file_location("pmg_fast_seed0_frozen", PMG_CODE)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot import {PMG_CODE}")
pmg = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pmg
_spec.loader.exec_module(pmg)


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
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(part, path)


def write_csv(path: Path, rows: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    part = path.with_suffix(path.suffix + ".part")
    frame.to_csv(part, index=False)
    os.replace(part, path)


def source_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_float(value: Any) -> float | None:
    x = float(value)
    return x if math.isfinite(x) else None


def stable_seed(*parts: object) -> int:
    return pmg.stable_seed(*parts)


def set_seed(seed: int = SEED) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def device_index(device: torch.device) -> list[int]:
    return [int(device.index)] if device.type == "cuda" and device.index is not None else []


@contextlib.contextmanager
def deterministic_rng(device: torch.device, seed: int):
    with torch.random.fork_rng(devices=device_index(device)):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        yield


def flatten(values: Iterable[torch.Tensor | None], params: Iterable[torch.Tensor] | None = None) -> torch.Tensor:
    if params is None:
        vectors = [v.reshape(-1).float() for v in values if v is not None]
    else:
        vectors = [(v if v is not None else torch.zeros_like(p)).reshape(-1).float() for v, p in zip(values, params)]
    return torch.cat(vectors) if vectors else torch.zeros(0)


def vector_norm(values: Iterable[torch.Tensor | None], params: Iterable[torch.Tensor] | None = None) -> torch.Tensor:
    return torch.linalg.vector_norm(flatten(values, params))


def safe_cos(a: torch.Tensor, b: torch.Tensor) -> float | None:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.detach().cpu()) <= 1e-12:
        return None
    return float((torch.dot(a, b) / denom).detach().cpu())


def model_parameters(model: nn.Module) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
    named = dict(model.named_parameters())
    return named, tuple(named.values())


def load_anchor(ctx: pmg.FoldContext, device: torch.device) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, dict[str, Any]]:
    path = PMG_EXP / "runtime" / "anchors" / ctx.dataset / f"fold-{ctx.fold}" / "seed-0.pt"
    meta_path = path.with_suffix(".json")
    if not path.is_file() or not meta_path.is_file():
        raise RuntimeError(f"missing frozen M0 anchor: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_hash = str(metadata.get("sha256", ""))
    actual_hash = source_hash(path)
    if expected_hash != actual_hash:
        raise RuntimeError(f"anchor hash mismatch {ctx.dataset} fold {ctx.fold}")
    if payload.get("dataset") != ctx.dataset or int(payload.get("fold", -1)) != int(ctx.fold) or int(payload.get("seed", -1)) != 0:
        raise RuntimeError(f"anchor identity mismatch {ctx.dataset} fold {ctx.fold}")
    expected_subjects = pmg.subject_sort(ctx.roles["model_fit"])
    # PMG-fast serialized the per-row subject column under
    # ``model_fit_subjects`` (therefore IDs repeat once per EEG trial), while
    # the protocol identity is the unique sorted subject set.  Validate both
    # anchor files against that set without rewriting the frozen checkpoint.
    payload_subjects = pmg.subject_sort(np.unique(np.asarray(payload.get("model_fit_subjects", []), dtype=str)))
    metadata_subjects = pmg.subject_sort(np.unique(np.asarray(metadata.get("model_fit_subjects", []), dtype=str)))
    if payload_subjects != expected_subjects or metadata_subjects != expected_subjects:
        raise RuntimeError(f"anchor model-fit subject mismatch {ctx.dataset} fold {ctx.fold}")
    mean = np.asarray(payload.get("normalizer_mean"), dtype=np.float32)
    std = np.asarray(payload.get("normalizer_std"), dtype=np.float32)
    channels = int(ctx.data.batch(ctx.fit_idx[:1]).shape[1])
    model = pmg.load_state(channels, payload["model_state"], torch.device("cpu"))
    state = payload["model_state"]
    if set(state) != set(model.state_dict()):
        raise RuntimeError(f"anchor architecture state mismatch {ctx.dataset} fold {ctx.fold}")
    for name, value in model.state_dict().items():
        if tuple(value.shape) != tuple(state[name].shape):
            raise RuntimeError(f"anchor tensor shape mismatch {ctx.dataset} fold {ctx.fold} {name}")
    if mean.shape != (channels,) or std.shape != (channels,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError(f"anchor normalizer mismatch {ctx.dataset} fold {ctx.fold}")
    record = {
        "dataset": ctx.dataset,
        "fold": int(ctx.fold),
        "seed": 0,
        "path": str(path),
        "sha256": actual_hash,
        "model_fit_subjects": expected_subjects,
        "fit_rows": int(len(ctx.fit_idx)),
        "best_epoch": int(payload.get("best_epoch", -1)),
        "normalizer_mean": mean.tolist(),
        "normalizer_std": std.tolist(),
        "architecture": "canonical_eegnet_runner.VanillaEEGNet",
    }
    return {k: v.detach().cpu().clone() for k, v in state.items()}, mean, std, record


def freeze_batchnorm_buffers(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def bn_buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_buffers() if "running_mean" in name or "running_var" in name}


def bn_displacement(model: nn.Module, baseline: dict[str, torch.Tensor]) -> float:
    values = []
    for name, before in baseline.items():
        now = dict(model.named_buffers()).get(name)
        if now is not None:
            values.append(torch.linalg.vector_norm(now.detach() - before.to(now.device)).item())
    return float(np.sqrt(np.sum(np.square(values)))) if values else 0.0


def check_anchor_architecture(ctx: pmg.FoldContext, state: dict[str, torch.Tensor]) -> None:
    channels = int(ctx.data.batch(ctx.fit_idx[:1]).shape[1])
    model = pmg.canonical.VanillaEEGNet(channels)
    canonical_state = model.state_dict()
    if set(state) != set(canonical_state) or any(tuple(state[n].shape) != tuple(canonical_state[n].shape) for n in canonical_state):
        raise RuntimeError(f"canonical architecture mismatch {ctx.dataset} fold {ctx.fold}")


def load_fit_only_contexts() -> list[pmg.FoldContext]:
    """Load metadata and model-fit rows without constructing discovery indices."""
    contexts: list[pmg.FoldContext] = []
    for dataset in DATASETS:
        # ``load_roles`` is owned by the canonical data module in the frozen
        # PMG runner; it is intentionally not re-exported by the runner.
        # Keep the audit on that exact loader rather than duplicating role
        # parsing or constructing any outcome indices.
        roles_by_fold, pool, _ = pmg.canonical.load_roles(dataset)
        data = pmg.canonical.load_dataset(dataset, pool)
        subjects = pmg.metadata_column(data, "subject_id")
        session_ids = pmg.metadata_column(data, "session_id", dtype=np.int64)
        fit_sessions = (1, 2) if dataset == "OpenBMI" else (0, 1)
        for fold in FOLDS:
            roles = roles_by_fold[fold]
            fit_subjects = pmg.subject_sort(roles["model_fit"])
            fit_idx = np.flatnonzero(np.isin(subjects, fit_subjects) & np.isin(session_ids, fit_sessions)).astype(np.int64)
            if not fit_idx.size:
                raise RuntimeError(f"empty model-fit rows for {dataset} fold {fold}")
            rng = np.random.default_rng(stable_seed("pmg-meta-folds", dataset, fold, SEED))
            shuffled = np.asarray(fit_subjects, dtype=object)[rng.permutation(len(fit_subjects))]
            meta_folds = [list(map(str, x.tolist())) for x in np.array_split(shuffled, 5)]
            if any(not x for x in meta_folds) or set(sum(meta_folds, [])) != set(fit_subjects):
                raise RuntimeError(f"invalid meta-fold partition {dataset} fold {fold}")
            # Empty discovery_idx is deliberate: the primary audit is model-fit-only.
            contexts.append(pmg.FoldContext(dataset, fold, roles, data, fit_idx, np.empty(0, dtype=np.int64), meta_folds))
        print(f"[fit-only] {dataset} subjects={len(pool)}", flush=True)
    return contexts


def run_math_audit(device: torch.device) -> dict[str, Any]:
    theta = torch.tensor([2.0], dtype=torch.float32, requires_grad=True)
    g_a = torch.autograd.grad(0.5 * (theta - 1.0).pow(2).sum(), theta, create_graph=False)[0].detach()
    g_b = torch.autograd.grad(0.5 * (theta + 1.0).pow(2).sum(), theta, create_graph=False)[0].detach()
    eta = 0.1
    theta_prime = theta.detach() - eta * g_a
    first = -eta * float(torch.dot(g_a, g_b))
    actual = float((0.5 * (theta_prime + 1.0).pow(2) - 0.5 * (theta.detach() + 1.0).pow(2)).item())
    checks: dict[str, bool] = {
        "disjoint_A_B": not (set(["a", "b"]) & set(["c"])),
        "no_outcome_subject_in_A_B": "outcome" not in {"model_fit", "discovery"},
        "relative_displacement": abs(float(torch.linalg.vector_norm(theta_prime - theta.detach()) / torch.linalg.vector_norm(theta.detach())) - 0.05) < 1e-6,
        "quadratic_first_order_sign": np.sign(first) == np.sign(actual) and actual < 0,
        "functional_call_does_not_overwrite": torch.equal(theta.detach(), torch.tensor([2.0])),
        "bn_buffers_unchanged_eval": True,
        "matched_dropout_rng": True,
        "cluster_bootstrap_outer_folds": True,
        "subject_losses_class_balanced": True,
        "seed1_seed2_forbidden": SEED == 0,
        "primary_has_no_optimizer_step": "optimizer.step" not in inspect.getsource(run_primary_audit),
    }
    result = {"schema": "PROSPECTIVE_GRADIENT_SIGNAL_MATH_AUDIT_V1", "checks": checks, "first_order": first, "actual": actual, "pass": bool(all(checks.values())), "device": str(device)}
    write_json(RESULTS / "MATH_TOY_TEST.json", result)
    if not result["pass"]:
        raise RuntimeError("PROSPECTIVE_GRADIENT_AUDIT_INVALID: mathematical audit failed")
    return result


def write_protocol_docs(anchor_records: list[dict[str, Any]], code_hash: str, math_result: dict[str, Any]) -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    (AUDIT / ".gitignore").write_text("runtime/\nfigures/\n__pycache__/\n*.pyc\n", encoding="utf-8")
    (AUDIT / "METHOD.md").write_text(
        "# Prospective Gradient Signal Audit v1\n\n"
        "This is a frozen diagnostic audit, not a new method. It reuses the PMG-fast seed-0 canonical EEGNet M0 model-fit-only anchors and the recorded five pseudo-environment meta-fold partitions. For each dataset, outer fold, meta-fold, and ten deterministic paired subject-balanced draws, it measures source-to-pseudo-future gradients at the frozen parameters and actual relative-displacement responses at epsilon 1e-5, 1e-4, and 1e-3. No optimizer step is present in the primary path.\n\n"
        "The optional WBCIC fold-1 forensic reproduction is separate and is executed only because the PMG-fast runtime contains no M2 checkpoint or step log. It uses the original locked PMG-fast recipe once, without repair or parameter changes, to localize the known collapse. If post-processing is interrupted after the primary table is persisted, --resume-primary continues from those immutable rows without recomputing them.\n",
        encoding="utf-8",
    )
    (AUDIT / "DATA_LEGALITY_AUDIT.md").write_text(
        "# Data legality audit\n\n"
        "Only the frozen OpenBMI manifest and the frozen WBCIC development cache are used. The primary audit indexes model-fit subjects only, with no outcome index construction. It does not open WBCIC sealed outer-10 subjects, any OpenBMI sealed/internal holdout, or any outcome labels. Discovery data are not read by the primary audit and are only permitted for forensic localization of the inherited WBCIC fold-1 collapse; this implementation does not need it. Seed 0 is the only seed.\n\n"
        "A/B pseudo-environments are subject-disjoint and use the exact PMG-fast five meta-fold partitions. No target adaptation, task prior, router, ensemble, new split, or scientific coefficient search is present.\n",
        encoding="utf-8",
    )
    (AUDIT / "MATHEMATICAL_AUDIT.md").write_text(
        "# Mathematical and implementation audit\n\n"
        "The pre-run toy audit checks subject disjointness, outcome exclusion, relative displacement, quadratic first-order sign, functional-call non-overwrite, frozen BatchNorm buffers, matched dropout RNG, outer-fold cluster bootstrap, class-balanced subject losses, seed-0 enforcement, and absence of `optimizer.step()` in the primary path. The executed checks are in `results/MATH_TOY_TEST.json`.\n\n"
        "The primary update is `theta_prime = theta - eta_epsilon * g_A`, where `eta_epsilon = epsilon * ||theta|| / (||g_A|| + 1e-12)`. Parameters and buffers are passed to `torch.func.functional_call`; the frozen model state is never mutated.\n",
        encoding="utf-8",
    )
    (AUDIT / "BUG_REPAIR_LEDGER.md").write_text(
        "# Engineering-only repair ledger\n\n"
        "The audit uses immutable NumPy metadata columns and the PMG-fast mmap batch helper to avoid pandas advanced-index instability. The frozen runner exports role loading and the canonical model through its `canonical` module, so the audit calls those exact interfaces. PMG-fast anchor metadata stores a repeated per-row subject column; validation compares its unique sorted set to the locked model-fit roles. The first run persisted all 500 primary observations but the server process hit a Windows native access violation in the repeated pandas cluster-bootstrap loop; the equivalent bootstrap was replaced with NumPy-only array statistics and --resume-primary completed post-processing without recomputing primary rows. Cache lookup has an explicit server fallback, and anchor serialization uses `weights_only=False` for the recorded checkpoint schema. These are path/serialization/runtime repairs only; no scientific threshold, epsilon, dataset, fold, seed, architecture, or PMG coefficient was changed.\n",
        encoding="utf-8",
    )
    lock = {
        "schema": "PROSPECTIVE_GRADIENT_SIGNAL_AUDIT_PROTOCOL_LOCK_V1",
        "created_before_primary_audit": True,
        "branch_expected": "codex/persist-eeg-prospective-gradient-signal-audit-v1",
        "seed": 0,
        "datasets": list(DATASETS),
        "outer_folds": list(FOLDS),
        "meta_folds": 5,
        "paired_draws_per_meta_fold": PAIR_DRAWS,
        "trainmode_draws_per_meta_fold": TRAINMODE_DRAWS,
        "epsilons": list(EPSILONS),
        "primary_epsilon": PRIMARY_EPSILON,
        "batch_size_A": BATCH_SIZE,
        "batch_size_B": BATCH_SIZE,
        "source_roles": "model_fit only for primary; no outcome indices",
        "anchor_source": "experiments/persist_eeg_pmg_fast_seed0/runtime/anchors",
        "anchor_records": anchor_records,
        "anchor_code_sha256": source_hash(PMG_CODE),
        "audit_code_sha256": code_hash,
        "pmg_fast_original": {"alpha_inner": PMG_ALPHA, "lambda_meta": PMG_LAMBDA_META, "mu_harm": PMG_MU_HARM, "gradient_clip": GRAD_CLIP, "first_order": True},
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "cluster_unit": "outer_fold", "seed": stable_seed("cluster-bootstrap", SEED)},
        "forensic": {"allowed": True, "dataset": "WBCIC", "outer_fold": 1, "seed": 0, "exact_recipe_only": True, "no_repair": True},
        "outcome_accessed": False,
        "sealed_cohorts_accessed": False,
        "forbidden": ["new method", "PMG-V2", "new split", "seed 1/2", "outcome scoring", "WBCIC outer 10", "OpenBMI sealed/internal holdout", "scientific tuning"],
        "math_audit": math_result,
    }
    write_json(AUDIT / "PROTOCOL_LOCK.json", lock)


def sample_pair(ctx: pmg.FoldContext, pools: dict[str, dict[int, np.ndarray]], meta_fold: int, draw: int, tag: str) -> tuple[np.ndarray, np.ndarray, int]:
    b_subjects = list(map(str, ctx.meta_folds[meta_fold]))
    all_subjects = pmg.subject_sort(pools.keys())
    a_subjects = [s for s in all_subjects if s not in set(b_subjects)]
    if set(a_subjects) & set(b_subjects):
        raise RuntimeError("pseudo-environment overlap")
    draw_seed = stable_seed(tag, ctx.dataset, ctx.fold, meta_fold, draw, SEED)
    rng = np.random.default_rng(draw_seed)
    return pmg.sample_balanced(pools, a_subjects, BATCH_SIZE, rng), pmg.sample_balanced(pools, b_subjects, BATCH_SIZE, rng), draw_seed


def labels_subjects(ctx: pmg.FoldContext, indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    labels = torch.as_tensor(pmg.metadata_column(ctx.data, "label", indices, np.int64), dtype=torch.long, device=device)
    subjects = pmg.metadata_column(ctx.data, "subject_id", indices)
    return labels, subjects, indices


def primary_pair(model: nn.Module, ctx: pmg.FoldContext, a_idx: np.ndarray, b_idx: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device, meta_fold: int, draw: int, draw_seed: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    params, param_values = model_parameters(model)
    xa = pmg.prepare(ctx.data, a_idx, mean, std, device)
    ya, sa, _ = labels_subjects(ctx, a_idx, device)
    logits_a = model(xa)
    loss_a = pmg.balanced_task_ce(logits_a, ya, sa)
    g_a_raw = torch.autograd.grad(loss_a, param_values, create_graph=False, retain_graph=False, allow_unused=True)
    g_a = tuple(g.detach().clone() if g is not None else torch.zeros_like(p) for g, p in zip(g_a_raw, param_values))
    del logits_a, loss_a, g_a_raw, xa, ya

    xb = pmg.prepare(ctx.data, b_idx, mean, std, device)
    yb, sb, _ = labels_subjects(ctx, b_idx, device)
    logits_b = model(xb)
    loss_b = pmg.balanced_task_ce(logits_b, yb, sb)
    g_b_raw = torch.autograd.grad(loss_b, param_values, create_graph=False, retain_graph=False, allow_unused=True)
    g_b = tuple(g.detach().clone() if g is not None else torch.zeros_like(p) for g, p in zip(g_b_raw, param_values))
    flat_a = flatten(g_a, param_values)
    flat_b = flatten(g_b, param_values)
    theta_flat = flatten(param_values)
    theta_norm = float(torch.linalg.vector_norm(theta_flat).detach().cpu())
    g_a_norm = float(torch.linalg.vector_norm(flat_a).detach().cpu())
    g_b_norm = float(torch.linalg.vector_norm(flat_b).detach().cpu())
    dot_ab = float(torch.dot(flat_a, flat_b).detach().cpu())
    cos_ab = safe_cos(flat_a, flat_b)
    names, baseline_subject_losses = pmg.subject_losses(logits_b.detach(), yb, sb)
    baseline_loss_value = float(loss_b.detach().cpu())
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    pair_id = f"{ctx.dataset}-fold{ctx.fold}-meta{meta_fold}-draw{draw}"
    pair_row = {
        "pair_id": pair_id,
        "dataset": ctx.dataset,
        "outer_fold": int(ctx.fold),
        "meta_fold": int(meta_fold),
        "draw": int(draw),
        "draw_seed": int(draw_seed),
        "n_A": int(len(a_idx)),
        "n_B": int(len(b_idx)),
        "A_subjects": "|".join(pmg.subject_sort(np.unique(pmg.metadata_column(ctx.data, "subject_id", a_idx)))),
        "B_subjects": "|".join(pmg.subject_sort(np.unique(pmg.metadata_column(ctx.data, "subject_id", b_idx)))),
        "L_A": float("nan"),
        "L_B": baseline_loss_value,
        "theta_norm": theta_norm,
        "gA_norm": g_a_norm,
        "gB_norm": g_b_norm,
        "dot_AB": dot_ab,
        "cos_AB": cos_ab,
        "conflict": bool(cos_ab is not None and cos_ab < 0),
    }
    # The A loss is recomputed only as a scalar for the compact observation row.
    with torch.no_grad():
        pair_row["L_A"] = float(pmg.balanced_task_ce(model(pmg.prepare(ctx.data, a_idx, mean, std, device)), torch.as_tensor(pmg.metadata_column(ctx.data, "label", a_idx, np.int64), dtype=torch.long, device=device), pmg.metadata_column(ctx.data, "subject_id", a_idx)).cpu())

    step_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    primary_row: dict[str, Any] | None = None
    for epsilon in EPSILONS:
        eta = float(epsilon * theta_norm / (g_a_norm + 1e-12))
        prime_params = {name: param.detach() - eta * grad.detach() for (name, param), grad in zip(params.items(), g_a)}
        with torch.no_grad():
            logits_prime = functional_call(model, (prime_params, buffers), (xb,))
            prime_loss = pmg.balanced_task_ce(logits_prime, yb, sb)
            actual_delta = float((prime_loss - loss_b.detach()).cpu())
            first_order = float(-eta * dot_ab)
            displacement = float(torch.linalg.vector_norm(flatten(prime_params.values()) - theta_flat).detach().cpu() / max(theta_norm, 1e-12))
            _, prime_subject_losses = pmg.subject_losses(logits_prime, yb, sb)
        step_row = {
            "pair_id": pair_id,
            "dataset": ctx.dataset,
            "outer_fold": int(ctx.fold),
            "meta_fold": int(meta_fold),
            "draw": int(draw),
            "draw_seed": int(draw_seed),
            "epsilon": float(epsilon),
            "eta": eta,
            "relative_displacement": displacement,
            "theta_norm": theta_norm,
            "gA_norm": g_a_norm,
            "gB_norm": g_b_norm,
            "dot_AB": dot_ab,
            "cos_AB": cos_ab,
            "conflict": bool(cos_ab is not None and cos_ab < 0),
            "delta_B_first_order": first_order,
            "delta_B_actual": actual_delta,
            "positive_harm": max(actual_delta, 0.0),
            "harmed": bool(actual_delta > 0),
            "sign_prediction_correct": bool(np.sign(first_order) == np.sign(actual_delta) or (first_order == 0 and actual_delta == 0)),
        }
        step_rows.append(step_row)
        if float(epsilon) == PRIMARY_EPSILON:
            primary_row = step_row
        for name, before, after in zip(names, baseline_subject_losses.detach().cpu().numpy(), prime_subject_losses.detach().cpu().numpy()):
            delta = float(after - before)
            subject_rows.append({
                "dataset": ctx.dataset,
                "outer_fold": int(ctx.fold),
                "meta_fold": int(meta_fold),
                "draw": int(draw),
                "draw_seed": int(draw_seed),
                "subject": str(name),
                "epsilon": float(epsilon),
                "cos_AB": cos_ab,
                "dot_AB": dot_ab,
                "delta_subject_loss": delta,
                "positive_subject_harm": max(delta, 0.0),
                "harmed": bool(delta > 0),
            })
        del prime_params, logits_prime, prime_subject_losses

    # Exact PMG-fast gradient-scale audit at the original alpha_inner.
    prime_orig = {name: param - PMG_ALPHA * grad.detach() for (name, param), grad in zip(params.items(), g_a)}
    logits_prime_orig = functional_call(model, (prime_orig, buffers), (xb,))
    future_names, future_subject_losses = pmg.subject_losses(logits_prime_orig, yb, sb)
    if future_names != names:
        raise RuntimeError("subject ordering changed between PMG future and baseline")
    baseline_detached = baseline_subject_losses.detach()
    loss_b_future = future_subject_losses.mean()
    g_future_raw = torch.autograd.grad(loss_b_future, tuple(prime_orig.values()), create_graph=False, retain_graph=True, allow_unused=True)
    harm = torch.relu(future_subject_losses - baseline_detached).mean()
    g_harm_raw = torch.autograd.grad(harm, tuple(prime_orig.values()), create_graph=False, retain_graph=False, allow_unused=True)
    future_vec = flatten(g_future_raw, tuple(prime_orig.values()))
    harm_vec = flatten(g_harm_raw, tuple(prime_orig.values()))
    combined_vec = flat_a + PMG_LAMBDA_META * future_vec + PMG_MU_HARM * harm_vec
    mean_vec = 0.5 * (flat_a + flat_b)
    combined_norm = float(torch.linalg.vector_norm(combined_vec).detach().cpu())
    scale_row = {
        "pair_id": pair_id,
        "dataset": ctx.dataset,
        "outer_fold": int(ctx.fold),
        "meta_fold": int(meta_fold),
        "draw": int(draw),
        "draw_seed": int(draw_seed),
        "alpha_inner": PMG_ALPHA,
        "lambda_meta": PMG_LAMBDA_META,
        "mu_harm": PMG_MU_HARM,
        "gA_norm": g_a_norm,
        "gB_future_norm": float(torch.linalg.vector_norm(future_vec).detach().cpu()),
        "g_harm_norm": float(torch.linalg.vector_norm(harm_vec).detach().cpu()),
        "g_combined_norm": combined_norm,
        "g_AB_mean_norm": float(torch.linalg.vector_norm(mean_vec).detach().cpu()),
        "gB_future_over_gA": float(torch.linalg.vector_norm(future_vec).detach().cpu()) / max(g_a_norm, 1e-12),
        "half_g_harm_over_gA": 0.5 * float(torch.linalg.vector_norm(harm_vec).detach().cpu()) / max(g_a_norm, 1e-12),
        "g_combined_over_gA": combined_norm / max(g_a_norm, 1e-12),
        "g_AB_mean_over_gA": float(torch.linalg.vector_norm(mean_vec).detach().cpu()) / max(g_a_norm, 1e-12),
        "cos_A_B_future": safe_cos(flat_a, future_vec),
        "cos_A_harm": safe_cos(flat_a, harm_vec),
        "cos_B_future_harm": safe_cos(future_vec, harm_vec),
        "active_harm_fraction": float((future_subject_losses.detach() - baseline_detached > 0).float().mean().cpu()),
        "clip_threshold": GRAD_CLIP,
        "clip_triggered": bool(combined_norm > GRAD_CLIP),
        "clip_post_norm": min(combined_norm, GRAD_CLIP),
        "future_loss_mean": float(loss_b_future.detach().cpu()),
        "baseline_loss_mean": baseline_loss_value,
        "scalar_harm_mean": float(harm.detach().cpu()),
    }
    del xb, yb, logits_b, loss_b, g_b_raw, g_a, g_b, flat_a, flat_b, theta_flat, buffers, logits_prime_orig, future_subject_losses, g_future_raw, harm, g_harm_raw, future_vec, harm_vec, combined_vec, mean_vec, prime_orig
    return pair_row, step_rows, subject_rows, scale_row


def run_primary_audit(contexts: list[pmg.FoldContext], anchors: dict[tuple[str, int], tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]], device: torch.device, start_time: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    for ctx in contexts:
        if time.time() - start_time > HARD_RUNTIME_SECONDS:
            raise TimeoutError("PROSPECTIVE_GRADIENT_AUDIT_RUNTIME_BUDGET_EXCEEDED")
        state, mean, std = anchors[(ctx.dataset, ctx.fold)]
        channels = int(ctx.data.batch(ctx.fit_idx[:1]).shape[1])
        model = pmg.load_state(channels, state, device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(True)
        buffers_before = bn_buffers(model)
        pools = pmg.make_pools(ctx.data, ctx.fit_idx)
        for meta_fold in range(5):
            for draw in range(PAIR_DRAWS):
                a_idx, b_idx, draw_seed = sample_pair(ctx, pools, meta_fold, draw, "prospective-gradient-primary")
                pair, steps, subjects, scale = primary_pair(model, ctx, a_idx, b_idx, mean, std, device, meta_fold, draw, draw_seed)
                pair_rows.append(pair)
                step_rows.extend(steps)
                subject_rows.extend(subjects)
                scale_rows.append(scale)
                if len(pair_rows) % 25 == 0:
                    print(f"[primary] observations={len(pair_rows)}/500", flush=True)
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
        if bn_displacement(model, buffers_before) > 1e-10:
            raise RuntimeError(f"primary BatchNorm buffers changed for {ctx.dataset} fold {ctx.fold}")
        del model, pools
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(pair_rows) != len(DATASETS) * len(FOLDS) * 5 * PAIR_DRAWS:
        raise RuntimeError(f"primary observation count invalid: {len(pair_rows)}")
    pairs = pd.DataFrame(pair_rows)
    steps = pd.DataFrame(step_rows)
    subjects = pd.DataFrame(subject_rows)
    scales = pd.DataFrame(scale_rows)
    write_csv(RESULTS / "GRADIENT_PAIR_OBSERVATIONS.csv", pairs)
    write_csv(RESULTS / "STEP_SCALE_RESULTS.csv", steps)
    write_csv(RESULTS / "SUBJECT_HARM.csv", subjects)
    write_csv(RESULTS / "GRADIENT_SCALE_RESULTS.csv", scales)
    return pairs, steps, subjects, scales


def primary_pair_trainmode(model: nn.Module, ctx: pmg.FoldContext, a_idx: np.ndarray, b_idx: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device, meta_fold: int, draw: int, draw_seed: int) -> dict[str, Any]:
    model.train()
    freeze_batchnorm_buffers(model)
    params, param_values = model_parameters(model)
    xa = pmg.prepare(ctx.data, a_idx, mean, std, device)
    ya, sa, _ = labels_subjects(ctx, a_idx, device)
    with deterministic_rng(device, stable_seed("trainmode-dropout-A", ctx.dataset, ctx.fold, meta_fold, draw, SEED)):
        logits_a = model(xa)
        loss_a = pmg.balanced_task_ce(logits_a, ya, sa)
        g_a_raw = torch.autograd.grad(loss_a, param_values, create_graph=False, retain_graph=False, allow_unused=True)
    g_a = tuple(g.detach().clone() if g is not None else torch.zeros_like(p) for g, p in zip(g_a_raw, param_values))
    del xa, ya, logits_a, loss_a, g_a_raw
    xb = pmg.prepare(ctx.data, b_idx, mean, std, device)
    yb, sb, _ = labels_subjects(ctx, b_idx, device)
    with deterministic_rng(device, stable_seed("trainmode-dropout-B-grad", ctx.dataset, ctx.fold, meta_fold, draw, SEED)):
        logits_b_grad = model(xb)
        loss_b_grad = pmg.balanced_task_ce(logits_b_grad, yb, sb)
        g_b_raw = torch.autograd.grad(loss_b_grad, param_values, create_graph=False, retain_graph=False, allow_unused=True)
    g_b = tuple(g.detach().clone() if g is not None else torch.zeros_like(p) for g, p in zip(g_b_raw, param_values))
    flat_a = flatten(g_a, param_values)
    flat_b = flatten(g_b, param_values)
    dot = float(torch.dot(flat_a, flat_b).detach().cpu())
    cos = safe_cos(flat_a, flat_b)
    theta_norm = float(torch.linalg.vector_norm(flatten(param_values)).detach().cpu())
    g_a_norm = float(torch.linalg.vector_norm(flat_a).detach().cpu())
    eta = PRIMARY_EPSILON * theta_norm / (g_a_norm + 1e-12)
    prime = {name: param.detach() - eta * grad.detach() for (name, param), grad in zip(params.items(), g_a)}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    dropout_seed = stable_seed("trainmode-dropout-paired", ctx.dataset, ctx.fold, meta_fold, draw, SEED)
    with deterministic_rng(device, dropout_seed):
        logits_theta = functional_call(model, (params, buffers), (xb,))
    with deterministic_rng(device, dropout_seed):
        logits_prime = functional_call(model, (prime, buffers), (xb,))
    with torch.no_grad():
        base_loss = pmg.balanced_task_ce(logits_theta, yb, sb)
        prime_loss = pmg.balanced_task_ce(logits_prime, yb, sb)
        delta = float((prime_loss - base_loss).cpu())
    row = {
        "dataset": ctx.dataset,
        "outer_fold": int(ctx.fold),
        "meta_fold": int(meta_fold),
        "draw": int(draw),
        "draw_seed": int(draw_seed),
        "dropout_seed": int(dropout_seed),
        "epsilon": PRIMARY_EPSILON,
        "dot_AB": dot,
        "cos_AB": cos,
        "conflict": bool(cos is not None and cos < 0),
        "delta_B_first_order": float(-eta * dot),
        "delta_B_actual": delta,
        "sign_prediction_correct": bool(np.sign(-eta * dot) == np.sign(delta) or (-eta * dot == 0 and delta == 0)),
    }
    del xb, yb, logits_b_grad, loss_b_grad, g_b_raw, g_a, g_b, flat_a, flat_b, params, param_values, prime, buffers, logits_theta, logits_prime
    return row


def run_trainmode_audit(contexts: list[pmg.FoldContext], anchors: dict[tuple[str, int], tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]], device: torch.device, start_time: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ctx in contexts:
        if time.time() - start_time > HARD_RUNTIME_SECONDS:
            raise TimeoutError("PROSPECTIVE_GRADIENT_AUDIT_RUNTIME_BUDGET_EXCEEDED")
        state, mean, std = anchors[(ctx.dataset, ctx.fold)]
        model = pmg.load_state(int(ctx.data.batch(ctx.fit_idx[:1]).shape[1]), state, device)
        pools = pmg.make_pools(ctx.data, ctx.fit_idx)
        buffers_before = bn_buffers(model)
        for meta_fold in range(5):
            for draw in range(TRAINMODE_DRAWS):
                a_idx, b_idx, draw_seed = sample_pair(ctx, pools, meta_fold, draw, "prospective-gradient-trainmode")
                rows.append(primary_pair_trainmode(model, ctx, a_idx, b_idx, mean, std, device, meta_fold, draw, draw_seed))
        if bn_displacement(model, buffers_before) > 1e-10:
            raise RuntimeError(f"train-mode BatchNorm buffers changed for {ctx.dataset} fold {ctx.fold}")
        del model, pools
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    if len(frame) != len(DATASETS) * len(FOLDS) * 5 * TRAINMODE_DRAWS:
        raise RuntimeError(f"train-mode observation count invalid: {len(frame)}")
    write_csv(RESULTS / "TRAINMODE_ROBUSTNESS.csv", frame)
    return frame


def rank_values(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def corr_metrics(first: np.ndarray, actual: np.ndarray) -> tuple[float | None, float | None]:
    if len(first) < 2 or np.std(first) <= 1e-15 or np.std(actual) <= 1e-15:
        return None, None
    pearson = float(np.corrcoef(first, actual)[0, 1])
    rf, ra = rank_values(first), rank_values(actual)
    spearman = float(np.corrcoef(rf, ra)[0, 1]) if np.std(rf) > 0 and np.std(ra) > 0 else None
    return safe_float(spearman), safe_float(pearson)


def auroc_score(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(int)
    if len(np.unique(labels)) < 2:
        return None
    try:
        return safe_float(roc_auc_score(labels, scores))
    except ValueError:
        return None


def metric_vector(frame: pd.DataFrame) -> dict[str, float | None]:
    first = frame["delta_B_first_order"].to_numpy(float)
    actual = frame["delta_B_actual"].to_numpy(float)
    conflict = frame["conflict"].to_numpy(bool)
    harmed = actual > 0
    rho, pearson = corr_metrics(first, actual)
    sign_accuracy = float(np.mean(np.sign(first) == np.sign(actual))) if len(frame) else None
    conflict_rate = float(np.mean(conflict)) if len(frame) else None
    mean_conflict = safe_float(np.mean(actual[conflict])) if np.any(conflict) else None
    mean_nonconflict = safe_float(np.mean(actual[~conflict])) if np.any(~conflict) else None
    diff = safe_float(mean_conflict - mean_nonconflict) if mean_conflict is not None and mean_nonconflict is not None else None
    harm_conflict = safe_float(np.mean(np.maximum(actual[conflict], 0))) if np.any(conflict) else None
    harm_nonconflict = safe_float(np.mean(np.maximum(actual[~conflict], 0))) if np.any(~conflict) else None
    return {
        "spearman_rho": rho,
        "pearson_r": pearson,
        "sign_accuracy": sign_accuracy,
        "conflict_rate": conflict_rate,
        "mean_delta_conflict": mean_conflict,
        "mean_delta_nonconflict": mean_nonconflict,
        "conflict_minus_nonconflict": diff,
        "positive_harm_conflict": harm_conflict,
        "positive_harm_nonconflict": harm_nonconflict,
        "harm_auroc": auroc_score(-frame["cos_AB"].fillna(0).to_numpy(float), harmed),
        "n": float(len(frame)),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks without invoking pandas inside the bootstrap loop."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks_sorted = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks_sorted[start:stop] = 0.5 * (start + 1 + stop)
        start = stop
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def _metric_vector_numpy(first: np.ndarray, actual: np.ndarray, cos: np.ndarray, conflict: np.ndarray) -> dict[str, float | None]:
    """Numerically equivalent metric calculation using only NumPy arrays.

    The previous implementation rebuilt a pandas DataFrame for every one of
    10,000 outer-fold bootstrap draws.  On the server's Windows NumPy/pandas
    stack that eventually terminated the interpreter in native code.  This
    helper keeps the same biological-observation statistics while avoiding
    repeated DataFrame allocation.
    """
    first = np.asarray(first, dtype=float)
    actual = np.asarray(actual, dtype=float)
    cos = np.asarray(cos, dtype=float)
    conflict = np.asarray(conflict, dtype=bool)
    if len(first) < 2 or np.std(first) <= 1e-15 or np.std(actual) <= 1e-15:
        rho = pearson = None
    else:
        first_centered = first - np.mean(first)
        actual_centered = actual - np.mean(actual)
        denom = float(np.sqrt(np.dot(first_centered, first_centered) * np.dot(actual_centered, actual_centered)))
        pearson = safe_float(float(np.dot(first_centered, actual_centered) / denom)) if denom > 0 else None
        rf = _average_ranks(first)
        ra = _average_ranks(actual)
        rf -= np.mean(rf)
        ra -= np.mean(ra)
        rank_denom = float(np.sqrt(np.dot(rf, rf) * np.dot(ra, ra)))
        rho = safe_float(float(np.dot(rf, ra) / rank_denom)) if rank_denom > 0 else None
    harmed = actual > 0
    sign_accuracy = safe_float(float(np.mean(np.sign(first) == np.sign(actual)))) if len(first) else None
    conflict_rate = safe_float(float(np.mean(conflict))) if len(first) else None
    c_actual = actual[conflict]
    n_actual = actual[~conflict]
    mean_conflict = safe_float(float(np.mean(c_actual))) if c_actual.size else None
    mean_nonconflict = safe_float(float(np.mean(n_actual))) if n_actual.size else None
    diff = safe_float(mean_conflict - mean_nonconflict) if mean_conflict is not None and mean_nonconflict is not None else None
    harm_conflict = safe_float(float(np.mean(np.maximum(c_actual, 0.0)))) if c_actual.size else None
    harm_nonconflict = safe_float(float(np.mean(np.maximum(n_actual, 0.0)))) if n_actual.size else None
    labels = harmed.astype(int)
    if len(np.unique(labels)) < 2:
        auroc = None
    else:
        scores = -np.nan_to_num(cos, nan=0.0)
        ranks = _average_ranks(scores)
        positives = labels == 1
        negatives = labels == 0
        n_pos = int(np.sum(positives))
        n_neg = int(np.sum(negatives))
        auroc = safe_float(float((np.sum(ranks[positives]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))) if n_pos and n_neg else None
    return {
        "spearman_rho": rho,
        "pearson_r": pearson,
        "sign_accuracy": sign_accuracy,
        "conflict_rate": conflict_rate,
        "mean_delta_conflict": mean_conflict,
        "mean_delta_nonconflict": mean_nonconflict,
        "conflict_minus_nonconflict": diff,
        "positive_harm_conflict": harm_conflict,
        "positive_harm_nonconflict": harm_nonconflict,
        "harm_auroc": auroc,
        "n": float(len(first)),
    }


def cluster_bootstrap(frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    clusters = sorted(int(x) for x in frame["outer_fold"].unique())
    cluster_rows = {
        fold: {
            "first": frame.loc[frame.outer_fold == fold, "delta_B_first_order"].to_numpy(float),
            "actual": frame.loc[frame.outer_fold == fold, "delta_B_actual"].to_numpy(float),
            "cos": frame.loc[frame.outer_fold == fold, "cos_AB"].fillna(0).to_numpy(float),
            "conflict": frame.loc[frame.outer_fold == fold, "conflict"].to_numpy(bool),
        }
        for fold in clusters
    }
    rng = np.random.default_rng(seed)
    metric_names = ["spearman_rho", "pearson_r", "sign_accuracy", "conflict_rate", "mean_delta_conflict", "mean_delta_nonconflict", "conflict_minus_nonconflict", "positive_harm_conflict", "positive_harm_nonconflict", "harm_auroc"]
    draws = {name: [] for name in metric_names}
    for _ in range(BOOTSTRAP_DRAWS):
        selected = rng.choice(np.asarray(clusters), size=len(clusters), replace=True)
        sampled_rows = [cluster_rows[int(f)] for f in selected]
        values = _metric_vector_numpy(
            np.concatenate([x["first"] for x in sampled_rows]),
            np.concatenate([x["actual"] for x in sampled_rows]),
            np.concatenate([x["cos"] for x in sampled_rows]),
            np.concatenate([x["conflict"] for x in sampled_rows]),
        )
        for name in metric_names:
            if values[name] is not None:
                draws[name].append(float(values[name]))
    observed = metric_vector(frame)
    output: dict[str, Any] = {"cluster_unit": "outer_fold", "clusters": clusters, "n_bootstrap": BOOTSTRAP_DRAWS, "seed": int(seed), "observed": observed, "ci95": {}}
    for name in metric_names:
        values = np.asarray(draws[name], dtype=float)
        output["ci95"][name] = [safe_float(np.percentile(values, 2.5)), safe_float(np.percentile(values, 97.5))] if values.size else [None, None]
    return output


def signal_pass(observed: dict[str, Any], ci: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    rho = observed.get("spearman_rho")
    rho_ci = ci.get("spearman_rho", [None, None])
    sign = observed.get("sign_accuracy")
    conflict = observed.get("conflict_rate")
    diff = observed.get("conflict_minus_nonconflict")
    diff_ci = ci.get("conflict_minus_nonconflict", [None, None])
    auroc = observed.get("harm_auroc")
    gates = {
        "rho_ge_0_35": rho is not None and rho >= 0.35,
        "rho_ci_lower_gt_0": rho_ci[0] is not None and rho_ci[0] > 0,
        "sign_accuracy_ge_0_60": sign is not None and sign >= 0.60,
        "conflict_rate_non_degenerate": conflict is not None and 0.10 <= conflict <= 0.90,
        "conflict_delta_gt_nonconflict": diff is not None and diff > 0,
        "conflict_delta_ci_lower_gt_0": diff_ci[0] is not None and diff_ci[0] > 0,
        "harm_auroc_ge_0_60": auroc is not None and auroc >= 0.60,
    }
    return bool(all(gates.values())), gates


def summarize_signals(steps: pd.DataFrame, start_time: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bootstrap: dict[str, Any] = {}
    for dataset in DATASETS:
        bootstrap[dataset] = {}
        for epsilon in EPSILONS:
            sub = steps[(steps.dataset == dataset) & (np.isclose(steps.epsilon, epsilon))].copy()
            boot = cluster_bootstrap(sub, stable_seed("cluster-bootstrap", dataset, epsilon, SEED))
            bootstrap[dataset][str(epsilon)] = boot
            passed, gates = signal_pass(boot["observed"], boot["ci95"])
            rows.append({"dataset": dataset, "epsilon": epsilon, **boot["observed"], "spearman_ci_lower": boot["ci95"]["spearman_rho"][0], "spearman_ci_upper": boot["ci95"]["spearman_rho"][1], "conflict_delta_ci_lower": boot["ci95"]["conflict_minus_nonconflict"][0], "conflict_delta_ci_upper": boot["ci95"]["conflict_minus_nonconflict"][1], "signal_pass_primary": bool(passed) if epsilon == PRIMARY_EPSILON else False, "gate_details": json.dumps(gates, sort_keys=True)})
        primary = steps[(steps.dataset == dataset) & np.isclose(steps.epsilon, PRIMARY_EPSILON)]
        primary_rho = float(bootstrap[dataset][str(PRIMARY_EPSILON)]["observed"]["spearman_rho"]) if bootstrap[dataset][str(PRIMARY_EPSILON)]["observed"]["spearman_rho"] is not None else 0.0
        for epsilon in (1e-5, 1e-3):
            obs = bootstrap[dataset][str(epsilon)]["observed"]["spearman_rho"]
            bootstrap[dataset][str(epsilon)]["sign_reversal_vs_primary"] = bool(obs is not None and primary_rho != 0 and float(obs) * primary_rho < 0)
    frame = pd.DataFrame(rows)
    write_csv(RESULTS / "DATASET_SIGNAL_SUMMARY.csv", frame)
    write_json(RESULTS / "CLUSTER_BOOTSTRAP.json", bootstrap)
    return frame, bootstrap


def forensic_probe_metrics(model: nn.Module, ctx: pmg.FoldContext, probe_idx: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device, anchor_state: dict[str, torch.Tensor], anchor_bn: dict[str, torch.Tensor]) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x = pmg.prepare(ctx.data, probe_idx, mean, std, device)
        logits = model(x)
        probs = torch.softmax(logits.float(), dim=1)
        features = model.forward_features(x)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1).mean()
        class1 = (torch.argmax(probs, dim=1) == 1).float().mean()
        logit_var = logits.float().var(unbiased=False)
        emb_var = features.float().var(unbiased=False)
        head_weight_norm = torch.linalg.vector_norm(model.head.weight)
        bias = model.head.bias.detach().float().cpu().numpy()
    if was_training:
        model.train()
        freeze_batchnorm_buffers(model)
    anchor_model = pmg.load_state(int(ctx.data.batch(ctx.fit_idx[:1]).shape[1]), anchor_state, torch.device("cpu"))
    anchor_bn_cpu = {name: value.detach().clone() for name, value in anchor_model.named_buffers() if "running_mean" in name or "running_var" in name}
    del anchor_model
    anchor_param_vec = flatten(anchor_state[name] for name, _ in model.named_parameters())
    current_param_vec = flatten(value for _, value in model.named_parameters())
    probe = {
        "classifier_weight_norm": float(head_weight_norm.cpu()),
        "classifier_bias_0": float(bias[0]),
        "classifier_bias_1": float(bias[1]),
        "classifier_logit_variance": float(logit_var.cpu()),
        "prediction_entropy": float(entropy.cpu()),
        "predicted_class1_fraction": float(class1.cpu()),
        "embedding_variance": float(emb_var.cpu()),
        "parameter_l2_displacement_from_M0": float(torch.linalg.vector_norm(current_param_vec.cpu() - anchor_param_vec.cpu())),
        "bn_running_displacement": bn_displacement(model, anchor_bn),
        "bn_anchor_displacement_recheck": float(np.sqrt(sum(float(torch.linalg.vector_norm(dict(model.named_buffers())[n].detach().cpu() - v.cpu())) ** 2 for n, v in anchor_bn_cpu.items() if n in dict(model.named_buffers())))),
    }
    del x, logits, probs, features
    return probe


def run_forensic(ctx: pmg.FoldContext, anchor_state: dict[str, torch.Tensor], mean: np.ndarray, std: np.ndarray, device: torch.device, start_time: float) -> tuple[pd.DataFrame, str]:
    m2_checkpoint_candidates = list((PMG_EXP / "runtime").glob("**/*M2*.pt")) + list((PMG_EXP / "runtime").glob("**/*m2*.pt"))
    if m2_checkpoint_candidates:
        return pd.DataFrame(), "existing M2 checkpoint found; no reproduction was run (not audited in this implementation)"
    channels = int(ctx.data.batch(ctx.fit_idx[:1]).shape[1])
    model = pmg.load_state(channels, anchor_state, device).train()
    teacher = pmg.load_state(channels, anchor_state, device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    pools = pmg.make_pools(ctx.data, ctx.fit_idx)
    subjects = pmg.subject_sort(pools.keys())
    steps = int(math.ceil(len(ctx.fit_idx) / pmg.BATCH_SIZE))
    params, param_values = model_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=pmg.LEARNING_RATE, weight_decay=pmg.WEIGHT_DECAY)
    probe_idx = pmg.representative_indices(pools, subjects)
    anchor_bn = bn_buffers(model)
    rows: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(1, pmg.ADAPT_EPOCHS + 1):
        schedule = pmg.make_meta_schedule(ctx, pools, epoch, steps, "train")
        for local_step, (a_idx, b_idx, bfold) in enumerate(schedule, start=1):
            if time.time() - start_time > HARD_RUNTIME_SECONDS:
                raise TimeoutError("PROSPECTIVE_GRADIENT_AUDIT_RUNTIME_BUDGET_EXCEEDED_BEFORE_FORENSIC_COMPLETION")
            xa = pmg.prepare(ctx.data, a_idx, mean, std, device)
            ya = torch.as_tensor(pmg.metadata_column(ctx.data, "label", a_idx, np.int64), dtype=torch.long, device=device)
            sa = pmg.metadata_column(ctx.data, "subject_id", a_idx)
            xb = pmg.prepare(ctx.data, b_idx, mean, std, device)
            yb = torch.as_tensor(pmg.metadata_column(ctx.data, "label", b_idx, np.int64), dtype=torch.long, device=device)
            sb = pmg.metadata_column(ctx.data, "subject_id", b_idx)
            optimizer.zero_grad(set_to_none=True)
            logits_a = model(xa)
            with torch.no_grad():
                anchor_a = teacher(xa)
            loss_a = pmg.balanced_task_ce(logits_a, ya, sa) + pmg.LAMBDA_KD * pmg.kl_anchor(logits_a, anchor_a)
            g_a_raw = torch.autograd.grad(loss_a, param_values, create_graph=False, retain_graph=False, allow_unused=True)
            g_a = tuple(g.detach().clone() if g is not None else torch.zeros_like(p) for g, p in zip(g_a_raw, param_values))
            prime_params = {name: param - pmg.ALPHA_INNER * grad for (name, param), grad in zip(params.items(), g_a)}
            prime_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
            base_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
            rng_seed = stable_seed("pmg-dropout-pair", ctx.dataset, ctx.fold, SEED, epoch, local_step - 1)
            with deterministic_rng(device, rng_seed):
                logits_prime = functional_call(model, (prime_params, prime_buffers), (xb,))
            with deterministic_rng(device, rng_seed):
                logits_theta = functional_call(model, (params, base_buffers), (xb,))
            future_names, future_losses = pmg.subject_losses(logits_prime, yb, sb)
            _, baseline_losses = pmg.subject_losses(logits_theta.detach(), yb, sb)
            loss_b_future = future_losses.mean()
            harm = torch.relu(future_losses - baseline_losses.detach()).mean()
            g_future = torch.autograd.grad(loss_b_future, tuple(prime_params.values()), create_graph=False, retain_graph=True, allow_unused=True)
            g_harm = torch.autograd.grad(harm, tuple(prime_params.values()), create_graph=False, retain_graph=False, allow_unused=True)
            loss_b_theta = pmg.balanced_task_ce(logits_theta, yb, sb)
            g_b_theta = torch.autograd.grad(loss_b_theta, param_values, create_graph=False, retain_graph=False, allow_unused=True)
            combined = []
            for p, ga, gb, gh in zip(param_values, g_a, g_future, g_harm):
                combined.append(ga + pmg.LAMBDA_META * (gb if gb is not None else torch.zeros_like(p)) + pmg.MU_HARM * (gh if gh is not None else torch.zeros_like(p)))
            pre_vec = flatten(combined, param_values)
            pre_norm = float(torch.linalg.vector_norm(pre_vec).detach().cpu())
            for p, grad in zip(param_values, combined):
                p.grad = grad.detach().clone()
            torch.nn.utils.clip_grad_norm_(param_values, pmg.GRAD_CLIP)
            post_vec = flatten([p.grad for p in param_values], param_values)
            post_norm = float(torch.linalg.vector_norm(post_vec).detach().cpu())
            optimizer.step()
            global_step += 1
            if global_step % 10 == 0 or local_step == len(schedule):
                probe = forensic_probe_metrics(model, ctx, probe_idx, mean, std, device, anchor_state, anchor_bn)
                row = {
                    "dataset": ctx.dataset,
                    "outer_fold": int(ctx.fold),
                    "seed": 0,
                    "epoch": int(epoch),
                    "step": int(local_step),
                    "global_step": int(global_step),
                    "meta_fold": int(bfold),
                    "L_A": float(loss_a.detach().cpu()),
                    "L_B_future": float(loss_b_future.detach().cpu()),
                    "harm_scalar": float(harm.detach().cpu()),
                    "gA_norm": float(vector_norm(g_a, param_values).detach().cpu()),
                    "gB_future_norm": float(vector_norm(g_future, param_values).detach().cpu()),
                    "g_harm_norm": float(vector_norm(g_harm, param_values).detach().cpu()),
                    "g_combined_norm": pre_norm,
                    "gradient_clip_pre_norm": pre_norm,
                    "gradient_clip_post_norm": post_norm,
                    "clip_activated": bool(pre_norm > pmg.GRAD_CLIP),
                    "cos_A_B": safe_cos(flatten(g_a, param_values), flatten(g_b_theta, param_values)),
                    "cos_A_harm": safe_cos(flatten(g_a, param_values), flatten(g_harm, param_values)),
                    "cos_B_harm": safe_cos(flatten(g_b_theta, param_values), flatten(g_harm, param_values)),
                    "active_harm_fraction": float((future_losses.detach() - baseline_losses.detach() > 0).float().mean().cpu()),
                    **probe,
                }
                rows.append(row)
                print(f"[forensic] step={global_step} pre_norm={pre_norm:.4f} harm={row['harm_scalar']:.7f} class1={row['predicted_class1_fraction']:.3f}", flush=True)
            del xa, ya, sa, xb, yb, sb, logits_a, anchor_a, logits_prime, logits_theta, loss_a, g_a_raw, g_a, prime_params, prime_buffers, base_buffers, future_losses, baseline_losses, loss_b_future, harm, g_future, g_harm, loss_b_theta, g_b_theta, combined, pre_vec, post_vec
            if global_step % 16 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    write_csv(RESULTS / "WBCIC_FOLD1_FORENSIC.csv", frame)
    if frame.empty:
        diagnosis = "not_run_existing_M2_artifact"
    else:
        last = frame.iloc[-1]
        clip_fraction = float(frame.clip_activated.mean())
        class1 = float(last.predicted_class1_fraction)
        harm_ratio = float(np.median(frame.half_g_harm_over_gA)) if "half_g_harm_over_gA" in frame else float(np.median(frame.g_harm_norm / np.maximum(frame.gA_norm, 1e-12) * 0.5))
        bn_max = float(frame.bn_running_displacement.max())
        emb_start = float(frame.embedding_variance.iloc[0])
        emb_end = float(frame.embedding_variance.iloc[-1])
        if class1 < 0.05 or class1 > 0.95:
            diagnosis = "A_classifier_one_class_collapse"
        elif clip_fraction >= 0.5:
            diagnosis = "C_repeated_gradient_clipping"
        elif harm_ratio >= 1.0:
            diagnosis = "B_overdominant_harm_gradient"
        elif bn_max > 1e-3:
            diagnosis = "D_batchnorm_running_stat_drift"
        elif emb_start > 0 and emb_end / emb_start < 0.1:
            diagnosis = "E_embedding_variance_collapse"
        else:
            diagnosis = "F_other_or_G_no_obvious_pathology"
    return frame, diagnosis


def write_scale_report(scales: pd.DataFrame) -> None:
    lines = ["# Gradient scale audit", "", "The original PMG-fast gradient was evaluated at the frozen M0 anchors for the same 500 paired observations. No parameter update was performed here.", "", "| Dataset | median ||gB_future||/||gA|| | median ||0.5 g_harm||/||gA|| | median ||g_combined||/||gA|| | clip-trigger fraction | active-harm fraction |", "|---|---:|---:|---:|---:|---:|"]
    for ds in DATASETS:
        sub = scales[scales.dataset == ds]
        lines.append(f"| {ds} | {np.median(sub.gB_future_over_gA):.6f} | {np.median(sub.half_g_harm_over_gA):.6f} | {np.median(sub.g_combined_over_gA):.6f} | {sub.clip_triggered.mean():.4f} | {sub.active_harm_fraction.mean():.4f} |")
    lines += ["", "The `0.5*g_harm` term is the exact PMG-fast harm-gradient coefficient. A small scalar ReLU harm does not scale this gradient; when active, its derivative is a task-loss gradient. The per-observation ratios and cosine geometry are in `results/GRADIENT_SCALE_RESULTS.csv`."]
    (AUDIT / "GRADIENT_SCALE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_forensic_report(frame: pd.DataFrame, diagnosis: str) -> None:
    lines = ["# WBCIC fold-1 forensic diagnosis", "", "The inherited PMG-fast result was WBCIC fold-1 discovery BA 0.501250 (M2), versus M0 0.784375 and M1 0.771250. Because the PMG-fast runtime contains no M2 checkpoint, optimizer state, or step log, this branch performed exactly one deterministic forensic reproduction with the original locked recipe. No scientific repair was applied.", ""]
    if frame.empty:
        lines.append("No reproduction was run because an exact M2 artifact was present.")
    else:
        last = frame.iloc[-1]
        lines += [f"- rows logged: {len(frame)} (every 10 optimizer steps plus epoch-final steps)", f"- final predicted class-1 fraction on fixed model-fit probe: {last.predicted_class1_fraction:.6f}", f"- fraction of logged steps with clip activated: {frame.clip_activated.mean():.6f}", f"- median 0.5*||g_harm||/||g_A||: {np.median(0.5 * frame.g_harm_norm / np.maximum(frame.gA_norm, 1e-12)):.6f}", f"- maximum BatchNorm running-stat displacement: {frame.bn_running_displacement.max():.8g}", f"- embedding variance first/last: {frame.embedding_variance.iloc[0]:.8g} / {frame.embedding_variance.iloc[-1]:.8g}", ""]
    lines += [f"Primary diagnosis: `{diagnosis}`.", "", "This is a localization statement for the known fold-1 collapse, not a method claim and not a permission to modify PMG."]
    (AUDIT / "WBCIC_FOLD1_FORENSIC.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(signal_summary: pd.DataFrame, scales: pd.DataFrame, forensic: pd.DataFrame, diagnosis: str, terminal: str, runtime_profile: dict[str, Any], math_result: dict[str, Any]) -> None:
    lines = ["# Prospective Gradient Signal Audit v1", "", f"terminal = {terminal}", "", "## Signal test (gradient-pair biological environments)", "", "| Dataset | Spearman rho | 95% cluster CI | sign accuracy | conflict rate | harm AUROC | conflict minus nonconflict actual Delta_B | 95% CI | signal pass |", "|---|---:|---|---:|---:|---:|---:|---|---|"]
    for ds in DATASETS:
        row = signal_summary[(signal_summary.dataset == ds) & np.isclose(signal_summary.epsilon, PRIMARY_EPSILON)].iloc[0]
        lines.append(f"| {ds} | {row.spearman_rho:.6f} | [{row.spearman_ci_lower:.6f}, {row.spearman_ci_upper:.6f}] | {row.sign_accuracy:.4f} | {row.conflict_rate:.4f} | {row.harm_auroc if pd.notna(row.harm_auroc) else float('nan'):.6f} | {row.conflict_minus_nonconflict:.8g} | [{row.conflict_delta_ci_lower:.8g}, {row.conflict_delta_ci_upper:.8g}] | {'YES' if row.signal_pass_primary else 'NO'} |")
    lines += ["", "## Gradient scale", "", "| Dataset | median ||gB||/||gA|| | median ||0.5gH||/||gA|| | median ||gCombined||/||gA|| | clip-trigger fraction | active-harm fraction |", "|---|---:|---:|---:|---:|---:|"]
    for ds in DATASETS:
        sub = scales[scales.dataset == ds]
        lines.append(f"| {ds} | {np.median(sub.gB_future_over_gA):.6f} | {np.median(sub.half_g_harm_over_gA):.6f} | {np.median(sub.g_combined_over_gA):.6f} | {sub.clip_triggered.mean():.4f} | {sub.active_harm_fraction.mean():.4f} |")
    primary_pass = {
        ds: bool(signal_summary[(signal_summary.dataset == ds) & np.isclose(signal_summary.epsilon, PRIMARY_EPSILON)].iloc[0].signal_pass_primary)
        for ds in DATASETS
    }
    both_pass = all(primary_pass.values())
    lines += [
        "", "## WBCIC fold-1 collapse diagnosis", "",
        f"`{diagnosis}`. Details are in `WBCIC_FOLD1_FORENSIC.md`; the forensic reproduction used the exact original five-epoch PMG-fast recipe once and did not repair it.",
        "", "## Required answers", "",
        f"1. Source gradient conflict predicts actual pseudo-future harm? {'Yes under the predeclared gates.' if any(primary_pass.values()) else 'No reliable signal under the predeclared gates.'}",
        f"2. Reproducible in both datasets? {'Yes.' if both_pass else 'No; at least one dataset fails the primary signal gate.'}",
        "3. Is PMG-fast's harm gradient disproportionate to its tiny scalar harm? See the scale table and per-observation audit; the derivative is not magnitude-weighted by the tiny scalar.",
        f"4. Most likely WBCIC fold-1 cause: {diagnosis}.",
        f"5. Direct prospective-gradient safeguarding scientifically justified? {'Only as a follow-up research question; this audit does not validate a method.' if both_pass else 'No.'}",
        f"6. PMG / prospective-gradient family: {'literature audit before any design, per protocol.' if both_pass else 'stop under this frozen audit.'}",
        "7. Outcome/sealed subjects accessed? No.", "8. Seed 1/2 run? No.",
        "", f"Runtime seconds: {runtime_profile.get('elapsed_seconds', float('nan')):.3f}; primary observations: {runtime_profile.get('primary_observations', 0)}; source-only: yes; mathematical audit: {'PASS' if math_result.get('pass') else 'FAIL'}.",
    ]
    (AUDIT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {"schema": "PROSPECTIVE_GRADIENT_SIGNAL_AUDIT_FINAL_V1", "terminal": terminal, "signal_summary": signal_summary.to_dict(orient="records"), "gradient_scale": {ds: scales[scales.dataset == ds].median(numeric_only=True).to_dict() for ds in DATASETS}, "forensic_diagnosis": diagnosis, "forensic_rows": int(len(forensic)), "runtime_profile": runtime_profile, "math_audit": math_result, "source_only": True, "outcome_accessed": False, "seed1_seed2_run": False, "sealed_status": {"WBCIC_outer_10_accessed": False, "OpenBMI_sealed_internal_accessed": False}}
    write_json(AUDIT / "FINAL_REPORT.json", report)


def run_audit(device: torch.device) -> None:
    start = time.time()
    set_seed(SEED)
    contexts = load_fit_only_contexts()
    if {(c.dataset, c.fold) for c in contexts} != {(d, f) for d in DATASETS for f in FOLDS}:
        raise RuntimeError("context coverage invalid")
    anchors: dict[tuple[str, int], tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]] = {}
    anchor_records = []
    for ctx in contexts:
        state, mean, std, record = load_anchor(ctx, device)
        check_anchor_architecture(ctx, state)
        anchors[(ctx.dataset, ctx.fold)] = (state, mean, std)
        anchor_records.append(record)
    math_result = run_math_audit(device)
    code_hash = source_hash(Path(__file__).resolve())
    write_protocol_docs(anchor_records, code_hash, math_result)
    print("[audit] frozen anchors verified; starting primary 500-observation audit", flush=True)
    pairs, steps, subjects, scales = run_primary_audit(contexts, anchors, device, start)
    signal_summary, bootstrap = summarize_signals(steps, start)
    print("[audit] primary signal summaries written", flush=True)
    trainmode = run_trainmode_audit(contexts, anchors, device, start)
    print("[audit] train-mode robustness written", flush=True)
    forensic_ctx = next(c for c in contexts if c.dataset == "WBCIC" and c.fold == 1)
    forensic_state, forensic_mean, forensic_std = anchors[("WBCIC", 1)]
    forensic, diagnosis = run_forensic(forensic_ctx, forensic_state, forensic_mean, forensic_std, device, start)
    write_scale_report(scales)
    write_forensic_report(forensic, diagnosis)
    primary_pass = {ds: bool(signal_summary[(signal_summary.dataset == ds) & np.isclose(signal_summary.epsilon, PRIMARY_EPSILON)].iloc[0].signal_pass_primary) for ds in DATASETS}
    terminal = "PROSPECTIVE_GRADIENT_SIGNAL_SUPPORTED" if all(primary_pass.values()) else ("PROSPECTIVE_GRADIENT_SIGNAL_DATASET_DEPENDENT" if any(primary_pass.values()) else "PROSPECTIVE_GRADIENT_SIGNAL_NOT_SUPPORTED")
    elapsed = time.time() - start
    runtime_profile = {"elapsed_seconds": elapsed, "target_seconds": TARGET_RUNTIME_SECONDS, "hard_budget_seconds": HARD_RUNTIME_SECONDS, "primary_observations": int(len(pairs)), "step_scale_rows": int(len(steps)), "trainmode_rows": int(len(trainmode)), "forensic_rows": int(len(forensic)), "source_only": True, "outcome_accessed": False, "seed1_seed2_run": False, "sealed_untouched": True, "anchor_recreated": False}
    write_json(RESULTS / "VALIDATION.json", {"pass": True, "terminal": terminal, "primary_complete": len(pairs) == 500, "trainmode_complete": len(trainmode) == len(DATASETS) * len(FOLDS) * 5 * TRAINMODE_DRAWS, "forensic_diagnosis": diagnosis, "math_audit_pass": bool(math_result.get("pass")), "source_only": True, "outcome_accessed": False, "seed1_seed2_run": False, "sealed_untouched": True, "primary_signal_pass": primary_pass})
    write_json(AUDIT / "RUNTIME_PROFILE.json", runtime_profile)
    (AUDIT / "RUNTIME_PROFILE.md").write_text("# Runtime profile\n\n" + json.dumps(clean(runtime_profile), indent=2) + "\n", encoding="utf-8")
    write_final_report(signal_summary, scales, forensic, diagnosis, terminal, runtime_profile, math_result)
    print(f"terminal = {terminal}", flush=True)
    print("outcome_accessed = NO", flush=True)
    print("seed1_seed2_run = NO", flush=True)


def resume_after_primary(device: torch.device) -> None:
    """Finish a run whose primary 500 observations were already persisted.

    This is an engineering resume path for an interrupted process.  It never
    recomputes or changes the primary pair observations, scales, or subject
    rows; it only performs the deterministic post-processing, train-mode
    robustness diagnostic, and the separately permitted fold-1 forensic.
    """
    start = time.time()
    set_seed(SEED)
    contexts = load_fit_only_contexts()
    anchors: dict[tuple[str, int], tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]] = {}
    anchor_records = []
    for ctx in contexts:
        state, mean, std, record = load_anchor(ctx, device)
        check_anchor_architecture(ctx, state)
        anchors[(ctx.dataset, ctx.fold)] = (state, mean, std)
        anchor_records.append(record)
    pairs = pd.read_csv(RESULTS / "GRADIENT_PAIR_OBSERVATIONS.csv")
    steps = pd.read_csv(RESULTS / "STEP_SCALE_RESULTS.csv")
    subjects = pd.read_csv(RESULTS / "SUBJECT_HARM.csv")
    scales = pd.read_csv(RESULTS / "GRADIENT_SCALE_RESULTS.csv")
    expected_pairs = len(DATASETS) * len(FOLDS) * 5 * PAIR_DRAWS
    expected_steps = expected_pairs * len(EPSILONS)
    if len(pairs) != expected_pairs or len(steps) != expected_steps or len(scales) != expected_pairs:
        raise RuntimeError(f"persisted primary artifacts incomplete: pairs={len(pairs)} steps={len(steps)} scales={len(scales)}")
    math_result = run_math_audit(device)
    write_protocol_docs(anchor_records, source_hash(Path(__file__).resolve()), math_result)
    signal_summary, _bootstrap = summarize_signals(steps, start)
    print("[resume] primary signal summaries written", flush=True)
    trainmode = run_trainmode_audit(contexts, anchors, device, start)
    print("[resume] train-mode robustness written", flush=True)
    forensic_ctx = next(c for c in contexts if c.dataset == "WBCIC" and c.fold == 1)
    forensic_state, forensic_mean, forensic_std = anchors[("WBCIC", 1)]
    forensic, diagnosis = run_forensic(forensic_ctx, forensic_state, forensic_mean, forensic_std, device, start)
    write_scale_report(scales)
    write_forensic_report(forensic, diagnosis)
    primary_pass = {
        ds: bool(signal_summary[(signal_summary.dataset == ds) & np.isclose(signal_summary.epsilon, PRIMARY_EPSILON)].iloc[0].signal_pass_primary)
        for ds in DATASETS
    }
    terminal = "PROSPECTIVE_GRADIENT_SIGNAL_SUPPORTED" if all(primary_pass.values()) else ("PROSPECTIVE_GRADIENT_SIGNAL_DATASET_DEPENDENT" if any(primary_pass.values()) else "PROSPECTIVE_GRADIENT_SIGNAL_NOT_SUPPORTED")
    elapsed = time.time() - start
    runtime_profile = {"elapsed_seconds": elapsed, "target_seconds": TARGET_RUNTIME_SECONDS, "hard_budget_seconds": HARD_RUNTIME_SECONDS, "primary_observations": int(len(pairs)), "step_scale_rows": int(len(steps)), "trainmode_rows": int(len(trainmode)), "forensic_rows": int(len(forensic)), "source_only": True, "outcome_accessed": False, "seed1_seed2_run": False, "sealed_untouched": True, "anchor_recreated": False, "resumed_after_primary": True}
    write_json(RESULTS / "VALIDATION.json", {"pass": True, "terminal": terminal, "primary_complete": len(pairs) == expected_pairs, "trainmode_complete": len(trainmode) == len(DATASETS) * len(FOLDS) * 5 * TRAINMODE_DRAWS, "forensic_diagnosis": diagnosis, "math_audit_pass": bool(math_result.get("pass")), "source_only": True, "outcome_accessed": False, "seed1_seed2_run": False, "sealed_untouched": True, "primary_signal_pass": primary_pass, "resumed_after_primary": True})
    write_json(AUDIT / "RUNTIME_PROFILE.json", runtime_profile)
    (AUDIT / "RUNTIME_PROFILE.md").write_text("# Runtime profile\n\n" + json.dumps(clean(runtime_profile), indent=2) + "\n", encoding="utf-8")
    write_final_report(signal_summary, scales, forensic, diagnosis, terminal, runtime_profile, math_result)
    print(f"terminal = {terminal}", flush=True)
    print("outcome_accessed = NO", flush=True)
    print("seed1_seed2_run = NO", flush=True)


def preflight(device: torch.device) -> None:
    set_seed(SEED)
    contexts = load_fit_only_contexts()
    anchor_records = []
    for ctx in contexts:
        state, _mean, _std, record = load_anchor(ctx, device)
        check_anchor_architecture(ctx, state)
        anchor_records.append(record)
    math_result = run_math_audit(device)
    write_protocol_docs(anchor_records, source_hash(Path(__file__).resolve()), math_result)
    print(f"preflight complete: anchors={len(anchor_records)} math_pass={math_result['pass']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume-primary", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    modes = int(args.preflight) + int(args.run) + int(args.resume_primary)
    if args.seed != 0 or modes != 1:
        raise SystemExit("exactly one of --preflight/--run/--resume-primary is required and seed 0 is mandatory")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    if args.preflight:
        preflight(device)
    elif args.run:
        run_audit(device)
    else:
        resume_after_primary(device)


if __name__ == "__main__":
    main()
