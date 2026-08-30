"""Repair-R1 source-only experiment: task-protected Bures subject transport.

This module deliberately reuses the audited V3 loaders, Bures bank, optimizer,
and source split.  The only scientific change is a source-training
class-centroid projection applied to the transport operator.  Runtime artifacts
stay on the server; only compact summaries are intended for publication.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


V3_CODE = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_bures_scst_v3\code")
if str(V3_CODE) not in sys.path:
    sys.path.insert(0, str(V3_CODE))

import common as c  # noqa: E402
from bures import BuresBank, anchor_excluded_neighbors, matched_random_displacement  # noqa: E402
from source_v3 import (  # noqa: E402
    AdapterHead,
    Geometry,
    _evaluate,
    _geometry_subject_frame,
    _margin_np,
    _matched_masks,
    _subject_batches,
    _teacher_np,
    _transport_loss,
)


BASE_REPO = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC")
EXP = BASE_REPO / "experiments" / "persist_eeg_scst_autonomous_repair"
CODE = EXP / "code"
PROTOCOL = EXP / "protocol"
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
RUNTIME = EXP / "runtime"

# V3 helpers use the common module object imported above.  Redirect only the
# experiment outputs; V2 detached source caches remain at the audited location.
c.EXP = EXP
c.CODE = CODE
c.PROTOCOL = PROTOCOL
c.RESULTS = RESULTS
c.FIGURES = FIGURES
c.RUNTIME = RUNTIME

PRIMARY = "R1-TaskProtected-Bures"
RANDOM = "R1-TaskProtected-Random"
PREVIOUS = "V3-Bures-HardSCST"
ERM = "ERM"
MIXUP = "Mixup"
RECIPES = tuple((q, lam) for q in (0.25, 0.50) for lam in (0.25, 0.50, 1.00))
ALPHA_LADDER = np.asarray((1.00, 0.75, 0.50, 0.25), np.float32)
WARMUP_EPOCHS = int(c.WARMUP_EPOCHS)
STAGE2_EPOCHS = int(c.STAGE2_EPOCHS)


def _task_direction(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return the frozen source-only two-class protected direction."""
    values = np.asarray(features, np.float64)
    y = np.asarray(labels, np.int64)
    classes = sorted(np.unique(y).tolist())
    if len(classes) != 2:
        raise RuntimeError(f"R1_EXPECTS_TWO_CLASSES:{classes}")
    delta = values[y == classes[1]].mean(0) - values[y == classes[0]].mean(0)
    norm = float(np.linalg.norm(delta))
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.zeros(values.shape[1], np.float64)
    return delta / norm


def _protected(direction: np.ndarray, task_direction: np.ndarray) -> np.ndarray:
    value = np.asarray(direction, np.float64)
    d = np.asarray(task_direction, np.float64)
    return (value - d * float(np.dot(d, value))).astype(np.float32)


def _geometry(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    row_ids: np.ndarray,
    dataset: str,
    fold: int,
    seed: int,
    teacher: AdapterHead,
    *,
    mode: str,
    bank: BuresBank,
    reference_directions: np.ndarray | None = None,
) -> tuple[Geometry, np.ndarray]:
    """Build a V3-compatible geometry with a frozen R1 operator."""
    features = np.asarray(features, np.float64)
    labels = np.asarray(labels, np.int64)
    subjects = np.asarray(subjects).astype(str)
    row_ids = np.asarray(row_ids, np.int64)
    n, dim = features.shape
    k = int(c.K_TARGETS)
    offsets = np.zeros((n, k, dim), np.float32)
    valid = np.zeros((n, k), bool)
    chosen_alpha = np.zeros((n, k), np.float32)
    targets = np.full((n, k), "", dtype="U32")
    support_pass = np.zeros((n, k), bool)
    class_pass = np.zeros((n, k), bool)
    before_d = np.full((n, k), np.nan)
    after_d = np.full((n, k), np.nan)
    before_nll = np.full((n, k), np.nan)
    after_nll = np.full((n, k), np.nan)
    displacement_ratio = np.full((n, k), np.nan)
    relative_margin_drop = np.full((n, k), np.nan)
    directions = np.zeros((n, k, dim), np.float32)
    task_direction = _task_direction(features, labels)
    rng = np.random.default_rng(c.stable_seed("r1-random", dataset, fold, seed, mode))
    neighbor_idx, neighbor_dist = anchor_excluded_neighbors(features, row_ids, c.KNN_K)
    local_radius = np.nanmean(np.where(np.isfinite(neighbor_dist), neighbor_dist, np.nan), axis=1)
    local_radius[~np.isfinite(local_radius)] = np.inf

    for position in range(n):
        source = str(subjects[position])
        label = int(labels[position])
        pool = np.asarray([s for s in bank.subject_list if s != source], dtype=str)
        target_rng = np.random.default_rng(c.stable_seed("r1-targets", dataset, fold, seed, int(row_ids[position])))
        if len(pool) > k:
            pool = pool[target_rng.choice(len(pool), k, replace=False)]
        for slot, target in enumerate(pool[:k]):
            targets[position, slot] = str(target)
            structured = bank.displacement(position, str(target)).astype(np.float32)
            if mode == "protected":
                direction = _protected(structured, task_direction)
            elif mode == "random":
                if reference_directions is None:
                    raise RuntimeError("R1_RANDOM_NEEDS_REFERENCE")
                direction = matched_random_displacement(reference_directions[position, slot], bank, rng)
            elif mode == "v3":
                direction = structured
            else:
                raise ValueError(mode)
            directions[position, slot] = direction

    anchors = features[:, None, None, :] + directions[:, :, None, :] * ALPHA_LADDER[None, None, :, None]
    teacher_logits = _teacher_np(teacher, anchors.reshape(-1, dim)).reshape(n, k, len(ALPHA_LADDER), 2)
    clean_logits = _teacher_np(teacher, features)
    for position in range(n):
        label = int(labels[position])
        local = neighbor_idx[position][neighbor_idx[position] >= 0]
        local_radius_position = float(local_radius[position])
        anchor_class = bool(len(local) and np.mean(labels[local] == label) >= 0.8)
        clean_margin = _margin_np(clean_logits[position], label)
        for slot, target in enumerate(targets[position]):
            if target == "":
                continue
            candidates = anchors[position, slot]
            local_dist = np.linalg.norm(features[local][None, :, :] - candidates[:, None, :], axis=2) if len(local) else np.full((len(ALPHA_LADDER), 0), np.inf)
            support_distance = local_dist.mean(1) if len(local) else np.full(len(ALPHA_LADDER), np.inf)
            support_ok = support_distance <= float(bank.radius.get(label, np.inf))
            class_ok = np.zeros(len(ALPHA_LADDER), bool)
            if len(local):
                nearest = np.take(local, np.argsort(local_dist, axis=1), axis=0)
                class_ok = np.sum(labels[nearest[:, : min(c.KNN_K, len(local))]] == label, axis=1) >= 4
            cached = bank.target_affinity_stats(position, str(target))
            if cached is None:
                continue
            values, mean, inv, logdet = cached
            before_d[position, slot] = float(np.sort(np.linalg.norm(values - features[position][None], axis=1))[: min(c.KNN_K, len(values))].mean())
            diff0 = features[position] - mean
            before_nll[position, slot] = float(0.5 * (diff0 @ inv @ inv @ diff0 + logdet))
            distances = np.linalg.norm(values[None, :, :] - candidates[:, None, :], axis=2)
            after_d_candidates = np.sort(distances, axis=1)[:, : min(c.KNN_K, len(values))].mean(1)
            diff = candidates - mean[None, :]
            after_nll_candidates = 0.5 * (np.einsum("ai,ij,aj->a", diff, inv @ inv, diff) + logdet)
            margin = teacher_logits[position, slot, :, label] - teacher_logits[position, slot, :, 1 - label]
            teacher_ok = (teacher_logits[position, slot].argmax(1) == label) & (margin > 0)
            ratios = np.linalg.norm(directions[position, slot][None, :] * ALPHA_LADDER[:, None], axis=1) / max(local_radius_position, 1e-6)
            guard_ok = ratios <= 1.0 + 1e-8
            affinity_ok = np.isfinite(after_d_candidates) & np.isfinite(after_nll_candidates) & (after_d_candidates < before_d[position, slot]) & (after_nll_candidates < before_nll[position, slot])
            margin_drop = (clean_margin - margin) / (abs(clean_margin) + 1e-6)
            gate = anchor_class & support_ok & class_ok & teacher_ok & guard_ok & affinity_ok
            picked = np.flatnonzero(gate)
            if len(picked):
                aidx = int(picked[0])
                offsets[position, slot] = directions[position, slot] * ALPHA_LADDER[aidx]
                valid[position, slot] = True
                chosen_alpha[position, slot] = ALPHA_LADDER[aidx]
                support_pass[position, slot] = bool(support_ok[aidx])
                class_pass[position, slot] = bool(class_ok[aidx])
                after_d[position, slot] = after_d_candidates[aidx]
                after_nll[position, slot] = after_nll_candidates[aidx]
                displacement_ratio[position, slot] = ratios[aidx]
                relative_margin_drop[position, slot] = margin_drop[aidx]
            else:
                offsets[position, slot] = directions[position, slot] * float(max(ALPHA_LADDER))
    return Geometry(offsets, valid, chosen_alpha, targets, support_pass, class_pass, before_d, after_d, before_nll, after_nll, displacement_ratio, relative_margin_drop, bank), directions


def _warmup(train_cache: dict[str, np.ndarray], dataset: str, fold: int, seed: int, device: torch.device) -> tuple[AdapterHead, AdapterHead]:
    c.set_seed(c.stable_seed("r1-warmup", dataset, fold, seed))
    dim = int(train_cache["features"].shape[1])
    model = AdapterHead(dim).to(device)
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for parameter in model.adapter.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=c.HEAD_LR, weight_decay=c.WEIGHT_DECAY)
    for epoch in range(WARMUP_EPOCHS):
        model.train()
        for positions in _subject_batches(train_cache["subjects"], seed, epoch):
            if not len(positions):
                continue
            x = torch.from_numpy(train_cache["features"][positions]).to(device)
            y = torch.from_numpy(train_cache["labels"][positions]).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model.logits(x).float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            with torch.no_grad():
                teacher.head.weight.mul_(c.EMA_DECAY).add_(model.head.weight, alpha=1 - c.EMA_DECAY)
                teacher.head.bias.mul_(c.EMA_DECAY).add_(model.head.bias, alpha=1 - c.EMA_DECAY)
    return model, teacher


def _train_from_warmup(
    base_model: AdapterHead,
    base_teacher: AdapterHead,
    train_cache: dict[str, np.ndarray],
    valid_cache: dict[str, np.ndarray],
    dataset: str,
    fold: int,
    seed: int,
    method: str,
    q: float,
    lam: float,
    geometry: Geometry | None,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    model = copy.deepcopy(base_model).to(device)
    teacher = copy.deepcopy(base_teacher).to(device)
    for parameter in model.adapter.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        [{"params": model.head.parameters(), "lr": c.HEAD_LR}, {"params": model.adapter.parameters(), "lr": c.ADAPTER_LR}],
        weight_decay=c.WEIGHT_DECAY,
    )
    diagnostics: list[dict[str, float]] = []
    for epoch in range(STAGE2_EPOCHS):
        model.train()
        for positions in _subject_batches(train_cache["subjects"], seed, WARMUP_EPOCHS + epoch):
            if not len(positions):
                continue
            x = torch.from_numpy(train_cache["features"][positions]).to(device)
            y = torch.from_numpy(train_cache["labels"][positions]).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            h = model.features(x)
            clean_logits = model.head(h)
            loss = F.cross_entropy(clean_logits.float(), y)
            if method == MIXUP:
                rng = np.random.default_rng(c.stable_seed("r1-mixup", dataset, fold, seed, epoch, int(positions[0])))
                perm = torch.as_tensor(rng.permutation(len(positions)), device=device)
                weight = float(rng.beta(0.4, 0.4))
                mixed = weight * h + (1 - weight) * h[perm]
                soft = weight * F.one_hot(y, 2).float() + (1 - weight) * F.one_hot(y[perm], 2).float()
                loss = loss + 0.5 * (-(soft * torch.log_softmax(model.head(mixed).float(), -1)).sum(-1).mean())
            elif geometry is not None:
                cf, _ = _transport_loss(model, h, clean_logits, y, geometry, positions, q, lam)
                loss = loss + cf
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            with torch.no_grad():
                teacher.adapter.weight.mul_(c.EMA_DECAY).add_(model.adapter.weight, alpha=1 - c.EMA_DECAY)
                teacher.head.weight.mul_(c.EMA_DECAY).add_(model.head.weight, alpha=1 - c.EMA_DECAY)
                teacher.head.bias.mul_(c.EMA_DECAY).add_(model.head.bias, alpha=1 - c.EMA_DECAY)
        if geometry is not None:
            def finite_mean(value: np.ndarray) -> float:
                value = np.asarray(value, np.float64).reshape(-1)
                value = value[np.isfinite(value)]
                return float(value.mean()) if len(value) else float("nan")
            diagnostics.append({
                "epoch": float(epoch + 1),
                "coverage": float(np.mean(geometry.valid.sum(1) >= 1)),
                "target_distance_improvement": finite_mean(geometry.target_distance_before - geometry.target_distance_after),
                "target_nll_improvement": finite_mean(geometry.target_nll_before - geometry.target_nll_after),
                "class_pass_rate": float(np.mean(geometry.class_pass)),
                "median_displacement_ratio": float(np.nanmedian(geometry.displacement_ratio)),
                "median_relative_margin_drop": float(np.nanmedian(geometry.relative_margin_drop)),
            })
    frame, ba, macro = _evaluate(model, valid_cache, device, method, dataset, fold, seed, q, lam)
    summary = {"dataset": dataset, "fold": fold, "seed": seed, "method": method, "q": q, "lambda_T": lam, "BA": ba, "macro_F1": macro, "diagnostic_last": diagnostics[-1] if diagnostics else {}, "epochs": WARMUP_EPOCHS + STAGE2_EPOCHS, "future_or_outer_opened": False}
    return frame, summary, pd.DataFrame(diagnostics)


def _summary_geometry(geometry: Geometry, train_cache: dict[str, np.ndarray], dataset: str, fold: int, seed: int, method: str) -> pd.DataFrame:
    return _geometry_subject_frame(geometry, train_cache, dataset, fold, seed, method, 0.50, 0.50)


def _match_audit(structured: Geometry, random: Geometry, dataset: str, fold: int, seed: int) -> dict[str, object]:
    (mask_s, mask_r), frame = _matched_masks(structured, random, dataset, fold, seed)
    paired = frame[frame["matched_pair"] == True] if len(frame) else frame  # noqa: E712
    return {
        "dataset": dataset,
        "fold": fold,
        "seed": seed,
        "rows": int(len(frame)),
        "matched_pairs": int(len(paired)),
        "mean_euclidean_norm_mismatch": float(paired["euclidean_norm_mismatch"].abs().mean()) if len(paired) else None,
        "mean_whitened_norm_mismatch": float(paired["whitened_norm_mismatch"].abs().mean()) if len(paired) else None,
        "alpha_mismatch_max": float(paired["alpha_mismatch"].abs().max()) if len(paired) else None,
        "per_anchor_count_match": bool((frame["structured_matched_count"] == frame["random_matched_count"]).all()) if len(frame) else False,
        "structured_mask_count": int(mask_s.sum()),
        "random_mask_count": int(mask_r.sum()),
    }


def unit_dir(dataset: str, fold: int, seed: int) -> Path:
    return RUNTIME / "r1_units" / dataset / f"fold-{fold}" / f"seed-{seed}"


def run_unit(dataset: str, fold: int, seed: int, device: torch.device) -> None:
    directory = unit_dir(dataset, fold, seed)
    marker = directory / "COMPLETE.json"
    if marker.is_file():
        return
    train = c.load_feature_cache(dataset, fold, seed, "train")
    valid = c.load_feature_cache(dataset, fold, seed, "validation")
    directory.mkdir(parents=True, exist_ok=True)
    base_model, base_teacher = _warmup(train, dataset, fold, seed, device)
    with torch.inference_mode():
        base_features = base_teacher.features(torch.from_numpy(train["features"]).to(device)).float().cpu().numpy()
    bank = BuresBank(base_features, train["labels"], train["subjects"], train["indices"], dataset=dataset, fold=fold, seed=seed)
    protected, protected_dirs = _geometry(base_features, train["labels"], train["subjects"], train["indices"], dataset, fold, seed, base_teacher, mode="protected", bank=bank)
    random_geometry, _ = _geometry(base_features, train["labels"], train["subjects"], train["indices"], dataset, fold, seed, base_teacher, mode="random", bank=bank, reference_directions=protected_dirs)
    (protected_mask, random_mask), _ = _matched_masks(protected, random_geometry, dataset, fold, seed)
    protected.valid &= protected_mask
    random_geometry.valid &= random_mask
    previous, _ = _geometry(base_features, train["labels"], train["subjects"], train["indices"], dataset, fold, seed, base_teacher, mode="v3", bank=bank)

    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    geometry_rows: list[pd.DataFrame] = []
    controls = [
        (ERM, 0.50, 0.50, None),
        (MIXUP, 0.50, 0.50, None),
        (PREVIOUS, 0.50, 0.50, previous),
    ]
    for method, q, lam, geometry in controls:
        frame, summary, _ = _train_from_warmup(base_model, base_teacher, train, valid, dataset, fold, seed, method, q, lam, geometry, device)
        frames.append(frame)
        summaries.append(summary)
    for q, lam in RECIPES:
        for method, geometry in ((PRIMARY, protected), (RANDOM, random_geometry)):
            frame, summary, _ = _train_from_warmup(base_model, base_teacher, train, valid, dataset, fold, seed, method, q, lam, geometry, device)
            frames.append(frame)
            summaries.append(summary)
    geometry_rows.append(_summary_geometry(protected, train, dataset, fold, seed, PRIMARY))
    geometry_rows.append(_summary_geometry(random_geometry, train, dataset, fold, seed, RANDOM))
    geometry_rows.append(_summary_geometry(previous, train, dataset, fold, seed, PREVIOUS))
    c.write_csv(directory / "per_subject.csv", pd.concat(frames, ignore_index=True))
    c.write_json(directory / "summary.json", summaries)
    c.write_csv(directory / "geometry_per_subject.csv", pd.concat(geometry_rows, ignore_index=True))
    c.write_json(directory / "match_audit.json", _match_audit(protected, random_geometry, dataset, fold, seed))
    c.write_json(marker, {"dataset": dataset, "fold": fold, "seed": seed, "methods": len(summaries), "future_or_outer_opened": False})


def _paired(frame: pd.DataFrame, dataset: str, method: str, control: str, q: float, lam: float) -> np.ndarray:
    left = frame[(frame.dataset == dataset) & (frame.method == method) & np.isclose(frame.q, q) & np.isclose(frame.lambda_T, lam)]
    right = frame[(frame.dataset == dataset) & (frame.method == control)]
    if not len(left) or not len(right):
        return np.asarray([], np.float64)
    l = left.groupby("subject_id").BA.mean()
    r = right.groupby("subject_id").BA.mean()
    return (l - r).dropna().to_numpy(np.float64)


def _gate(frame: pd.DataFrame, geometry: pd.DataFrame, match: pd.DataFrame) -> dict[str, object]:
    recipe_rows = []
    for q, lam in RECIPES:
        checks: dict[str, bool] = {}
        by_dataset: dict[str, dict[str, float | int | None]] = {}
        for dataset in c.DATASETS:
            delta = _paired(frame, dataset, PRIMARY, ERM, q, lam)
            ci = c.bootstrap_ci(delta, seed=c.stable_seed("r1-gate", dataset, q, lam)) if len(delta) else (float("nan"), float("nan"), float("nan"))
            random_delta = _paired(frame, dataset, PRIMARY, RANDOM, q, lam)
            random_ci = c.bootstrap_ci(random_delta, seed=c.stable_seed("r1-random-gate", dataset, q, lam)) if len(random_delta) else (float("nan"), float("nan"), float("nan"))
            mixup_delta = _paired(frame, dataset, PRIMARY, MIXUP, q, lam)
            by_dataset[dataset] = {"delta": ci[0], "ci95_l": ci[1], "ci95_u": ci[2], "n_subjects": int(len(delta)), "vs_random_delta": random_ci[0], "vs_random_ci95_l": random_ci[1], "vs_mixup_delta": float(mixup_delta.mean()) if len(mixup_delta) else None}
            checks[f"{dataset}_delta_ge_002"] = bool(np.isfinite(ci[0]) and ci[0] >= 0.002)
            checks[f"{dataset}_ci_lower_vs_erm_positive"] = bool(np.isfinite(ci[1]) and ci[1] > 0)
            checks[f"{dataset}_ci_lower_vs_random_positive"] = bool(np.isfinite(random_ci[1]) and random_ci[1] > 0)
            checks[f"{dataset}_mean_vs_mixup_positive"] = bool(np.isfinite(by_dataset[dataset]["vs_mixup_delta"]) and float(by_dataset[dataset]["vs_mixup_delta"]) > 0)
        recipe_rows.append({"q": q, "lambda_T": lam, "by_dataset": by_dataset, "checks": checks, "pass": bool(all(checks.values()))})

    pooled_delta = _paired(frame.assign(dataset=frame.dataset.astype(str)), "OpenBMI", PRIMARY, ERM, 0.50, 0.50)
    pooled_wbcic = _paired(frame, "WBCIC", PRIMARY, ERM, 0.50, 0.50)
    pooled = np.concatenate([pooled_delta, pooled_wbcic]) if len(pooled_delta) or len(pooled_wbcic) else np.asarray([], np.float64)
    subject_fraction = float(np.mean(pooled >= 0)) if len(pooled) else 0.0
    affinity = {}
    for dataset in c.DATASETS:
        subset = geometry[(geometry.dataset == dataset) & (geometry.method == PRIMARY)]
        subject = subset.groupby("subject_id").agg(target_distance_improvement=("target_distance_improvement", "mean"), target_nll_improvement=("target_nll_improvement", "mean"), coverage=("coverage", "mean"), class_pass_rate=("class_pass_rate", "mean"), displacement_ratio=("median_displacement_ratio", "median")) if len(subset) else pd.DataFrame()
        if len(subject):
            dci = c.bootstrap_ci(subject.target_distance_improvement.to_numpy(float), seed=c.stable_seed("r1-affinity", dataset, "distance"))
            nci = c.bootstrap_ci(subject.target_nll_improvement.to_numpy(float), seed=c.stable_seed("r1-affinity", dataset, "nll"))
            affinity[dataset] = {"subjects": int(len(subject)), "target_distance_mean": dci[0], "target_distance_ci95_l": dci[1], "target_nll_mean": nci[0], "target_nll_ci95_l": nci[1], "coverage_mean": float(subject.coverage.mean()), "class_fidelity_mean": float(subject.class_pass_rate.mean()), "median_displacement_ratio": float(subject.displacement_ratio.median()), "checks": {"target_affinity_ci_lower_positive": bool(dci[1] > 0 and nci[1] > 0), "coverage": bool(subject.coverage.mean() >= 0.50), "class_fidelity": bool(subject.class_pass_rate.mean() >= 0.90), "displacement": bool(subject.displacement_ratio.median() >= 0.15)}}
        else:
            affinity[dataset] = {"subjects": 0, "checks": {"target_affinity_ci_lower_positive": False, "coverage": False, "class_fidelity": False, "displacement": False}}
    candidate_survival = float(geometry.coverage.mean()) if len(geometry) and "coverage" in geometry else 0.0
    match_summary = {"rows": int(match.rows.sum()) if len(match) else 0, "matched_pairs": int(match.matched_pairs.sum()) if len(match) else 0, "mean_euclidean_norm_mismatch": float(np.nanmean(match.mean_euclidean_norm_mismatch)) if len(match) else None, "mean_whitened_norm_mismatch": float(np.nanmean(match.mean_whitened_norm_mismatch)) if len(match) else None, "alpha_mismatch_max": float(np.nanmax(match.alpha_mismatch_max)) if len(match) else None, "per_anchor_count_match": bool(match.per_anchor_count_match.all()) if len(match) else False}
    # A source gate must be satisfied by one *same* primary recipe.  The
    # previous implementation combined per-dataset deltas from different
    # recipes, which could report a pass for a recipe that never passed all
    # preregistered checks jointly.  Keep the scientific thresholds unchanged
    # and require the existing row-level checks to hold for one recipe.
    source_pass = bool(
        any(row["pass"] for row in recipe_rows)
        and subject_fraction >= 0.60
        and all(
            value["checks"]["target_affinity_ci_lower_positive"]
            and value["checks"]["coverage"]
            and value["checks"]["class_fidelity"]
            and value["checks"]["displacement"]
            for value in affinity.values()
        )
        and candidate_survival <= 0.95
    )
    return {"schema": "SCST_AUTONOMOUS_R1_GATE_V1", "method": PRIMARY, "recipe_rows": recipe_rows, "subject_nonnegative_fraction": subject_fraction, "affinity": affinity, "candidate_survival_mean": candidate_survival, "candidate_survival_guard": bool(candidate_survival <= 0.95), "match_audit": match_summary, "source_gate_pass": source_pass, "future_or_outer_opened": False, "outer_or_sealed_opened": False, "terminal_if_stop": "R1_SOURCE_GATE_PASSED" if source_pass else "R1_SOURCE_GATE_FAILED"}


def aggregate() -> None:
    files = sorted((RUNTIME / "r1_units").rglob("per_subject.csv"))
    if not files:
        raise RuntimeError("R1_NO_SOURCE_RESULTS")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    geometry = pd.concat([pd.read_csv(path) for path in sorted((RUNTIME / "r1_units").rglob("geometry_per_subject.csv"))], ignore_index=True)
    match = pd.DataFrame([json.loads(path.read_text(encoding="utf-8")) for path in sorted((RUNTIME / "r1_units").rglob("match_audit.json"))])
    c.write_csv(RESULTS / "R1_SOURCE_PER_SUBJECT.csv", frame)
    grouped = frame.groupby(["dataset", "method", "q", "lambda_T", "fold", "seed"], as_index=False).agg(BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), subjects=("subject_id", "nunique"))
    c.write_csv(RESULTS / "R1_SOURCE_PER_FOLD.csv", grouped)
    c.write_csv(RESULTS / "R1_GEOMETRY_PER_SUBJECT.csv", geometry)
    c.write_csv(RESULTS / "R1_MATCH_AUDIT.csv", match)
    c.write_csv(RESULTS / "R1_METHOD_SUMMARY.csv", grouped)
    gate = _gate(frame, geometry, match)
    c.write_json(RESULTS / "R1_GATE.json", gate)
    stats = {"schema": "SCST_AUTONOMOUS_R1_STATISTICS_V1", "source_units": int(len(files)), "rows": int(len(frame)), "future_or_outer_opened": False, "outer_or_sealed_opened": False, "source_gate_pass": bool(gate["source_gate_pass"]), "terminal": gate["terminal_if_stop"]}
    c.write_json(RESULTS / "R1_STATISTICS.json", stats)
    print(json.dumps({"source_units": len(files), "source_gate_pass": gate["source_gate_pass"], "terminal": gate["terminal_if_stop"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=c.DATASETS)
    parser.add_argument("--fold", type=int, choices=c.FOLDS)
    parser.add_argument("--seed", type=int, choices=c.SEEDS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    c.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.all:
        for dataset in c.DATASETS:
            for fold in c.FOLDS:
                for seed in c.SEEDS:
                    print(f"[r1] START {dataset} f={fold} s={seed}", flush=True)
                    run_unit(dataset, fold, seed, device)
                    print(f"[r1] DONE {dataset} f={fold} s={seed}", flush=True)
    elif args.dataset is not None and args.fold is not None and args.seed is not None:
        run_unit(args.dataset, args.fold, args.seed, device)
    if args.aggregate:
        aggregate()


if __name__ == "__main__":
    main()
