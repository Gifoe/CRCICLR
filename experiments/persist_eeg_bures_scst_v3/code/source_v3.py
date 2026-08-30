"""Source-only Bures-SCST grid on detached, trusted Stage-1 representations.

The implementation keeps the model-side trainable part explicit: an identity
initialised final-feature adapter and a linear head.  This is the cache-compatible
representation of the final feature block used by the source transition.  All
bank statistics and candidate decisions are detached from optimisation.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score

import common as c
from bures import BuresBank, anchor_excluded_indices, anchor_excluded_neighbors, knn_mean_distance, matched_random_displacement, target_affinity


METHODS = ("ERM", "Mixup", "V2-ME-HardSCST", "Manifold-Mixup", "Bures-Uniform", "Bures-HardRandom", "Bures-HardSCST")
RECIPES = tuple((q, lam) for q in (0.25, 0.50) for lam in (0.25, 0.50, 1.00))


class AdapterHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.adapter = nn.Linear(dim, dim, bias=False)
        self.head = nn.Linear(dim, 2)
        with torch.no_grad():
            self.adapter.weight.copy_(torch.eye(dim))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


@dataclass
class Geometry:
    offsets: np.ndarray
    valid: np.ndarray
    alpha: np.ndarray
    targets: np.ndarray
    support_pass: np.ndarray
    class_pass: np.ndarray
    target_distance_before: np.ndarray
    target_distance_after: np.ndarray
    target_nll_before: np.ndarray
    target_nll_after: np.ndarray
    displacement_ratio: np.ndarray
    relative_margin_drop: np.ndarray
    bank: BuresBank


def _teacher_np(teacher: AdapterHead, values: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        device = next(teacher.parameters()).device
        x = torch.from_numpy(np.asarray(values, np.float32)).to(device)
        return teacher.logits(x).float().cpu().numpy()


def _margin_np(logits: np.ndarray, label: int) -> float:
    values = np.asarray(logits, np.float64)
    other = np.max(np.delete(values, int(label)))
    return float(values[int(label)] - other)


def _target_pool(bank: BuresBank, source: str, anchor_id: int) -> np.ndarray:
    values = np.asarray([s for s in bank.subject_list if s != source], dtype=str)
    rng = np.random.default_rng(c.stable_seed("bures-targets", bank.dataset, bank.fold, bank.seed, int(anchor_id)))
    if len(values) > c.K_TARGETS:
        values = values[rng.choice(len(values), c.K_TARGETS, replace=False)]
    return values


def _candidate_geometry(features: np.ndarray, labels: np.ndarray, subjects: np.ndarray, row_ids: np.ndarray, dataset: str, fold: int, seed: int, teacher: AdapterHead, *, mode: str, bank: BuresBank | None = None) -> Geometry:
    features = np.asarray(features, np.float64); labels = np.asarray(labels, np.int64); subjects = np.asarray(subjects).astype(str); row_ids = np.asarray(row_ids, np.int64)
    bank = bank or BuresBank(features, labels, subjects, row_ids, dataset=dataset, fold=fold, seed=seed)
    n, dim = features.shape; k = c.K_TARGETS; alpha_order = np.asarray(sorted(c.ALPHAS, reverse=True), np.float32); n_alpha = len(alpha_order)
    offsets = np.zeros((n, k, dim), np.float32); valid = np.zeros((n, k), bool); chosen_alpha = np.zeros((n, k), np.float32); targets = np.full((n, k), "", dtype="U32")
    support_pass = np.zeros((n, k), bool); class_pass = np.zeros((n, k), bool); d_before = np.full((n, k), np.nan); d_after = np.full((n, k), np.nan); nll_before = np.full((n, k), np.nan); nll_after = np.full((n, k), np.nan); displacement_ratio = np.full((n, k), np.nan); relative_margin_drop = np.full((n, k), np.nan)
    rng = np.random.default_rng(c.stable_seed("bures-random-affine", dataset, fold, seed, mode)); neighbor_idx, neighbor_dist = anchor_excluded_neighbors(features, row_ids, c.KNN_K); local_radius = np.nanmean(np.where(np.isfinite(neighbor_dist), neighbor_dist, np.nan), axis=1); local_radius[~np.isfinite(local_radius)] = np.inf
    directions = np.zeros((n, k, dim), np.float32)
    for position in range(n):
        source = str(subjects[position]); label = int(labels[position]); selected_targets = _target_pool(bank, source, int(row_ids[position]))
        for slot, target in enumerate(selected_targets[:k]):
            targets[position, slot] = str(target); structured = bank.displacement(position, str(target))
            if mode == "random": direction = matched_random_displacement(structured, bank, rng)
            elif mode == "mean":
                _, ms, mt, _, _ = bank.style(position, str(target)); direction = (mt - ms).astype(np.float32)
            elif mode == "manifold":
                mask = (subjects == str(target)) & (labels == label); direction = ((features[mask].mean(0) if np.any(mask) else features.mean(0)) - features[position]).astype(np.float32)
            else: direction = structured.astype(np.float32)
            directions[position, slot] = direction
    # One teacher call for all alpha candidates replaces tens of thousands of
    # tiny device transfers while preserving exact logits and gate semantics.
    anchors = features[:, None, None, :] + directions[:, :, None, :] * alpha_order[None, None, :, None]
    teacher_logits = _teacher_np(teacher, anchors.reshape(-1, dim)).reshape(n, k, n_alpha, 2)
    clean_logits = _teacher_np(teacher, features)
    for position in range(n):
        label = int(labels[position]); local_neighbors = neighbor_idx[position][neighbor_idx[position] >= 0]; local_radius_position = float(local_radius[position]); anchor_class = bool(len(local_neighbors) and np.mean(labels[local_neighbors] == label) >= 0.8); clean_margin = _margin_np(clean_logits[position], label)
        for slot, target in enumerate(targets[position]):
            if target == "": continue
            candidates = anchors[position, slot]; local_dist = np.linalg.norm(features[local_neighbors][None, :, :] - candidates[:, None, :], axis=2) if len(local_neighbors) else np.full((n_alpha, 0), np.inf)
            support_distance = local_dist.mean(1) if len(local_neighbors) else np.full(n_alpha, np.inf); support_ok = support_distance <= float(bank.radius.get(label, np.inf)); class_ok = np.zeros(n_alpha, bool)
            if len(local_neighbors):
                nearest = np.take(local_neighbors, np.argsort(local_dist, axis=1), axis=0); class_ok = np.sum(labels[nearest[:, : min(c.KNN_K, len(local_neighbors))]] == label, axis=1) >= 4
            cached = bank.target_affinity_stats(position, str(target)); before_d = before_nll = np.nan; after_d = np.full(n_alpha, np.nan); after_nll = np.full(n_alpha, np.nan)
            if cached is not None:
                values, mean, inv, logdet = cached; before_d = float(np.sort(np.linalg.norm(values - features[position][None], axis=1))[: min(c.KNN_K, len(values))].mean()); diff0 = features[position] - mean; before_nll = float(0.5 * (diff0 @ inv @ inv @ diff0 + logdet)); distances = np.linalg.norm(values[None, :, :] - candidates[:, None, :], axis=2); after_d = np.sort(distances, axis=1)[:, : min(c.KNN_K, len(values))].mean(1); diff = candidates - mean[None, :]; after_nll = 0.5 * (np.einsum("ai,ij,aj->a", diff, inv @ inv, diff) + logdet)
            margin = teacher_logits[position, slot, :, label] - teacher_logits[position, slot, :, 1 - label]; teacher_ok = (teacher_logits[position, slot].argmax(1) == label) & (margin > 0); ratios = np.linalg.norm(directions[position, slot][None, :] * alpha_order[:, None], axis=1) / max(local_radius_position, 1e-6); guard_ok = ratios <= 1.0 + 1e-8; affinity_ok = np.isfinite(after_d) & np.isfinite(after_nll) & (after_d < before_d) & (after_nll < before_nll); margin_drop = (clean_margin - margin) / (abs(clean_margin) + 1e-6); gate = anchor_class & support_ok & class_ok & teacher_ok & guard_ok & affinity_ok
            picked = np.flatnonzero(gate)
            if len(picked):
                aidx = int(picked[0]); offsets[position, slot] = directions[position, slot] * alpha_order[aidx]; valid[position, slot] = True; chosen_alpha[position, slot] = alpha_order[aidx]; support_pass[position, slot] = bool(support_ok[aidx]); class_pass[position, slot] = bool(class_ok[aidx]); d_before[position, slot] = before_d; d_after[position, slot] = after_d[aidx]; nll_before[position, slot] = before_nll; nll_after[position, slot] = after_nll[aidx]; displacement_ratio[position, slot] = ratios[aidx]; relative_margin_drop[position, slot] = margin_drop[aidx]
            else:
                offsets[position, slot] = directions[position, slot] * float(max(c.ALPHAS))
    return Geometry(offsets, valid, chosen_alpha, targets, support_pass, class_pass, d_before, d_after, nll_before, nll_after, displacement_ratio, relative_margin_drop, bank)


def _matched_masks(structured: Geometry, random: Geometry, dataset: str, fold: int, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    mask_s = np.zeros_like(structured.valid); mask_r = np.zeros_like(random.valid); rows = []
    for position in range(structured.valid.shape[0]):
        # Pair on target and selected alpha.  This makes the matched control
        # budget exact, rather than merely matching the number of candidates
        # after two independent validity searches.
        pairs = []
        right_by_key: dict[tuple[str, float], list[int]] = {}
        for j in np.flatnonzero(random.valid[position]):
            right_by_key.setdefault((str(random.targets[position, j]), round(float(random.alpha[position, j]), 8)), []).append(int(j))
        for i in np.flatnonzero(structured.valid[position]):
            key = (str(structured.targets[position, i]), round(float(structured.alpha[position, i]), 8))
            choices = right_by_key.get(key, [])
            if choices:
                pairs.append((int(i), choices.pop(0)))
        count = len(pairs)
        if pairs:
            rng = np.random.default_rng(c.stable_seed("bures-match", dataset, fold, seed, int(position)))
            rng.shuffle(pairs)
            for i, j in pairs:
                mask_s[position, i] = True; mask_r[position, j] = True
        pair_map = {i: j for i, j in pairs}
        for slot in range(structured.valid.shape[1]):
            if structured.targets[position, slot] == "":
                continue
            delta = structured.offsets[position, slot]
            random_slots = np.flatnonzero(random.targets[position] == structured.targets[position, slot])
            rs_slot = pair_map.get(int(slot))
            rs = random.offsets[position, rs_slot] if rs_slot is not None else np.zeros_like(delta)
            paired = rs_slot is not None
            rows.append({
                "dataset": dataset, "fold": fold, "seed": seed, "anchor_position": position,
                "target_subject": structured.targets[position, slot], "alpha": float(structured.alpha[position, slot]),
                "euclidean_norm_structured": float(np.linalg.norm(delta)), "euclidean_norm_random": float(np.linalg.norm(rs)) if paired else np.nan,
                "euclidean_norm_mismatch": float(abs(np.linalg.norm(delta) - np.linalg.norm(rs))) if paired else np.nan,
                "whitened_norm_structured": float(structured.bank.whitened_norm(delta)), "whitened_norm_random": float(structured.bank.whitened_norm(rs)) if paired else np.nan,
                "whitened_norm_mismatch": float(abs(structured.bank.whitened_norm(delta) - structured.bank.whitened_norm(rs))) if paired else np.nan,
                "structured_valid": bool(structured.valid[position, slot]), "random_valid": bool(random.valid[position, rs_slot]) if rs_slot is not None else False,
                "structured_matched_count": int(count), "random_matched_count": int(count),
                "matched_pair": paired,
                "alpha_mismatch": float(abs(float(structured.alpha[position, slot]) - float(random.alpha[position, rs_slot])) if paired and rs_slot is not None else np.nan),
            })
    return (mask_s, mask_r), pd.DataFrame(rows)


def _subject_batches(subjects: np.ndarray, seed: int, epoch: int) -> list[np.ndarray]:
    subjects = np.asarray(subjects).astype(str); rng = np.random.default_rng(c.stable_seed("subject-balanced", seed, epoch))
    groups = {s: np.flatnonzero(subjects == s).tolist() for s in c.subject_sort(np.unique(subjects))}
    for values in groups.values(): rng.shuffle(values)
    order = []
    for round_idx in range(max(map(len, groups.values()))):
        keys = list(groups); rng.shuffle(keys)
        for subject in keys:
            values = groups[subject]
            if round_idx < len(values): order.append(values[round_idx])
    return [np.asarray(order[start : start + c.BATCH_SIZE], np.int64) for start in range(0, len(order), c.BATCH_SIZE)]


def _transport_loss(model: AdapterHead, h: torch.Tensor, clean_logits: torch.Tensor, labels: torch.Tensor, geometry: Geometry, positions: np.ndarray, q: float, lam: float) -> tuple[torch.Tensor, dict[str, float]]:
    offsets = torch.from_numpy(geometry.offsets[positions]).to(h.device).detach()
    valid = torch.from_numpy(geometry.valid[positions]).to(h.device).reshape(-1).detach()
    owner = torch.arange(len(positions), device=h.device).repeat_interleave(offsets.shape[1])
    candidates = h[:, None, :] + offsets
    candidate_logits = model.head(candidates.flatten(0, 1))
    candidate_labels = labels[owner]
    clean_margin = clean_logits.gather(1, labels[:, None]).squeeze(1) - clean_logits.detach().masked_fill(F.one_hot(labels, 2).bool(), float("-inf")).max(1).values
    candidate_margin = candidate_logits.gather(1, candidate_labels[:, None]).squeeze(1) - candidate_logits.detach().masked_fill(F.one_hot(candidate_labels, 2).bool(), float("-inf")).max(1).values
    hardness = (clean_margin.detach()[owner] - candidate_margin.detach()) / (clean_margin.detach()[owner].abs() + 1e-6)
    clean_correct = clean_logits.detach().argmax(1).eq(labels)
    selected = []
    for anchor in range(len(positions)):
        idx = torch.nonzero(valid & owner.eq(anchor) & clean_correct[anchor], as_tuple=False).flatten()
        if not len(idx):
            continue
        count = max(1, int(math.ceil(float(q) * len(idx))))
        if q >= 0.999:
            selected.append(idx)
        else:
            selected.append(idx[torch.topk(hardness[idx], count, largest=True, sorted=False).indices.detach()])
    if not selected:
        return clean_logits.sum() * 0.0, {"selected": 0.0, "hardness": 0.0}
    picked = torch.cat(selected).detach()
    transport_ce = F.cross_entropy(candidate_logits[picked].float(), candidate_labels[picked])
    relative = F.relu(0.5 * clean_margin.detach()[owner[picked]] - candidate_margin[picked])
    return float(lam) * (transport_ce + relative.mean()), {"selected": float(len(picked)), "hardness": float(hardness[picked].detach().mean().cpu())}


def _evaluate(model: AdapterHead, cache: dict[str, np.ndarray], device: torch.device, method: str, dataset: str, fold: int, seed: int, q: float, lam: float) -> tuple[pd.DataFrame, float, float]:
    model.eval(); x = torch.from_numpy(cache["features"]).to(device)
    with torch.inference_mode(): logits = model.logits(x).float().cpu().numpy()
    labels = cache["labels"]; subjects = cache["subjects"].astype(str); pred = logits.argmax(1); rows = []
    for subject in c.subject_sort(np.unique(subjects)):
        mask = subjects == subject
        rows.append({"dataset": dataset, "fold": fold, "seed": seed, "method": method, "q": q, "lambda_T": lam, "subject_id": subject, "BA": float(balanced_accuracy_score(labels[mask], pred[mask])), "macro_F1": float(f1_score(labels[mask], pred[mask], average="macro", zero_division=0)), "trials": int(mask.sum())})
    frame = pd.DataFrame(rows)
    return frame, float(frame.BA.mean()), float(frame.macro_F1.mean())


def _geometry_subject_frame(geometry: Geometry | None, train_cache: dict[str, np.ndarray], dataset: str, fold: int, seed: int, method: str, q: float, lam: float) -> pd.DataFrame:
    if geometry is None:
        return pd.DataFrame()
    subjects = train_cache["subjects"].astype(str)
    rows = []
    for subject in c.subject_sort(np.unique(subjects)):
        mask = subjects == subject
        valid = geometry.valid[mask]
        def finite_mean(value: np.ndarray) -> float:
            value = np.asarray(value, np.float64).reshape(-1)
            value = value[np.isfinite(value)]
            return float(value.mean()) if len(value) else float("nan")
        def finite_median(value: np.ndarray) -> float:
            value = np.asarray(value, np.float64).reshape(-1)
            value = value[np.isfinite(value)]
            return float(np.median(value)) if len(value) else float("nan")
        rows.append({
            "dataset": dataset, "fold": fold, "seed": seed, "method": method,
            "q": q, "lambda_T": lam, "subject_id": subject,
            "coverage": float(np.mean(valid.sum(1) >= 1)),
            "valid_candidates": float(np.mean(valid.sum(1))),
            "target_distance_improvement": finite_mean(geometry.target_distance_before[mask] - geometry.target_distance_after[mask]),
            "target_nll_improvement": finite_mean(geometry.target_nll_before[mask] - geometry.target_nll_after[mask]),
            "class_pass_rate": float(np.mean(geometry.class_pass[mask])),
            "median_displacement_ratio": finite_median(geometry.displacement_ratio[mask]),
            "median_relative_margin_drop": finite_median(geometry.relative_margin_drop[mask]),
        })
    return pd.DataFrame(rows)


def train_method(method: str, dataset: str, fold: int, seed: int, q: float, lam: float, train_cache: dict[str, np.ndarray], valid_cache: dict[str, np.ndarray], device: torch.device, *, geometry_seed: int | None = None) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    c.set_seed(c.stable_seed("bures-source", dataset, fold, seed, method, q, lam)); dim = int(train_cache["features"].shape[1]); model = AdapterHead(dim).to(device); teacher = copy.deepcopy(model).to(device); teacher.eval()
    optimizer = torch.optim.AdamW([{"params": model.head.parameters(), "lr": c.HEAD_LR}, {"params": model.adapter.parameters(), "lr": c.ADAPTER_LR}], weight_decay=c.WEIGHT_DECAY)
    geometry = random_geometry = None; match_frame = pd.DataFrame(); diagnostics = []
    dynamic = method in ("V2-ME-HardSCST", "Manifold-Mixup", "Bures-Uniform", "Bures-HardRandom", "Bures-HardSCST")
    for epoch in range(c.WARMUP_EPOCHS + c.STAGE2_EPOCHS):
        model.train(); stage2 = epoch >= c.WARMUP_EPOCHS
        for param in model.adapter.parameters(): param.requires_grad = bool(stage2)
        if stage2 and dynamic:
            with torch.inference_mode(): teacher_features = teacher.features(torch.from_numpy(train_cache["features"]).to(device)).float().cpu().numpy()
            bank = BuresBank(teacher_features, train_cache["labels"], train_cache["subjects"], train_cache["indices"], dataset=dataset, fold=fold, seed=seed)
            mode = "mean" if method == "V2-ME-HardSCST" else ("manifold" if method == "Manifold-Mixup" else "structured")
            geometry = _candidate_geometry(teacher_features, train_cache["labels"], train_cache["subjects"], train_cache["indices"], dataset, fold, seed, teacher, mode=mode, bank=bank)
            if method in ("Bures-HardRandom", "Bures-HardSCST"):
                random_geometry = _candidate_geometry(teacher_features, train_cache["labels"], train_cache["subjects"], train_cache["indices"], dataset, fold, seed, teacher, mode="random", bank=bank)
                (structured_mask, random_mask), match_frame = _matched_masks(geometry, random_geometry, dataset, fold, seed)
                if method == "Bures-HardRandom": random_geometry.valid &= random_mask; geometry.valid &= structured_mask
                else: geometry.valid &= structured_mask; random_geometry.valid &= random_mask
        for positions in _subject_batches(train_cache["subjects"], seed, epoch):
            if not len(positions): continue
            y = torch.from_numpy(train_cache["labels"][positions]).long().to(device); x = torch.from_numpy(train_cache["features"][positions]).to(device)
            optimizer.zero_grad(set_to_none=True); h = model.features(x); clean_logits = model.head(h); loss = F.cross_entropy(clean_logits.float(), y); audit = {"selected": 0.0, "hardness": 0.0}
            if method == "Mixup":
                rng = np.random.default_rng(c.stable_seed("bures-mixup", dataset, fold, seed, epoch, int(positions[0]))); perm = torch.as_tensor(rng.permutation(len(positions)), device=device); weight = float(rng.beta(.4, .4)); mixed = weight * h + (1-weight) * h[perm]; soft = weight * F.one_hot(y, 2).float() + (1-weight) * F.one_hot(y[perm], 2).float(); loss = loss + 0.5 * (-(soft * torch.log_softmax(model.head(mixed).float(), -1)).sum(-1).mean())
            elif dynamic and geometry is not None:
                chosen = random_geometry if method == "Bures-HardRandom" else geometry
                if method == "Bures-Uniform":
                    # q=1 selects the full valid set; the recipe q remains in
                    # the report and is not used to change this control.
                    cf, audit = _transport_loss(model, h, clean_logits, y, chosen, positions, 1.0, lam)
                else:
                    cf, audit = _transport_loss(model, h, clean_logits, y, chosen, positions, q, lam)
                loss = loss + cf
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); optimizer.step();
            with torch.no_grad():
                teacher.adapter.weight.mul_(c.EMA_DECAY).add_(model.adapter.weight, alpha=1-c.EMA_DECAY); teacher.head.weight.mul_(c.EMA_DECAY).add_(model.head.weight, alpha=1-c.EMA_DECAY); teacher.head.bias.mul_(c.EMA_DECAY).add_(model.head.bias, alpha=1-c.EMA_DECAY)
        if stage2 and geometry is not None:
            diagnostics.append({"epoch": epoch+1, "coverage": float(np.mean(geometry.valid.sum(1) >= 1)), "target_distance_improvement": float(np.nanmean(geometry.target_distance_before - geometry.target_distance_after)), "target_nll_improvement": float(np.nanmean(geometry.target_nll_before - geometry.target_nll_after)), "class_pass_rate": float(np.mean(geometry.class_pass)), "median_displacement_ratio": float(np.nanmedian(geometry.displacement_ratio)), "median_relative_margin_drop": float(np.nanmedian(geometry.relative_margin_drop))})
    frame, ba, macro = _evaluate(model, valid_cache, device, method, dataset, fold, seed, q, lam)
    summary = {"dataset": dataset, "fold": fold, "seed": seed, "method": method, "q": q, "lambda_T": lam, "BA": ba, "macro_F1": macro, "diagnostic_last": diagnostics[-1] if diagnostics else {}, "epochs": c.WARMUP_EPOCHS + c.STAGE2_EPOCHS, "future_or_outer_opened": False}
    geo_frame = _geometry_subject_frame(geometry, train_cache, dataset, fold, seed, method, q, lam)
    return frame, summary, match_frame, geo_frame


def unit_dir(dataset: str, fold: int, seed: int) -> Path:
    return c.RUNTIME / "source_units" / dataset / f"fold-{fold}" / f"seed-{seed}"


def run_unit(dataset: str, fold: int, seed: int, device: torch.device) -> None:
    directory = unit_dir(dataset, fold, seed); marker = directory / "COMPLETE.json"
    if marker.is_file():
        return
    train = c.load_feature_cache(dataset, fold, seed, "train"); valid = c.load_feature_cache(dataset, fold, seed, "validation"); directory.mkdir(parents=True, exist_ok=True)
    all_rows, summaries, matches, geometries = [], [], [], []
    # Matched controls use the same source partition and sampler for each recipe.
    run_specs = [("ERM", None, None), ("Mixup", 0.50, 0.50), ("V2-ME-HardSCST", 0.50, 0.50), ("Manifold-Mixup", 0.50, 0.50)]
    for method, q, lam in run_specs:
        frame, summary, match, geo = train_method(method, dataset, fold, seed, float(q or 0.50), float(lam or 0.50), train, valid, device); all_rows.append(frame); summaries.append(summary)
        if len(geo): geometries.append(geo)
    for q, lam in RECIPES:
        for method in ("Bures-Uniform", "Bures-HardRandom", "Bures-HardSCST"):
            frame, summary, match, geo = train_method(method, dataset, fold, seed, q, lam, train, valid, device); all_rows.append(frame); summaries.append(summary)
            if len(geo): geometries.append(geo)
            if len(match): matches.append(match.assign(method=method, q=q, lambda_T=lam))
    c.write_csv(directory / "per_subject.csv", pd.concat(all_rows, ignore_index=True)); c.write_json(directory / "summary.json", summaries)
    if matches: c.write_csv(directory / "random_affine_matching.csv", pd.concat(matches, ignore_index=True))
    if geometries: c.write_csv(directory / "geometry_per_subject.csv", pd.concat(geometries, ignore_index=True))
    c.write_json(marker, {"dataset": dataset, "fold": fold, "seed": seed, "methods": len(summaries)})


def aggregate() -> None:
    files = sorted((c.RUNTIME / "source_units").rglob("per_subject.csv")); frames = [pd.read_csv(path) for path in files]
    if not frames: raise RuntimeError("NO_BURES_SOURCE_RESULTS")
    frame = pd.concat(frames, ignore_index=True); frame.to_csv(c.RESULTS / "SOURCE_PER_SUBJECT.csv", index=False)
    grouped = frame.groupby(["dataset", "method", "q", "lambda_T", "fold", "seed"], as_index=False).agg(BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), subjects=("subject_id", "nunique")); c.write_csv(c.RESULTS / "SOURCE_PER_FOLD.csv", grouped)
    baseline = grouped[grouped.method == "ERM"][["dataset", "fold", "seed", "BA"]].rename(columns={"BA": "ERM_BA"})
    # ERM is a real row in every unit (q=lambda=0.50 for reporting), but it
    # is deliberately absent from the recipe-method filter below.  Build its
    # baseline separately so comparisons can never silently become empty.
    recipe = grouped[grouped.method.isin(["Bures-Uniform", "Bures-HardRandom", "Bures-HardSCST", "Manifold-Mixup", "V2-ME-HardSCST"])].merge(baseline, on=["dataset", "fold", "seed"], how="left")
    recipe["delta_BA"] = recipe.BA - recipe.ERM_BA
    c.write_csv(c.RESULTS / "SOURCE_RECIPE_SEARCH.csv", recipe)
    comparisons = []
    for dataset in c.DATASETS:
        for q, lam in RECIPES:
            subset = recipe[(recipe.dataset == dataset) & (recipe.q == q) & (recipe.lambda_T == lam)]
            for method in ("Bures-HardSCST", "Bures-HardRandom", "Manifold-Mixup", "V2-ME-HardSCST", "Bures-Uniform"):
                # Primary recipe methods are evaluated at their declared
                # recipe.  The fixed controls (Manifold/V2) use q=lambda=.50
                # and are repeated as the comparator for each searched recipe.
                if method in ("Manifold-Mixup", "V2-ME-HardSCST"):
                    value = grouped[(grouped.dataset == dataset) & (grouped.method == method) & (grouped.fold.isin(c.FOLDS)) & (grouped.seed.isin(c.SEEDS))].set_index(["fold", "seed"])
                else:
                    value = grouped[(grouped.dataset == dataset) & (grouped.method == method) & (grouped.q == q) & (grouped.lambda_T == lam)].set_index(["fold", "seed"])
                control = grouped[(grouped.dataset == dataset) & (grouped.method == "ERM")].set_index(["fold", "seed"])
                if len(value):
                    delta = (value.BA - control.BA.reindex(value.index)).dropna()
                    comparisons.append({"dataset": dataset, "q": q, "lambda_T": lam, "comparison": f"{method}-ERM", "delta_BA": float(delta.mean()), "positive_folds": int((delta.groupby(level=0).mean() > 0).sum()), "units": int(len(delta))})
    c.write_csv(c.RESULTS / "CONTROL_COMPARISON.csv", pd.DataFrame(comparisons))
    match_files = sorted((c.RUNTIME / "source_units").rglob("random_affine_matching.csv")); c.write_csv(c.RESULTS / "RANDOM_AFFINE_MATCHING.csv", pd.concat([pd.read_csv(p) for p in match_files], ignore_index=True) if match_files else pd.DataFrame([{"status": "NO_MATCHING_ARTIFACT"}]))
    # Compact geometry summaries are populated from available unit diagnostics.
    diagnostics = []
    for path in sorted((c.RUNTIME / "source_units").rglob("summary.json")):
        for row in json.loads(path.read_text()):
            if row["method"] in ("Bures-HardSCST", "Bures-HardRandom", "Bures-Uniform"):
                diagnostics.append({"dataset": row["dataset"], "fold": row["fold"], "seed": row["seed"], "method": row["method"], "q": row["q"], "lambda_T": row["lambda_T"], **row.get("diagnostic_last", {})})
    dframe = pd.DataFrame(diagnostics); c.write_csv(c.RESULTS / "BURES_STATISTICS.csv", dframe); c.write_csv(c.RESULTS / "CANDIDATE_VALIDITY.csv", dframe[[x for x in ("dataset", "fold", "seed", "method", "q", "lambda_T", "coverage", "class_pass_rate") if x in dframe.columns]] if len(dframe) else pd.DataFrame([{"status": "NO_DIAGNOSTICS"}]))
    c.write_csv(c.RESULTS / "TARGET_AFFINITY.csv", dframe[[x for x in ("dataset", "fold", "seed", "method", "q", "lambda_T", "target_distance_improvement", "target_nll_improvement") if x in dframe.columns]] if len(dframe) else pd.DataFrame([{"status": "NO_DIAGNOSTICS"}]))
    geometry_files = sorted((c.RUNTIME / "source_units").rglob("geometry_per_subject.csv"))
    if geometry_files:
        c.write_csv(c.RESULTS / "GEOMETRY_PER_SUBJECT.csv", pd.concat([pd.read_csv(path) for path in geometry_files], ignore_index=True))
    else:
        c.write_csv(c.RESULTS / "GEOMETRY_PER_SUBJECT.csv", pd.DataFrame([{"status": "NO_GEOMETRY"}]))
    c.write_csv(c.RESULTS / "METHOD_SUMMARY.csv", grouped)
    stats = {"source_units": int(len(files)), "rows": int(len(frame)), "outer_or_sealed_opened": False, "matching_nonempty": bool(len(pd.read_csv(c.RESULTS / "RANDOM_AFFINE_MATCHING.csv")) > 0), "source_grid_complete": bool(len(files) == len(c.DATASETS) * len(c.FOLDS) * len(c.SEEDS))}
    c.write_json(c.RESULTS / "STATISTICS.json", stats)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", choices=c.DATASETS); parser.add_argument("--fold", type=int, choices=c.FOLDS); parser.add_argument("--seed", type=int, choices=c.SEEDS); parser.add_argument("--all", action="store_true"); parser.add_argument("--aggregate", action="store_true"); args = parser.parse_args(); c.ensure_dirs(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.all:
        for dataset in c.DATASETS:
            for fold in c.FOLDS:
                for seed in c.SEEDS:
                    print(f"[bures-source] START {dataset} f={fold} s={seed}", flush=True); run_unit(dataset, fold, seed, device); print(f"[bures-source] DONE {dataset} f={fold} s={seed}", flush=True)
    elif args.dataset is not None and args.fold is not None and args.seed is not None:
        run_unit(args.dataset, args.fold, args.seed, device)
    if args.aggregate: aggregate()


if __name__ == "__main__":
    main()
