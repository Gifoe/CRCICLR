"""Bounded 12-recipe source-only ME-HardSCST search.

Only development transitions are opened: OpenBMI session 1 -> session 2 and
WBCIC S1 -> S2.  WBCIC S3 and every outer/sealed resource are inaccessible
from this program by construction.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score

import v2_common as c
from candidate_engine import margins, upper_tail_loss
from mixed_effects import MixedEffectsBank
from training_components import BankRefreshTracker, EMATeacher, configure_scope, primary_total_loss


RECIPES = tuple((scope, q, lam) for scope in ("A", "B") for q in (0.25, 0.50) for lam in (0.25, 0.50, 1.00))


@dataclass
class Cache:
    indices: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    final: np.ndarray
    preblock: np.ndarray


@dataclass
class Geometry:
    offsets: np.ndarray
    base_valid: np.ndarray
    support_pass: np.ndarray
    semantic_pass: np.ndarray
    norm_pass: np.ndarray
    whitened_norm: np.ndarray
    alpha: np.ndarray
    target: np.ndarray
    bank: MixedEffectsBank


def preblock(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Frozen CleanRoom ATCNet coordinates immediately before norm+TCN."""
    y = model.drop1(F.elu(model.bn1(model.spatial(model.temporal(x[:, None])))))
    y = F.avg_pool2d(y, (1, 8))
    y = model.drop2(F.elu(model.bn2(model.sep_pw(model.sep_dw(y)))))
    y = F.avg_pool2d(y, (1, 8)).squeeze(2).transpose(1, 2)
    window = min(12, y.shape[1])
    starts = sorted(set((0, max(0, (y.shape[1] - window) // 2), max(0, y.shape[1] - window))))
    values = []
    for start in starts:
        token = y[:, start : start + window]
        attended, _ = model.attn(token, token, token, need_weights=False)
        values.append(token + attended)
    return torch.stack(values, dim=1)


def from_preblock(model: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    outputs = []
    for window in value.unbind(1):
        token = model.norm(window)
        outputs.append(model.tcn(token.transpose(1, 2))[:, :, -1])
    return torch.stack(outputs).mean(0)


def cache_path(dataset: str, fold: int, seed: int, role: str) -> Path:
    return c.RUNTIME / "source_cache" / dataset / f"fold-{fold}" / f"seed-{seed}" / f"{role}.npz"


def load_cache(path: Path) -> Cache:
    with np.load(path) as values:
        return Cache(values["indices"], values["labels"], values["subjects"].astype(str), values["final"], values["preblock"])


def build_cache(dataset: str, fold: int, seed: int, device: torch.device) -> tuple[Cache, Cache]:
    paths = {role: cache_path(dataset, fold, seed, role) for role in ("train", "validation")}
    if all(path.is_file() for path in paths.values()):
        return load_cache(paths["train"]), load_cache(paths["validation"])
    raw, metadata, _ = c.load_development_data(dataset)
    train_idx, valid_idx = c.source_indices(dataset, fold)
    net, _ = c.load_anchor("ATCNet-CleanRoom", dataset, fold, seed, device)
    net.eval()
    for role, indices in (("train", train_idx), ("validation", valid_idx)):
        finals, inputs = [], []
        with torch.inference_mode():
            for start in range(0, len(indices), c.BATCH_SIZE):
                idx = indices[start : start + c.BATCH_SIZE]
                x = torch.from_numpy(c.normalize_raw(raw[idx])).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    before = preblock(net, x)
                    final = from_preblock(net, before)
                inputs.append(before.float().cpu().numpy())
                finals.append(final.float().cpu().numpy())
        picked = metadata.iloc[indices]
        path = paths[role]; path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npz.part")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                indices=indices,
                labels=picked.label.to_numpy(np.int64),
                subjects=picked.subject_id.astype(str).to_numpy().astype("U"),
                final=np.concatenate(finals).astype(np.float32),
                preblock=np.concatenate(inputs).astype(np.float32),
            )
        os.replace(temporary, path)
    del net
    torch.cuda.empty_cache()
    return load_cache(paths["train"]), load_cache(paths["validation"])


@torch.no_grad()
def all_features(net, scope: str, cache: Cache, device: torch.device) -> np.ndarray:
    if scope == "A":
        return cache.final.astype(np.float32, copy=False)
    net.eval(); output = []
    for start in range(0, len(cache.labels), c.BATCH_SIZE * 2):
        value = torch.from_numpy(cache.preblock[start : start + c.BATCH_SIZE * 2]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output.append(from_preblock(net, value).float().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def _support_radius(features: torch.Tensor, labels: np.ndarray) -> dict[int, float]:
    result = {}
    for label in sorted(np.unique(labels).tolist()):
        positions = np.flatnonzero(labels == label)
        value = features[torch.as_tensor(positions, device=features.device)]
        means = []
        for start in range(0, len(value), 512):
            distance = torch.cdist(value[start : start + 512].float(), value.float())
            row = torch.arange(len(distance), device=value.device)
            col = torch.arange(start, start + len(distance), device=value.device)
            distance[row, col] = float("inf")
            means.append(distance.topk(3, largest=False).values.mean(1).cpu())
        result[int(label)] = float(torch.quantile(torch.cat(means), c.SUPPORT_QUANTILE))
    return result


def build_geometry(features: np.ndarray, labels: np.ndarray, subjects: np.ndarray, indices: np.ndarray, dataset: str, fold: int, seed: int, device: torch.device, *, factorized: bool = True) -> Geometry:
    bank = MixedEffectsBank(features, labels, subjects, indices)
    n, dim = features.shape
    directions = np.zeros((n, c.K_TARGETS, dim), np.float32)
    targets = np.empty((n, c.K_TARGETS), dtype="U32")
    for position in range(n):
        source = str(subjects[position])
        pool = np.asarray([value for value in bank.subjects if value != source])
        rng = np.random.default_rng(c.stable_seed("me-hard-target", dataset, fold, seed, int(indices[position])))
        selected = pool if len(pool) <= c.K_TARGETS else pool[rng.choice(len(pool), c.K_TARGETS, replace=False)]
        if len(selected) < c.K_TARGETS:
            selected = np.resize(selected, c.K_TARGETS)
        targets[position] = selected
        snapshot = bank.snapshot(position)
        source_index = bank.subject_index[source]
        label_index = bank.label_index[int(labels[position])]
        for target_idx, target in enumerate(selected):
            target_index = bank.subject_index[str(target)]
            if factorized:
                directions[position, target_idx] = snapshot.b[target_index] - snapshot.b[source_index]
            else:
                directions[position, target_idx] = snapshot.residual[target_index, label_index] - snapshot.residual[source_index, label_index]
    alpha = np.tile(np.asarray(c.ALPHAS, np.float32), c.K_TARGETS)
    offsets = (directions[:, :, None, :] * np.asarray(c.ALPHAS, np.float32)[None, None, :, None]).reshape(n, -1, dim)
    target_flat = np.repeat(targets, len(c.ALPHAS), axis=1)
    whitened = np.asarray([[bank.whitened_norm(value) for value in row] for row in offsets], np.float32)
    norm_pass = whitened <= bank.norm_radius + 1e-7
    feature_tensor = torch.from_numpy(features).to(device)
    radius = _support_radius(feature_tensor, labels)
    support_pass = np.zeros((n, offsets.shape[1]), bool)
    semantic_pass = np.zeros_like(support_pass)
    # Exact same all-training-point 3NN query, vectorized in larger chunks to
    # use the 32-GB server GPU efficiently without changing any neighbor.
    for start in range(0, n, 128):
        stop = min(n, start + 128)
        candidate = feature_tensor[start:stop, None, :] + torch.from_numpy(offsets[start:stop]).to(device)
        flat = candidate.reshape(-1, dim)
        distance = torch.cdist(flat.float(), feature_tensor.float())
        values, near = distance.topk(3, largest=False)
        owner_labels = torch.as_tensor(np.repeat(labels[start:stop], offsets.shape[1]), device=device)
        near_labels = torch.as_tensor(labels, device=device)[near]
        support = values.mean(1) <= torch.as_tensor([radius[int(value)] for value in owner_labels.cpu().tolist()], device=device)
        semantic = near_labels.eq(owner_labels[:, None]).sum(1) >= 2
        support_pass[start:stop] = support.reshape(stop - start, -1).cpu().numpy()
        semantic_pass[start:stop] = semantic.reshape(stop - start, -1).cpu().numpy()
    base = support_pass & semantic_pass & norm_pass
    return Geometry(offsets, base, support_pass, semantic_pass, norm_pass, whitened, alpha, target_flat, bank)


def unit_dir(scope: str, q: float | None, lam: float | None, dataset: str, fold: int, seed: int) -> Path:
    tag = "erm" if q is None else f"q{q:.2f}_l{lam:.2f}"
    return c.RUNTIME / "source_units" / scope / tag / dataset / f"fold-{fold}" / f"seed-{seed}"


def _batch_features(net, scope: str, cache: Cache, positions: np.ndarray, device: torch.device) -> torch.Tensor:
    if scope == "A":
        return torch.from_numpy(cache.final[positions]).to(device)
    return from_preblock(net, torch.from_numpy(cache.preblock[positions]).to(device))


@torch.no_grad()
def evaluate(net, scope: str, cache: Cache, device: torch.device) -> tuple[list[dict[str, object]], float, float]:
    net.eval(); logits = []
    for start in range(0, len(cache.labels), c.BATCH_SIZE * 2):
        pos = np.arange(start, min(len(cache.labels), start + c.BATCH_SIZE * 2))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = _batch_features(net, scope, cache, pos, device)
            logits.append(net.head(h).float().cpu().numpy())
    logits = np.concatenate(logits); pred = logits.argmax(1)
    rows = []
    for subject in c.subject_sort(np.unique(cache.subjects)):
        mask = cache.subjects.astype(str) == subject
        rows.append({"subject_id": subject, "BA": float(balanced_accuracy_score(cache.labels[mask], pred[mask])), "macro_F1": float(f1_score(cache.labels[mask], pred[mask], average="macro", zero_division=0))})
    return rows, float(np.mean([row["BA"] for row in rows])), float(np.mean([row["macro_F1"] for row in rows]))


def train_unit(dataset: str, fold: int, seed: int, scope: str, q: float | None, lam: float | None, train: Cache, valid: Cache, device: torch.device) -> dict[str, object]:
    directory = unit_dir(scope, q, lam, dataset, fold, seed)
    result_path = directory / "metrics.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    c.set_seed(c.stable_seed("me-hard-source-unit", dataset, fold, seed, scope, q, lam))
    net, anchor = c.load_anchor("ATCNet-CleanRoom", dataset, fold, seed, device)
    parameters = configure_scope("ATCNet-CleanRoom", net, scope)
    optimizer = torch.optim.AdamW(parameters, lr=c.LEARNING_RATE, weight_decay=c.WEIGHT_DECAY)
    teacher = EMATeacher(net, c.EMA_DECAY) if q is not None else None
    order_rng = np.random.default_rng(c.stable_seed("me-hard-source-order", dataset, fold, seed, scope))
    refresh = BankRefreshTracker()
    geometry = None
    bank_cosines: list[float] = []
    previous_b = None
    final_anchor_rows: list[dict[str, object]] = []
    for epoch in range(c.EPOCHS):
        net.train()
        if q is not None:
            if scope == "A" and geometry is not None:
                teacher_features = train.final.astype(np.float32, copy=False)
            else:
                refresh.refresh(epoch)
                teacher_features = all_features(teacher.model, scope, train, device)
                geometry = build_geometry(teacher_features, train.labels, train.subjects, train.indices, dataset, fold, seed, device, factorized=True)
                current_b = geometry.bank.full.b
                if previous_b is not None:
                    numerator = np.sum(previous_b * current_b, axis=1)
                    denominator = np.linalg.norm(previous_b, axis=1) * np.linalg.norm(current_b, axis=1)
                    bank_cosines.append(float(np.mean(numerator / np.maximum(denominator, 1e-8))))
                previous_b = current_b.copy()
            if scope == "A" and refresh.refreshes == 0:
                refresh.refresh(0)
                teacher_features = train.final.astype(np.float32, copy=False)
                geometry = build_geometry(teacher_features, train.labels, train.subjects, train.indices, dataset, fold, seed, device, factorized=True)
        losses = []
        final_anchor_rows = []
        epoch_order = order_rng.permutation(len(train.labels))
        for start in range(0, len(train.labels), c.BATCH_SIZE):
            positions = epoch_order[start : start + c.BATCH_SIZE]
            y = torch.as_tensor(train.labels[positions], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = _batch_features(net, scope, train, positions, device)
                clean_logits = net.head(h)
                if q is None:
                    loss = F.cross_entropy(clean_logits.float(), y)
                else:
                    offsets = torch.from_numpy(geometry.offsets[positions]).to(device).detach()
                    teacher_clean = torch.from_numpy(teacher_features[positions]).to(device)
                    teacher_candidates = teacher_clean[:, None, :] + offsets
                    teacher_logits = teacher.model.head(teacher_candidates.flatten(0, 1))
                    teacher_y = y[:, None].expand(-1, offsets.shape[1]).reshape(-1)
                    teacher_valid = (teacher_logits.detach().argmax(1) == teacher_y) & (margins(teacher_logits.detach(), teacher_y) > 0)
                    base_valid = torch.from_numpy(geometry.base_valid[positions]).to(device).reshape(-1)
                    valid_mask = base_valid & teacher_valid
                    student_candidates = h[:, None, :] + offsets
                    candidate_logits = net.head(student_candidates.flatten(0, 1))
                    owner = torch.arange(len(positions), device=device).repeat_interleave(offsets.shape[1])
                    cf_loss, _ = upper_tail_loss(clean_logits, candidate_logits, y, owner, valid_mask, q=float(q))
                    loss = primary_total_loss(clean_logits, y, cf_loss, float(lam))
                    if epoch == c.EPOCHS - 1:
                        clean_margin = margins(clean_logits.detach(), y)
                        candidate_margin = margins(candidate_logits.detach(), y[owner])
                        hardness = clean_margin[owner] - candidate_margin
                        correct = clean_logits.detach().argmax(1).eq(y)
                        for local, global_pos in enumerate(positions):
                            mask = valid_mask & owner.eq(local)
                            values = hardness[mask]
                            selected = int(math.ceil(float(q) * len(values))) if len(values) and bool(correct[local]) else 0
                            tail = torch.topk(values, selected).values if selected else values[:0]
                            final_anchor_rows.append({
                                "subject_id": str(train.subjects[global_pos]),
                                "clean_correct": bool(correct[local]),
                                "valid_count": int(mask.sum()),
                                "coverage_ge2": bool(correct[local] and mask.sum() >= 2),
                                "uniform_hardness": float(values.mean().cpu()) if len(values) else np.nan,
                                "tail_hardness": float(tail.mean().cpu()) if len(tail) else np.nan,
                                "semantic_pass_rate": float(geometry.semantic_pass[global_pos].mean()),
                            })
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 3.0); optimizer.step()
            if teacher is not None: teacher.update(net)
            losses.append(float(loss.detach().cpu()))
        print(f"[source] {dataset} f={fold} s={seed} scope={scope} q={q} l={lam} epoch={epoch+1} loss={np.mean(losses):.5f}", flush=True)
    subject_rows, ba, macro_f1 = evaluate(net, scope, valid, device)
    audit = pd.DataFrame(final_anchor_rows)
    clean = audit[audit.clean_correct] if len(audit) else audit
    coverage = float(clean.coverage_ge2.mean()) if len(clean) else 0.0
    median_valid = float(clean.valid_count.median()) if len(clean) else 0.0
    gap_by_subject = (clean.assign(gap=clean.tail_hardness - clean.uniform_hardness).groupby("subject_id").gap.mean().dropna() if len(clean) else pd.Series(dtype=float))
    if len(gap_by_subject):
        rng = np.random.default_rng(c.stable_seed("source-hardness-bootstrap", dataset, fold, seed, scope, q, lam))
        values = gap_by_subject.to_numpy(np.float64)
        draws = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(1)
        gap, gap_low = float(values.mean()), float(np.quantile(draws, .025))
    else:
        gap = gap_low = float("nan")
    result = {
        "dataset": dataset, "fold": fold, "seed": seed, "scope": scope,
        "q": q, "lambda_H": lam, "method": "ERM" if q is None else "ME-HardSCST",
        "BA": ba, "macro_F1": macro_f1, "subjects": len(subject_rows),
        "coverage_ge2": coverage, "median_valid_candidates": median_valid,
        "hardness_gap": gap, "hardness_gap_CI95_L": gap_low,
        "semantic_pass_rate": float(clean.semantic_pass_rate.mean()) if len(clean) else (1.0 if q is None else 0.0),
        "bank_stability": float(np.mean(bank_cosines)) if bank_cosines else 1.0,
        "bank_refreshes": refresh.refreshes,
        "epochs": c.EPOCHS, "anchor": str(anchor), "future_or_outer_opened": False,
    }
    directory.mkdir(parents=True, exist_ok=True)
    c.write_json(result_path, result)
    if q is not None:
        c.write_csv(directory / "anchor_audit.csv", audit)
        c.write_csv(directory / "bank_audit.csv", pd.DataFrame(geometry.bank.audit_rows()))
    checkpoint = directory / "model.pt"
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in net.state_dict().items()}, "result": result}, checkpoint)
    del net, teacher
    torch.cuda.empty_cache()
    return result


def aggregate() -> None:
    files = sorted((c.RUNTIME / "source_units").rglob("metrics.json"))
    rows = [json.loads(path.read_text()) for path in files]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("NO_SOURCE_RESULTS")
    baseline = frame[frame.method == "ERM"][["dataset", "fold", "seed", "scope", "BA"]].rename(columns={"BA": "ERM_BA"})
    # Each fold/seed/scope has one ERM row and six ME-HardSCST q/lambda
    # rows; the baseline therefore joins many recipes to one control.
    recipes = frame[frame.method == "ME-HardSCST"].merge(baseline, on=["dataset", "fold", "seed", "scope"], validate="many_to_one")
    recipes["delta_BA"] = recipes.BA - recipes.ERM_BA
    grouped = recipes.groupby(["scope", "q", "lambda_H", "dataset"], as_index=False).agg(
        BA=("BA", "mean"), ERM_BA=("ERM_BA", "mean"), delta_BA=("delta_BA", "mean"),
        valid_candidate_coverage=("coverage_ge2", "mean"), median_valid_candidates=("median_valid_candidates", "median"),
        hardness_gap=("hardness_gap", "mean"), hardness_gap_CI95_L=("hardness_gap_CI95_L", "mean"),
        semantic_pass_rate=("semantic_pass_rate", "mean"), bank_stability=("bank_stability", "mean"), units=("BA", "size"),
    )
    # Replace the mean of unit-level lower bounds with one biological-subject
    # bootstrap across the full fold/seed grid. Repeated anchors and fold/seed
    # appearances are averaged within each biological subject before draws.
    for row_index, row in grouped.iterrows():
        tag = f"q{float(row.q):.2f}_l{float(row.lambda_H):.2f}"
        root = c.RUNTIME / "source_units" / str(row.scope) / tag / str(row.dataset)
        audit_files = sorted(root.rglob("anchor_audit.csv"))
        audit = pd.concat([pd.read_csv(path) for path in audit_files], ignore_index=True)
        audit = audit[audit.clean_correct.astype(bool)].copy()
        audit["hardness_gap_subject"] = audit.tail_hardness - audit.uniform_hardness
        subject_values = audit.groupby("subject_id").hardness_gap_subject.mean().dropna().to_numpy(np.float64)
        if not len(subject_values):
            grouped.loc[row_index, ["hardness_gap", "hardness_gap_CI95_L"]] = np.nan
            continue
        rng = np.random.default_rng(c.stable_seed("source-grid-hardness-bootstrap", row.scope, row.q, row.lambda_H, row.dataset))
        draws = subject_values[rng.integers(0, len(subject_values), size=(10_000, len(subject_values)))].mean(1)
        grouped.loc[row_index, "hardness_gap"] = float(subject_values.mean())
        grouped.loc[row_index, "hardness_gap_CI95_L"] = float(np.quantile(draws, .025))
        grouped.loc[row_index, "hardness_gap_CI95_U"] = float(np.quantile(draws, .975))
    c.write_csv(c.RESULTS / "SOURCE_RECIPE_SEARCH.csv", grouped)
    c.write_csv(c.RESULTS / "CANDIDATE_COVERAGE.csv", recipes[["dataset", "fold", "seed", "scope", "q", "lambda_H", "coverage_ge2", "median_valid_candidates", "semantic_pass_rate"]])
    c.write_csv(c.RESULTS / "HARDNESS_DISTRIBUTION.csv", recipes[["dataset", "fold", "seed", "scope", "q", "lambda_H", "hardness_gap", "hardness_gap_CI95_L"]])
    bank_files = sorted((c.RUNTIME / "source_units").rglob("bank_audit.csv"))
    bank_rows = []
    for path in bank_files:
        value = pd.read_csv(path); value["unit"] = str(path.relative_to(c.RUNTIME)); bank_rows.append(value)
    c.write_csv(c.RESULTS / "BANK_DECOMPOSITION.csv", pd.concat(bank_rows, ignore_index=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=c.DATASETS, required=True)
    parser.add_argument("--fold", type=int, choices=c.FOLDS, required=True)
    parser.add_argument("--seed", type=int, choices=c.SEEDS, required=True)
    parser.add_argument("--scope", choices=("A", "B"), required=True)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    c.ensure_dirs(); device = torch.device("cuda")
    train, valid = build_cache(args.dataset, args.fold, args.seed, device)
    train_unit(args.dataset, args.fold, args.seed, args.scope, None, None, train, valid, device)
    for scope, q, lam in RECIPES:
        if scope == args.scope:
            train_unit(args.dataset, args.fold, args.seed, scope, q, lam, train, valid, device)
    if args.aggregate:
        aggregate()


if __name__ == "__main__":
    main()
