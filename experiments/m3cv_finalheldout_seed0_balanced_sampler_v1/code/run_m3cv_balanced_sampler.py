"""M3CV seed-0 outer-development comparison with a hierarchical balanced sampler.

Only the train sampler differs from the frozen trial-equal reference.  The
final-heldout cohort is represented by subject IDs for split construction but
its metadata, signals and labels are never opened by this runner.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path("/root/p4_m3cv_finalheldout_seed0_balanced_sampler_v1")
CACHE = Path("/root/m3cv_nm000166_external_replication/cache/m3cv_lr_4s_250hz")
FINAL = Path("/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK")
SIRE_SOURCE = FINAL / "experiments/persist_eeg_carrier_dualdataset_screen_v1/code/run_carrier_screen.py"
EEG_SOURCE = FINAL / "experiments/persist_eeg_carrier_dualdataset_screen_v1/code/eegnet_locked.py"
SIRE_SHA = "920af131aabc272317da128f42be9961d5592619ce85ca99192029d1181f126f"
EEG_SHA = "f7c513c3f3cd1f326a74b4e419bd693e15ee8980a7c378f1c0bee8215b8b89dd"

C, T, K = 64, 1000, 2
SESSIONS = ("ses-01", "ses-02")
EPOCHS, FIRST_ELIGIBLE, PATIENCE, BATCH_SIZE = 60, 10, 8, 128
LR, WEIGHT_DECAY, CLIP = 3e-4, 5e-4, 5.0
HELDOUT_RULE = "rank SHA256('m3cv-final-heldout-v1|' + subject_id), take first 18"
SAMPLER_VERSION = "hierarchical-balanced-cyclic-v1"
REFERENCE_OUTER_METRICS = Path(__file__).resolve().parents[3] / "experiments/m3cv_finalheldout_seed0_v1/seed0_outer_development_subject_metrics.csv"
OPENED_ARRAY_CELLS: set[tuple[str, str]] = set()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    b = io.BytesIO()
    torch.save(state, b)
    return hashlib.sha256(b.getvalue()).hexdigest()


def jwrite(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cwrite(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        out = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        out.writeheader()
        out.writerows(rows)


def twrite(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def subject_key(subject: str) -> int:
    return int(subject.split("-")[1])


def set_seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def direct(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    old_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path


def load_models() -> tuple[Any, Any, dict[str, Any]]:
    if digest(SIRE_SOURCE) != SIRE_SHA:
        raise RuntimeError("authoritative SIRE source hash drift")
    if digest(EEG_SOURCE) != EEG_SHA:
        raise RuntimeError("authoritative EEGNet source hash drift")
    sire, eeg = direct("m3cv_final_sire", SIRE_SOURCE), direct("m3cv_final_eeg", EEG_SOURCE)
    models = {
        "SIRE-EEG": sire.CompactLite(C, "bn"),
        "EEGNet": eeg.EEGNet(C, T),
    }
    audit: dict[str, Any] = {}
    dummy = torch.zeros(2, C, T)
    for name, model in models.items():
        logits, embedding = model(dummy)
        if tuple(logits.shape) != (2, K) or tuple(embedding.shape) != (2, 64):
            raise RuntimeError(f"{name} output schema drift")
        audit[name] = {
            "source": str(SIRE_SOURCE if name == "SIRE-EEG" else EEG_SOURCE),
            "source_sha256": SIRE_SHA if name == "SIRE-EEG" else EEG_SHA,
            "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "input": [C, T],
            "output": [K],
            "embedding": 64,
            "state_tensor_shapes": {k: list(v.shape) for k, v in model.state_dict().items()},
        }
    expected = 44_872 + 48 * C + 65 * K
    if audit["SIRE-EEG"]["parameters"] != expected:
        raise RuntimeError(f"SIRE parameter mismatch {audit['SIRE-EEG']['parameters']} != {expected}")
    return sire, eeg, audit


@dataclass(frozen=True)
class Cell:
    subject: str
    session: str
    x: Path
    y: Path
    channel_order: tuple[str, ...]
    shape: tuple[int, int, int]


def subject_inventory() -> list[str]:
    subjects = sorted((p.name for p in CACHE.glob("sub-*") if p.is_dir()), key=subject_key)
    if len(subjects) != 93:
        raise RuntimeError(f"expected 93 eligible cached M3CV subjects, got {len(subjects)}")
    return subjects


def inventory(subjects: list[str]) -> tuple[dict[tuple[str, str], Cell], tuple[str, ...]]:
    cells: dict[tuple[str, str], Cell] = {}
    orders: set[tuple[str, ...]] = set()
    for subject in subjects:
        for session in SESSIONS:
            meta_path = CACHE / subject / f"{session}_cache_metadata.json"
            if not meta_path.is_file():
                raise RuntimeError(f"missing metadata {meta_path}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            x, y = Path(meta["X"]), Path(meta["y"])
            shape = tuple(int(v) for v in meta["shape"])
            order = tuple(meta["channel_order"])
            if meta["subject"] != subject or meta["session"] != session or shape[1:] != (C, T):
                raise RuntimeError(f"cache metadata schema drift {meta_path}")
            if meta["class_order"] != ["left_hand", "right_hand"] or not x.is_file() or not y.is_file():
                raise RuntimeError(f"cache data path/schema drift {meta_path}")
            orders.add(order)
            cells[(subject, session)] = Cell(subject, session, x, y, order, shape)
    if len(orders) != 1 or len(next(iter(orders))) != C:
        raise RuntimeError("M3CV channel-order drift")
    return cells, next(iter(orders))


def open_cell(cell: Cell) -> tuple[np.ndarray, np.ndarray]:
    OPENED_ARRAY_CELLS.add((cell.subject, cell.session))
    x = np.load(cell.x, mmap_mode="r", allow_pickle=False)
    y = np.load(cell.y, mmap_mode="r", allow_pickle=False)
    if tuple(x.shape) != cell.shape or x.dtype != np.float32 or y.shape != (x.shape[0],):
        raise RuntimeError(f"cache array schema drift {cell.x}")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise RuntimeError(f"non-binary cache labels {cell.y}")
    return x, y


def heldout_split(subjects: list[str]) -> tuple[list[str], list[str]]:
    rank = lambda s: hashlib.sha256(f"m3cv-final-heldout-v1|{s}".encode()).hexdigest()
    heldout = sorted(sorted(subjects, key=rank)[:18], key=subject_key)
    development = sorted(set(subjects) - set(heldout), key=subject_key)
    if len(heldout) != 18 or len(development) != 75 or set(heldout) & set(development):
        raise RuntimeError("heldout split construction failure")
    return development, heldout


def folds(development: list[str], seed: int) -> list[dict[str, Any]]:
    p = np.asarray(development, dtype=object).copy()
    np.random.default_rng(seed).shuffle(p)
    out = []
    for fold in range(5):
        outer = sorted(p[fold * 15:(fold + 1) * 15].tolist(), key=subject_key)
        remaining = np.asarray(sorted(set(development) - set(outer), key=subject_key), dtype=object)
        np.random.default_rng(10_000 + seed * 100 + fold).shuffle(remaining)
        validation = sorted(remaining[:12].tolist(), key=subject_key)
        train = sorted(remaining[12:].tolist(), key=subject_key)
        if len(train) != 48 or len(validation) != 12 or len(outer) != 15:
            raise RuntimeError("development fold size drift")
        if set(train) & set(validation) or set(train) & set(outer) or set(validation) & set(outer):
            raise RuntimeError("development subject overlap")
        out.append({"fold": fold, "train": train, "validation": validation, "outer_development": outer})
    if set().union(*(set(x["outer_development"]) for x in out)) != set(development):
        raise RuntimeError("outer-development coverage drift")
    return out


def normalize(cells: dict[tuple[str, str], Cell], subjects: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    total = np.zeros(C, dtype=np.float64)
    square = np.zeros(C, dtype=np.float64)
    count = 0
    class_counts = {0: 0, 1: 0}
    for subject in subjects:
        for session in SESSIONS:
            x, y = open_cell(cells[(subject, session)])
            for start in range(0, len(x), 64):
                z = np.asarray(x[start:start + 64], dtype=np.float64)
                total += z.sum(axis=(0, 2))
                square += np.square(z).sum(axis=(0, 2))
                count += z.shape[0] * T
            class_counts[0] += int((y == 0).sum())
            class_counts[1] += int((y == 1).sum())
    mean = (total / count).astype(np.float32)
    std = np.sqrt(np.maximum(square / count - np.square(mean.astype(np.float64)), 1e-12)).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise RuntimeError("nonfinite normalizer")
    info = {"subjects": subjects, "sessions": list(SESSIONS), "samples_per_channel": int(count), "class_counts": class_counts,
            "mean_std_sha256": hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest()}
    return mean, std, info


def training_pool(cells: dict[tuple[str, str], Cell], subjects: list[str], mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[tuple[str, str, int], np.ndarray]]:
    blocks, labels = [], []
    per_cell: dict[tuple[str, str, int], np.ndarray] = {}
    offset = 0
    for subject in subjects:
        for session in SESSIONS:
            x, y = open_cell(cells[(subject, session)])
            xb, yb = np.asarray(x, dtype=np.float32).copy(), np.asarray(y, dtype=np.int64).copy()
            for cls in range(K):
                local = np.flatnonzero(yb == cls).astype(np.int64, copy=False)
                if not len(local):
                    raise RuntimeError(f"empty train sampling cell {(subject, session, cls)}")
                per_cell[(subject, session, cls)] = offset + local
            blocks.append(xb)
            labels.append(yb)
            offset += len(yb)
    x = np.concatenate(blocks, axis=0)
    y = np.concatenate(labels, axis=0)
    x -= mean[None, :, None]
    x /= np.maximum(std[None, :, None], 1e-6)
    if not np.isfinite(x).all() or set(np.unique(y).tolist()) != {0, 1}:
        raise RuntimeError("invalid supervised train pool")
    expected_cells = len(subjects) * len(SESSIONS) * K
    if len(per_cell) != expected_cells or offset != len(y):
        raise RuntimeError("hierarchical train-cell construction drift")
    return x, y, per_cell


def deterministic_rng(*parts: object) -> np.random.Generator:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False))


def even_draws_per_cell(original_samples: int, cells_count: int) -> int:
    if original_samples <= 0 or cells_count <= 0:
        raise RuntimeError("invalid sampler target")
    center = max(1, int(round((original_samples / cells_count) / 2.0)))
    candidates = sorted({2 * max(1, center + delta) for delta in range(-3, 4)})
    return min(candidates, key=lambda n: (abs(cells_count * n - original_samples), n))


def balanced_sampler(seed: int, fold: dict[str, Any], per_cell: dict[tuple[str, str, int], np.ndarray], original_samples: int) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    ordered_cells = [(subject, session, cls) for subject in fold["train"] for session in SESSIONS for cls in range(K)]
    if set(ordered_cells) != set(per_cell) or len(ordered_cells) != len(fold["train"]) * len(SESSIONS) * K:
        raise RuntimeError("sampler cell coverage drift")
    draws = even_draws_per_cell(original_samples, len(ordered_cells))
    sampled_samples = len(ordered_cells) * draws
    if sampled_samples % BATCH_SIZE:
        raise RuntimeError("balanced epoch does not preserve full batch size")
    plans: dict[int, dict[str, Any]] = {}
    epoch_audits: list[dict[str, Any]] = []
    manifest_dir = ROOT / "runtime/sampler_manifests" / f"seed{seed}" / f"fold{fold['fold']}"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        cyclic: dict[tuple[str, str, int], np.ndarray] = {}
        for key in ordered_cells:
            source = np.asarray(per_cell[key], dtype=np.int64)
            sequence: list[np.ndarray] = []
            needed, cycle = draws, 0
            while needed:
                permuted = deterministic_rng(SAMPLER_VERSION, seed, fold["fold"], epoch, *key, cycle).permutation(source)
                take = min(needed, len(permuted))
                sequence.append(permuted[:take])
                needed -= take
                cycle += 1
            selected = np.concatenate(sequence)
            if len(selected) != draws:
                raise RuntimeError("cyclic sampler length drift")
            for start in range(0, draws, len(source)):
                fragment = selected[start:start + len(source)]
                if len(np.unique(fragment)) != len(fragment):
                    raise RuntimeError("cyclic sampler repeated a trial within a shuffle cycle")
            cyclic[key] = selected
        indices: list[int] = []
        cell_ids: list[int] = []
        cell_id = {key: i for i, key in enumerate(ordered_cells)}
        for draw in range(draws):
            subject_order = deterministic_rng(SAMPLER_VERSION, "subjects", seed, fold["fold"], epoch, draw).permutation(np.asarray(fold["train"], dtype=object)).tolist()
            for subject in subject_order:
                for session in SESSIONS:
                    for cls in range(K):
                        key = (str(subject), session, cls)
                        indices.append(int(cyclic[key][draw]))
                        cell_ids.append(cell_id[key])
        index_array = np.asarray(indices, dtype=np.int64)
        cell_array = np.asarray(cell_ids, dtype=np.int16)
        if len(index_array) != sampled_samples or len(cell_array) != sampled_samples:
            raise RuntimeError("hierarchical sampler sequence size drift")
        sequence_sha = hashlib.sha256(index_array.tobytes() + cell_array.tobytes()).hexdigest()
        subject_counts = {subject: int(sum(1 for cid in cell_array if ordered_cells[int(cid)][0] == subject)) for subject in fold["train"]}
        session_counts = {session: int(sum(1 for cid in cell_array if ordered_cells[int(cid)][1] == session)) for session in SESSIONS}
        class_counts = {str(cls): int(sum(1 for cid in cell_array if ordered_cells[int(cid)][2] == cls)) for cls in range(K)}
        cell_counts = np.bincount(cell_array, minlength=len(ordered_cells)).astype(int)
        for start in range(0, sampled_samples, BATCH_SIZE):
            batch_cells = cell_array[start:start + BATCH_SIZE]
            batch_sessions = [ordered_cells[int(cid)][1] for cid in batch_cells]
            batch_classes = [ordered_cells[int(cid)][2] for cid in batch_cells]
            if batch_sessions.count(SESSIONS[0]) != BATCH_SIZE // 2 or batch_sessions.count(SESSIONS[1]) != BATCH_SIZE // 2 or batch_classes.count(0) != BATCH_SIZE // 2 or batch_classes.count(1) != BATCH_SIZE // 2:
                raise RuntimeError("batch-level session/class balance drift")
        if len(set(subject_counts.values())) != 1 or len(set(session_counts.values())) != 1 or len(set(class_counts.values())) != 1 or not np.all(cell_counts == draws):
            raise RuntimeError("epoch-level hierarchical exposure inequality")
        file = manifest_dir / f"epoch{epoch:03d}.npz"
        np.savez_compressed(file, indices=index_array, cell_ids=cell_array, ordered_cells=np.asarray(["|".join((s, se, str(c))) for s, se, c in ordered_cells]))
        plans[epoch] = {"indices": index_array, "sequence_sha256": sequence_sha, "file": str(file), "file_sha256": digest(file)}
        epoch_audits.append({"epoch": epoch, "sequence_sha256": sequence_sha, "file": str(file), "file_sha256": digest(file), "subject_exposure": subject_counts, "session_exposure": session_counts, "class_exposure": class_counts, "cell_exposure": {"min": int(cell_counts.min()), "max": int(cell_counts.max())}, "batches": sampled_samples // BATCH_SIZE})
    manifest_sha = hashlib.sha256(json.dumps([plans[e]["sequence_sha256"] for e in range(1, EPOCHS + 1)], separators=(",", ":")).encode()).hexdigest()
    audit = {"sampler": SAMPLER_VERSION, "fold": fold["fold"], "seed": seed, "hierarchy": ["subject", "session", "class", "trial"], "sampling": "cyclic independent shuffled permutations within each subject/session/class trial cell", "train_subjects": fold["train"], "cells": len(ordered_cells), "draws_per_cell_per_epoch": draws, "original_trial_equal_samples_per_epoch": original_samples, "balanced_samples_per_epoch": sampled_samples, "sample_difference": sampled_samples - original_samples, "original_updates_per_epoch": int(math.ceil(original_samples / BATCH_SIZE)), "balanced_updates_per_epoch": sampled_samples // BATCH_SIZE, "batch_size": BATCH_SIZE, "full_batch_only": True, "manifest_sha256": manifest_sha, "epochs": epoch_audits}
    return plans, audit


def evaluate_subject(model: torch.nn.Module, cell: Cell, mean: np.ndarray, std: np.ndarray, device: torch.device) -> dict[str, float]:
    x, y = open_cell(cell)
    logits: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), BATCH_SIZE):
            z = np.asarray(x[start:start + BATCH_SIZE], dtype=np.float32).copy()
            z -= mean[None, :, None]
            z /= np.maximum(std[None, :, None], 1e-6)
            tx = torch.from_numpy(z).to(device, non_blocking=True)
            logits.append(model(tx)[0].float().cpu().numpy())
    pred = np.concatenate(logits, axis=0).argmax(1)
    return {"BA": float(balanced_accuracy_score(y, pred)), "Macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)), "trials": int(len(y))}


def validation_score(model: torch.nn.Module, cells: dict[tuple[str, str], Cell], subjects: list[str], mean: np.ndarray, std: np.ndarray, device: torch.device) -> float:
    values = [evaluate_subject(model, cells[(subject, "ses-02")], mean, std, device)["BA"] for subject in subjects]
    return float(np.mean(values))


def construct(name: str, sire: Any, eeg: Any) -> torch.nn.Module:
    return sire.CompactLite(C, "bn") if name == "SIRE-EEG" else eeg.EEGNet(C, T)


def train_one(name: str, seed: int, fold: dict[str, Any], cells: dict[tuple[str, str], Cell], mean: np.ndarray, std: np.ndarray, norm_info: dict[str, Any], sire: Any, eeg: Any, audit: dict[str, Any], device: torch.device, x_train: np.ndarray, y_train: np.ndarray, plans: dict[int, dict[str, Any]], sampler_audit: dict[str, Any]) -> dict[str, Any]:
    directory = ROOT / "runtime/checkpoints" / f"seed{seed}" / name / f"fold{fold['fold']}"
    directory.mkdir(parents=True, exist_ok=True)
    latest, selected = directory / "latest.pt", directory / "selected.pt"
    fold_sha = hashlib.sha256(json.dumps(fold, sort_keys=True).encode()).hexdigest()
    set_seed(seed)
    model = construct(name, sire, eeg).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters != audit[name]["parameters"]:
        raise RuntimeError(f"{name} architecture audit mismatch")
    init = state_digest(copy.deepcopy(model.state_dict()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, history, best, best_epoch, best_state, no_improve = 1, [], -float("inf"), None, None, 0
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        required = {"init": init, "fold_sha256": fold_sha, "normalizer_sha256": norm_info["mean_std_sha256"], "seed": seed, "model": name, "sampler_manifest_sha256": sampler_audit["manifest_sha256"]}
        if any(saved.get(k) != v for k, v in required.items()):
            raise RuntimeError("unsafe resume")
        model.load_state_dict(saved["state"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start, history, best, best_epoch, best_state, no_improve = int(saved["epoch"]) + 1, saved["history"], float(saved["best"]), saved["best_epoch"], saved["best_state"], int(saved["no_improve"])
    began = time.perf_counter()
    for epoch in range(start, EPOCHS + 1):
        model.train()
        order = plans[epoch]["indices"]
        losses = []
        for start_i in range(0, len(order), BATCH_SIZE):
            idx = order[start_i:start_i + BATCH_SIZE]
            xb = torch.from_numpy(np.ascontiguousarray(x_train[idx])).to(device, non_blocking=True)
            yb = torch.from_numpy(y_train[idx]).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits, _ = model(xb)
                loss = F.cross_entropy(logits, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite CE")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        score = validation_score(model, cells, fold["validation"], mean, std, device)
        selected_now = epoch >= FIRST_ELIGIBLE and score > best + 1e-12
        if selected_now:
            best, best_epoch, best_state, no_improve = score, epoch, copy.deepcopy(model.state_dict()), 0
        elif epoch >= FIRST_ELIGIBLE:
            no_improve += 1
        history.append({"epoch": epoch, "CE": float(np.mean(losses)), "validation_ses02_subject_equal_BA": score, "selected": selected_now, "sampler_sequence_sha256": plans[epoch]["sequence_sha256"]})
        torch.save({"epoch": epoch, "state": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "history": history,
                    "best": best, "best_epoch": best_epoch, "best_state": best_state, "no_improve": no_improve, "init": init, "fold_sha256": fold_sha,
                    "normalizer_sha256": norm_info["mean_std_sha256"], "seed": seed, "model": name, "sampler_manifest_sha256": sampler_audit["manifest_sha256"]}, latest)
        if epoch == 1 or epoch % 5 == 0 or selected_now:
            print(f"[seed={seed} {name} fold={fold['fold']}] e={epoch:02d} ce={history[-1]['CE']:.4f} valS2={score:.4f}", flush=True)
        if epoch >= FIRST_ELIGIBLE and no_improve >= PATIENCE:
            print(f"[seed={seed} {name} fold={fold['fold']}] early-stop e={epoch:02d} patience={PATIENCE}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("no eligible checkpoint")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), selected)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"model": name, "seed": seed, "fold": fold["fold"], "selected_epoch": int(best_epoch), "best_validation_ses02_subject_equal_BA": float(best),
            "epochs_completed": int(history[-1]["epoch"]), "checkpoint": str(selected), "checkpoint_sha256": digest(selected), "parameters": parameters,
            "normalizer_sha256": norm_info["mean_std_sha256"], "fold_sha256": fold_sha, "elapsed_seconds": time.perf_counter() - began, "sampler_manifest_sha256": sampler_audit["manifest_sha256"], "all_epoch_sampler_sequence_sha256": [plans[e]["sequence_sha256"] for e in range(1, EPOCHS + 1)], "history": history}


def checkpoint_model(record: dict[str, Any], sire: Any, eeg: Any, device: torch.device) -> torch.nn.Module:
    model = construct(record["model"], sire, eeg).to(device)
    model.load_state_dict(torch.load(record["checkpoint"], map_location=device, weights_only=False))
    model.eval()
    return model


def bootstrap(delta: np.ndarray, key: str) -> tuple[float, float, float]:
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
    draws = delta[rng.integers(0, len(delta), size=(20_000, len(delta)))].mean(1)
    return float(delta.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def summarize(subject_rows: list[dict[str, Any]], population: str) -> list[dict[str, Any]]:
    per: dict[str, dict[str, dict[str, float]]] = {}
    for row in subject_rows:
        per.setdefault(row["model"], {}).setdefault(row["subject"], {})[row["session"]] = row
    models = ("EEGNet", "SIRE-EEG")
    summaries: dict[str, dict[str, float]] = {}
    for model in models:
        if len(per.get(model, {})) == 0:
            raise RuntimeError("empty subject population")
        values = []
        for subject, sessions in per[model].items():
            if set(sessions) != set(SESSIONS):
                raise RuntimeError(f"session coverage drift {population}/{model}/{subject}")
            values.append({"subject": subject, "future BA": sessions["ses-02"]["BA"], "future Macro-F1": sessions["ses-02"]["Macro_F1"], "WS-BA": min(sessions[s]["BA"] for s in SESSIONS)})
        summaries[model] = {metric: float(np.mean([v[metric] for v in values])) for metric in ("future BA", "future Macro-F1", "WS-BA")}
    rows = [{"row_type": "model", "population": population, "model": model, "metric": "", "mean": "", "CI95_low": "", "CI95_high": "", "subjects": len(per[model]), **summaries[model]} for model in models]
    subjects = sorted(set(per["EEGNet"]) & set(per["SIRE-EEG"]), key=subject_key)
    for metric in ("future BA", "WS-BA"):
        delta = np.asarray([((per["SIRE-EEG"][s]["ses-02"]["BA"] if metric == "future BA" else min(per["SIRE-EEG"][s][x]["BA"] for x in SESSIONS)) - (per["EEGNet"][s]["ses-02"]["BA"] if metric == "future BA" else min(per["EEGNet"][s][x]["BA"] for x in SESSIONS))) for s in subjects])
        mean, lo, hi = bootstrap(delta, f"m3cv-finalheldout-seed0-{population}-{metric}")
        rows.append({"row_type": "paired_SIRE_minus_EEGNet", "population": population, "model": "SIRE-EEG minus EEGNet", "metric": metric, "mean": mean, "CI95_low": lo, "CI95_high": hi, "subjects": len(subjects), "future BA": "", "future Macro-F1": "", "WS-BA": ""})
    return rows


def per_fold_summary(subject_rows: list[dict[str, Any]], setting: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], dict[str, dict[str, Any]]] = {}
    for row in subject_rows:
        key = (int(row["fold"]), row["model"], row["subject"])
        grouped.setdefault(key, {})[row["session"]] = row
    output: list[dict[str, Any]] = []
    for fold in range(5):
        for model in ("EEGNet", "SIRE-EEG"):
            subjects = sorted({subject for (f, m, subject) in grouped if f == fold and m == model}, key=subject_key)
            values = []
            for subject in subjects:
                sessions = grouped[(fold, model, subject)]
                if set(sessions) != set(SESSIONS):
                    raise RuntimeError("per-fold session coverage drift")
                values.append((sessions["ses-02"]["BA"], sessions["ses-02"]["Macro_F1"], min(sessions[s]["BA"] for s in SESSIONS)))
            output.append({"setting": setting, "fold": fold, "model": model, "subjects": len(subjects), "ses02_BA": float(np.mean([v[0] for v in values])), "ses02_Macro_F1": float(np.mean([v[1] for v in values])), "WS_BA": float(np.mean([v[2] for v in values]))})
    return output


def original_outer_reference() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not REFERENCE_OUTER_METRICS.is_file():
        raise RuntimeError(f"missing frozen original outer-development reference {REFERENCE_OUTER_METRICS}")
    rows: list[dict[str, Any]] = []
    with REFERENCE_OUTER_METRICS.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({"model": row["model"], "seed": int(row["seed"]), "fold": int(row["fold"]), "subject": row["subject"], "session": row["session"], "selected_epoch": int(row["selected_epoch"]), "checkpoint_sha256": row["checkpoint_sha256"], "BA": float(row["BA"]), "Macro_F1": float(row["Macro_F1"]), "trials": int(row["trials"])})
    if len(rows) != 300 or {r["model"] for r in rows} != {"EEGNet", "SIRE-EEG"}:
        raise RuntimeError("frozen original outer-development reference schema drift")
    return rows, {"path": str(REFERENCE_OUTER_METRICS), "sha256": digest(REFERENCE_OUTER_METRICS), "rows": len(rows), "population": "outer_development only"}


def run(seed: int) -> None:
    if seed != 0:
        raise RuntimeError("this authorized run is seed 0 only")
    ROOT.mkdir(parents=True, exist_ok=True)
    sire, eeg, audit = load_models()
    subjects = subject_inventory()
    development, heldout = heldout_split(subjects)
    cells, channel_order = inventory(development)
    if set(subject for subject, _ in cells) != set(development) or any(subject in heldout for subject, _ in cells):
        raise RuntimeError("heldout cell isolation drift")
    dev_folds = folds(development, seed)
    original_rows, original_reference_audit = original_outer_reference()
    if {r["subject"] for r in original_rows} != set(development):
        raise RuntimeError("original reference development cohort drift")
    jwrite(ROOT / "DEVELOPMENT_SPLIT_MANIFEST.json", {"dataset": "M3CV / NEMAR nm000166", "cache": str(CACHE), "eligible_subjects": subjects, "development_subjects": development, "final_heldout_subject_ids": heldout, "heldout_rule": HELDOUT_RULE, "seed": seed, "folds": dev_folds, "heldout_metadata_opened": False, "heldout_arrays_opened": False})
    twrite(ROOT / "OUTER_DEVELOPMENT_ONLY_PROTOCOL.md", ["# M3CV seed-0 hierarchical balanced-sampler variant", "", "- The frozen 93-subject cohort, 75/18 development/final-heldout split, and five 48/12/15 development folds are unchanged.", "- This run uses development data only: final-heldout metadata, signals, labels, checkpoint selection, and evaluation are excluded.", "- The only changed component is the training sampler: subject -> session -> class -> trial, with equal train-subject/session/class/cell exposure in every epoch.", "- Each cell uses deterministic cyclic shuffled trial permutations; a trial is never repeated inside one cycle.", "- One precomputed fold/epoch manifest is consumed by both EEGNet and SIRE-EEG. Batch size remains 128 with full batches only.", "- Architecture, optimizer, normalization, validation, early stopping, folds, outer-development evaluation, and metrics exactly match the frozen seed-0 reference.", "- Scope: seed 0 only; no final-heldout data access, seed 1/2, or tuning."])
    twrite(ROOT / "MODEL_AUDIT.md", ["# Final-model audit", "", f"SIRE source: `{SIRE_SOURCE}`", f"SIRE SHA-256: `{SIRE_SHA}`", f"EEGNet source: `{EEG_SOURCE}`", f"EEGNet SHA-256: `{EEG_SHA}`", "", f"SIRE trainable parameters: {audit['SIRE-EEG']['parameters']:,} = 44,872 + 48*64 + 65*2.", f"EEGNet trainable parameters: {audit['EEGNet']['parameters']:,}.", "SIRE remains CompactLite_BN: temporal kernels 15/63/127; 8-to-16 grouped spatial branches; width 48; refinements 15 and 31; pool 8; 512-to-64 embedding; LayerNorm(64); 2-class head.", "Only the sampler differs from the frozen reference.", "", "## Preserved channel order", "", ", ".join(f"`{x}`" for x in channel_order)])
    jwrite(ROOT / "MODEL_AUDIT.json", audit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records: list[dict[str, Any]] = []
    normalizers: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    sampler_audits: dict[str, dict[str, Any]] = {}
    for fold in dev_folds:
        mean, std, info = normalize(cells, fold["train"])
        normalizers[fold["fold"]] = (mean, std, info)
        (ROOT / "normalizers").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ROOT / "normalizers" / f"seed{seed}_fold{fold['fold']}.npz", mean=mean, std=std, info=json.dumps(info, sort_keys=True))
        x_train, y_train, per_cell = training_pool(cells, fold["train"], mean, std)
        plans, sampler_audit = balanced_sampler(seed, fold, per_cell, len(y_train))
        sampler_audits[str(fold["fold"])] = sampler_audit
        for name in ("EEGNet", "SIRE-EEG"):
            records.append(train_one(name, seed, fold, cells, mean, std, info, sire, eeg, audit, device, x_train, y_train, plans, sampler_audit))
        del x_train, y_train, per_cell, plans
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for fold in range(5):
        fold_records = [r for r in records if r["fold"] == fold]
        if len(fold_records) != 2 or fold_records[0]["all_epoch_sampler_sequence_sha256"] != fold_records[1]["all_epoch_sampler_sequence_sha256"] or fold_records[0]["sampler_manifest_sha256"] != fold_records[1]["sampler_manifest_sha256"]:
            raise RuntimeError("EEGNet/SIRE sampler-manifest mismatch")
        sampler_audits[str(fold)]["model_manifest_usage"] = {r["model"]: {"full_60_epoch_manifest_sha256": r["sampler_manifest_sha256"], "epochs_consumed": r["epochs_completed"], "consumed_prefix_sequence_sha256": [h["sampler_sequence_sha256"] for h in r["history"]]} for r in fold_records}
        sampler_audits[str(fold)]["full_manifest_identical_for_EEGNet_and_SIRE"] = True
    jwrite(ROOT / "BALANCED_SAMPLER_AUDIT.json", {"sampler": SAMPLER_VERSION, "seed": seed, "folds": sampler_audits, "same_precomputed_manifests_used_by_models": True})
    jwrite(ROOT / "seed0_training_records.json", {"seed": seed, "records": records})
    jwrite(ROOT / "TRAINING_AUDIT.json", {"normalizers": {str(k): v[2] for k, v in normalizers.items()}, "checkpoints": [{k: r[k] for k in ("model", "fold", "selected_epoch", "checkpoint", "checkpoint_sha256", "normalizer_sha256", "fold_sha256", "sampler_manifest_sha256")} for r in records], "original_outer_development_reference": original_reference_audit})
    outer_rows: list[dict[str, Any]] = []
    record_by = {(r["model"], r["fold"]): r for r in records}
    for fold in dev_folds:
        mean, std, _ = normalizers[fold["fold"]]
        for name in ("EEGNet", "SIRE-EEG"):
            record = record_by[(name, fold["fold"])]
            model = checkpoint_model(record, sire, eeg, device)
            for subject in fold["outer_development"]:
                for session in SESSIONS:
                    outer_rows.append({"model": name, "seed": seed, "fold": fold["fold"], "subject": subject, "session": session, "selected_epoch": record["selected_epoch"], "checkpoint_sha256": record["checkpoint_sha256"], **evaluate_subject(model, cells[(subject, session)], mean, std, device)})
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if any(subject in heldout for subject, _ in OPENED_ARRAY_CELLS) or set(subject for subject, _ in OPENED_ARRAY_CELLS) - set(development):
        raise RuntimeError("heldout array access detected")
    if {r["subject"] for r in outer_rows} != set(development) or len(outer_rows) != 300:
        raise RuntimeError("outer-development coverage drift")
    fields = ["model", "seed", "fold", "subject", "session", "selected_epoch", "checkpoint_sha256", "BA", "Macro_F1", "trials"]
    cwrite(ROOT / "seed0_outer_development_subject_metrics.csv", outer_rows, fields)
    balanced_summary = summarize(outer_rows, "outer_development")
    original_summary = summarize(original_rows, "outer_development")
    summary_rows = [{"setting": "original_trial_equal", **row} for row in original_summary] + [{"setting": "hierarchical_balanced", **row} for row in balanced_summary]
    cwrite(ROOT / "seed0_outer_development_summary.csv", summary_rows, ["setting", "row_type", "population", "model", "metric", "mean", "CI95_low", "CI95_high", "subjects", "future BA", "future Macro-F1", "WS-BA"])
    per_fold = per_fold_summary(original_rows, "original_trial_equal") + per_fold_summary(outer_rows, "hierarchical_balanced")
    cwrite(ROOT / "seed0_outer_development_per_fold_comparison.csv", per_fold, ["setting", "fold", "model", "subjects", "ses02_BA", "ses02_Macro_F1", "WS_BA"])
    original_models = {r["model"]: r for r in original_summary if r["row_type"] == "model"}
    balanced_models = {r["model"]: r for r in balanced_summary if r["row_type"] == "model"}
    original_delta = {r["metric"]: r for r in original_summary if r["row_type"] == "paired_SIRE_minus_EEGNet"}
    balanced_delta = {r["metric"]: r for r in balanced_summary if r["row_type"] == "paired_SIRE_minus_EEGNet"}
    twrite(ROOT / "ORIGINAL_VS_BALANCED_COMPARISON.md", ["# M3CV outer-development: original trial-equal versus hierarchical balanced sampling", "", "All rows use the same frozen seed-0 development folds. The new run changes only the sampler and does not access final-heldout arrays.", "", "| Setting | Model | S2 BA | S2 Macro-F1 | WS-BA |", "|---|---|---:|---:|---:|"] + [f"| {setting} | {model} | {float(row['future BA']):.4f} | {float(row['future Macro-F1']):.4f} | {float(row['WS-BA']):.4f} |" for setting, values in (("Original trial-equal", original_models), ("Hierarchical balanced", balanced_models)) for model, row in values.items()] + ["", "| Setting | SIRE-EEG minus EEGNet | S2 BA | 95% CI | WS-BA | 95% CI |", "|---|---|---:|---:|---:|---:|"] + [f"| {setting} | paired subject-level | {float(values['future BA']['mean']):+.4f} | [{float(values['future BA']['CI95_low']):+.4f}, {float(values['future BA']['CI95_high']):+.4f}] | {float(values['WS-BA']['mean']):+.4f} | [{float(values['WS-BA']['CI95_low']):+.4f}, {float(values['WS-BA']['CI95_high']):+.4f}] |" for setting, values in (("Original trial-equal", original_delta), ("Hierarchical balanced", balanced_delta))] + ["", f"Balanced sampling improves SIRE relative to EEGNet on S2 BA: `{float(balanced_delta['future BA']['mean']) > float(original_delta['future BA']['mean'])}`.", f"Balanced sampling improves SIRE relative to EEGNet on WS-BA: `{float(balanced_delta['WS-BA']['mean']) > float(original_delta['WS-BA']['mean'])}`."])
    jwrite(ROOT / "HELDOUT_ISOLATION_AUDIT.json", {"final_heldout_subject_ids": heldout, "final_heldout_metadata_opened": False, "final_heldout_signal_or_label_arrays_opened": False, "opened_array_cells": [{"subject": subject, "session": session} for subject, session in sorted(OPENED_ARRAY_CELLS, key=lambda x: (subject_key(x[0]), x[1]))], "opened_array_subjects": sorted({subject for subject, _ in OPENED_ARRAY_CELLS}, key=subject_key), "all_opened_arrays_are_development_only": True})
    twrite(ROOT / "SEED0_OUTER_DEVELOPMENT_RESULT_SUMMARY.md", ["# M3CV hierarchical balanced-sampler seed-0 result", "", "Development-only result: final-heldout data were not read or evaluated.", "", "| Model | S2 BA | S2 Macro-F1 | WS-BA |", "|---|---:|---:|---:|"] + [f"| {model} | {float(row['future BA']):.4f} | {float(row['future Macro-F1']):.4f} | {float(row['WS-BA']):.4f} |" for model, row in balanced_models.items()] + ["", "| SIRE-EEG minus EEGNet | Mean | 95% subject bootstrap CI |", "|---|---:|---:"] + [f"| {metric} | {float(row['mean']):+.4f} | [{float(row['CI95_low']):+.4f}, {float(row['CI95_high']):+.4f}] |" for metric, row in balanced_delta.items()] + ["", "See `ORIGINAL_VS_BALANCED_COMPARISON.md` and `BALANCED_SAMPLER_AUDIT.json` for the frozen-reference comparison and sampler proof."])
    print("M3CV_BALANCED_SAMPLER_SEED0_OUTER_DEVELOPMENT_COMPLETE", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args.seed)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"M3CV_BALANCED_SAMPLER_INVALID: {type(exc).__name__}: {exc}", flush=True)
        raise
