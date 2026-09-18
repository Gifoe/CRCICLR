"""Frozen M3CV final-heldout seed-0 supervised replication.

This runner intentionally consumes the validated M3CV cache only.  It never
opens final-heldout signal/label arrays until all five development checkpoints
have been selected and their provenance has been frozen.
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

ROOT = Path("/root/p4_m3cv_finalheldout_seed0_v1")
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


def inventory() -> tuple[list[str], dict[tuple[str, str], Cell], tuple[str, ...]]:
    subjects = sorted((p.name for p in CACHE.glob("sub-*") if p.is_dir()), key=subject_key)
    if len(subjects) != 93:
        raise RuntimeError(f"expected 93 eligible cached M3CV subjects, got {len(subjects)}")
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
    return subjects, cells, next(iter(orders))


def open_cell(cell: Cell) -> tuple[np.ndarray, np.ndarray]:
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


def training_pool(cells: dict[tuple[str, str], Cell], subjects: list[str], mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blocks, labels = [], []
    for subject in subjects:
        for session in SESSIONS:
            x, y = open_cell(cells[(subject, session)])
            blocks.append(np.asarray(x, dtype=np.float32).copy())
            labels.append(np.asarray(y, dtype=np.int64).copy())
    x = np.concatenate(blocks, axis=0)
    y = np.concatenate(labels, axis=0)
    x -= mean[None, :, None]
    x /= np.maximum(std[None, :, None], 1e-6)
    if not np.isfinite(x).all() or set(np.unique(y).tolist()) != {0, 1}:
        raise RuntimeError("invalid supervised train pool")
    return x, y


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


def train_one(name: str, seed: int, fold: dict[str, Any], cells: dict[tuple[str, str], Cell], mean: np.ndarray, std: np.ndarray, norm_info: dict[str, Any], sire: Any, eeg: Any, audit: dict[str, Any], device: torch.device) -> dict[str, Any]:
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
    x_train, y_train = training_pool(cells, fold["train"], mean, std)
    start, history, best, best_epoch, best_state, no_improve = 1, [], -float("inf"), None, None, 0
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        required = {"init": init, "fold_sha256": fold_sha, "normalizer_sha256": norm_info["mean_std_sha256"], "seed": seed, "model": name}
        if any(saved.get(k) != v for k, v in required.items()):
            raise RuntimeError("unsafe resume")
        model.load_state_dict(saved["state"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start, history, best, best_epoch, best_state, no_improve = int(saved["epoch"]) + 1, saved["history"], float(saved["best"]), saved["best_epoch"], saved["best_state"], int(saved["no_improve"])
    began = time.perf_counter()
    for epoch in range(start, EPOCHS + 1):
        model.train()
        order = np.random.default_rng(1_000_000 * seed + 10_000 * fold["fold"] + epoch).permutation(len(y_train))
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
        history.append({"epoch": epoch, "CE": float(np.mean(losses)), "validation_ses02_subject_equal_BA": score, "selected": selected_now})
        torch.save({"epoch": epoch, "state": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "history": history,
                    "best": best, "best_epoch": best_epoch, "best_state": best_state, "no_improve": no_improve, "init": init, "fold_sha256": fold_sha,
                    "normalizer_sha256": norm_info["mean_std_sha256"], "seed": seed, "model": name}, latest)
        if epoch == 1 or epoch % 5 == 0 or selected_now:
            print(f"[seed={seed} {name} fold={fold['fold']}] e={epoch:02d} ce={history[-1]['CE']:.4f} valS2={score:.4f}", flush=True)
        if epoch >= FIRST_ELIGIBLE and no_improve >= PATIENCE:
            print(f"[seed={seed} {name} fold={fold['fold']}] early-stop e={epoch:02d} patience={PATIENCE}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("no eligible checkpoint")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), selected)
    del x_train, y_train
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"model": name, "seed": seed, "fold": fold["fold"], "selected_epoch": int(best_epoch), "best_validation_ses02_subject_equal_BA": float(best),
            "epochs_completed": int(history[-1]["epoch"]), "checkpoint": str(selected), "checkpoint_sha256": digest(selected), "parameters": parameters,
            "normalizer_sha256": norm_info["mean_std_sha256"], "fold_sha256": fold_sha, "elapsed_seconds": time.perf_counter() - began, "history": history}


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


def run(seed: int) -> None:
    if seed != 0:
        raise RuntimeError("this authorized run is seed 0 only")
    ROOT.mkdir(parents=True, exist_ok=True)
    sire, eeg, audit = load_models()
    subjects, cells, channel_order = inventory()
    development, heldout = heldout_split(subjects)
    dev_folds = folds(development, seed)
    jwrite(ROOT / "FINAL_HELDOUT_MANIFEST.json", {"dataset": "M3CV / NEMAR nm000166", "cache": str(CACHE), "eligible_subjects": subjects, "development_subjects": development,
           "final_heldout_subjects": heldout, "heldout_rule": HELDOUT_RULE, "seed": seed, "folds": dev_folds, "heldout_labels_opened_before_checkpoint_freeze": False})
    twrite(ROOT / "PROTOCOL.md", ["# M3CV final-heldout supervised protocol", "", "- Analysis cohort: 93 cached eligible subjects; 75 development and 18 final-heldout.",
           f"- Final-heldout assignment: `{HELDOUT_RULE}`. It uses subject IDs only and is fixed before model construction.", "- Every train subject contributes both ses-01 and ses-02 LH/RH trials; LH=0, RH=1.",
           "- Per development fold: 48 train, 12 validation, 15 outer-development subjects; all partitions are subject-disjoint.", "- Input: native direct 64x1000 float32 EEG cache at 250 Hz, with preserved 64-channel order.",
           "- Models: direct imports of final CompactLite_BN/SIRE-EEG and canonical EEGNet. Only C=64 is adapted.", "- Optimizer: ordinary CE, AdamW lr=3e-4, weight_decay=5e-4, gradient clip=5.0, maximum 60 epochs.",
           "- Selection: validation ses-02 subject-equal BA; eligible epoch >=10; earliest strict tie. Stop after 8 non-improving eligible epochs.",
           "- Normalizer: channel mean/std fit on fold TRAIN subjects using both sessions only; frozen for validation, outer-development and final-heldout.",
           "- Final-heldout signals/labels are opened only after all five development checkpoints and their normalizers are frozen.", "- Scope: seed 0 only; no seed 1/2 is launched by this run."])
    twrite(ROOT / "MODEL_AUDIT.md", ["# Final-model audit", "", f"SIRE source: `{SIRE_SOURCE}`", f"SIRE SHA-256: `{SIRE_SHA}`", f"EEGNet source: `{EEG_SOURCE}`", f"EEGNet SHA-256: `{EEG_SHA}`", "",
           f"SIRE trainable parameters: {audit['SIRE-EEG']['parameters']:,} = 44,872 + 48*64 + 65*2.", f"EEGNet trainable parameters: {audit['EEGNet']['parameters']:,}.",
           "SIRE remains CompactLite_BN: temporal kernels 15/63/127; 8-to-16 grouped spatial branches; width 48; refinements 15 and 31; pool 8; 512-to-64 embedding; LayerNorm(64); 2-class head.",
           "Only M3CV input C=64 is adapted. The 64D embedding and 2-class head are unchanged.", "", "## Preserved channel order", "", ", ".join(f"`{x}`" for x in channel_order)])
    jwrite(ROOT / "MODEL_AUDIT.json", audit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, normalizers = [], {}
    for fold in dev_folds:
        mean, std, info = normalize(cells, fold["train"])
        normalizers[fold["fold"]] = (mean, std, info)
        (ROOT / "normalizers").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ROOT / "normalizers" / f"seed{seed}_fold{fold['fold']}.npz", mean=mean, std=std, info=json.dumps(info, sort_keys=True))
        for name in ("EEGNet", "SIRE-EEG"):
            records.append(train_one(name, seed, fold, cells, mean, std, info, sire, eeg, audit, device))
    jwrite(ROOT / "seed0_training_records.json", {"seed": seed, "records": records})
    jwrite(ROOT / "CHECKPOINT_FREEZE_AUDIT.json", {"final_heldout_labels_opened_before_freeze": False, "normalizers": {str(k): v[2] for k, v in normalizers.items()},
           "checkpoints": [{k: r[k] for k in ("model", "fold", "selected_epoch", "checkpoint", "checkpoint_sha256", "normalizer_sha256", "fold_sha256")} for r in records]})
    outer_rows: list[dict[str, Any]] = []
    held_checkpoint_rows: list[dict[str, Any]] = []
    record_by = {(r["model"], r["fold"]): r for r in records}
    for fold in dev_folds:
        mean, std, info = normalizers[fold["fold"]]
        for name in ("EEGNet", "SIRE-EEG"):
            record = record_by[(name, fold["fold"])]
            model = checkpoint_model(record, sire, eeg, device)
            for subject in fold["outer_development"]:
                for session in SESSIONS:
                    outer_rows.append({"model": name, "seed": seed, "fold": fold["fold"], "subject": subject, "session": session, "selected_epoch": record["selected_epoch"], "checkpoint_sha256": record["checkpoint_sha256"], **evaluate_subject(model, cells[(subject, session)], mean, std, device)})
            for subject in heldout:
                for session in SESSIONS:
                    held_checkpoint_rows.append({"model": name, "seed": seed, "fold": fold["fold"], "subject": subject, "session": session, "selected_epoch": record["selected_epoch"], "checkpoint_sha256": record["checkpoint_sha256"], **evaluate_subject(model, cells[(subject, session)], mean, std, device)})
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    fields = ["model", "seed", "fold", "subject", "session", "selected_epoch", "checkpoint_sha256", "BA", "Macro_F1", "trials"]
    cwrite(ROOT / "seed0_outer_development_subject_metrics.csv", outer_rows, fields)
    cwrite(ROOT / "seed0_final_heldout_checkpoint_metrics.csv", held_checkpoint_rows, fields)
    aggregated: list[dict[str, Any]] = []
    for name in ("EEGNet", "SIRE-EEG"):
        for subject in heldout:
            for session in SESSIONS:
                part = [r for r in held_checkpoint_rows if r["model"] == name and r["subject"] == subject and r["session"] == session]
                if len(part) != 5:
                    raise RuntimeError("final-heldout checkpoint coverage drift")
                aggregated.append({"model": name, "seed": seed, "subject": subject, "session": session, "checkpoints_aggregated": 5,
                                   "BA": float(np.mean([r["BA"] for r in part])), "Macro_F1": float(np.mean([r["Macro_F1"] for r in part])), "trials_per_checkpoint": int(part[0]["trials"])})
    cwrite(ROOT / "seed0_final_heldout_subject_metrics.csv", aggregated, ["model", "seed", "subject", "session", "checkpoints_aggregated", "BA", "Macro_F1", "trials_per_checkpoint"])
    jwrite(ROOT / "FINAL_HELDOUT_EVALUATION_AUDIT.json", {"checkpoint_freeze_completed_before_heldout_array_access": True, "heldout_subjects": heldout, "checkpoints_per_subject_model_session": 5, "weights_input_normalizers_and_batchnorm_statistics_fixed_at_inference": True})
    final_summary = summarize(aggregated, "final_heldout")
    outer_summary = summarize(outer_rows, "outer_development")
    all_summary = final_summary + outer_summary
    cwrite(ROOT / "seed0_summary.csv", all_summary, ["row_type", "population", "model", "metric", "mean", "CI95_low", "CI95_high", "subjects", "future BA", "future Macro-F1", "WS-BA"])
    lookup = {(r["row_type"], r["population"], r["model"], r["metric"]): r for r in final_summary}
    twrite(ROOT / "SEED0_RESULT_SUMMARY.md", ["# M3CV final-heldout seed-0 result", "", "Final-heldout reporting unit: each of 18 subjects after averaging that subject's two-session metric separately across all 5 frozen fold checkpoints.", "",
           "| Model | S2 BA | S2 Macro-F1 | WS-BA |", "|---|---:|---:|---:|"] + [f"| {r['model']} | {float(r['future BA']):.4f} | {float(r['future Macro-F1']):.4f} | {float(r['WS-BA']):.4f} |" for r in final_summary if r["row_type"] == "model"] + ["", "| SIRE-EEG minus EEGNet | Mean | 95% subject bootstrap CI |", "|---|---:|---:"] + [f"| {r['metric']} | {float(r['mean']):+.4f} | [{float(r['CI95_low']):+.4f}, {float(r['CI95_high']):+.4f}] |" for r in final_summary if r["row_type"] == "paired_SIRE_minus_EEGNet"] + ["", "No seed 1/2 was run."])
    print("M3CV_FINAL_HELDOUT_SEED0_COMPLETE", flush=True)


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
        print(f"M3CV_FINAL_HELDOUT_INVALID: {type(exc).__name__}: {exc}", flush=True)
        raise
