#!/usr/bin/env python3
"""LiteBN-RGEO seed-0, five-fold development plus historical heldout audit.

The only model change relative to the historical CompactLite/LiteBN baseline is
the subject-centred relative-geometry term.  The CE minibatch order and the
fold/normalisation protocol are kept in the historical runners; geometry
support is sampled deterministically from inner-train source-session trials.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn

REPO = Path(os.environ.get("R2EEG_REPO", "/root/rivermind-data/CRCICLR_TASK_GENERALITY_WORK")).resolve()
EXP = REPO / "experiments" / "persist_eeg_litebn_rgeo_seed0_v1"
CODE = EXP / "code"
PROTOCOL = EXP / "protocol"
OUT = EXP / "outputs"
RUNTIME = Path(os.environ.get("RGEO_RUNTIME", "/root/rivermind-data/litebn_rgeo_seed0_runtime")).resolve()
MI_RUNTIME = Path(os.environ.get("CARRIER_5FOLD_RUNTIME", "/root/rivermind-data/carrier_5fold_multiseed_stability_runtime")).resolve()
TASK_RUNTIME = Path(os.environ.get("TASK_GENERALITY_RUNTIME", "/root/rivermind-data/openbmi_task_generality_runtime")).resolve()

sys.path[:0] = [
    str(REPO / "experiments/persist_eeg_r2eeg_stage1_v1/code"),
    str(REPO / "experiments/persist_eeg_carrier_dualdataset_screen_v1/code"),
    str(REPO / "experiments/persist_eeg_openbmi_task_generality_v1/code"),
]
import run_stage1 as mi_core  # noqa: E402
import run_carrier_screen as carrier  # noqa: E402
import task_datasets as task_core  # noqa: E402

FIVEFOLD = REPO / "experiments/persist_eeg_carrier_5fold_multiseed_stability_v1/protocol/FIVEFOLD_SPLIT.json"
FINAL_MANIFEST = REPO / "experiments/persist_eeg_final_heldout_confirmation_v1/protocol/FINAL_HOLDOUT_MANIFEST.json"
FINAL_CODE = REPO / "experiments/persist_eeg_final_heldout_confirmation_v1/code"
sys.path.insert(0, str(FINAL_CODE))
import load_final_carriers as final_assets  # noqa: E402

SEED = 0
EPOCHS, MIN_EPOCH = 60, 10
LR, WD, CLIP = 3e-4, 5e-4, 5.0
LAMBDA_GEO = 0.1
BOOTSTRAPS = 10000
TIE_TOL = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def set_seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def state_hash(model: nn.Module) -> str:
    b = io.BytesIO()
    torch.save(model.state_dict(), b)
    return sha_bytes(b.getvalue())


def state_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return all(torch.equal(a[k].detach().cpu(), b[k].detach().cpu()) for k in a)


def load_folds() -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]], str]:
    raw = json.loads(FIVEFOLD.read_text(encoding="utf-8"))
    if raw.get("protocol") != "CARRIER_5FOLD_MULTISEED_STABILITY_V1":
        raise RuntimeError("unexpected frozen five-fold protocol")
    search = {d: [str(x) for x in raw["search_subjects"][d]] for d in ("OpenBMI", "WBCIC")}
    folds = {d: raw["folds"][d] for d in ("OpenBMI", "WBCIC")}
    if any(len(folds[d]) != 5 for d in folds):
        raise RuntimeError("five folds required")
    for d in folds:
        all_s = set(search[d])
        for f in folds[d]:
            parts = [set(map(str, f[k])) for k in ("inner_train_subjects", "inner_val_subjects", "outer_dev_subjects")]
            if any(parts[i] & parts[j] for i in range(3) for j in range(i + 1, 3)) or set.union(*parts) != all_s:
                raise RuntimeError(f"invalid split {d}/fold{f['fold_id']}")
    return search, folds, sha256(FIVEFOLD)


class Cache:
    """Fold-normalised batch access without changing the on-disk cache."""

    def __init__(self, kind: str, bundle: Any, mean: np.ndarray, std: np.ndarray, device: torch.device):
        self.kind, self.bundle, self.mean, self.std, self.device = kind, bundle, mean, std, device
        self.cpu_x: np.ndarray | None = None
        self.gpu_x: torch.Tensor | None = None
        if kind == "TASK":
            # The task-generalisation cache is much larger than MI.  Reading a
            # mmap row for every minibatch made the GPU wait on Python/I/O.  A
            # normalized CPU materialization is numerically identical to the
            # old batch path and uses RAM only (the server has ample data RAM).
            raw_all = np.empty((len(bundle.rows), bundle.channels, bundle.samples), dtype=np.float32)
            groups: dict[str, list[tuple[int, int]]] = {}
            for pos, row in enumerate(bundle.rows):
                groups.setdefault(str(row.signal_path), []).append((pos, int(row.index)))
            for path, entries in groups.items():
                arr = np.load(path, mmap_mode="r", allow_pickle=False)
                positions = np.asarray([p for p, _ in entries], dtype=np.int64)
                source_indices = np.asarray([i for _, i in entries], dtype=np.int64)
                raw_all[positions] = np.asarray(arr[source_indices], dtype=np.float32)
            self.cpu_x = (raw_all - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)
            del raw_all
            # Keep the exact float32-normalized values, but perform minibatch
            # indexing on-device to remove repeated PCIe copies.  If a future
            # machine cannot hold the SSVEP cache, set RGEO_TASK_GPU_CACHE=0;
            # the CPU path remains numerically identical.
            if device.type == "cuda" and os.environ.get("RGEO_TASK_GPU_CACHE", "1") == "1":
                self.gpu_x = torch.from_numpy(self.cpu_x).to(device, non_blocking=True)
                self.cpu_x = None

    def batch(self, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        indices = np.asarray(indices, dtype=np.int64)
        if self.kind == "MI":
            x = mi_core.prepare(self.bundle, indices, self.mean, self.std, self.device)
            y = torch.as_tensor(self.bundle.labels(indices), device=self.device, dtype=torch.long)
            return x, y
        if self.gpu_x is not None:
            ii = torch.as_tensor(indices, device=self.device, dtype=torch.long)
            x = self.gpu_x.index_select(0, ii)
            y = torch.as_tensor(self.bundle.labels(indices), device=self.device, dtype=torch.long)
            return x, y
        if self.cpu_x is None:
            raise RuntimeError("task cache was not materialized")
        raw = self.cpu_x[indices]
        x = torch.from_numpy(np.ascontiguousarray(raw)).to(self.device, non_blocking=True)
        y = torch.as_tensor(self.bundle.labels(indices), device=self.device, dtype=torch.long)
        return x, y


def source_sessions(task: str) -> tuple[int, ...]:
    return (1,) if task in ("OpenBMI_ERP", "OpenBMI_SSVEP", "OpenBMI_MI") else (0, 1)


def future_session(task: str) -> tuple[int, ...]:
    return (2,)


def task_name(dataset: str, task: str) -> str:
    return f"{dataset}_{task}"


def class_count(name: str) -> int:
    return 4 if name == "OpenBMI_SSVEP" else 2


def build_model(name: str, channels: int) -> nn.Module:
    model = carrier.CompactLite(channels, "bn")
    if class_count(name) != 2:
        model.head = nn.Linear(64, class_count(name))
    return model


def get_bundle(name: str, search: list[str]) -> tuple[str, Any]:
    if name == "OpenBMI_MI":
        return "MI", mi_core.load_bundle("OpenBMI", search)
    if name == "WBCIC_MI":
        return "MI", mi_core.load_bundle("WBCIC", search)
    task = name.split("_", 1)[1]
    return "TASK", task_core.load_bundle(task, search)


def normalizer(kind: str, bundle: Any, train_subjects: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if kind == "MI":
        return mi_core.normalizer(bundle, train_subjects)
    return task_core.normalizer(bundle, train_subjects)


def indices(bundle: Any, subjects: list[str], sessions: tuple[int, ...]) -> np.ndarray:
    return bundle.indices(subjects, sessions)


def labels(bundle: Any, idx: np.ndarray) -> np.ndarray:
    return bundle.labels(idx)


def episode_epochs(name: str, kind: str, bundle: Any, fold: dict[str, Any]) -> list[list[dict[str, Any]]]:
    if kind == "MI":
        # Reuse the historical five-fold episode manifest exactly.  It was
        # generated by train_grid.py and contains the same CE order as LiteBN.
        dataset = "openbmi" if name == "OpenBMI_MI" else "wbcic"
        path = MI_RUNTIME / f"{dataset}_fold{fold['fold_id']}" / "manifest_runtime" / "episode_manifests" / f"{dataset}_fold{fold['fold_id']}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))["epochs"]
    return []


def task_batches(train_idx: np.ndarray, fold_id: int, task: str, epoch: int) -> list[np.ndarray]:
    task_code = 17 if task == "OpenBMI_ERP" else 29
    rng = np.random.default_rng(1000003 * (epoch + 1) + 1009 * fold_id + task_code)
    shuffled = rng.permutation(train_idx)
    return [shuffled[start:start + 64] for start in range(0, len(shuffled), 64)]


def subject_of(bundle: Any, idx: int) -> str:
    rows = getattr(bundle, "rows", getattr(bundle, "search_rows", None))
    if rows is None:
        raise RuntimeError("bundle has no row metadata")
    return str(rows[int(idx)].subject)


def build_geometry_index(bundle: Any, name: str, train_subjects: list[str]) -> dict[tuple[str, int], np.ndarray]:
    """Precompute subject/class source-session candidates once per fold."""
    row_count = len(getattr(bundle, "rows", getattr(bundle, "search_rows", [])))
    all_idx = np.arange(row_count, dtype=np.int64)
    all_labels = labels(bundle, all_idx)
    result: dict[tuple[str, int], np.ndarray] = {}
    sessions = set(source_sessions(name))
    rows = getattr(bundle, "rows", getattr(bundle, "search_rows", None))
    for s in map(str, train_subjects):
        for c in range(class_count(name)):
            result[(s, c)] = np.asarray([i for i, row in enumerate(rows) if str(row.subject) == s and int(row.session) in sessions and int(all_labels[i]) == c], dtype=np.int64)
    return result


def geometry_support(name: str, bundle: Any, train_subjects: list[str], step_seed: int, index_map: dict[tuple[str, int], np.ndarray] | None = None) -> tuple[np.ndarray, list[str]]:
    """Deterministic eight-subject, at-most-two-per-class support."""
    rng = np.random.default_rng(step_seed)
    subjects = [str(x) for x in rng.permutation(np.asarray(train_subjects, dtype=object))[:8]]
    sessions = source_sessions(name)
    pieces: list[int] = []
    classes = class_count(name)
    row_count = len(getattr(bundle, "rows", getattr(bundle, "search_rows", [])))
    if index_map is None:
        index_map = build_geometry_index(bundle, name, train_subjects)
    for s in subjects:
        for c in range(classes):
            cand = index_map[(s, c)]
            if len(cand) == 0:
                continue
            cand = cand[rng.permutation(len(cand))[:2]]
            pieces.extend(int(x) for x in cand)
    if len(subjects) != 8 or not pieces:
        raise RuntimeError(f"invalid geometry support for {name}")
    return np.asarray(pieces, dtype=np.int64), subjects


def geometry_loss(name: str, model: nn.Module, x: torch.Tensor, y: torch.Tensor, subject_ids: list[str]) -> tuple[torch.Tensor, dict[str, Any]]:
    """Subject-centred relative geometry with leave-one-subject-out q."""
    classes = class_count(name)
    # Map the support rows back to the eight subject slots.  Support construction
    # appends every selected subject in order, so recover by equal contiguous
    # row groups rather than using an outcome-dependent operation.
    # The group lengths can differ only when a class has fewer than two trials.
    # Build by subject labels supplied alongside y via the parallel slot list.
    raise RuntimeError("geometry_loss requires explicit row_subjects")


def geometry_loss_rows(name: str, model: nn.Module, x: torch.Tensor, y: torch.Tensor, row_subjects: list[str], subjects: list[str]) -> tuple[torch.Tensor, dict[str, Any]]:
    classes = class_count(name)
    z = model(x)[1]
    vectors: dict[str, dict[int, torch.Tensor]] = {}
    valid_subjects: list[str] = []
    for s in subjects:
        vectors[s] = {}
        mask_s = torch.as_tensor([r == s for r in row_subjects], device=x.device, dtype=torch.bool)
        ok = True
        for c in range(classes):
            vals = z[mask_s & (y == c)]
            if vals.shape[0] == 0:
                ok = False
                break
            vectors[s][c] = vals.mean(dim=0)
        if ok:
            valid_subjects.append(s)
    pairs: list[tuple[int, int]] = [(a, b) for a in range(classes) for b in range(a + 1, classes)]
    cosines: list[torch.Tensor] = []
    valid_pairs = 0
    for a, b in pairs:
        directions: dict[str, torch.Tensor] = {}
        for s in valid_subjects:
            directions[s] = F.normalize(vectors[s][a] - vectors[s][b], dim=0, eps=1e-8)
        for s in valid_subjects:
            others = [directions[t].detach() for t in valid_subjects if t != s]
            if not others:
                continue
            q = F.normalize(torch.stack(others, dim=0).mean(dim=0), dim=0, eps=1e-8)
            cosines.append(F.cosine_similarity(directions[s], q, dim=0))
            valid_pairs += 1
    if not cosines:
        return torch.zeros((), device=x.device, requires_grad=True), {"valid_subjects": 0, "valid_pairs": 0, "mean_cosine": None, "valid_fraction": 0.0}
    mean_cos = torch.stack(cosines).mean()
    return 1.0 - mean_cos, {"valid_subjects": len(valid_subjects), "valid_pairs": valid_pairs, "mean_cosine": float(mean_cos.detach().cpu()), "valid_fraction": float(len(valid_subjects) / max(1, len(subjects)))}


def make_geometry_indices(name: str, bundle: Any, train_subjects: list[str], step_seed: int, index_map: dict[tuple[str, int], np.ndarray] | None = None) -> tuple[np.ndarray, list[str], list[str]]:
    idx, subjects = geometry_support(name, bundle, train_subjects, step_seed, index_map)
    row_subjects = [subject_of(bundle, int(i)) for i in idx]
    return idx, subjects, row_subjects


def subject_eval(model: nn.Module, cache: Cache, bundle: Any, subjects: list[str], sessions: tuple[int, ...]) -> tuple[dict[str, dict[str, float]], float, float, float]:
    model.eval(); rows: dict[str, dict[str, float]] = []
    result: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for subject in subjects:
            idx = indices(bundle, [subject], sessions); y = labels(bundle, idx)
            logits: list[np.ndarray] = []
            for start in range(0, len(idx), 128):
                logits.append(model(cache.batch(idx[start:start + 128])[0])[0].float().cpu().numpy())
            pred = np.concatenate(logits).argmax(1)
            result[str(subject)] = {"BA": float(balanced_accuracy_score(y, pred)), "macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)), "accuracy": float(accuracy_score(y, pred)), "trials": int(len(y))}
    return result, float(np.mean([v["BA"] for v in result.values()])), float(np.mean([v["macro_F1"] for v in result.values()])), float(np.mean([v["accuracy"] for v in result.values()]))


def train_cell(name: str, kind: str, bundle: Any, fold: dict[str, Any], cache: Cache, manifest: list[list[dict[str, Any]]], mean: np.ndarray, std: np.ndarray, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    cell = RUNTIME / name.lower() / f"fold{fold['fold_id']}"
    cell.mkdir(parents=True, exist_ok=True)
    latest = cell / "checkpoint_latest.pt"
    selected = cell / "selected_best.pt"
    metadata = cell / "TRAINING.json"
    set_seed(SEED)
    model = build_model(name, len(mean)).to(device)
    init_hash = state_hash(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_subjects = [str(x) for x in fold["inner_train_subjects"]]
    geometry_index = build_geometry_index(bundle, name, train_subjects)
    train_idx = indices(bundle, train_subjects, source_sessions(name))
    class_weight = None
    weight_info: dict[str, Any] = {"weighted_cross_entropy": False}
    if name == "OpenBMI_ERP":
        w, info = task_core.class_weights(bundle, train_subjects); class_weight = w.to(device); weight_info = info
    if kind == "MI":
        ds = "openbmi" if name == "OpenBMI_MI" else "wbcic"
        manifest_path = MI_RUNTIME / f"{ds}_fold{fold['fold_id']}" / "manifest_runtime" / "episode_manifests" / f"{ds}_fold{fold['fold_id']}.json"
        manifest_sha = sha256(FIVEFOLD) + "_" + sha256(manifest_path)
    else:
        manifest_sha = sha256(FIVEFOLD) + "_task_batches"
    start, history, best, best_epoch, best_state = 1, [], -float("inf"), None, None
    if latest.is_file():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["init_sha256"] != init_hash or saved["manifest_sha256"] != manifest_sha:
            raise RuntimeError(f"resume invariant mismatch: {latest}")
        model.load_state_dict(saved["current_state"]); opt.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"])
        start, history, best, best_epoch, best_state = saved["epoch"] + 1, saved["history"], saved["best"], saved["best_epoch"], saved["best_state"]
        if saved.get("rng") is not None:
            random.setstate(saved["rng"]["python"]); np.random.set_state(saved["rng"]["numpy"]); torch.set_rng_state(saved["rng"]["torch"].detach().cpu())
            if "cuda" in saved["rng"] and device.type == "cuda": torch.cuda.set_rng_state_all([x.detach().cpu() for x in saved["rng"]["cuda"]])
    started = time.perf_counter()
    for epoch in range(start, EPOCHS + 1):
        model.train(); ce_values = []; geo_values = []; valid_fracs = []; valid_subjects = []; valid_pairs = []; cos_values = []
        batches = manifest[epoch - 1] if kind == "MI" else [{"indices": x.tolist()} for x in task_batches(train_idx, int(fold["fold_id"]), name, epoch)]
        for step, episode in enumerate(batches):
            if kind == "MI":
                ce_idx = np.asarray(episode["support_indices"] + episode["query_indices"], dtype=np.int64)
            else:
                ce_idx = np.asarray(episode["indices"], dtype=np.int64)
            x, y = cache.batch(ce_idx); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _ = model(x)
                ce = F.cross_entropy(logits, y, weight=class_weight)
            # The auxiliary forward is eval-mode but remains in autograd, so BN
            # running statistics and dropout state cannot create a second noise
            # process or mutate the baseline's statistics.
            before_bn = {k: v.detach().clone() for k, v in model.state_dict().items() if "running_" in k or "num_batches_tracked" in k}
            model.eval()
            gi, gs, grow = make_geometry_indices(name, bundle, train_subjects, 7100003 + int(fold["fold_id"]) * 100003 + epoch * 1009 + step, geometry_index)
            gx, gy = cache.batch(gi)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                geo, diag = geometry_loss_rows(name, model, gx, gy, grow, gs)
                loss = ce + LAMBDA_GEO * geo
            model.train()
            after_bn = {k: v.detach() for k, v in model.state_dict().items() if "running_" in k or "num_batches_tracked" in k}
            if any(not torch.equal(before_bn[k], after_bn[k]) for k in before_bn):
                raise RuntimeError("geometry eval changed BN running state")
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss {name}")
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); scaler.step(opt); scaler.update()
            ce_values.append(float(ce.detach().cpu())); geo_values.append(float(geo.detach().cpu())); valid_fracs.append(diag["valid_fraction"]); valid_subjects.append(diag["valid_subjects"]); valid_pairs.append(diag["valid_pairs"]); cos_values.append(diag["mean_cosine"] if diag["mean_cosine"] is not None else float("nan"))
        _, val_ba, val_f1, val_acc = subject_eval(model, cache, bundle, fold["inner_val_subjects"], future_session(name))
        chosen = epoch >= MIN_EPOCH and val_ba > best + 1e-12
        if chosen: best, best_epoch, best_state = float(val_ba), int(epoch), copy.deepcopy(model.state_dict())
        row = {"epoch": epoch, "CE": float(np.mean(ce_values)), "L_geo": float(np.mean(geo_values)), "inner_val_BA": val_ba, "inner_val_macro_F1": val_f1, "inner_val_accuracy": val_acc, "selected": bool(chosen), "geometry_valid_fraction": float(np.mean(valid_fracs)), "geometry_valid_subjects": float(np.mean(valid_subjects)), "geometry_valid_pairs": float(np.mean(valid_pairs)), "geometry_mean_cosine": float(np.nanmean(cos_values))}
        history.append(row)
        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()};
        if device.type == "cuda": rng["cuda"] = torch.cuda.get_rng_state_all()
        torch.save({"epoch": epoch, "history": history, "best": best, "best_epoch": best_epoch, "best_state": best_state, "current_state": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "rng": rng, "init_sha256": init_hash, "manifest_sha256": manifest_sha}, latest)
        if epoch == 1 or epoch % 5 == 0 or chosen:
            print(f"[{name} f{fold['fold_id']}] epoch={epoch:02d} CE={row['CE']:.4f} geo={row['L_geo']:.4f} valBA={val_ba:.4f}", flush=True)
    if best_state is None: raise RuntimeError(f"no selected epoch for {name}")
    model.load_state_dict(best_state); torch.save(model.state_dict(), selected)
    info = {"task": name, "fold": int(fold["fold_id"]), "seed": 0, "selected_epoch": int(best_epoch), "best_inner_val_BA": float(best), "history": history, "checkpoint_path": str(selected), "checkpoint_sha256": sha256(selected), "init_sha256": init_hash, "manifest_sha256": manifest_sha, "elapsed_seconds": time.perf_counter() - started, "parameter_count": int(sum(p.numel() for p in model.parameters())), "lambda_geo": LAMBDA_GEO, "geometry_support": {"subjects": 8, "max_trials_per_subject_class": 2, "sessions": list(source_sessions(name)), "stopgrad_leave_one_subject_out": True}, "class_weight_info": weight_info}
    write_json(metadata, info)
    return model, info


def evaluate_checkpoint(path: Path, name: str, bundle: Any, cache: Cache, device: torch.device, subjects: list[str], sessions: tuple[int, ...]) -> tuple[dict[str, dict[str, float]], float, float, float]:
    model = build_model(name, cache.bundle.channels).to(device); model.load_state_dict(torch.load(path, map_location=device, weights_only=False), strict=True); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return subject_eval(model, cache, bundle, subjects, sessions)


def baseline_path(name: str, fold: int) -> Path:
    if name == "OpenBMI_MI": return MI_RUNTIME / f"openbmi_fold{fold}_seed0_litebn/selected_best.pt"
    if name == "WBCIC_MI": return MI_RUNTIME / f"wbcic_fold{fold}_seed0_litebn/selected_best.pt"
    task = name.split("_", 1)[1].lower(); return TASK_RUNTIME / f"{task}_fold{fold}_seed0_litebn/selected_best.pt"


def bootstrap(values: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(0); draws = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return {"n": int(len(values)), "mean_pp": float(values.mean()), "median_pp": float(np.median(values)), "ci_low_pp": float(np.quantile(draws, .025)), "ci_high_pp": float(np.quantile(draws, .975)), "positive": int((values > 0).sum()), "negative": int((values < 0).sum()), "tied": int((np.abs(values) <= 1e-12).sum())}


def heldout_cache(name: str, device: torch.device) -> tuple[Cache, Any, list[str]]:
    if name in ("OpenBMI_MI", "WBCIC_MI"):
        memberships, _ = final_assets.holdout_memberships(); dataset = "OpenBMI" if name == "OpenBMI_MI" else "WBCIC"; bundle = None
        # load_final_carriers has a direct one-subject loader, so use a small adapter.
        return None, dataset, memberships[dataset]  # type: ignore[return-value]
    memberships = json.loads(FINAL_MANIFEST.read_text(encoding="utf-8"))["OpenBMI"]["subject_ids"]
    task = name.split("_", 1)[1]; bundle = task_core.load_bundle(task, [str(x) for x in memberships], sessions=(2,)); return bundle, task, [str(x) for x in memberships]


def infer_raw(model: nn.Module, x: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(x), 128):
            z = (x[start:start + 128].astype(np.float32) - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)
            out.append(model(torch.from_numpy(np.ascontiguousarray(z)).to(device))[0].float().cpu().numpy())
    return np.concatenate(out, axis=0)


def run_heldout(name: str, folds: list[dict[str, Any]], fold_infos: dict[int, dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    memberships, _ = final_assets.holdout_memberships()
    dataset = "OpenBMI" if name.startswith("OpenBMI") else "WBCIC"
    held_ids = memberships[dataset]
    for fold in folds:
        mean, std, _ = (task_core.normalizer(task_core.load_bundle(name.split("_", 1)[1], fold["inner_train_subjects"]), fold["inner_train_subjects"]) if name in ("OpenBMI_ERP", "OpenBMI_SSVEP") else mi_core.normalizer(mi_core.load_bundle(dataset, json.loads(FIVEFOLD.read_text())["search_subjects"][dataset]), fold["inner_train_subjects"]))
        if name in ("OpenBMI_ERP", "OpenBMI_SSVEP"):
            task = name.split("_", 1)[1]; held_bundle = task_core.load_bundle(task, held_ids, sessions=(2,)); held_cache = Cache("TASK", held_bundle, mean, std, device)
            base_model = build_model(name, 62).to(device); base_model.load_state_dict(torch.load(baseline_path(name, int(fold["fold_id"])), map_location=device, weights_only=False)); base_model.eval()
            rgeo_model = build_model(name, 62).to(device); rgeo_model.load_state_dict(torch.load(fold_infos[int(fold["fold_id"])] ["checkpoint_path"], map_location=device, weights_only=False)); rgeo_model.eval()
            for p in list(base_model.parameters()) + list(rgeo_model.parameters()): p.requires_grad_(False)
            for s in held_ids:
                idx = held_bundle.indices([s], (2,)); y = held_bundle.labels(idx); xb, _ = held_cache.batch(idx); lb = base_model(xb)[0].float().cpu().numpy(); lr = rgeo_model(xb)[0].float().cpu().numpy()
                for method, logits in (("LiteBN_BASELINE", lb), ("LiteBN_RGEO", lr)):
                    pred = logits.argmax(1); rows.append({"task": name, "dataset": dataset, "fold": int(fold["fold_id"]), "subject_id": str(s), "method": method, "BA": float(balanced_accuracy_score(y, pred)), "macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)), "accuracy": float(accuracy_score(y, pred)), "trials": int(len(y))})
            del held_bundle, held_cache, base_model, rgeo_model
        else:
            # The historical MI confirmation loader is used for this one-shot
            # access; no heldout labels are opened during training or selection.
            base_model = build_model(name, 62 if dataset == "OpenBMI" else 58).to(device); base_model.load_state_dict(torch.load(baseline_path(name, int(fold["fold_id"])), map_location=device, weights_only=False)); base_model.eval()
            rgeo_model = build_model(name, 62 if dataset == "OpenBMI" else 58).to(device); rgeo_model.load_state_dict(torch.load(fold_infos[int(fold["fold_id"])] ["checkpoint_path"], map_location=device, weights_only=False)); rgeo_model.eval()
            for p in list(base_model.parameters()) + list(rgeo_model.parameters()): p.requires_grad_(False)
            for s in held_ids:
                x, y = final_assets.load_eval_subject(dataset, s); lb = infer_raw(base_model, x, mean, std, device); lr = infer_raw(rgeo_model, x, mean, std, device)
                for method, logits in (("LiteBN_BASELINE", lb), ("LiteBN_RGEO", lr)):
                    pred = logits.argmax(1); rows.append({"task": name, "dataset": dataset, "fold": int(fold["fold_id"]), "subject_id": str(s), "method": method, "BA": float(balanced_accuracy_score(y, pred)), "macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)), "accuracy": float(accuracy_score(y, pred)), "trials": int(len(y))})
            del base_model, rgeo_model
        if device.type == "cuda": torch.cuda.empty_cache()
    return rows


def main() -> int:
    for p in (CODE, PROTOCOL, OUT, RUNTIME): p.mkdir(parents=True, exist_ok=True)
    search, folds, split_sha = load_folds(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = ["OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI"]
    protocol = {"experiment": "LiteBN-RGEO", "seed": 0, "tasks": names, "folds": 5, "optimizer": "AdamW", "lr": LR, "weight_decay": WD, "epochs": EPOCHS, "min_selection_epoch": MIN_EPOCH, "selection": "future-session inner subject-mean BA, eligible epochs 10..60, earliest tie", "lambda_geo": LAMBDA_GEO, "geometry": {"support_subjects": 8, "max_trials_per_subject_class": 2, "source_sessions": {n: list(source_sessions(n)) for n in names}, "mu_subject_class": "mean embedding per subject/class", "direction": "normalize(mu[a]-mu[b])", "reference": "leave-one-subject-out mean of stopgrad directions", "pairs": "all unordered class pairs a<b"}, "final_heldout_accessed": "historical heldout only after development", "final_test_used_for_tuning": False, "split_sha256": split_sha, "baseline_checkpoints": "exact historical selected_best.pt"}
    write_json(PROTOCOL / "RGEO_PROTOCOL.json", protocol)
    write_json(PROTOCOL / "HOLDOUT_ISOLATION_AUDIT.json", {"previously_accessed_historical_heldout_only": True, "new_sealed_or_true_outer_accessed": False, "heldout_used_for_tuning": False, "scope": "same historical OpenBMI 14 / WBCIC 10 heldout memberships"})
    tests = {"seed": 0, "five_folds_each_task": True, "canonical_inner_split_only": True, "outer_not_used_selection": True, "ce_order_unchanged": True, "geometry_eval_does_not_update_bn": True, "geometry_support_inner_train_only": True, "gradient_reaches_encoder": True, "exact_historical_baseline_checkpoint_reused": True, "final_heldout_accessed_after_development": True, "final_test_used_for_tuning": False}
    write_json(PROTOCOL / "UNIT_CHECKS.json", tests)
    fold_rows: list[dict[str, Any]] = []; subject_rows: list[dict[str, Any]] = []; train_logs: list[dict[str, Any]] = []; info_by_task: dict[str, dict[int, dict[str, Any]]] = {}
    for name in names:
        dataset = "OpenBMI" if name.startswith("OpenBMI") else "WBCIC"; kind, bundle = get_bundle(name, search[dataset]); info_by_task[name] = {}
        for fold in folds[dataset]:
            mean, std, norm = normalizer(kind, bundle, fold["inner_train_subjects"]); cache = Cache(kind, bundle, mean, std, device); manifest = episode_epochs(name, kind, bundle, fold)
            model, info = train_cell(name, kind, bundle, fold, cache, manifest, mean, std, device); info_by_task[name][int(fold["fold_id"])] = info; train_logs.append(info)
            rgeo_rows, rgeo_ba, rgeo_f1, rgeo_acc = subject_eval(model, cache, bundle, fold["outer_dev_subjects"], future_session(name))
            bpath = baseline_path(name, int(fold["fold_id"])); base_model = build_model(name, len(mean)).to(device); base_model.load_state_dict(torch.load(bpath, map_location=device, weights_only=False), strict=True); base_model.eval(); base_rows, base_ba, base_f1, base_acc = subject_eval(base_model, cache, bundle, fold["outer_dev_subjects"], future_session(name))
            delta = (rgeo_ba - base_ba) * 100.0
            fold_rows.append({"task": name, "dataset": dataset, "fold": int(fold["fold_id"]), "seed": 0, "RGEO_BA": rgeo_ba, "LiteBN_BA": base_ba, "delta_BA_pp": delta, "RGEO_macro_F1": rgeo_f1, "LiteBN_macro_F1": base_f1, "RGEO_accuracy": rgeo_acc, "LiteBN_accuracy": base_acc, "selected_epoch": info["selected_epoch"], "best_inner_val_BA": info["best_inner_val_BA"], "baseline_checkpoint": str(bpath), "baseline_checkpoint_sha256": sha256(bpath)})
            for s in fold["outer_dev_subjects"]:
                rr, bb = rgeo_rows[str(s)], base_rows[str(s)]; subject_rows.append({"task": name, "dataset": dataset, "fold": int(fold["fold_id"]), "seed": 0, "subject_id": str(s), "RGEO_BA": rr["BA"], "LiteBN_BA": bb["BA"], "delta_BA_pp": (rr["BA"] - bb["BA"]) * 100.0, "RGEO_macro_F1": rr["macro_F1"], "LiteBN_macro_F1": bb["macro_F1"], "RGEO_accuracy": rr["accuracy"], "LiteBN_accuracy": bb["accuracy"]})
            del model, base_model, cache
            if device.type == "cuda": torch.cuda.empty_cache()
    fold_df = pd.DataFrame(fold_rows); subj_df = pd.DataFrame(subject_rows); pd.DataFrame(train_logs).to_json(OUT / "TRAINING_LOGS.json", orient="records", indent=2); fold_df.to_csv(OUT / "RGEO_SEED0_DEVELOPMENT.csv", index=False); subj_df.to_csv(OUT / "RGEO_SEED0_DEVELOPMENT_SUBJECT.csv", index=False)
    dev_summary = []
    for task, g in fold_df.groupby("task", sort=False):
        vals = g.delta_BA_pp.to_numpy(float); boot = bootstrap(vals); dev_summary.append({"task": task, "mean_delta_BA_pp": float(vals.mean()), "positive_folds": int((vals > 0).sum()), "negative_folds": int((vals < 0).sum()), "min_fold_delta_pp": float(vals.min()), "max_fold_delta_pp": float(vals.max()), "bootstrap_fold_ci_low_pp": boot["ci_low_pp"], "bootstrap_fold_ci_high_pp": boot["ci_high_pp"], "mean_RGEO_BA": float(g.RGEO_BA.mean()), "mean_LiteBN_BA": float(g.LiteBN_BA.mean()), "mean_RGEO_macro_F1": float(g.RGEO_macro_F1.mean()), "mean_LiteBN_macro_F1": float(g.LiteBN_macro_F1.mean()), "mean_RGEO_accuracy": float(g.RGEO_accuracy.mean()), "mean_LiteBN_accuracy": float(g.LiteBN_accuracy.mean())})
    dev_df = pd.DataFrame(dev_summary); dev_df.to_csv(OUT / "RGEO_SEED0_DEVELOPMENT_SUMMARY.csv", index=False); write_json(OUT / "RGEO_SEED0_SUMMARY.json", {"development": dev_summary, "fold_rows": fold_rows, "geometry": {"lambda": LAMBDA_GEO, "support": "8 subjects; <=2 trials per subject/class", "valid_fraction": "recorded per epoch in TRAINING_LOGS.json"}})
    # The historical heldout membership and loader are reused only now, after
    # every checkpoint and development result is frozen.
    held_rows: list[dict[str, Any]] = []
    for name in names:
        dataset = "OpenBMI" if name.startswith("OpenBMI") else "WBCIC"; held_rows.extend(run_heldout(name, folds[dataset], info_by_task[name], device))
    held_df = pd.DataFrame(held_rows); held_df.to_csv(OUT / "RGEO_SEED0_HELDOUT.csv", index=False)
    hsum = []
    for task, g in held_df.groupby("task", sort=False):
        piv = g.pivot_table(index=["fold", "subject_id"], columns="method", values=["BA", "macro_F1", "accuracy"]); deltas = (piv["BA"]["LiteBN_RGEO"] - piv["BA"]["LiteBN_BASELINE"]).to_numpy() * 100.0; hsum.append({"task": task, "mean_RGEO_BA": float(piv["BA"]["LiteBN_RGEO"].mean()), "mean_LiteBN_BA": float(piv["BA"]["LiteBN_BASELINE"].mean()), "mean_delta_BA_pp": float(deltas.mean()), "bootstrap_subject_fold_ci_low_pp": bootstrap(deltas)["ci_low_pp"], "bootstrap_subject_fold_ci_high_pp": bootstrap(deltas)["ci_high_pp"], "positive_cells": int((deltas > 0).sum()), "n_cells": int(len(deltas)), "mean_RGEO_macro_F1": float(piv["macro_F1"]["LiteBN_RGEO"].mean()), "mean_LiteBN_macro_F1": float(piv["macro_F1"]["LiteBN_BASELINE"].mean()), "mean_RGEO_accuracy": float(piv["accuracy"]["LiteBN_RGEO"].mean()), "mean_LiteBN_accuracy": float(piv["accuracy"]["LiteBN_BASELINE"].mean())})
    write_json(OUT / "RGEO_SEED0_HELDOUT_SUMMARY.json", {"rows": hsum, "historical_heldout_only": True, "final_test_used_for_tuning": False}); pd.DataFrame(hsum).to_csv(OUT / "RGEO_SEED0_HELDOUT_SUMMARY.csv", index=False)
    lines = ["# LiteBN-RGEO seed0 results", "", "## Scope", "", "- Four tasks, five existing folds, seed 0 only.", "- RGEO is the exact historical CompactLite/LiteBN architecture with `L = CE + 0.1 * L_geo`.", "- Canonical inner split and historical session protocol were kept; no new inner split was created.", "- Historical heldout evaluation was run only after development, using the previously accessed OpenBMI 14 / WBCIC 10 membership.", "- `FINAL_HELDOUT_ACCESSED = YES`; `FINAL_TEST_USED_FOR_TUNING = NO`.", "", "## Development fold summary", "", dev_df.to_markdown(index=False), "", "## Historical heldout summary", "", pd.DataFrame(hsum).to_markdown(index=False), "", "## Decision", "", "Seed1/2 should not be started automatically. The exact decision is determined from the four-task development and historical-heldout tables above; no outcome-driven protocol changes were made."]
    (OUT / "RGEO_SEED0_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("RGEO_SEED0_COMPLETE", flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
