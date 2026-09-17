"""Resumable 120-cell source-only training queue for two frozen baselines."""
from __future__ import annotations

import copy
import gc
import hashlib
import io
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from filterbank import FILTER_SHA256, make_memmap
from models import build_model


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
RUNTIME = Path(os.environ["NEW_BASELINE_RUNTIME"]).resolve()
SCRATCH = Path(os.environ["NEW_BASELINE_SCRATCH"]).resolve()
os.environ.setdefault("SEVEN_REPO", str(REPO))
os.environ.setdefault("SEVEN_RUNTIME", str(RUNTIME))
os.environ.setdefault("MODERN_REPO", str(REPO))
os.environ.setdefault("TASK_GENERALITY_REPO", str(REPO))
os.environ.setdefault("FULL_OPENBMI_CACHE", r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi")
os.environ.setdefault("FULL_WBCIC_CACHE", r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache\wbcic_epochs")
os.environ.setdefault("PERSIST_OPENBMI_CACHE", os.environ["FULL_OPENBMI_CACHE"])
PROTOCOL = EXP / "protocol" / "PROTOCOL.json"
SIDECAR = EXP / "protocol" / "PROTOCOL.sha256"
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
MODELS = ("EEGConformer", "FBCNet")
MODEL_SHARD = os.environ.get("NEW_BASELINE_MODEL", "ALL")
if MODEL_SHARD != "ALL" and MODEL_SHARD not in MODELS:
    raise ValueError(f"invalid model shard {MODEL_SHARD}")
ACTIVE_MODELS = MODELS if MODEL_SHARD == "ALL" else (MODEL_SHARD,)
TASK_SHARD = os.environ.get("NEW_BASELINE_TASK", "ALL")
if TASK_SHARD != "ALL" and TASK_SHARD not in TASKS:
    raise ValueError(f"invalid task shard {TASK_SHARD}")
ACTIVE_TASKS = TASKS if TASK_SHARD == "ALL" else (TASK_SHARD,)
SEEDS = (0, 1, 2)
FOLDS = range(5)
MAX_EPOCHS = 60
MIN_EPOCH = 10
PATIENCE = 8
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 5.0

sys.path.insert(0, str(REPO / "experiments" / "persist_eeg_seven_backbone_fourtask_3seed_v1" / "code"))
from benchmark_data import load_search_fold  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + f".{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(part, path)


def atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    torch.save(payload, part, _use_new_zipfile_serialization=False)
    os.replace(part, path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def frozen_protocol() -> dict:
    if not PROTOCOL.is_file() or not SIDECAR.is_file() or sha(PROTOCOL) != SIDECAR.read_text().strip():
        raise RuntimeError("training requires intact committed protocol lock")
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if value["tasks"] != list(TASKS) or value["models"] != list(MODELS):
        raise RuntimeError("protocol matrix changed")
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(value: dict) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if torch.cuda.is_available() and value["cuda"] is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in value["cuda"]])


def metrics(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> dict:
    predictions = logits.argmax(axis=1)
    details = []
    for subject in sorted(set(map(str, subjects)), key=lambda x: int(x.replace("sub-", ""))):
        mask = subjects.astype(str) == subject
        details.append({"subject": subject,
                        "BA": float(balanced_accuracy_score(labels[mask], predictions[mask])),
                        "macro_F1": float(f1_score(labels[mask], predictions[mask],
                                                  average="macro", zero_division=0)),
                        "accuracy": float(accuracy_score(labels[mask], predictions[mask])),
                        "trials": int(mask.sum())})
    return {"subject_equal_BA": float(np.mean([r["BA"] for r in details])),
            "subject_equal_macro_F1": float(np.mean([r["macro_F1"] for r in details])),
            "subjects": details}


def tensor_batch(values, indices: np.ndarray | slice, device: torch.device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        index = torch.as_tensor(indices, dtype=torch.long, device=device) if not isinstance(indices, slice) else indices
        return values.index_select(0, index) if not isinstance(index, slice) else values[index]
    # For FBCNet ERP, the fixed filter-bank cache is mmap-backed and too large
    # for GPU residency. Only the requested batch crosses PCIe.
    return torch.from_numpy(np.array(values[indices], copy=True, order="C")).to(device)


def evaluate(model: torch.nn.Module, values, labels: np.ndarray, subjects: np.ndarray,
             device: torch.device) -> dict:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(labels), BATCH_SIZE):
            batch = tensor_batch(values, slice(start, start + BATCH_SIZE), device)
            parts.append(model(batch).float().cpu().numpy())
    return metrics(labels, np.concatenate(parts), subjects)


def ordered_batches(length: int, task: str, model: str, fold: int, seed: int, epoch: int):
    token = f"SEVEN-BACKBONE|{task}|{model}|{fold}|{seed}|{epoch}"
    digest = hashlib.sha256(token.encode()).digest()
    order = np.random.default_rng(int.from_bytes(digest[:8], "little")).permutation(length)
    for start in range(0, length, BATCH_SIZE):
        yield order[start:start + BATCH_SIZE]


def cell_dir(model: str, task: str, fold: int, seed: int) -> Path:
    return RUNTIME / "cells" / model.lower() / task.lower() / f"fold{fold}_seed{seed}"


def train_cell(model_name: str, task: str, fold: int, seed: int, data: dict,
               arrays: dict, device: torch.device) -> None:
    cell = cell_dir(model_name, task, fold, seed)
    record_path, selected_path, latest_path = (cell / name for name in
                                               ("record.json", "selected.pt", "latest.pt"))
    if record_path.is_file() and selected_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if sha(selected_path) == record["checkpoint_sha256"]:
            print(f"SKIP_COMPLETE {model_name} {task} fold{fold} seed{seed}", flush=True)
            return
        raise RuntimeError(f"completed checkpoint hash mismatch: {cell}")
    cell.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seed_everything(seed)
    model = build_model(model_name, data["channels"], data["samples"], data["classes"]).to(device)
    initial = io.BytesIO()
    torch.save(model.state_dict(), initial)
    initial_hash = hashlib.sha256(initial.getvalue()).hexdigest()
    seed_everything(seed + 100000)
    recipe = {"max_epochs": MAX_EPOCHS, "min_epoch": MIN_EPOCH, "patience": PATIENCE,
              "batch_size": BATCH_SIZE, "optimizer": "AdamW", "lr": LEARNING_RATE,
              "weight_decay": WEIGHT_DECAY, "gradient_clip": GRAD_CLIP,
              "selection": "inner-validation subject-equal BA"}
    invariant = hashlib.sha256(json.dumps({"model": model_name, "task": task, "fold": fold,
                                            "seed": seed, "split": data["split_sha256"],
                                            "normalizer": data["normalizer"],
                                            "protocol": sha(PROTOCOL), "recipe": recipe,
                                            "initial": initial_hash,
                                            "filter": FILTER_SHA256 if model_name == "FBCNet" else None},
                                           sort_keys=True).encode()).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_y = torch.as_tensor(data["train_y"], dtype=torch.long, device=device)
    class_weight = None
    if data["weighted_cross_entropy"]:
        counts = np.bincount(data["train_y"], minlength=data["classes"])
        class_weight = torch.as_tensor(len(train_y) / (data["classes"] * counts),
                                       dtype=torch.float32, device=device)
    epoch_start, best, best_epoch, best_state, no_gain, history = 1, -float("inf"), 0, None, 0, []
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        if saved["invariant"] != invariant:
            raise RuntimeError(f"resume invariant mismatch: {cell}")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        epoch_start = saved["epoch"] + 1
        best, best_epoch, best_state, no_gain = saved["best"], saved["best_epoch"], saved["best_state"], saved["no_gain"]
        history = saved["history"]
        del saved
    stopped = no_gain >= PATIENCE and epoch_start > MIN_EPOCH
    for epoch in ([] if stopped else range(epoch_start, MAX_EPOCHS + 1)):
        tick = time.perf_counter()
        model.train()
        losses = []
        for order in ordered_batches(len(train_y), task, model_name, fold, seed, epoch):
            features = tensor_batch(arrays["train"], order, device)
            target = train_y.index_select(0, torch.as_tensor(order, dtype=torch.long, device=device))
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(features), target, weight=class_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite CE {model_name} {task} f{fold} s{seed} e{epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        inner = evaluate(model, arrays["val"], data["val_y"], data["val_subjects"], device)
        improved = epoch >= MIN_EPOCH and inner["subject_equal_BA"] > best + 1e-12
        if improved:
            best, best_epoch, best_state = inner["subject_equal_BA"], epoch, cpu_tree(model.state_dict())
            no_gain = 0
        elif epoch >= MIN_EPOCH:
            no_gain += 1
        history.append({"epoch": epoch, "train_CE": float(np.mean(losses)),
                        "inner_BA": inner["subject_equal_BA"], "selected": improved,
                        "no_gain": no_gain, "seconds": time.perf_counter() - tick})
        stopped = epoch >= MIN_EPOCH and no_gain >= PATIENCE
        if improved or epoch % 5 == 0 or stopped or epoch == MAX_EPOCHS:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            atomic_torch(latest_path, cpu_tree({"epoch": epoch, "model": model.state_dict(),
                                                "optimizer": optimizer.state_dict(), "rng": rng_state(),
                                                "best": best, "best_epoch": best_epoch,
                                                "best_state": best_state, "no_gain": no_gain,
                                                "history": history, "invariant": invariant}))
        if epoch == 1 or epoch % 5 == 0 or improved or stopped:
            print(f"EPOCH {model_name} {task} f{fold} s{seed} e{epoch} "
                  f"CE={history[-1]['train_CE']:.5f} innerBA={inner['subject_equal_BA']:.5f} "
                  f"sec={history[-1]['seconds']:.1f}", flush=True)
        if stopped:
            break
    if best_state is None:
        raise RuntimeError(f"no eligible checkpoint: {cell}")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    atomic_torch(selected_path, {"state_dict": best_state, "invariant": invariant,
                                 "selected_epoch": best_epoch})
    outer = evaluate(model, arrays["outer"], data["outer_y"], data["outer_subjects"], device)
    record = {"schema": "EEGCONFORMER_FBCNET_SOURCE_CELL_V1", "model": model_name,
              "task": task, "fold": fold, "seed": seed, "channels": data["channels"],
              "samples": data["samples"], "classes": data["classes"],
              "split_sha256": data["split_sha256"], "normalizer": data["normalizer"],
              "protocol_sha256": sha(PROTOCOL), "recipe": recipe, "initial_sha256": initial_hash,
              "invariant_sha256": invariant, "checkpoint_sha256": sha(selected_path),
              "selected_epoch": best_epoch, "epochs_completed": history[-1]["epoch"],
              "early_stopped": stopped, "inner_validation_BA": best,
              "outer_development": outer, "history": history,
              "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "seconds": time.perf_counter() - started}
    atomic_json(record_path, record)
    print(f"CELL_COMPLETE {model_name} {task} f{fold} s{seed} "
          f"epochs={record['epochs_completed']} seconds={record['seconds']:.1f}", flush=True)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def filtered_arrays(task: str, fold: int, data: dict) -> dict:
    root = SCRATCH / task.lower() / f"fold{fold}"
    identity = {"task": task, "fold": fold, "split_sha256": data["split_sha256"],
                "normalizer_sha256": data["normalizer"]["mean_std_sha256"]}
    return {name: make_memmap(data[f"{name}_x"], root / f"{name}.npy",
                                   {**identity, "part": name}) for name in ("train", "val", "outer")}


def cell_complete(model: str, task: str, fold: int, seed: int) -> bool:
    root = cell_dir(model, task, fold, seed)
    record_path, checkpoint = root / "record.json", root / "selected.pt"
    if not record_path.is_file() or not checkpoint.is_file():
        return False
    record = json.loads(record_path.read_text(encoding="utf-8"))
    return sha(checkpoint) == record.get("checkpoint_sha256")


def main() -> None:
    frozen_protocol()
    torch.set_num_threads(int(os.environ.get("NEW_BASELINE_CPU_THREADS", "16")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU baseline queue requires CUDA")
    for task in ACTIVE_TASKS:
        for fold in FOLDS:
            needed = [(model, seed) for model in ACTIVE_MODELS for seed in SEEDS
                      if not cell_complete(model, task, fold, seed)]
            if not needed:
                continue
            print(f"LOAD_FOLD {task} f{fold} pending={len(needed)}", flush=True)
            data = load_search_fold(task, fold)
            for model_name in ACTIVE_MODELS:
                pending = [seed for model, seed in needed if model == model_name]
                if not pending:
                    continue
                if model_name == "EEGConformer":
                    arrays = {name: torch.from_numpy(np.ascontiguousarray(data[f"{name}_x"])).to(device)
                              for name in ("train", "val", "outer")}
                else:
                    arrays = filtered_arrays(task, fold, data)
                for seed in pending:
                    train_cell(model_name, task, fold, seed, data, arrays, device)
                    total_completed = sum(cell_complete(m, t, f, s) for m in MODELS for t in TASKS
                                          for f in FOLDS for s in SEEDS)
                    atomic_json(RUNTIME / f"TRAINING_PROGRESS_{model_name}.json", {"completed_training_cells": total_completed,
                                                                        "expected_training_cells": 120,
                                                                        "current_model": model_name,
                                                                        "current_task": task,
                                                                        "current_fold": fold,
                                                                        "current_seed": seed})
                del arrays
                gc.collect()
                torch.cuda.empty_cache()
            del data
            gc.collect()
            # Fixed-filter arrays are scratch, not experimental checkpoints.
            # Release only this completed fold's known files to keep E: bounded.
            scratch = SCRATCH / task.lower() / f"fold{fold}"
            if "FBCNet" in ACTIVE_MODELS and all(cell_complete("FBCNet", task, fold, seed) for seed in SEEDS):
                for part in ("train", "val", "outer"):
                    for suffix in (".npy", ".json"):
                        path = scratch / f"{part}{suffix}"
                        if path.is_file():
                            path.unlink()
    if not all(cell_complete(model, task, fold, seed) for model in ACTIVE_MODELS
               for task in ACTIVE_TASKS for fold in FOLDS for seed in SEEDS):
        raise RuntimeError(f"training shard incomplete: {ACTIVE_MODELS}, {ACTIVE_TASKS}")
    shard_name = MODEL_SHARD if TASK_SHARD == "ALL" else f"{MODEL_SHARD}_{TASK_SHARD}"
    expected = len(ACTIVE_MODELS) * len(ACTIVE_TASKS) * len(FOLDS) * len(SEEDS)
    atomic_json(RUNTIME / f"TRAINING_SHARD_{shard_name}_COMPLETED.json",
                {"model_shard": MODEL_SHARD, "task_shard": TASK_SHARD,
                 "expected": expected, "completed": expected})
    if all(cell_complete(model, task, fold, seed) for model in MODELS
           for task in TASKS for fold in FOLDS for seed in SEEDS):
        atomic_json(RUNTIME / "TRAINING_COMPLETED.json", {"expected": 120, "completed": 120})
        print("ALL_120_SOURCE_TRAINING_CELLS_COMPLETE", flush=True)
    else:
        print(f"MODEL_SHARD_COMPLETE {shard_name}", flush=True)


if __name__ == "__main__":
    main()
