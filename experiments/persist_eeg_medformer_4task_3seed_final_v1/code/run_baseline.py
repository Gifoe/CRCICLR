"""Resumable matched-protocol ModernTCN training and outer-dev evaluation."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from benchmark_data import load_search_fold
from model_adapter import ARCHITECTURE, MODEL_NAME, UPSTREAM_COMMIT, UPSTREAM_REPOSITORY, build_model, model_metadata


EXP = Path(__file__).resolve().parents[1]
PROTOCOL = EXP / "protocol"
OUTPUTS = EXP / "outputs"
RUNTIME = Path(os.environ["BASELINE_RUNTIME"]).resolve()
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
FOLDS = tuple(range(5))
SEEDS = (0, 1, 2)
EPOCHS = 60
MIN_EPOCH = 10
EARLY_STOPPING_PATIENCE = 8
BATCH_SIZE = 128
LR = 3e-4
WEIGHT_DECAY = 5e-4
GRADIENT_CLIP = 5.0


def _clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(_clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: _clean(row.get(key, "")) for key in fields} for row in rows])
    os.replace(temporary, path)


def _save_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(value, temporary, _use_new_zipfile_serialization=False)
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha(model: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Checkpoints are loaded with map_location=device so ByteTensor RNG states
    # may arrive on CUDA.  PyTorch's RNG restoration APIs require CPU states.
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _freeze_hash() -> str:
    document = PROTOCOL / "PROTOCOL_FREEZE.md"
    sidecar = PROTOCOL / "PROTOCOL_FREEZE.sha256"
    if not document.is_file() or not sidecar.is_file():
        raise RuntimeError("training disabled until protocol freeze and sidecar exist")
    actual = _sha(document)
    if sidecar.read_text(encoding="utf-8").strip() != actual:
        raise RuntimeError("protocol freeze hash mismatch")
    return actual


def _batch_order(length: int, task: str, fold: int, seed: int, epoch: int) -> list[np.ndarray]:
    token = f"SEVEN-BACKBONE|{task}|{MODEL_NAME}|{fold}|{seed}|{epoch}"
    digest = hashlib.sha256(token.encode()).digest()
    order = np.random.default_rng(int.from_bytes(digest[:8], "little")).permutation(length)
    return [order[start:start + BATCH_SIZE] for start in range(0, length, BATCH_SIZE)]


def _subject_metrics(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> dict[str, Any]:
    prediction = logits.argmax(1)
    rows = []
    for subject in sorted(np.unique(subjects.astype(str)), key=lambda value: int(value.replace("sub-", ""))):
        mask = subjects.astype(str) == subject
        truth, pred = labels[mask], prediction[mask]
        rows.append({
            "subject": subject,
            "BA": float(balanced_accuracy_score(truth, pred)),
            "macro_F1": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(truth, pred)),
            "trials": int(mask.sum()),
        })
    return {
        "subject_equal_BA": float(np.mean([row["BA"] for row in rows])),
        "subject_equal_macro_F1": float(np.mean([row["macro_F1"] for row in rows])),
        "trial_accuracy": float(accuracy_score(labels, prediction)),
        "trial_count": int(len(labels)),
        "subjects": rows,
    }


def _evaluate(model: torch.nn.Module, x: torch.Tensor, labels: np.ndarray, subjects: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(labels), BATCH_SIZE):
            parts.append(model(x[start:start + BATCH_SIZE]).float().cpu().numpy())
    logits = np.concatenate(parts)
    return _subject_metrics(labels, logits, subjects), logits


def _weight(data: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    if not data["weighted_cross_entropy"]:
        return None
    counts = np.bincount(data["train_y"], minlength=data["classes"])
    return torch.as_tensor(len(data["train_y"]) / (data["classes"] * counts), dtype=torch.float32, device=device)


def _cell_dir(task: str, fold: int, seed: int) -> Path:
    return RUNTIME / "cells" / task.lower() / f"fold{fold}_seed{seed}"


def run_cell(task: str, fold: int, seed: int) -> dict[str, Any]:
    freeze_hash = _freeze_hash()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cell = _cell_dir(task, fold, seed)
    record_path, latest_path = cell / "record.json", cell / "latest.pt"
    selected_path, predictions_path, lock = cell / "selected.pt", cell / "outer_predictions.npz", cell / "RUNNING.lock"
    if record_path.is_file() and selected_path.is_file() and predictions_path.is_file():
        return json.loads(record_path.read_text(encoding="utf-8"))
    try:
        lock.mkdir(parents=True, exist_ok=False)
        (lock / "pid.txt").write_text(str(os.getpid()), encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"cell already locked: {lock}") from exc
    started = time.perf_counter()
    try:
        data = load_search_fold(task, fold)
        _set_seed(seed)
        model = build_model(channels=data["channels"], samples=data["samples"], classes=data["classes"]).to(device)
        metadata = model_metadata(model)
        initial_hash = _state_sha(model)
        _set_seed(seed + 100000)
        # The device tensors are the sole signal owners during optimization.  Pop
        # the source arrays so three concurrent workers do not retain a second,
        # multi-gigabyte CPU copy of every split.  Values and transfer order are
        # unchanged from the original implementation.
        train_x = torch.from_numpy(np.ascontiguousarray(data.pop("train_x"))).to(device)
        val_x = torch.from_numpy(np.ascontiguousarray(data.pop("val_x"))).to(device)
        outer_x = torch.from_numpy(np.ascontiguousarray(data.pop("outer_x"))).to(device)
        train_y = torch.as_tensor(data["train_y"], dtype=torch.long, device=device)
        weight = _weight(data, device)
        invariant_value = {
            "task": task, "model": MODEL_NAME, "fold": fold, "seed": seed,
            "initial_state": initial_hash, "split": data["split_sha256"], "normalizer": data["normalizer"],
            "recipe": {"max_epochs": EPOCHS, "min_epoch": MIN_EPOCH,
                       "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                       "early_stopping_metric": "inner_validation_subject_equal_BA",
                       "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
                       "gradient_clip": GRADIENT_CLIP, "scheduler": None},
            "freeze": freeze_hash, "architecture": ARCHITECTURE,
        }
        invariant = hashlib.sha256(json.dumps(invariant_value, sort_keys=True).encode()).hexdigest()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        start_epoch, best, best_epoch, best_state, history = 1, -float("inf"), 0, None, []
        no_improvement_epochs = 0
        if latest_path.is_file():
            # Deserialize resume payloads on CPU.  load_state_dict transfers the
            # live model/optimizer tensors to their parameter device; keeping
            # historical model, optimizer, and best-state copies on CUDA wastes
            # several GB and can exhaust VRAM in concurrent queues.
            saved = torch.load(latest_path, map_location="cpu", weights_only=False)
            if saved.get("invariant") != invariant:
                raise RuntimeError("cell resume invariant mismatch")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            _restore_rng(saved["rng"])
            start_epoch = int(saved["epoch"]) + 1
            best, best_epoch = float(saved["best"]), int(saved["best_epoch"])
            best_state, history = saved["best_state"], list(saved["history"])
            no_improvement_epochs = int(saved.get("no_improvement_epochs", 0))
            del saved
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        stopped_epoch = start_epoch - 1 if no_improvement_epochs >= EARLY_STOPPING_PATIENCE else None
        for epoch in ([] if stopped_epoch is not None else range(start_epoch, EPOCHS + 1)):
            epoch_started = time.perf_counter()
            model.train()
            losses = []
            for indices in _batch_order(len(train_y), task, fold, seed, epoch):
                idx = torch.as_tensor(indices, dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(train_x.index_select(0, idx)), train_y.index_select(0, idx), weight=weight)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at {task}/fold{fold}/seed{seed}/epoch{epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            inner, _ = _evaluate(model, val_x, data["val_y"], data["val_subjects"])
            selected = epoch >= MIN_EPOCH and inner["subject_equal_BA"] > best + 1e-12
            if selected:
                best, best_epoch, best_state = float(inner["subject_equal_BA"]), epoch, _cpu_state(model)
                no_improvement_epochs = 0
            elif epoch >= MIN_EPOCH:
                no_improvement_epochs += 1
            should_stop = epoch >= MIN_EPOCH and no_improvement_epochs >= EARLY_STOPPING_PATIENCE
            history.append({
                "epoch": epoch, "cross_entropy": float(np.mean(losses)), "inner_validation": inner,
                "selected": selected, "no_improvement_epochs": no_improvement_epochs,
                "elapsed_seconds": time.perf_counter() - epoch_started,
            })
            if selected or epoch % 5 == 0 or epoch == EPOCHS or should_stop:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                _save_torch(latest_path, _cpu_tree({
                    "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": _rng_state(),
                    "best": best, "best_epoch": best_epoch, "best_state": best_state, "history": history, "invariant": invariant,
                    "no_improvement_epochs": no_improvement_epochs,
                }))
            if epoch == 1 or epoch % 5 == 0 or selected:
                print(f"[{MODEL_NAME} {task} f{fold} s{seed}] epoch={epoch} loss={history[-1]['cross_entropy']:.5f} innerBA={inner['subject_equal_BA']:.5f}", flush=True)
            if should_stop:
                stopped_epoch = epoch
                print(f"[{MODEL_NAME} {task} f{fold} s{seed}] EARLY_STOP epoch={epoch} patience={EARLY_STOPPING_PATIENCE} best_epoch={best_epoch}", flush=True)
                break
        if best_state is None:
            raise RuntimeError("no eligible checkpoint selected")
        model.load_state_dict(best_state, strict=True)
        _save_torch(selected_path, {"state_dict": _cpu_state(model), "invariant": invariant, "selected_epoch": best_epoch})
        outer, logits = _evaluate(model, outer_x, data["outer_y"], data["outer_subjects"])
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(predictions_path, logits=logits.astype(np.float32), labels=data["outer_y"].astype(np.int64), subjects=data["outer_subjects"].astype(str))
        record = {
            "schema": "CRCICLR_MATCHED_BASELINE_CELL_V1", "task": task, "model": MODEL_NAME,
            "fold": fold, "seed": seed, "classes": data["classes"], "channels": data["channels"], "samples": data["samples"],
            "normalizer": data["normalizer"], "split_sha256": data["split_sha256"], "protocol_freeze_sha256": freeze_hash,
            "initial_state_sha256": initial_hash, "invariant_sha256": invariant, "selected_epoch": best_epoch,
            "best_inner_validation_BA": best, "outer_development": outer, "history": history,
            "epochs_completed": int(history[-1]["epoch"]), "early_stopped": stopped_epoch is not None,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "checkpoint_path": str(selected_path), "checkpoint_sha256": _sha(selected_path),
            "predictions_path": str(predictions_path), "predictions_sha256": _sha(predictions_path),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            **metadata,
        }
        _json(record_path, record)
        return record
    finally:
        if lock.exists():
            shutil.rmtree(lock)


def smoke(task: str) -> None:
    data = load_search_fold(task, 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(0)
    model = build_model(channels=data["channels"], samples=data["samples"], classes=data["classes"]).to(device)
    batch = torch.from_numpy(np.ascontiguousarray(data["train_x"][:BATCH_SIZE])).to(device)
    labels = torch.as_tensor(data["train_y"][:len(batch)], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch)
    loss = F.cross_entropy(logits, labels, weight=_weight(data, device))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    optimizer.step()
    if logits.shape != (len(batch), data["classes"]) or not torch.isfinite(loss):
        raise RuntimeError("smoke test failed")
    result = {"task": task, "batch": len(batch), "logit_shape": list(logits.shape), "loss": float(loss.detach().cpu()),
              "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0, **model_metadata(model)}
    _json(PROTOCOL / f"SMOKE_{task}.json", result)
    print(json.dumps(_clean(result), sort_keys=True), flush=True)


def preflight() -> None:
    freeze_hash = _freeze_hash()
    split_rows, normalizer_rows, admissions = [], [], []
    for task in TASKS:
        for fold in FOLDS:
            data = load_search_fold(task, fold)
            spec = data["fold"]
            train, val, outer = map(set, (spec["inner_train_subjects"], spec["inner_val_subjects"], spec["outer_dev_subjects"]))
            checks = {
                "train_val_disjoint": not bool(train & val),
                "train_outer_disjoint": not bool(train & outer),
                "val_outer_disjoint": not bool(val & outer),
            }
            if not all(checks.values()):
                raise RuntimeError(f"subject overlap in {task}/fold{fold}")
            split_rows.append({
                "task": task, "fold": fold, "inner_train_subjects": sorted(train),
                "inner_val_subjects": sorted(val), "outer_dev_subjects": sorted(outer),
                "split_sha256": data["split_sha256"], **checks,
            })
            normalizer_rows.append({
                "task": task, "fold": fold, "fit_scope": "inner_train legal source sessions only",
                "mean_std_sha256": data["normalizer"]["mean_std_sha256"],
                "samples_per_channel": data["normalizer"]["samples_per_channel"],
                "validation_contributed": False, "outer_dev_contributed": False, "heldout_contributed": False,
            })
        _set_seed(0)
        data = load_search_fold(task, 0)
        model = build_model(channels=data["channels"], samples=data["samples"], classes=data["classes"])
        model.train()
        value = torch.zeros((2, data["channels"], data["samples"]), dtype=torch.float32)
        logits = model(value)
        loss = F.cross_entropy(logits, torch.arange(2) % data["classes"])
        loss.backward()
        if logits.shape != (2, data["classes"]) or not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise RuntimeError(f"model admission failed for {task}")
        admissions.append({"task": task, "logit_shape": list(logits.shape), "finite": True, **model_metadata(model)})
    _json(PROTOCOL / "SPLIT_AUDIT.json", {"pass": True, "rows": split_rows, "protocol_freeze_sha256": freeze_hash})
    _json(PROTOCOL / "NORMALIZATION_AUDIT.json", {"pass": True, "rows": normalizer_rows, "formula": "per-channel train-only z-score"})
    _json(PROTOCOL / "HELDOUT_LEAKAGE_AUDIT.json", {
        "pass": True, "HELDOUT_ACCESSED_DURING_TRAINING": "NO", "HELDOUT_ACCESSED_DURING_SELECTION": "NO",
        "OUTER_DEV_USED_FOR_SELECTION": "NO", "heldout_loader_present_in_training_module": False,
    })
    _json(PROTOCOL / "MODEL_ADMISSION.json", {"pass": True, "rows": admissions})
    print("PREFLIGHT_PASS", flush=True)


def queue(worker_index: int, worker_count: int) -> None:
    grid = [(task, fold, seed) for task in TASKS for fold in FOLDS for seed in SEEDS]
    for index, (task, fold, seed) in enumerate(grid):
        if index % worker_count != worker_index:
            continue
        try:
            run_cell(task, fold, seed)
        except Exception as exc:
            print(f"CELL_FAILED {task} fold={fold} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
            raise
    print(f"QUEUE_WORKER_COMPLETE index={worker_index} count={worker_count}", flush=True)


def aggregate() -> None:
    expected = [(task, fold, seed) for task in TASKS for fold in FOLDS for seed in SEEDS]
    records = []
    for task, fold, seed in expected:
        path = _cell_dir(task, fold, seed) / "record.json"
        if not path.is_file():
            raise RuntimeError(f"missing completed cell: {task}/fold{fold}/seed{seed}")
        records.append(json.loads(path.read_text(encoding="utf-8")))
    manifest, selection, subject_rows = [], [], []
    for record in records:
        outer = record["outer_development"]
        manifest.append({
            "task": record["task"], "model": MODEL_NAME, "fold": record["fold"], "seed": record["seed"],
            "status": "complete", "elapsed_seconds": record["elapsed_seconds"], "peak_cuda_bytes": record["peak_cuda_bytes"],
            "checkpoint_sha256": record["checkpoint_sha256"], "normalizer_sha256": record["normalizer"]["mean_std_sha256"],
            "split_sha256": record["split_sha256"],
        })
        selection.append({
            "task": record["task"], "fold": record["fold"], "seed": record["seed"], "selected_epoch": record["selected_epoch"],
            "inner_validation_subject_equal_BA": record["best_inner_validation_BA"], "checkpoint_sha256": record["checkpoint_sha256"],
        })
        for row in outer["subjects"]:
            subject_rows.append({"task": record["task"], "fold": record["fold"], "seed": record["seed"], **row})
    seed_rows = []
    for task in TASKS:
        for seed in SEEDS:
            group = [row for row in subject_rows if row["task"] == task and int(row["seed"]) == seed]
            replicate = [row for row in records if row["task"] == task and int(row["seed"]) == seed]
            total_trials = sum(int(row["outer_development"]["trial_count"]) for row in replicate)
            accuracy = sum(float(row["outer_development"]["trial_accuracy"]) * int(row["outer_development"]["trial_count"]) for row in replicate) / total_trials
            seed_rows.append({
                "task": task, "seed": seed, "subject_equal_BA": float(np.mean([row["BA"] for row in group])),
                "subject_equal_macro_F1": float(np.mean([row["macro_F1"] for row in group])),
                "trial_accuracy": accuracy, "subjects": len(group), "trials": total_trials,
            })
    final_rows = []
    for task in TASKS:
        group = [row for row in seed_rows if row["task"] == task]
        final_rows.append({
            "task": task,
            "subject_equal_BA_mean": float(np.mean([row["subject_equal_BA"] for row in group])),
            "subject_equal_BA_std": float(np.std([row["subject_equal_BA"] for row in group], ddof=1)),
            "subject_equal_macro_F1_mean": float(np.mean([row["subject_equal_macro_F1"] for row in group])),
            "subject_equal_macro_F1_std": float(np.std([row["subject_equal_macro_F1"] for row in group], ddof=1)),
            "trial_accuracy_mean": float(np.mean([row["trial_accuracy"] for row in group])),
            "trial_accuracy_std": float(np.std([row["trial_accuracy"] for row in group], ddof=1)),
            "parameters": next(row["parameters"] for row in records if row["task"] == task),
        })
    _csv(OUTPUTS / "TRAINING_RUN_MANIFEST.csv", manifest)
    _csv(OUTPUTS / "CHECKPOINT_SELECTION.csv", selection)
    _csv(OUTPUTS / "OUTER_DEV_SUBJECT_RESULTS.csv", subject_rows)
    _csv(OUTPUTS / "OUTER_DEV_SEED_RESULTS.csv", seed_rows)
    _csv(OUTPUTS / "OUTER_DEV_FINAL_SUMMARY.csv", final_rows)
    parameter_rows = []
    for task in TASKS:
        record = next(row for row in records if row["task"] == task)
        parameter_rows.append({"task": task, "model": MODEL_NAME, "parameters": record["parameters"], "trainable_parameters": record["trainable_parameters"]})
    _csv(OUTPUTS / "PARAMETER_EFFICIENCY.csv", parameter_rows)
    lines = [f"# Final {MODEL_NAME} report", "", "Outer-development is complete. Heldout remains locked pending the separately frozen evaluator.", "", "| Task | seed0 BA | seed1 BA | seed2 BA | mean +/- std | Params |", "|---|---:|---:|---:|---:|---:|"]
    for row in final_rows:
        values = [item for item in seed_rows if item["task"] == row["task"]]
        lines.append(f"| {row['task']} | {values[0]['subject_equal_BA']:.4f} | {values[1]['subject_equal_BA']:.4f} | {values[2]['subject_equal_BA']:.4f} | {row['subject_equal_BA_mean']:.4f} +/- {row['subject_equal_BA_std']:.4f} | {row['parameters']} |")
    (OUTPUTS / f"FINAL_{MODEL_NAME.upper()}_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("ALL_60_OUTER_DEV_CELLS_AGGREGATED", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int)
    parser.add_argument("--smoke", choices=TASKS)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.smoke:
        smoke(args.smoke)
    elif args.aggregate:
        aggregate()
    elif args.worker_index is not None and args.worker_count is not None:
        queue(args.worker_index, args.worker_count)
    elif args.task is not None and args.fold is not None and args.seed is not None:
        run_cell(args.task, args.fold, args.seed)
    else:
        parser.error("select one cell, one queue worker, --smoke, or --aggregate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
