"""Inner-validation-only selection of the two preregistered TeCh recipes.

No outer-development or fixed-held-out subject is ever materialised here.  The
only score written by this program is inner-validation subject-equal BA, used
to choose TECH-T versus TECH-TC before the benchmark's outer-evaluation freeze.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

from tech_official import TeChAdapter, recipe_config


REPO = Path(os.environ.get("SEVEN_REPO", Path(__file__).resolve().parents[3])).resolve()
EXP = Path(__file__).resolve().parents[1]
PROTOCOL = EXP / "protocol"
RUNTIME = Path(os.environ["SEVEN_RUNTIME"]).resolve()
TASK_IDS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    # Dataclass annotation resolution in the inherited frozen loaders requires
    # their import identity to exist during module execution.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def modules() -> tuple[Any, Any]:
    modern = load_module("seven_backbone_modern_common", REPO / "experiments" / "persist_eeg_outcome_blind_modern_backbone_seed0_v1" / "code" / "modern_common.py")
    tasks = load_module("seven_backbone_task_datasets", REPO / "experiments" / "persist_eeg_openbmi_task_generality_v1" / "code" / "task_datasets.py")
    return modern, tasks


def sha_bytes(*values: bytes) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def save_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    torch.save(value, temp)
    os.replace(temp, path)


def restore_rng(saved: dict[str, Any]) -> None:
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch"].detach().cpu())
    if torch.cuda.is_available() and "cuda" in saved:
        torch.cuda.set_rng_state_all([state.detach().cpu() for state in saved["cuda"]])


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    # Benchmark-relevant stochasticity is seeded; no method change is hidden in
    # a performance flag.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cell_seed(task: str, recipe: str, fold: int) -> int:
    return int.from_bytes(hashlib.sha256(f"TeCh-selection|seed0|{task}|{recipe}|{fold}".encode()).digest()[:4], "little")


def subject_ba(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> float:
    values = []
    for subject in sorted(np.unique(subjects.astype(str)), key=lambda x: int(x.replace("sub-", ""))):
        mask = subjects.astype(str) == subject
        values.append(float(balanced_accuracy_score(labels[mask], logits[mask].argmax(1))))
    if not values: raise RuntimeError("inner-validation subject list is empty")
    return float(np.mean(values))


def _task_data(task_id: str, fold_id: int) -> dict[str, Any]:
    """Load only frozen inner-train and frozen inner-val cache slices."""
    modern, task_code = modules()
    if task_id in ("OpenBMI_MI", "WBCIC_MI"):
        dataset = "OpenBMI" if task_id == "OpenBMI_MI" else "WBCIC"
        folds, _, split_sha = modern.load_split()
        fold = next(value for value in folds[dataset] if int(value["fold_id"]) == int(fold_id))
        train = modern.load_subset(dataset, fold["inner_train_subjects"], modern.SOURCE_SESSIONS[dataset])
        val = modern.load_subset(dataset, fold["inner_val_subjects"], (modern.EVAL_SESSION,))
        mean, std, normalizer = modern.normalizer(dataset, fold["inner_train_subjects"])
        train_x, val_x = modern.normalize(train, mean, std), modern.normalize(val, mean, std)
        train_y, val_y = train.labels, val.labels
        train_subjects, val_subjects = train.subjects, val.subjects
        return {"task": task_id, "fold": fold, "split_sha256": split_sha, "normalizer": normalizer,
                "train_x": train_x, "train_y": train_y, "train_subjects": train_subjects,
                "val_x": val_x, "val_y": val_y, "val_subjects": val_subjects,
                "channels": int(train.channels), "samples": int(train_x.shape[-1]),
                "classes": int(np.unique(train_y).size), "source_sessions": list(modern.SOURCE_SESSIONS[dataset]),
                "future_session": int(modern.EVAL_SESSION)}
    task_name = "ERP" if task_id == "OpenBMI_ERP" else "SSVEP"
    search, _, reference, _ = task_code.split_reference()
    fold = next(value for value in reference["folds"] if int(value["fold_id"]) == int(fold_id))
    bundle = task_code.load_bundle(task_name, search, sessions=(1, 2))
    train_i = bundle.indices(fold["inner_train_subjects"], (1,))
    val_i = bundle.indices(fold["inner_val_subjects"], (2,))
    mean, std, normalizer = task_code.normalizer(bundle, fold["inner_train_subjects"])
    train_x = ((bundle.signal_batch(train_i) - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)).astype(np.float32)
    val_x = ((bundle.signal_batch(val_i) - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)).astype(np.float32)
    train_y, val_y = bundle.labels(train_i), bundle.labels(val_i)
    train_subjects = np.asarray([bundle.rows[int(i)].subject for i in train_i], dtype=object)
    val_subjects = np.asarray([bundle.rows[int(i)].subject for i in val_i], dtype=object)
    return {"task": task_id, "fold": fold, "split_sha256": reference["source_sha256"], "normalizer": normalizer,
            "train_x": train_x, "train_y": train_y, "train_subjects": train_subjects,
            "val_x": val_x, "val_y": val_y, "val_subjects": val_subjects,
            "channels": int(bundle.channels), "samples": int(bundle.samples),
            "classes": int(np.unique(train_y).size), "source_sessions": [1], "future_session": 2}


def _validate_data(data: dict[str, Any]) -> None:
    for key in ("train_x", "val_x"):
        value = data[key]
        if value.ndim != 3 or not np.isfinite(value).all(): raise RuntimeError(f"invalid {key}: {value.shape}")
    for key in ("train_y", "val_y"):
        value = data[key]
        if set(map(int, np.unique(value))) != set(range(data["classes"])):
            raise RuntimeError(f"label map is not contiguous for {data['task']}: {np.unique(value)}")
    if set(map(str, data["fold"]["inner_train_subjects"])) & set(map(str, data["fold"]["inner_val_subjects"])):
        raise RuntimeError("frozen train/validation subject overlap")


def evaluate(model: torch.nn.Module, x: torch.Tensor, labels: np.ndarray, subjects: np.ndarray, batch_size: int) -> float:
    model.eval(); chunks = []
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            chunks.append(model(x[start:start + batch_size]).float().cpu().numpy())
    return subject_ba(labels, np.concatenate(chunks), subjects)


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_cell(data: dict[str, Any], recipe: str, batch_size: int, device: torch.device, allow_fallback: bool = True) -> dict[str, Any]:
    task, fold_id = data["task"], int(data["fold"]["fold_id"])
    seed = cell_seed(task, recipe, fold_id)
    cell = RUNTIME / "tech_recipe_selection" / task.lower() / recipe.lower() / f"fold{fold_id}_seed0"
    record_path, latest_path, selected_path = cell / "record.json", cell / "latest.pt", cell / "selected.pt"
    if record_path.is_file() and selected_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("recipe") == recipe and int(record.get("batch_size", -1)) == batch_size:
            return record
    _validate_data(data); set_seed(seed)
    config = recipe_config(channels=data["channels"], samples=data["samples"], classes=data["classes"], recipe=recipe)
    config.batch_size = int(batch_size)
    model = TeChAdapter(config).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable: raise RuntimeError("TeCh has no trainable parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    invariant = sha_bytes(json.dumps({"task":task, "recipe":recipe, "fold":fold_id, "batch_size":batch_size,
                                      "split":data["split_sha256"], "normalizer":data["normalizer"]}, sort_keys=True).encode())
    start, best, best_epoch, best_state, history = 1, -float("inf"), 0, None, []
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        if saved.get("invariant") != invariant: raise RuntimeError(f"resume invariant mismatch: {latest_path}")
        model.load_state_dict(saved["model"], strict=True); optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"]); start = int(saved["epoch"]) + 1; best = float(saved["best"]); best_epoch = int(saved["best_epoch"])
        best_state, history = saved["best_state"], list(saved["history"])
    x_train = torch.from_numpy(np.ascontiguousarray(data["train_x"])).to(device, non_blocking=True)
    y_train = torch.as_tensor(data["train_y"], dtype=torch.long, device=device)
    x_val = torch.from_numpy(np.ascontiguousarray(data["val_x"])).to(device, non_blocking=True)
    # This records a hard fail before a costly grid if input/model is incompatible.
    model.train(); optimizer.zero_grad(set_to_none=True)
    sanity_logits = model(x_train[:min(len(x_train), batch_size)])
    sanity_loss = F.cross_entropy(sanity_logits, y_train[:len(sanity_logits)])
    if not torch.isfinite(sanity_loss) or not torch.isfinite(sanity_logits).all(): raise RuntimeError("TeCh non-finite synthetic/data sanity")
    sanity_loss.backward()
    if not any(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable): raise RuntimeError("TeCh gradient sanity failed")
    optimizer.zero_grad(set_to_none=True); model.eval()
    began = time.perf_counter(); peak_before = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    try:
        for epoch in range(start, config.train_epochs + 1):
            model.train(); order = np.random.default_rng(seed + 1000003 * epoch).permutation(len(y_train)); losses = []
            for start_i in range(0, len(order), batch_size):
                ii = torch.as_tensor(order[start_i:start_i + batch_size], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x_train.index_select(0, ii))
                loss = F.cross_entropy(logits, y_train.index_select(0, ii))
                if not torch.isfinite(loss): raise RuntimeError(f"non-finite TeCh loss at {task}/{recipe}/fold{fold_id}/epoch{epoch}")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0); optimizer.step()
                losses.append(float(loss.detach().cpu()))
            val_ba = evaluate(model, x_val, data["val_y"], data["val_subjects"], batch_size)
            selected = val_ba > best + 1e-12  # exact earliest-tie convention
            if selected: best, best_epoch, best_state = float(val_ba), int(epoch), cpu_state(model)
            history.append({"epoch":int(epoch), "train_cross_entropy":float(np.mean(losses)), "inner_val_subject_equal_BA":float(val_ba), "selected":bool(selected)})
            save_torch(latest_path, {"invariant":invariant, "epoch":epoch, "model":cpu_state(model), "optimizer":optimizer.state_dict(), "rng":rng_state(), "best":best, "best_epoch":best_epoch, "best_state":best_state, "history":history})
    except RuntimeError as exc:
        is_oom = device.type == "cuda" and "out of memory" in str(exc).lower()
        del model, x_train, y_train, x_val
        if device.type == "cuda": torch.cuda.empty_cache()
        if is_oom and allow_fallback and batch_size > 1 and not latest_path.exists():
            return train_cell(data, recipe, max(1, batch_size // 2), device, allow_fallback=False)
        raise
    if best_state is None: raise RuntimeError("TeCh selection has no selected epoch")
    save_torch(selected_path, best_state)
    peak = (torch.cuda.max_memory_allocated(device) - peak_before) if device.type == "cuda" else 0
    record = {"task":task, "model":"TeCh", "recipe":recipe, "seed":0, "fold":fold_id, "batch_size":batch_size,
              "parameter_count":int(sum(p.numel() for p in model.parameters())), "selected_epoch":best_epoch,
              "inner_val_subject_equal_BA":best, "epochs_completed":len(history), "train_seconds":time.perf_counter()-began,
              "peak_gpu_memory_bytes_delta":int(max(0, peak)), "checkpoint_path":str(selected_path),
              "checkpoint_sha256":sha256_file(selected_path), "split_sha256":data["split_sha256"],
              "normalizer":data["normalizer"], "source_sessions":data["source_sessions"], "future_session":data["future_session"],
              "history":history, "outer_data_accessed":False, "fixed_heldout_accessed":False}
    write_json(record_path, record)
    del model, x_train, y_train, x_val
    if device.type == "cuda": torch.cuda.empty_cache()
    return record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def write_selection(records: list[dict[str, Any]]) -> None:
    if len(records) != 40: raise RuntimeError(f"TeCh selection incomplete: expected 40 records, got {len(records)}")
    rows = []
    for task in TASK_IDS:
        scores = {recipe: np.mean([r["inner_val_subject_equal_BA"] for r in records if r["task"] == task and r["recipe"] == recipe]) for recipe in ("TECH-T", "TECH-TC")}
        margin_pp = float((scores["TECH-TC"] - scores["TECH-T"]) * 100.0)
        # Positive margin beyond +0.10 pp selects TECH-TC; otherwise the preregistered smaller TECH-T wins.
        selected = "TECH-TC" if margin_pp > 0.10 else "TECH-T"
        parameter_count = next(r["parameter_count"] for r in records if r["task"] == task and r["recipe"] == selected)
        rows.append({"task":task, "TECH-T_inner_val_BA":scores["TECH-T"], "TECH-TC_inner_val_BA":scores["TECH-TC"],
                     "selected_recipe":selected, "selection_margin_pp_TECH_TC_minus_TECH_T":margin_pp,
                     "selected_parameter_count":parameter_count, "selection_uses_outer_data":False})
    import csv
    path = PROTOCOL / "TECH_RECIPE_SELECTION.csv"; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(PROTOCOL / "TECH_RECIPE_SELECTION_LOCK.json", {"selection_seed":0, "folds":5, "records":[{k:v for k,v in r.items() if k != "history"} for r in records], "selection":rows,
        "tie_rule":"TECH-T when TECH-TC minus TECH-T is <= +0.10 percentage points", "outer_data_accessed":False, "fixed_heldout_accessed":False})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", choices=TASK_IDS, default=list(TASK_IDS))
    parser.add_argument("--folds", nargs="*", type=int, choices=range(5), default=list(range(5)))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("TeCh recipe selection is GPU-required for the declared execution protocol")
    records = []
    for task in args.tasks:
        for fold in args.folds:
            data = _task_data(task, fold)
            for recipe in ("TECH-T", "TECH-TC"):
                print(json.dumps({"event":"start", "task":task, "fold":fold, "recipe":recipe,
                                  "train_trials":int(len(data["train_y"])), "val_trials":int(len(data["val_y"])),
                                  "channels":data["channels"], "samples":data["samples"], "classes":data["classes"]}, sort_keys=True), flush=True)
                record = train_cell(data, recipe, 128, device)
                records.append(record)
                print(json.dumps({k:record[k] for k in ("task","fold","recipe","selected_epoch","inner_val_subject_equal_BA","train_seconds")}, sort_keys=True), flush=True)
    if set(args.tasks) == set(TASK_IDS) and set(args.folds) == set(range(5)):
        write_selection(records)
        print("TECH_RECIPE_SELECTION_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
