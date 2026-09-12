"""Locked LiteBN seed-0 outer aggregation and one-shot V8 internal-holdout audit.

This program reuses only the selected checkpoints from the seven-backbone
runtime.  It never trains a model and refuses old carrier LiteBN checkpoints:
their parameter schema is not the seven-backbone LiteBN schema.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


EXP = Path(__file__).resolve().parents[1]
OUT = EXP / "outputs"
PROTOCOL = EXP / "protocol"
RUNTIME = Path(os.environ["SEVEN_RUNTIME"]).resolve()
REPO = Path(os.environ["SEVEN_REPO"]).resolve()
CODE = REPO / "experiments" / "persist_eeg_seven_backbone_fourtask_3seed_v1" / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
DATASET = {"OpenBMI_MI": "OpenBMI", "OpenBMI_ERP": "OpenBMI", "OpenBMI_SSVEP": "OpenBMI", "WBCIC_MI": "WBCIC"}
CLASS_COUNT = {"OpenBMI_MI": 2, "OpenBMI_ERP": 2, "OpenBMI_SSVEP": 4, "WBCIC_MI": 2}
HELDOUT_SPLIT = REPO / "experiments" / "persist_eeg_final_model_v8" / "outputs" / "protocol" / "V8_SEARCH_SPLIT.json"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value)
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if isinstance(value, dict): return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_clean(v) for v in value]
    return value


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader(); writer.writerows(_clean(rows))


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def _sort(values: list[str]) -> list[str]:
    return sorted(map(str, values), key=lambda x: int(x.replace("sub-", "")))


def _cell(task: str, fold: int) -> tuple[dict[str, Any], Path]:
    root = RUNTIME / "search_cells" / task.lower() / "litebn" / f"fold{fold}_seed0"
    record, checkpoint = root / "record.json", root / "selected.pt"
    if not record.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"seven-backbone LiteBN seed0 cell is incomplete: {root}")
    payload = json.loads(record.read_text(encoding="utf-8"))
    if payload.get("task") != task or payload.get("model") != "LiteBN" or int(payload.get("fold", -1)) != fold or int(payload.get("seed", -1)) != 0:
        raise RuntimeError(f"cell identity mismatch: {root}")
    if payload.get("checkpoint_sha256") != _sha(checkpoint):
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
    return payload, checkpoint


def _records() -> list[dict[str, Any]]:
    rows = []
    for task in TASKS:
        for fold in range(5):
            payload, checkpoint = _cell(task, fold)
            outer = payload["outer_development"]
            rows.append({"task": task, "dataset": DATASET[task], "fold": fold, "seed": 0,
                         "outer_subject_equal_BA": outer["subject_equal_BA"],
                         "outer_subject_equal_macro_F1": outer["subject_equal_macro_F1"],
                         "outer_trial_accuracy": outer["trial_accuracy"],
                         "outer_subjects": len(outer["subjects"]),
                         "selected_epoch": payload["selected_epoch"],
                         "checkpoint_path": str(checkpoint), "checkpoint_sha256": _sha(checkpoint),
                         "normalizer_sha256": payload["normalizer"]["mean_std_sha256"],
                         "invariant_sha256": payload["invariant_sha256"]})
    return rows


def _holdout_membership() -> dict[str, list[str]]:
    raw = json.loads(HELDOUT_SPLIT.read_text(encoding="utf-8"))
    result = {"OpenBMI": _sort(raw["openbmi"]["V8_INTERNAL_HOLDOUT"]),
              "WBCIC": _sort(raw["wbcic"]["V8_INTERNAL_HOLDOUT"])}
    if len(result["OpenBMI"]) != 14 or len(result["WBCIC"]) != 10:
        raise RuntimeError(f"unexpected V8 internal-holdout membership: {result}")
    return result


def outer_lock() -> None:
    """Aggregate already-written outer records, then lock all heldout inputs."""
    rows = _records()
    _csv(OUT / "OUTER_DEVELOPMENT_FOLD_RESULTS.csv", rows)
    summary = []
    for task in TASKS:
        group = [row for row in rows if row["task"] == task]
        summary.append({"task": task, "dataset": DATASET[task], "seed": 0, "folds": len(group),
                        "outer_subject_equal_BA_mean": float(np.mean([r["outer_subject_equal_BA"] for r in group])),
                        "outer_subject_equal_BA_std": float(np.std([r["outer_subject_equal_BA"] for r in group], ddof=1)),
                        "outer_subject_equal_macro_F1_mean": float(np.mean([r["outer_subject_equal_macro_F1"] for r in group])),
                        "outer_trial_accuracy_mean": float(np.mean([r["outer_trial_accuracy"] for r in group]))})
    _csv(OUT / "OUTER_DEVELOPMENT_TASK_RESULTS.csv", summary)
    membership = _holdout_membership()
    lock = {"protocol": "SEVEN_BACKBONE_LITEBN_SEED0_OUTER_TO_V8_INTERNAL_HELDOUT_V1",
            "purpose": "one-shot frozen selected-checkpoint evaluation; no training, tuning, calibration, or refit",
            "heldout_scope": "V8_INTERNAL_HOLDOUT only; final/true outer cohort is not accessed",
            "seed": 0, "tasks": list(TASKS), "heldout_subjects": membership,
            "heldout_split_path": str(HELDOUT_SPLIT), "heldout_split_sha256": _sha(HELDOUT_SPLIT),
            "outer_records": rows, "pre_lock_outcomes": {"outer_development_aggregated": True, "heldout_labels_read": False},
            "requirements": {"records": 20, "selected_checkpoints": 20, "each_task_folds": 5,
                             "model": "seven-backbone LiteBN", "old_carrier_litebn_checkpoints_allowed": False}}
    _json(PROTOCOL / "OUTER_TO_HELDOUT_LOCK.json", lock)
    (PROTOCOL / "OUTER_TO_HELDOUT_LOCK.sha256").write_text(_sha(PROTOCOL / "OUTER_TO_HELDOUT_LOCK.json") + "\n", encoding="utf-8")
    _json(PROTOCOL / "OUTER_TO_HELDOUT_PREFLIGHT.json", {"pass": True, "heldout_labels_read": False,
          "record_count": len(rows), "checkpoint_count": len(rows), "final_true_outer_accessed": False,
          "lock_sha256": _sha(PROTOCOL / "OUTER_TO_HELDOUT_LOCK.json")})
    print("OUTER_AGGREGATED_AND_HELDOUT_LOCKED", flush=True)


def _metrics(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> tuple[dict[str, float], list[dict[str, Any]]]:
    prediction = logits.argmax(axis=1); rows = []
    for subject in _sort(list(np.unique(subjects.astype(str)))):
        where = subjects.astype(str) == subject
        rows.append({"subject_id": subject, "BA": float(balanced_accuracy_score(labels[where], prediction[where])),
                     "macro_F1": float(f1_score(labels[where], prediction[where], average="macro", zero_division=0)),
                     "accuracy": float(accuracy_score(labels[where], prediction[where])), "trials": int(where.sum())})
    return {"subject_equal_BA": float(np.mean([r["BA"] for r in rows])),
            "subject_equal_macro_F1": float(np.mean([r["macro_F1"] for r in rows])),
            "trial_accuracy": float(accuracy_score(labels, prediction))}, rows


def _infer(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    values = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 128):
            batch = torch.from_numpy(np.ascontiguousarray(x[start:start + 128], dtype=np.float32)).to(device, non_blocking=True)
            values.append(model(batch).float().cpu().numpy())
    return np.concatenate(values, axis=0)


def _load_model(task: str, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    models = _module("seven_litebn_models", CODE / "backbone_models.py")
    samples = {"OpenBMI_MI": 1000, "OpenBMI_ERP": 250, "OpenBMI_SSVEP": 1000, "WBCIC_MI": 1000}[task]
    channels = 58 if task == "WBCIC_MI" else 62
    model = models.build_model("LiteBN", dataset=DATASET[task], channels=channels, samples=samples, classes=CLASS_COUNT[task])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("invalid eval state")
    return model


def _mi_holdout(task: str, fold: int, heldout: dict[str, list[str]]):
    data = _module("seven_litebn_benchmark_data", CODE / "benchmark_data.py")
    inner = data._load_module("seven_litebn_inner_loader", CODE / "tech_recipe_selection.py")
    modern, _ = data._sources(); dataset = DATASET[task]
    folds, _, _ = modern.load_split(); current = next(row for row in folds[dataset] if int(row["fold_id"]) == fold)
    if task == "OpenBMI_MI":
        root = Path(os.environ["FULL_OPENBMI_CACHE"]).resolve()
        train, _, _, mapping = inner._openbmi_rows(root, current["inner_train_subjects"], tuple(modern.SOURCE_SESSIONS[dataset]), "mi")
        x, y, s, _ = inner._openbmi_rows(root, heldout[dataset], (modern.EVAL_SESSION,), "mi", mapping)
    else:
        root = Path(os.environ["FULL_WBCIC_CACHE"]).resolve()
        train, _, _, mapping = inner._wbcic_rows(root, current["inner_train_subjects"], tuple(modern.SOURCE_SESSIONS[dataset]))
        x, y, s, _ = inner._wbcic_rows(root, heldout[dataset], (modern.EVAL_SESSION,), mapping)
    _, [x], normalizer = data._normalise(train, x)
    return x, y, s, normalizer


def _task_holdout(task: str, fold: int, heldout: dict[str, list[str]]):
    # Use the seven-backbone row reader and normalizer rather than the older
    # task experiment's chunked normalizer.  Their mathematical formula is the
    # same, but this preserves the exact byte-hashed normalizer of selected.pt.
    data = _module("seven_litebn_benchmark_data", CODE / "benchmark_data.py")
    inner = data._load_module("seven_litebn_inner_loader", CODE / "tech_recipe_selection.py")
    os.environ["PERSIST_OPENBMI_CACHE"] = os.environ["FULL_OPENBMI_CACHE"]
    os.environ["TASK_GENERALITY_REPO"] = str(REPO)
    task_data = _module("seven_litebn_task_datasets", REPO / "experiments/persist_eeg_openbmi_task_generality_v1/code/task_datasets.py")
    name = "ERP" if task == "OpenBMI_ERP" else "SSVEP"
    search, _, reference, _ = task_data.split_reference(); current = next(row for row in reference["folds"] if int(row["fold_id"]) == fold)
    spec = task_data.TASKS[name]; root = Path(os.environ["FULL_OPENBMI_CACHE"]).resolve()
    train, _, _, mapping = inner._openbmi_rows(root, current["inner_train_subjects"], (spec["source_session"],), spec["cache_name"])
    x, y, s, _ = inner._openbmi_rows(root, heldout["OpenBMI"], (spec["future_session"],), spec["cache_name"], mapping)
    _, [x], normalizer = data._normalise(train, x)
    return x, y, s, normalizer


def heldout() -> None:
    lock_path, sidecar = PROTOCOL / "OUTER_TO_HELDOUT_LOCK.json", PROTOCOL / "OUTER_TO_HELDOUT_LOCK.sha256"
    if not lock_path.is_file() or not sidecar.is_file() or _sha(lock_path) != sidecar.read_text(encoding="utf-8").strip():
        raise RuntimeError("heldout evaluation requires the precommitted lock and SHA sidecar")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["pre_lock_outcomes"]["heldout_labels_read"] or lock["requirements"]["records"] != 20:
        raise RuntimeError("invalid heldout lock")
    records = _records(); expected = {(r["task"], r["fold"]): r for r in lock["outer_records"]}
    if {(r["task"],r["fold"]) for r in records} != set(expected): raise RuntimeError("checkpoint grid changed after lock")
    for row in records:
        if expected[(row["task"],row["fold"])]["checkpoint_sha256"] != row["checkpoint_sha256"]: raise RuntimeError("checkpoint changed after lock")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject_rows: list[dict[str, Any]] = []
    for row in records:
        task, fold = row["task"], int(row["fold"])
        x, y, subjects, normalizer = _mi_holdout(task, fold, lock["heldout_subjects"]) if task in ("OpenBMI_MI", "WBCIC_MI") else _task_holdout(task, fold, lock["heldout_subjects"])
        if normalizer["mean_std_sha256"] != row["normalizer_sha256"]:
            raise RuntimeError(f"normalizer mismatch with selected cell: {task}/fold{fold}")
        model = _load_model(task, Path(row["checkpoint_path"]), device)
        _, details = _metrics(y, _infer(model, x, device), subjects)
        for metric in details:
            subject_rows.append({"task": task, "dataset": DATASET[task], "fold": fold, "seed": 0, **metric,
                                 "checkpoint_sha256": row["checkpoint_sha256"]})
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    _csv(OUT / "HELDOUT_SUBJECT_RESULTS.csv", subject_rows)
    repl = []
    for task in TASKS:
        for fold in range(5):
            group = [r for r in subject_rows if r["task"] == task and r["fold"] == fold]
            repl.append({"task": task, "dataset": DATASET[task], "fold": fold, "seed": 0, "subjects": len(group),
                         "subject_equal_BA": float(np.mean([r["BA"] for r in group])),
                         "subject_equal_macro_F1": float(np.mean([r["macro_F1"] for r in group])),
                         "subject_equal_accuracy": float(np.mean([r["accuracy"] for r in group]))})
    _csv(OUT / "HELDOUT_REPLICATE_RESULTS.csv", repl)
    table = []
    for task in TASKS:
        group = [r for r in repl if r["task"] == task]
        table.append({"task": task, "dataset": DATASET[task], "seed": 0, "fold_replicates": len(group),
                      "heldout_subject_equal_BA_mean": float(np.mean([r["subject_equal_BA"] for r in group])),
                      "heldout_subject_equal_BA_std": float(np.std([r["subject_equal_BA"] for r in group], ddof=1)),
                      "heldout_subject_equal_macro_F1_mean": float(np.mean([r["subject_equal_macro_F1"] for r in group])),
                      "heldout_subject_equal_accuracy_mean": float(np.mean([r["subject_equal_accuracy"] for r in group]))})
    _csv(OUT / "HELDOUT_TASK_RESULTS.csv", table)
    _json(OUT / "RUN_METADATA.json", {"pass": True, "seed": 0, "tasks": list(TASKS), "records": 20,
          "heldout_scope": "V8_INTERNAL_HOLDOUT", "final_true_outer_accessed": False,
          "heldout_labels_read_once_after_lock": True, "lock_sha256": _sha(lock_path), "device": str(device)})
    print("LITEBN_SEED0_HELDOUT_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=("outer-lock", "heldout"), required=True)
    args = parser.parse_args(); outer_lock() if args.stage == "outer-lock" else heldout()
