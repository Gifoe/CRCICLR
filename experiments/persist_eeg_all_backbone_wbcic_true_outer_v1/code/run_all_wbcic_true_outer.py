"""Re-evaluate every completed WBCIC backbone on the fixed 10-person true outer cohort.

This script performs inference only.  It locks exact checkpoint hashes before
loading any true-outer arrays, uses the frozen development-fold normalizer, and
stores only compact metrics (never logits or labels).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


EXP = Path(__file__).resolve().parents[1]
OUT = EXP / "outputs"
PROTOCOL = EXP / "protocol"
SEVEN_REPO = Path(os.environ["SEVEN_REPO"]).resolve()
SEVEN_CODE = SEVEN_REPO / "experiments" / "persist_eeg_seven_backbone_fourtask_3seed_v1" / "code"
if str(SEVEN_CODE) not in sys.path:
    sys.path.insert(0, str(SEVEN_CODE))
SEVEN_RUNTIME = Path(os.environ["SEVEN_RUNTIME"]).resolve()
DEV_CACHE = Path(os.environ["FULL_WBCIC_CACHE"]).resolve()
TRUE_CACHE = Path(os.environ["TRUE_OUTER_WBCIC_CACHE"]).resolve()
TRUE_SUBJECTS = ("sub-4", "sub-8", "sub-10", "sub-15", "sub-20", "sub-39", "sub-40", "sub-43", "sub-46", "sub-51")
RECENT = {
    "ModernTCN": (Path(os.environ["MODERNTCN_CODE"]).resolve(), Path(os.environ["MODERNTCN_RUNTIME"]).resolve()),
    "Medformer": (Path(os.environ["MEDFORMER_CODE"]).resolve(), Path(os.environ["MEDFORMER_RUNTIME"]).resolve()),
    "SGN": (Path(os.environ["SGN_CODE"]).resolve(), Path(os.environ["SGN_RUNTIME"]).resolve()),
}
OLD_FULL = ("EEGNet", "TCFormer", "CBraMod", "TeCh")
OLD_SEED0 = ("LiteBN",)
MODELS = OLD_FULL + OLD_SEED0 + tuple(RECENT)
TASK = "WBCIC_MI"
_DATA: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}
_MODULES: dict[str, Any] = {}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, dict): return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [clean(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: raise RuntimeError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: clean(row.get(key, "")) for key in fields} for row in rows])
    os.replace(temporary, path)


def module(name: str, path: Path):
    key = f"{name}:{path}"
    if key in _MODULES: return _MODULES[key]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    _MODULES[key] = result
    return result


def cell(model: str, fold: int, seed: int) -> tuple[dict[str, Any], Path]:
    if model in OLD_FULL + OLD_SEED0:
        root = SEVEN_RUNTIME / "search_cells" / "wbcic_mi" / model.lower().replace("-", "_") / f"fold{fold}_seed{seed}"
    else:
        root = RECENT[model][1] / "cells" / "wbcic_mi" / f"fold{fold}_seed{seed}"
    record_path, checkpoint = root / "record.json", root / "selected.pt"
    if not record_path.is_file() or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"missing completed checkpoint cell: {root}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    identity = (record.get("task"), record.get("model"), int(record.get("fold", -1)), int(record.get("seed", -1)))
    if identity != (TASK, model, fold, seed): raise RuntimeError(f"identity mismatch {root}: {identity}")
    actual = sha(checkpoint)
    if record.get("checkpoint_sha256") != actual: raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
    return record, checkpoint


def all_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        seeds = (0, 1, 2) if model in OLD_FULL else (0,)
        for fold in range(5):
            for seed in seeds:
                record, checkpoint = cell(model, fold, seed)
                rows.append({
                    "model": model, "task": TASK, "fold": fold, "seed": seed,
                    "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                    "normalizer_sha256": record["normalizer"]["mean_std_sha256"],
                    "selected_epoch": int(record["selected_epoch"]),
                    "batch_size": int(record.get("recipe", {}).get("batch_size", 128)),
                    "recipe_name": record.get("recipe", {}).get("name"),
                    "channels": int(record.get("channels") or 58),
                    "samples": int(record.get("samples") or 1000),
                    "classes": int(record.get("classes") or 2),
                })
    if len(rows) != 80: raise RuntimeError(f"expected 80 frozen checkpoint records, got {len(rows)}")
    return rows


def prelock() -> None:
    lock_path = PROTOCOL / "TRUE_OUTER_EVALUATION_LOCK.json"
    if lock_path.exists(): raise RuntimeError("refusing to overwrite true-outer lock")
    records = all_records()
    found = tuple(sorted((p.name for p in TRUE_CACHE.iterdir() if p.is_dir() and p.name.startswith("sub-")), key=lambda x: int(x[4:])))
    expected = tuple(sorted(TRUE_SUBJECTS, key=lambda x: int(x[4:])))
    if found != expected: raise RuntimeError(f"true outer cache membership mismatch: {found}")
    for subject in TRUE_SUBJECTS:
        for session in (0, 1, 2):
            for suffix in ("epochs.npy", "labels.npy", "metadata.json"):
                if not (TRUE_CACHE / subject / f"ses-{session}_{suffix}").is_file():
                    raise FileNotFoundError(f"missing true outer cache member: {subject}/ses-{session}_{suffix}")
    lock = {
        "protocol": "ALL_COMPLETED_WBCIC_BACKBONES_TRUE_OUTER_REEVALUATION_V1",
        "purpose": "inference-only rerun on fixed true outer session 2; no training, selection, tuning, calibration, or ensembling",
        "true_outer_subjects": list(TRUE_SUBJECTS), "evaluation_session": 2,
        "development_cache": str(DEV_CACHE), "true_outer_cache": str(TRUE_CACHE),
        "models": list(MODELS), "checkpoint_count": len(records), "checkpoints": records,
        "evidence_scope": {model: ("5 folds x 3 seeds" if model in OLD_FULL else "5 folds x seed0 only") for model in MODELS},
        "prelock_access": {"true_outer_arrays_loaded": False, "true_outer_labels_loaded": False, "predictions_generated": False},
    }
    write_json(lock_path, lock)
    (PROTOCOL / "TRUE_OUTER_EVALUATION_LOCK.sha256").write_text(sha(lock_path) + "\n", encoding="utf-8")
    print("TRUE_OUTER_INPUTS_LOCKED_WITHOUT_ARRAY_OR_LABEL_ACCESS", flush=True)


def outer_data(fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if fold in _DATA: return _DATA[fold]
    benchmark = module("all_true_outer_benchmark_data", SEVEN_CODE / "benchmark_data.py")
    inner = benchmark._load_module("all_true_outer_inner_loader", SEVEN_CODE / "tech_recipe_selection.py")
    modern, _ = benchmark._sources()
    folds, _, _ = modern.load_split()
    current = next(row for row in folds["WBCIC"] if int(row["fold_id"]) == fold)
    train, _, _, mapping = inner._wbcic_rows(DEV_CACHE, current["inner_train_subjects"], tuple(modern.SOURCE_SESSIONS["WBCIC"]))
    heldout, labels, subjects, _ = inner._wbcic_rows(TRUE_CACHE, list(TRUE_SUBJECTS), (modern.EVAL_SESSION,), mapping)
    _, [heldout], normalizer = benchmark._normalise(train, heldout)
    _DATA[fold] = np.ascontiguousarray(heldout), labels, subjects, normalizer
    return _DATA[fold]


def build(row: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_name = row["model"]
    if model_name in OLD_FULL + OLD_SEED0:
        models = module("all_true_outer_old_models", SEVEN_CODE / "backbone_models.py")
        model = models.build_model(
            model_name, dataset="WBCIC", channels=58, samples=1000, classes=2,
            tech_recipe=row["recipe_name"] if model_name == "TeCh" else None,
        )
    else:
        code_dir = RECENT[model_name][0]
        if str(code_dir) not in sys.path: sys.path.insert(0, str(code_dir))
        # The upstream ModernTCN and Medformer vendors both expose generic
        # top-level ``models``/``layers`` packages.  Evict only those vendor
        # namespaces before switching adapters so Python cannot reuse the
        # preceding backbone's package path.  Already constructed classes keep
        # direct references to their imported objects.
        for namespace in ("models", "layers", "utils"):
            for imported in [key for key in sys.modules if key == namespace or key.startswith(namespace + ".")]:
                del sys.modules[imported]
        adapter = module(f"all_true_outer_{model_name.lower()}_adapter", code_dir / "model_adapter.py")
        model = adapter.build_model(channels=row["channels"], samples=row["samples"], classes=row["classes"])
    payload = torch.load(row["checkpoint_path"], map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    model.eval().to(device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model


def metrics(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> tuple[dict[str, float], list[dict[str, Any]]]:
    prediction = logits.argmax(1)
    details = []
    for subject in sorted(np.unique(subjects.astype(str)), key=lambda value: int(value.replace("sub-", ""))):
        mask = subjects.astype(str) == subject
        details.append({
            "subject_id": subject, "BA": float(balanced_accuracy_score(labels[mask], prediction[mask])),
            "macro_F1": float(f1_score(labels[mask], prediction[mask], average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(labels[mask], prediction[mask])), "trials": int(mask.sum()),
        })
    return {
        "subject_equal_BA": float(np.mean([row["BA"] for row in details])),
        "subject_equal_macro_F1": float(np.mean([row["macro_F1"] for row in details])),
        "trial_accuracy": float(accuracy_score(labels, prediction)),
    }, details


def evaluate_cell(row: dict[str, Any], device: torch.device) -> dict[str, Any]:
    path = OUT / "true_outer_cells" / f"{row['model'].lower()}_f{row['fold']}_s{row['seed']}.json"
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached["checkpoint_sha256"] != row["checkpoint_sha256"]: raise RuntimeError(f"resume hash mismatch: {path}")
        return cached
    x, labels, subjects, normalizer = outer_data(row["fold"])
    if normalizer["mean_std_sha256"] != row["normalizer_sha256"]: raise RuntimeError(f"normalizer mismatch: {row['model']} fold{row['fold']}")
    if row["model"] in OLD_FULL + OLD_SEED0:
        runner = module("all_true_outer_old_runner", SEVEN_CODE / "run_search.py")
        x = runner._resample(row["model"], x)
    model = build(row, device)
    tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(labels), row["batch_size"]):
            outputs.append(model(tensor[start:start + row["batch_size"]]).float().cpu().numpy())
    summary, details = metrics(labels, np.concatenate(outputs), subjects)
    result = {"identity": {key: row[key] for key in ("model", "task", "fold", "seed")}, "checkpoint_sha256": row["checkpoint_sha256"], "metrics": summary, "subject_rows": details}
    write_json(path, result)
    del tensor, model
    if device.type == "cuda": torch.cuda.empty_cache()
    print(f"TRUE_OUTER_CELL_COMPLETED {row['model']} fold={row['fold']} seed={row['seed']}", flush=True)
    return result


def evaluate() -> None:
    lock_path, sidecar = PROTOCOL / "TRUE_OUTER_EVALUATION_LOCK.json", PROTOCOL / "TRUE_OUTER_EVALUATION_LOCK.sha256"
    if not lock_path.is_file() or not sidecar.is_file() or sha(lock_path) != sidecar.read_text(encoding="utf-8").strip():
        raise RuntimeError("intact pre-evaluation lock required")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    records = all_records()
    expected = {(row["model"], row["fold"], row["seed"]): row["checkpoint_sha256"] for row in lock["checkpoints"]}
    if len(expected) != 80 or any(expected[(row["model"], row["fold"], row["seed"])] != row["checkpoint_sha256"] for row in records):
        raise RuntimeError("checkpoint family changed after lock")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replicate_rows, subject_rows = [], []
    for row in records:
        result = evaluate_cell(row, device)
        replicate_rows.append({**result["identity"], **result["metrics"], "subjects": len(result["subject_rows"]), "checkpoint_sha256": result["checkpoint_sha256"]})
        subject_rows.extend([{**result["identity"], **detail, "checkpoint_sha256": result["checkpoint_sha256"]} for detail in result["subject_rows"]])
    model_rows = []
    for model in MODELS:
        rows = [row for row in replicate_rows if row["model"] == model]
        seed_values = []
        for seed in sorted({row["seed"] for row in rows}):
            seed_group = [row for row in rows if row["seed"] == seed]
            seed_values.append(float(np.mean([row["subject_equal_BA"] for row in seed_group])))
        model_rows.append({
            "model": model, "evidence_scope": lock["evidence_scope"][model],
            "seeds": len(seed_values), "folds_per_seed": 5, "checkpoint_replicates": len(rows),
            "true_outer_subject_equal_BA_mean": float(np.mean([row["subject_equal_BA"] for row in rows])),
            "true_outer_checkpoint_BA_std": float(np.std([row["subject_equal_BA"] for row in rows], ddof=1)),
            "true_outer_seed_BA_std": float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0,
            "true_outer_subject_equal_macro_F1_mean": float(np.mean([row["subject_equal_macro_F1"] for row in rows])),
            "true_outer_trial_accuracy_mean": float(np.mean([row["trial_accuracy"] for row in rows])),
        })
    write_csv(OUT / "TRUE_OUTER_REPLICATE_RESULTS.csv", replicate_rows)
    write_csv(OUT / "TRUE_OUTER_SUBJECT_RESULTS.csv", subject_rows)
    write_csv(OUT / "TRUE_OUTER_MODEL_RESULTS.csv", model_rows)
    lines = ["# WBCIC True-Outer Backbone Re-evaluation", "", "Cohort: `sub-4, sub-8, sub-10, sub-15, sub-20, sub-39, sub-40, sub-43, sub-46, sub-51`; session 2 only.", "", "| Model | Evidence | BA | Macro-F1 |", "|---|---:|---:|---:|"]
    for row in model_rows:
        lines.append(f"| {row['model']} | {row['evidence_scope']} | {100*row['true_outer_subject_equal_BA_mean']:.2f}% | {100*row['true_outer_subject_equal_macro_F1_mean']:.2f}% |")
    lines.extend(["", "Means over the five frozen fold checkpoints (and over three seeds where available). No retraining, tuning, calibration, or ensembling was performed on true outer.", ""])
    (EXP / "TRUE_OUTER_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(OUT / "RUN_METADATA.json", {"pass": True, "device": str(device), "models": list(MODELS), "checkpoint_records": len(records), "true_outer_subjects": list(TRUE_SUBJECTS), "evaluation_session": 2, "lock_sha256": sha(lock_path), "training_performed": False, "selection_performed": False})
    print("ALL_WBCIC_TRUE_OUTER_REEVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prelock", "evaluate"), required=True)
    args = parser.parse_args()
    prelock() if args.stage == "prelock" else evaluate()
