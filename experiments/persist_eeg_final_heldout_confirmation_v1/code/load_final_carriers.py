"""Read-only assets for the final FROZEN_LOGIT50 confirmation.

This module deliberately separates signal metadata/preflight from label access.
``preflight`` never opens an held-out label array; labels are available only
through :func:`load_eval_subject` while the single fixed evaluation is running.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(os.environ.get("FINAL_CONFIRM_REPO", Path(__file__).resolve().parents[3])).resolve()
EXP = REPO / "experiments" / "persist_eeg_final_heldout_confirmation_v1"
PROTOCOL = EXP / "protocol"
OUTPUTS = EXP / "outputs"
CARRIER_RUNTIME = Path(os.environ.get("CARRIER_RUNTIME", "/root/rivermind-data/carrier_5fold_multiseed_stability_runtime")).resolve()
OPENBMI_ROOT = Path(os.environ.get("PERSIST_OPENBMI_CACHE", "/root/rivermind-data/persist_eeg_cache/openbmi/openbmi")).resolve()
WBCIC_ROOT = Path(os.environ.get("PERSIST_WBCIC_CACHE", "/root/rivermind-data/persist_eeg_cache/wbcic/wbcic_epochs")).resolve()
V8_SPLIT = REPO / "experiments" / "persist_eeg_final_model_v8" / "outputs" / "protocol" / "V8_SEARCH_SPLIT.json"
FIVEFOLD = REPO / "experiments" / "persist_eeg_carrier_5fold_multiseed_stability_v1" / "protocol" / "FIVEFOLD_SPLIT.json"
MANIFEST_HASHES = REPO / "experiments" / "persist_eeg_carrier_5fold_multiseed_stability_v1" / "protocol" / "MANIFEST_HASHES.json"
FUSION_PROVENANCE = REPO / "experiments" / "persist_eeg_frozen_fusion_headroom_v1" / "protocol" / "CHECKPOINT_PROVENANCE.json"
CARRIER_CODE = REPO / "experiments" / "persist_eeg_carrier_dualdataset_screen_v1" / "code"
DATASETS = ("OpenBMI", "WBCIC")
MODEL_NAMES = ("EEGNet", "LiteBN")
FUTURE_SESSION = {"OpenBMI": 2, "WBCIC": 2}
SOURCE_SESSIONS = {"OpenBMI": (1,), "WBCIC": (0, 1)}
CHANNELS = {"OpenBMI": 62, "WBCIC": 58}


def clean(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_subjects(dataset: str, values: list[str]) -> list[str]:
    return sorted((str(v) for v in values), key=lambda x: int(x.replace("sub-", "")))


def holdout_memberships() -> tuple[dict[str, list[str]], dict[str, Any]]:
    raw = json.loads(V8_SPLIT.read_text(encoding="utf-8"))
    names = {"OpenBMI": "openbmi", "WBCIC": "wbcic"}
    result = {dataset: canonical_subjects(dataset, raw[names[dataset]]["V8_INTERNAL_HOLDOUT"]) for dataset in DATASETS}
    expected = {"OpenBMI": 14, "WBCIC": 10}
    if {d: len(result[d]) for d in DATASETS} != expected:
        raise RuntimeError(f"final held-out membership count mismatch: {result}")
    return result, raw


def signal_path(dataset: str, subject: str, session: int) -> Path:
    if dataset == "OpenBMI":
        return OPENBMI_ROOT / f"sub-{int(subject):02d}" / f"ses-{session}" / "mi_1train_signals.npy"
    return WBCIC_ROOT / subject / f"ses-{session}_epochs.npy"


def label_path(dataset: str, subject: str, session: int) -> Path:
    if dataset == "OpenBMI":
        return OPENBMI_ROOT / f"sub-{int(subject):02d}" / f"ses-{session}" / "mi_1train_codes.npy"
    return WBCIC_ROOT / subject / f"ses-{session}_labels.npy"


def signal_schema(dataset: str, subject: str, session: int) -> dict[str, Any]:
    """Open a signal mmap only. This function never touches label contents."""
    path, target = signal_path(dataset, subject, session), label_path(dataset, subject, session)
    if not path.is_file() or not target.is_file():
        raise FileNotFoundError(f"cache pair absent for {dataset}/{subject}/session={session}")
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    wanted_dtype = np.float32 if dataset == "OpenBMI" else np.float16
    if x.ndim != 3 or x.shape[1:] != (CHANNELS[dataset], 1000) or x.dtype != wanted_dtype:
        raise RuntimeError(f"signal schema mismatch: {path}: {x.shape}/{x.dtype}")
    if dataset == "OpenBMI" and x.shape[0] != 100:
        raise RuntimeError(f"OpenBMI trial count mismatch: {path}: {x.shape[0]}")
    return {"signal_path": str(path), "label_path": str(target), "shape": list(x.shape), "dtype": str(x.dtype), "trials": int(x.shape[0])}


def _source_signal(dataset: str, subject: str, session: int) -> np.ndarray:
    path = signal_path(dataset, subject, session)
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    wanted_dtype = np.float32 if dataset == "OpenBMI" else np.float16
    if x.ndim != 3 or x.shape[1:] != (CHANNELS[dataset], 1000) or x.dtype != wanted_dtype:
        raise RuntimeError(f"source-only normalizer input schema mismatch: {path}")
    return x


def source_normalizer(dataset: str, subjects: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Deterministically reproduce a carrier normalizer from source subjects only."""
    total = np.zeros(CHANNELS[dataset], dtype=np.float64)
    square = np.zeros(CHANNELS[dataset], dtype=np.float64)
    samples = trials = 0
    for subject in canonical_subjects(dataset, subjects):
        for session in SOURCE_SESSIONS[dataset]:
            x = _source_signal(dataset, subject, session).astype(np.float64)
            total += x.sum(axis=(0, 2)); square += np.square(x).sum(axis=(0, 2))
            samples += x.shape[0] * x.shape[2]; trials += x.shape[0]
    mean = (total / samples).astype(np.float32)
    std = np.sqrt(np.maximum(square / samples - mean.astype(np.float64) ** 2, 1e-12)).astype(np.float32)
    digest = hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest()
    return mean, std, {"subjects": canonical_subjects(dataset, subjects), "sessions": list(SOURCE_SESSIONS[dataset]), "trials": int(trials), "samples_per_channel": int(samples), "mean_std_sha256": digest}


def carrier_folds() -> dict[str, list[dict[str, Any]]]:
    raw = json.loads(FIVEFOLD.read_text(encoding="utf-8"))
    if raw.get("protocol") != "CARRIER_5FOLD_MULTISEED_STABILITY_V1": raise RuntimeError("unexpected carrier fold protocol")
    folds = raw["folds"]
    if any(len(folds[d]) != 5 for d in DATASETS): raise RuntimeError("five carrier folds required")
    return folds


def normalizer_provenance() -> tuple[dict[str, list[dict[str, Any]]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    expected_raw = json.loads(MANIFEST_HASHES.read_text(encoding="utf-8"))
    expected = {(r["dataset"], int(r["fold"])): r["normalizer"] for r in expected_raw}
    folds = carrier_folds(); report: dict[str, list[dict[str, Any]]] = {d: [] for d in DATASETS}; values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for dataset in DATASETS:
        for fold in folds[dataset]:
            fid = int(fold["fold_id"]); mean, std, actual = source_normalizer(dataset, fold["inner_train_subjects"])
            wanted = expected.get((dataset, fid))
            if wanted is None or actual != wanted: raise RuntimeError(f"source normalizer provenance mismatch: {dataset} fold={fid}")
            report[dataset].append({"fold": fid, **actual, "expected_from": str(MANIFEST_HASHES), "verified": True})
            values[f"{dataset}:{fid}"] = (mean, std)
    return report, values


def checkpoint_records() -> list[dict[str, Any]]:
    raw = json.loads(FUSION_PROVENANCE.read_text(encoding="utf-8"))
    expected = {(r["dataset"], int(r["fold"]), int(r["seed"]), r["model"]): r for r in raw["verified_rows"] if r.get("variant") == "selected_best"}
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for fold in range(5):
            for seed in range(3):
                for model in MODEL_NAMES:
                    key = (dataset, fold, seed, model); provenance = expected.get(key)
                    if provenance is None: raise RuntimeError(f"missing frozen fusion provenance: {key}")
                    suffix = "eegnet" if model == "EEGNet" else "litebn"
                    path = CARRIER_RUNTIME / f"{dataset.lower()}_fold{fold}_seed{seed}_{suffix}" / "selected_best.pt"
                    records.append({"dataset": dataset, "fold": fold, "seed": seed, "model": model, "checkpoint_path": str(path), "expected_sha256": provenance["expected_sha256"], "provenance_source": str(FUSION_PROVENANCE), "provenance_variant": "selected_best"})
    if len(records) != 60: raise RuntimeError("exactly 60 selected checkpoints are required")
    return records


def audit_checkpoints(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audited = []
    for row in records:
        path = Path(row["checkpoint_path"]); actual = sha256(path) if path.is_file() else None
        audited.append({**row, "exists": path.is_file(), "actual_sha256": actual, "sha_match": actual == row["expected_sha256"]})
    if not all(row["exists"] and row["sha_match"] for row in audited): raise RuntimeError("frozen selected checkpoint hash audit failed")
    return audited


def model_classes() -> tuple[type[torch.nn.Module], type[torch.nn.Module]]:
    if str(CARRIER_CODE) not in sys.path: sys.path.insert(0, str(CARRIER_CODE))
    eegnet = importlib.import_module("eegnet_locked").EEGNet
    compact = importlib.import_module("run_carrier_screen").CompactLite
    return eegnet, compact


def load_frozen_model(model_name: str, channels: int, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    EEGNet, CompactLite = model_classes()
    model = EEGNet(channels) if model_name == "EEGNet" else CompactLite(channels, "bn")
    model = model.to(device)
    incompatible = model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys: raise RuntimeError(f"strict checkpoint load failure: {checkpoint}")
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    if model.training or any(p.requires_grad for p in model.parameters()): raise RuntimeError(f"eval/frozen invariant failure: {checkpoint}")
    return model


def load_eval_subject(dataset: str, subject: str) -> tuple[np.ndarray, np.ndarray]:
    """The only label-reading entry point; called only by the fixed final run."""
    session = FUTURE_SESSION[dataset]
    x = np.load(signal_path(dataset, subject, session), mmap_mode="r", allow_pickle=False).astype(np.float32)
    y = np.load(label_path(dataset, subject, session), mmap_mode="r", allow_pickle=False)
    if y.ndim != 1 or y.shape[0] != x.shape[0]: raise RuntimeError(f"held-out label shape mismatch: {dataset}/{subject}")
    y = np.asarray(y, dtype=np.int64)
    if dataset == "OpenBMI":
        if set(np.unique(y).tolist()) != {1, 2}: raise RuntimeError(f"OpenBMI held-out codes are not 1/2: {subject}")
        y = y - 1
    elif set(np.unique(y).tolist()) != {0, 1}:
        raise RuntimeError(f"WBCIC held-out codes are not 0/1: {subject}")
    return x, y


def normalize_to_device(x: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    z = (x - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)
    return torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32)).to(device, non_blocking=True)
