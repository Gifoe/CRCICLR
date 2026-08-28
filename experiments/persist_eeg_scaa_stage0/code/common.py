from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
PROTOCOL = EXP / "protocol"
RUNTIME = EXP / "runtime"

P3_ROOT = REPO / "experiments" / "persist_eeg_wbcic_independent_replication_v1"
P4A_ROOT = REPO / "experiments" / "persist_eeg_p4a_cross_setting_expansion_v1"

BACKBONES = ("EEGNet", "EEGConformer")
FOLDS = tuple(range(5))
SEEDS = (0, 1, 2)
LR_GRID = (1e-4, 3e-4, 1e-3)
S1_TRAIN_FRACTION = 0.70
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
MIN_EPOCHS = 10
PATIENCE = 8


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P3 = _load_module("scaa_p3_common", P3_ROOT / "code" / "common.py")
P4A = _load_module("scaa_p4a_common", P4A_ROOT / "code" / "common.py")


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def ensure_dirs() -> None:
    for path in (RESULTS, FIGURES, PROTOCOL, RUNTIME):
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def subject_sort(values) -> list[str]:
    return sorted(map(str, values), key=lambda x: (int(x) if str(x).isdigit() else 10**9, str(x)))


def load_data():
    return P3.load_data()


def roles(fold: int) -> dict[str, tuple[str, ...]]:
    p3 = P3.frozen_fold(fold)
    p4 = P4A.roles_for("S4", fold)
    normalized = {
        "model_fit": tuple(subject_sort(p3["model_fit"])),
        "validation": tuple(subject_sort(p3["validation_discovery"])),
        "outcome": tuple(subject_sort(p3["outcome"])),
    }
    if normalized != {k: tuple(subject_sort(p4[k])) for k in ("model_fit", "validation", "outcome")}:
        raise RuntimeError(f"P3/P4A fold mismatch at fold {fold}")
    return normalized


def target_fold_map() -> dict[str, int]:
    mapping: dict[str, int] = {}
    for fold in FOLDS:
        for subject in roles(fold)["outcome"]:
            if subject in mapping:
                raise RuntimeError(f"subject repeated as outcome: {subject}")
            mapping[subject] = fold
    if tuple(subject_sort(mapping)) != tuple(P3.frozen_subjects()):
        raise RuntimeError("outcome folds do not exhaust the 41-subject pool")
    return mapping


def anchor_paths(backbone: str, fold: int, seed: int) -> tuple[Path, Path, Path]:
    if backbone == "EEGNet":
        unit = P3.unit_dir("eegnet", fold, seed)
    elif backbone == "EEGConformer":
        unit = P4A.run_dir("S4", fold, seed)
    else:
        raise KeyError(backbone)
    return unit, unit / "checkpoints" / "erm__lambda-0.00.pt", unit / "normalizer.npz"


def load_anchor(backbone: str, fold: int, seed: int, device: torch.device):
    unit, checkpoint, normalizer = anchor_paths(backbone, fold, seed)
    unit_protocol = read_json(unit / "UNIT_PROTOCOL.json")
    initialization_seed = int(unit_protocol["initialization_seed"])
    if backbone == "EEGNet":
        model = P3.build_model("eegnet", initialization_seed)
    else:
        model = P4A.build_model("S4", initialization_seed)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().to(device)
    norm = np.load(normalizer, allow_pickle=False)
    mean = torch.as_tensor(norm["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(norm["std"], dtype=torch.float32, device=device)
    return model, mean, std, unit_protocol


def row_indices(metadata: pd.DataFrame, subjects, sessions) -> np.ndarray:
    mask = metadata.subject_id.astype(str).isin(set(map(str, subjects))).to_numpy(copy=True)
    mask &= metadata.session_id.astype(int).isin(set(map(int, sessions))).to_numpy()
    return np.flatnonzero(mask).astype(np.int64)


def extract(model, raw: torch.Tensor, metadata: pd.DataFrame, indices: np.ndarray, mean: torch.Tensor, std: torch.Tensor, batch_size: int = 512):
    model.eval()
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            idx_np = indices[start:start + batch_size]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=raw.device)
            x = (raw[idx].float() - mean[None, :, None]) / torch.clamp(std[None, :, None], min=1e-6)
            with torch.autocast(device_type=raw.device.type, dtype=torch.bfloat16, enabled=raw.device.type == "cuda"):
                h = model.forward_features(x)
                z = model.head(h)
            features.append(h.float().cpu().numpy())
            logits.append(z.float().cpu().numpy())
    selected = metadata.iloc[indices]
    return {
        "features": np.concatenate(features).astype(np.float32),
        "logits": np.concatenate(logits).astype(np.float32),
        "labels": selected.label.to_numpy(np.int64),
        "subjects": selected.subject_id.astype(str).to_numpy(),
        "sessions": selected.session_id.to_numpy(np.int64),
        "indices": indices.copy(),
    }


def chronological_class_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train: list[int] = []
    validation: list[int] = []
    for label in sorted(np.unique(labels)):
        positions = np.flatnonzero(labels == label)
        cut = int(np.floor(len(positions) * S1_TRAIN_FRACTION))
        cut = min(max(cut, 1), len(positions) - 1)
        train.extend(positions[:cut].tolist())
        validation.extend(positions[cut:].tolist())
    train_idx = np.asarray(sorted(train), dtype=np.int64)
    val_idx = np.asarray(sorted(validation), dtype=np.int64)
    if set(np.unique(labels[train_idx])) != {0, 1} or set(np.unique(labels[val_idx])) != {0, 1}:
        raise RuntimeError("S1 chronological within-class split lost a class")
    return train_idx, val_idx


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    prediction = logits.argmax(1)
    return {
        "BA": float(balanced_accuracy_score(labels, prediction)),
        "macro_F1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "NLL": float(log_loss(labels, softmax_np(logits), labels=[0, 1])),
    }


def softmax_np(logits: np.ndarray) -> np.ndarray:
    value = logits.astype(np.float64)
    value -= value.max(axis=1, keepdims=True)
    exp = np.exp(value)
    return exp / exp.sum(axis=1, keepdims=True)


def adapt_head(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    initial_weight: np.ndarray,
    initial_bias: np.ndarray,
    lr: float,
    adaptation_seed: int,
) -> dict[str, Any]:
    set_seed(adaptation_seed)
    head = torch.nn.Linear(features.shape[1], 2)
    with torch.no_grad():
        head.weight.copy_(torch.as_tensor(initial_weight, dtype=torch.float32))
        head.bias.copy_(torch.as_tensor(initial_bias, dtype=torch.float32))
    x_train = torch.as_tensor(features[train_idx], dtype=torch.float32)
    y_train = torch.as_tensor(labels[train_idx], dtype=torch.long)
    x_val = torch.as_tensor(features[val_idx], dtype=torch.float32)
    counts = np.bincount(labels[train_idx], minlength=2).astype(np.float64)
    class_weight = torch.as_tensor(counts.sum() / np.maximum(2.0 * counts, 1.0), dtype=torch.float32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(lr), weight_decay=WEIGHT_DECAY)
    best_state = copy.deepcopy(head.state_dict())
    best_key: tuple[float, float, int] | None = None
    best_epoch = 0
    stale = 0
    for epoch in range(MAX_EPOCHS):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(x_train), y_train, weight=class_weight)
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            val_logits = head(x_val).numpy()
        val_metric = metrics(labels[val_idx], val_logits)
        if epoch + 1 >= MIN_EPOCHS:
            key = (val_metric["BA"], -val_metric["NLL"], -(epoch + 1))
            if best_key is None or key > best_key:
                best_key = key
                best_state = copy.deepcopy(head.state_dict())
                best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
            if stale >= PATIENCE:
                break
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        all_logits = head(torch.as_tensor(features, dtype=torch.float32)).numpy()
    weight = head.weight.detach().numpy().copy()
    bias = head.bias.detach().numpy().copy()
    numerator = np.sqrt(np.square(weight - initial_weight).sum() + np.square(bias - initial_bias).sum())
    denominator = max(np.sqrt(np.square(initial_weight).sum() + np.square(initial_bias).sum()), 1e-12)
    return {
        "logits": all_logits.astype(np.float32),
        "weight": weight.astype(np.float32),
        "bias": bias.astype(np.float32),
        "best_epoch": int(best_epoch),
        "parameter_relative_change": float(numerator / denominator),
    }
