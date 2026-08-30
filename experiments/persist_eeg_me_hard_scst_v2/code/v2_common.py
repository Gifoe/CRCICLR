"""Shared, fail-closed utilities for ME-HardSCST V2.

The module reuses the frozen Stage-1 loaders/model adapters without changing
any V1 artifact.  Every public data-loading function checks that requested
resources are development-authorized before delegating to Stage-1.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


REPO = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC")
EXP = REPO / "experiments" / "persist_eeg_me_hard_scst_v2"
CODE = EXP / "code"
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
PROTOCOL = EXP / "protocol"
RUNTIME = EXP / "runtime"
V1 = REPO / "experiments" / "persist_eeg_scst_utility_stage1"
STAGE1_CODE = V1 / "code"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if str(STAGE1_CODE) not in sys.path:
    sys.path.insert(0, str(STAGE1_CODE))
S1 = _load("me_hard_scst_v2_stage1_common", STAGE1_CODE / "stage1_common.py")

FOLDS = tuple(range(5))
SEEDS = tuple(range(3))
DATASETS = ("OpenBMI", "WBCIC")
SOURCE_TRAIN_SESSION = {"OpenBMI": 1, "WBCIC": 0}
SOURCE_VALID_SESSION = {"OpenBMI": 2, "WBCIC": 1}
DISCOVERY_SESSION = 2
ALPHAS = (1.0 / 64.0, 2.0 / 64.0, 3.0 / 64.0)
K_TARGETS = 8
EMA_DECAY = 0.99
SUPPORT_QUANTILE = 0.95
KNN_K = 3
EPOCHS = 15
BATCH_SIZE = 192
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    S1.write_json(path, value)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    S1.write_csv(path, frame)


def read_json(path: Path) -> dict[str, Any]:
    return S1.read_json(path)


def ensure_dirs() -> None:
    for path in (CODE, RESULTS, FIGURES, PROTOCOL, RUNTIME):
        path.mkdir(parents=True, exist_ok=True)


def reject_reserved_path(path: str | Path) -> None:
    text = str(path).replace("\\", "/").lower()
    forbidden = ("outer", "sealed", "outer-10", "outer_10", "holdout")
    if any(token in text for token in forbidden):
        raise RuntimeError(f"RESERVED_RESOURCE_REJECTED:{path}")


def load_development_data(dataset: str):
    if dataset not in DATASETS:
        raise ValueError(dataset)
    raw, metadata, root = S1.load_data(dataset)
    reject_reserved_path(root)
    allowed = {1, 2} if dataset == "OpenBMI" else {0, 1, 2}
    observed = set(metadata.session_id.astype(int).unique().tolist())
    if not observed.issubset(allowed):
        raise RuntimeError(f"UNAUTHORIZED_SESSION:{dataset}:{sorted(observed - allowed)}")
    return raw, metadata, root


def roles(dataset: str, fold: int) -> dict[str, tuple[str, ...]]:
    if fold not in FOLDS:
        raise ValueError(fold)
    role = S1.roles(dataset, fold)
    sets = [set(role[key]) for key in ("model_fit", "validation", "outcome")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError(f"SUBJECT_ROLE_OVERLAP:{dataset}:{fold}")
    return role


def source_indices(dataset: str, fold: int) -> tuple[np.ndarray, np.ndarray]:
    _, metadata, _ = load_development_data(dataset)
    role = roles(dataset, fold)
    train = S1.row_indices(metadata, role["model_fit"], (SOURCE_TRAIN_SESSION[dataset],))
    valid = S1.row_indices(metadata, role["validation"], (SOURCE_VALID_SESSION[dataset],))
    if not len(train) or not len(valid) or np.intersect1d(train, valid).size:
        raise RuntimeError(f"INVALID_SOURCE_SPLIT:{dataset}:{fold}")
    return train, valid


def discovery_indices(fold: int, *, lock_verified: bool) -> np.ndarray:
    if not lock_verified:
        raise RuntimeError("PROTOCOL_LOCK_NOT_VERIFIED")
    _, metadata, _ = load_development_data("WBCIC")
    role = roles("WBCIC", fold)
    values = S1.row_indices(metadata, role["outcome"], (DISCOVERY_SESSION,))
    if not len(values):
        raise RuntimeError(f"EMPTY_DISCOVERY_SPLIT:{fold}")
    return values


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def verify_lock(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("PROTOCOL_LOCK_MISSING")
    lock = read_json(path)
    recorded = str(lock.get("code_tree_sha256", ""))
    files = [CODE / name for name in lock.get("code_files", [])]
    if not files or any(not path.is_file() for path in files):
        raise RuntimeError("PROTOCOL_CODE_FILE_MISSING")
    digest = hashlib.sha256()
    for file in sorted(files):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    if digest.hexdigest() != recorded:
        raise RuntimeError("PROTOCOL_LOCK_HASH_MISMATCH")
    return lock


def subject_sort(values: Iterable[object]) -> list[str]:
    return S1.subject_sort(values)


def build_model(model: str, channels: int, n_times: int = 1000):
    return S1.build_model(model, channels, n_times)


def model_features(model: str, net, x: torch.Tensor) -> torch.Tensor:
    return S1.model_features(model, net, x)


def feature_logits(model: str, net, h: torch.Tensor) -> torch.Tensor:
    return S1.feature_logits(model, net, h)


def normalize_raw(x: np.ndarray) -> np.ndarray:
    return S1.normalize_raw(x)


def anchor_checkpoint(model: str, dataset: str, fold: int, seed: int) -> Path:
    if model == "ATCNet-CleanRoom":
        selection = pd.read_csv(REPO / "experiments" / "persist_eeg_scst_competence_generality_v1" / "results" / "SPECIALIST_SELECTION.csv")
        recipe = str(selection[(selection.model == "ATCNet") & (selection.dataset == dataset)].iloc[0].config)
        return REPO / "experiments" / "persist_eeg_scst_competence_generality_v1" / "runtime" / "specialist_checkpoints" / "ATCNet" / dataset / recipe / f"fold-{fold}" / f"seed-{seed}.pt"
    tag = model.lower().replace("-", "_")
    selection = pd.read_csv(V1 / "results" / f"SOURCE_SELECTION_{tag}.csv")
    recipe = str(selection[(selection.model == model) & (selection.dataset == dataset)].iloc[0].recipe)
    return V1 / "runtime" / "source_grid" / model / dataset / recipe / f"fold-{fold}" / f"seed-{seed}.pt"


def load_anchor(model: str, dataset: str, fold: int, seed: int, device: torch.device):
    path = anchor_checkpoint(model, dataset, fold, seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw, _, _ = load_development_data(dataset)
    net = build_model(model, raw.shape[1], raw.shape[2])
    net.load_state_dict(payload["state_dict"])
    return net.to(device), path

