"""Shared paths, deterministic helpers, and trusted Stage-1 adapters for V3.

The V3 experiment is deliberately source-first.  It may reuse detached feature
caches produced by the committed V2 source run, but it never reads V2 S3
outputs as a training input and never opens an outer/sealed resource.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch


REPO = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC")
EXP = REPO / "experiments" / "persist_eeg_bures_scst_v3"
CODE = EXP / "code"
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
PROTOCOL = EXP / "protocol"
RUNTIME = EXP / "runtime"
V2_EXP = REPO / "experiments" / "persist_eeg_me_hard_scst_v2"
V2_CACHE = V2_EXP / "runtime" / "source_cache"
V1 = REPO / "experiments" / "persist_eeg_scst_utility_stage1"

DATASETS = ("OpenBMI", "WBCIC")
FOLDS = tuple(range(5))
SEEDS = tuple(range(3))
SOURCE_SESSIONS = {"OpenBMI": (1, 2), "WBCIC": (0, 1)}
ALPHAS = (0.25, 0.50, 0.75, 1.00)
K_TARGETS = 8
KNN_K = 5
SUPPORT_QUANTILE = 0.95
EMA_DECAY = 0.99
WARMUP_EPOCHS = 3
STAGE2_EPOCHS = 15
HEAD_LR = 1e-4
ADAPTER_LR = 1e-5
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 256
RECIPES = tuple((q, lam) for q in (0.25, 0.50) for lam in (0.25, 0.50, 1.00))


def _load_stage1():
    path = V1 / "code" / "stage1_common.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("bures_v3_stage1_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def ensure_dirs() -> None:
    for path in (CODE, RESULTS, FIGURES, PROTOCOL, RUNTIME):
        path.mkdir(parents=True, exist_ok=True)


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def subject_sort(values: Iterable[object]) -> list[str]:
    return sorted((str(v).replace("sub-", "") for v in values), key=lambda x: (int(x) if x.isdigit() else 10**9, x))


def reject_reserved_path(path: str | Path) -> None:
    text = str(path).replace("\\", "/").lower()
    if any(token in text for token in ("outer", "sealed", "outer-10", "outer_10", "holdout")):
        raise RuntimeError(f"RESERVED_RESOURCE_REJECTED:{path}")


def load_development_data(dataset: str):
    if dataset not in DATASETS:
        raise ValueError(dataset)
    s1 = _load_stage1()
    raw, metadata, root = s1.load_data(dataset)
    reject_reserved_path(root)
    observed = set(metadata.session_id.astype(int).unique().tolist())
    allowed = {1, 2} if dataset == "OpenBMI" else {0, 1, 2}
    if not observed.issubset(allowed):
        raise RuntimeError(f"UNAUTHORIZED_SESSION:{dataset}:{sorted(observed - allowed)}")
    return raw, metadata, root


def roles(dataset: str, fold: int) -> dict[str, tuple[str, ...]]:
    if dataset not in DATASETS or fold not in FOLDS:
        raise ValueError((dataset, fold))
    role = _load_stage1().roles(dataset, fold)
    return {key: tuple(subject_sort(value)) for key, value in role.items()}


def source_indices(dataset: str, fold: int) -> tuple[np.ndarray, np.ndarray]:
    _, metadata, _ = load_development_data(dataset)
    role = roles(dataset, fold)
    s1 = _load_stage1()
    train = s1.row_indices(metadata, role["model_fit"], (SOURCE_SESSIONS[dataset][0],))
    valid = s1.row_indices(metadata, role["validation"], (SOURCE_SESSIONS[dataset][1],))
    if not len(train) or not len(valid) or np.intersect1d(train, valid).size:
        raise RuntimeError(f"INVALID_SOURCE_SPLIT:{dataset}:{fold}")
    return train, valid


def cache_path(dataset: str, fold: int, seed: int, role: str) -> Path:
    return V2_CACHE / dataset / f"fold-{fold}" / f"seed-{seed}" / f"{role}.npz"


def load_feature_cache(dataset: str, fold: int, seed: int, role: str):
    """Load detached V2 source features; validation is never used for a bank."""
    path = cache_path(dataset, fold, seed, role)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as values:
        required = {"indices", "labels", "subjects", "final", "preblock"}
        if not required.issubset(values.files):
            raise RuntimeError(f"CACHE_SCHEMA:{path}")
        return {
            "indices": values["indices"].astype(np.int64),
            "labels": values["labels"].astype(np.int64),
            "subjects": values["subjects"].astype(str),
            "features": values["final"].astype(np.float32),
            "preblock": values["preblock"].astype(np.float32),
        }


def normalize_raw(x: np.ndarray) -> np.ndarray:
    value = np.asarray(x, np.float32)
    value = value - value.mean(axis=-1, keepdims=True)
    return value / np.maximum(value.std(axis=-1, keepdims=True), 1e-6)


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def sha256_path(path: Path) -> str:
    """Hash a file in bounded chunks for protocol locks and audits."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_tree_sha256(files: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(p) for p in files), key=lambda p: p.name):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def bootstrap_ci(values: Iterable[float], *, seed: int, draws: int = 10000) -> tuple[float, float, float]:
    """Deterministic percentile bootstrap for subject-level paired values."""
    values = np.asarray([float(v) for v in values if np.isfinite(v)], np.float64)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    sample = values[rng.integers(0, len(values), size=(int(draws), len(values)))]
    means = sample.mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_subject_delta(frame: pd.DataFrame, method: str, control: str, *, dataset: str, q: float | None = None, lambda_T: float | None = None, value: str = "BA") -> tuple[float, float, float, int]:
    """Aggregate by biological subject, then bootstrap paired deltas.

    Fold and seed replicates are averaged within subject before resampling;
    trials therefore never become bootstrap units.
    """
    subset = frame[frame["dataset"].astype(str) == str(dataset)].copy()
    left_subset = subset[subset["method"].astype(str) == str(method)]
    right_subset = subset[subset["method"].astype(str) == str(control)]
    if q is not None:
        left_subset = left_subset[(left_subset["q"].astype(float) - float(q)).abs() < 1e-8]
        # ERM is recorded once at q=lambda=.50 and is the matched baseline for
        # every searched recipe; other fixed controls have their declared
        # q/lambda and are likewise not filtered to the candidate recipe.
        if str(control) != "ERM":
            right_subset = right_subset[(right_subset["q"].astype(float) - 0.50).abs() < 1e-8]
    if lambda_T is not None:
        left_subset = left_subset[(left_subset["lambda_T"].astype(float) - float(lambda_T)).abs() < 1e-8]
        if str(control) != "ERM":
            right_subset = right_subset[(right_subset["lambda_T"].astype(float) - 0.50).abs() < 1e-8]
    left = left_subset.groupby("subject_id", as_index=True)[value].mean()
    right = right_subset.groupby("subject_id", as_index=True)[value].mean()
    delta = (left - right).dropna().to_numpy(np.float64)
    return (*bootstrap_ci(delta, seed=stable_seed("subject-bootstrap", dataset, method, control, q, lambda_T), draws=10000), int(len(delta)))
