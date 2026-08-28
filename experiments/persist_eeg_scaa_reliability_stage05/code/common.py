from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
PROTOCOL = EXP / "protocol"
RUNTIME = EXP / "runtime"

STAGE0_ROOT = REPO / "experiments" / "persist_eeg_scaa_stage0"
STAGE0_CODE = STAGE0_ROOT / "code"
STAGE0_RESULTS = STAGE0_ROOT / "results"
STAGE0_PROTOCOL = STAGE0_ROOT / "protocol"

BACKBONES = ("EEGNet", "EEGConformer")
FOLDS = tuple(range(5))
SEEDS = (0, 1, 2)
FEATURE_BOOTSTRAPS = 2000
SUBJECT_BOOTSTRAPS = 10000
LCB_Z_90 = 1.2815515655446004

FEATURE_COLUMNS = (
    "adaptation_effect_stability",
    "decision_stability",
    "certificate_snr",
    "certificate_lcb90",
    "representation_stability",
    "raw_delta2",
    "s1_parameter_relative_change",
    "s1_anchor_confidence",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STAGE0 = _load_module("scaa_stage05_stage0_common", STAGE0_CODE / "common.py")


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(item) for item in value]
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
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    if path.suffix.lower() in {".py", ".md", ".json", ".csv", ".txt"}:
        # Git checkouts on the Windows server may materialize CRLF while the
        # local worktree uses LF. Canonicalize text line endings so a protocol
        # lock identifies content rather than checkout policy.
        payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        return hashlib.sha256(payload).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def subject_sort(values: Sequence[Any]) -> list[str]:
    return sorted(map(str, values), key=lambda item: (int(item) if item.isdigit() else 10**9, item))


def target_fold_map() -> dict[str, int]:
    return STAGE0.target_fold_map()


def roles(fold: int) -> dict[str, tuple[str, ...]]:
    return STAGE0.roles(fold)


def centered_correct_margin(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[1] != 2 or len(logits) != len(labels):
        raise ValueError("binary logits and aligned labels are required")
    rows = np.arange(len(labels))
    return logits[rows, labels] - logits[rows, 1 - labels]


def _safe_scale(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_variance = float(np.var(left, ddof=1)) if len(left) > 1 else 0.0
    right_variance = float(np.var(right, ddof=1)) if len(right) > 1 else 0.0
    variance = 0.5 * (left_variance + right_variance)
    return max(math.sqrt(max(variance, 0.0)), 1e-8)


def class_conditioned_shift_stability(
    left_values: np.ndarray,
    left_labels: np.ndarray,
    right_values: np.ndarray,
    right_labels: np.ndarray,
) -> float:
    """Negative mean absolute standardized shift; zero is perfectly stable."""
    shifts: list[float] = []
    for label in (0, 1):
        left = np.asarray(left_values)[np.asarray(left_labels) == label]
        right = np.asarray(right_values)[np.asarray(right_labels) == label]
        if min(len(left), len(right)) < 2:
            raise RuntimeError(f"insufficient class-{label} observations")
        shifts.append(abs(float(np.mean(left) - np.mean(right))) / _safe_scale(left, right))
    return -float(np.mean(shifts))


def class_conditioned_representation_stability(
    left_features: np.ndarray,
    left_labels: np.ndarray,
    right_features: np.ndarray,
    right_labels: np.ndarray,
) -> float:
    """Negative normalized final-embedding centroid drift; zero is stable."""
    drifts: list[float] = []
    for label in (0, 1):
        left = np.asarray(left_features, dtype=np.float64)[np.asarray(left_labels) == label]
        right = np.asarray(right_features, dtype=np.float64)[np.asarray(right_labels) == label]
        if min(len(left), len(right)) < 2:
            raise RuntimeError(f"insufficient class-{label} representations")
        left_center = np.mean(left, axis=0)
        right_center = np.mean(right, axis=0)
        numerator = float(np.linalg.norm(left_center - right_center))
        left_dispersion = float(np.mean(np.sum((left - left_center) ** 2, axis=1)))
        right_dispersion = float(np.mean(np.sum((right - right_center) ** 2, axis=1)))
        denominator = max(math.sqrt(0.5 * (left_dispersion + right_dispersion)), 1e-8)
        drifts.append(numerator / denominator)
    return -float(np.mean(drifts))


def balanced_accuracy_delta(labels: np.ndarray, anchor_pred: np.ndarray, adapted_pred: np.ndarray) -> float:
    values = []
    for label in (0, 1):
        mask = np.asarray(labels) == label
        values.append(
            float(np.mean(np.asarray(adapted_pred)[mask] == label))
            - float(np.mean(np.asarray(anchor_pred)[mask] == label))
        )
    return 0.5 * float(np.sum(values))


def certificate_precision(
    labels: np.ndarray,
    anchor_pred: np.ndarray,
    adapted_pred: np.ndarray,
    seed: int,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    anchor_pred = np.asarray(anchor_pred, dtype=np.int64)
    adapted_pred = np.asarray(adapted_pred, dtype=np.int64)
    if not (len(labels) == len(anchor_pred) == len(adapted_pred)):
        raise ValueError("unaligned S2 predictions")
    delta = balanced_accuracy_delta(labels, anchor_pred, adapted_pred)
    paired = (adapted_pred == labels).astype(np.float64) - (anchor_pred == labels).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    draws = np.zeros(FEATURE_BOOTSTRAPS, dtype=np.float64)
    for label in (0, 1):
        positions = np.flatnonzero(labels == label)
        sampled = rng.choice(positions, size=(FEATURE_BOOTSTRAPS, len(positions)), replace=True)
        draws += 0.5 * np.mean(paired[sampled], axis=1)
    standard_error = max(float(np.std(draws, ddof=1)), 1e-8)
    return {
        "raw_delta2": float(delta),
        "certificate_se": standard_error,
        "certificate_snr": float(delta / standard_error),
        "certificate_lcb90": float(delta - LCB_Z_90 * standard_error),
    }


def verify_feature_lock(require_committed: bool = True) -> dict[str, Any]:
    lock_path = PROTOCOL / "RELIABILITY_FEATURE_PROTOCOL_LOCK.json"
    data_path = PROTOCOL / "DATA_ACCESS_LOCK.json"
    if not lock_path.is_file() or not data_path.is_file():
        raise RuntimeError("Stage-0.5 protocol locks are absent")
    lock = read_json(lock_path)
    if require_committed:
        relative = lock_path.relative_to(REPO)
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=REPO,
            check=True,
            capture_output=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", str(relative), str(data_path.relative_to(REPO))],
            cwd=REPO,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if dirty:
            raise RuntimeError("protocol locks must be committed and clean")
    if sha256(data_path) != lock["data_access_lock_sha256"]:
        raise RuntimeError("DATA_ACCESS_LOCK changed after freeze")
    for relative, expected in lock["frozen_code_hashes"].items():
        if sha256(EXP / relative) != expected:
            raise RuntimeError(f"post-freeze code modification: {relative}")
    for relative, expected in lock["stage0_provenance_hashes"].items():
        if sha256(STAGE0_ROOT / relative) != expected:
            raise RuntimeError(f"Stage-0 provenance changed: {relative}")
    if lock["outer_10"]["identifiers_present"] is not False:
        raise RuntimeError("outer identifiers must be absent")
    return lock
