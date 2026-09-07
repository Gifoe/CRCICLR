from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "protocol_v1.json"
OUTPUTS = EXPERIMENT_ROOT / "outputs"
CACHE = OUTPUTS / "cache"
CHECKPOINTS = OUTPUTS / "checkpoints"
SMOKE = OUTPUTS / "smoke"
FIGURES = EXPERIMENT_ROOT / "figures"
BOOTSTRAP_DRAWS = 10_000
OUTER_TEST_USED = False


def ensure_directories() -> None:
    for path in (OUTPUTS, CACHE, CHECKPOINTS, SMOKE, FIGURES):
        path.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload.get("outer_test_used") is not False or payload.get("outer_split_field_read") is not False:
        raise RuntimeError("Protocol outer lock is not false")
    return payload


def stable_uint64(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def stable_seed(*parts: object) -> int:
    return stable_uint64(*parts) % (2**31 - 1)


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


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
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else float(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: Iterable[object]) -> str:
    return sha256_bytes(("\n".join(map(str, values)) + "\n").encode("utf-8"))


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def subject_sort(values: Iterable[object]) -> list[str]:
    def key(value: str) -> tuple[int, str]:
        digits = "".join(character for character in value if character.isdigit())
        return (int(digits) if digits else 10**9, value)

    return sorted({str(value) for value in values}, key=key)


def subject_code(value: object) -> int:
    digits = "".join(character for character in str(value) if character.isdigit())
    if not digits:
        raise ValueError(f"Cannot encode subject: {value}")
    return int(digits)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    recalls = [float(np.mean(pred[truth == label] == label)) for label in np.unique(truth)]
    return float(np.mean(recalls)) if recalls else float("nan")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    scores: list[float] = []
    for label in np.unique(truth):
        tp = float(np.sum((truth == label) & (pred == label)))
        fp = float(np.sum((truth != label) & (pred == label)))
        fn = float(np.sum((truth == label) & (pred != label)))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(scores)) if scores else float("nan")


def softmax(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64)
    value = value - value.max(axis=1, keepdims=True)
    exp = np.exp(value)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def ce_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    return float(-np.mean(np.log(np.clip(probability[np.arange(len(truth)), truth], 1e-12, 1.0))))


def append_ledger(path: Path, row: Mapping[str, Any], columns: Sequence[str] | None = None) -> None:
    frame = pd.DataFrame([clean(dict(row))])
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True, sort=False)
    if columns is not None:
        for column in columns:
            if column not in frame:
                frame[column] = None
        frame = frame[list(columns) + [column for column in frame if column not in columns]]
    write_csv(path, frame)

