from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_seed(*parts: object) -> int:
    token = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32 - 1)


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    prediction = logits.argmax(axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    recalls: list[float] = []
    f1_values: list[float] = []
    for class_id in range(logits.shape[1]):
        positive = labels == class_id
        predicted_positive = prediction == class_id
        true_positive = int(np.sum(positive & predicted_positive))
        false_negative = int(np.sum(positive & ~predicted_positive))
        false_positive = int(np.sum(~positive & predicted_positive))
        recall = true_positive / max(true_positive + false_negative, 1)
        precision = true_positive / max(true_positive + false_positive, 1)
        recalls.append(recall)
        f1_values.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
    log_probability = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return {
        "BA": float(np.mean(recalls)),
        "F1": float(np.mean(f1_values)),
        "CE": float(-log_probability[np.arange(len(labels)), labels].mean()),
    }


def control_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics for logits shaped trials x controls x classes."""
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    prediction = logits.argmax(axis=2)
    recalls: list[np.ndarray] = []
    f1_values: list[np.ndarray] = []
    for class_id in range(logits.shape[2]):
        positive = labels[:, None] == class_id
        predicted_positive = prediction == class_id
        tp = np.sum(positive & predicted_positive, axis=0)
        fn = np.sum(positive & ~predicted_positive, axis=0)
        fp = np.sum(~positive & predicted_positive, axis=0)
        recall = tp / np.maximum(tp + fn, 1)
        precision = tp / np.maximum(tp + fp, 1)
        f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(recall, dtype=float), where=(precision + recall) != 0)
        recalls.append(recall)
        f1_values.append(f1)
    shifted = logits - logits.max(axis=2, keepdims=True)
    log_probability = shifted - np.log(np.exp(shifted).sum(axis=2, keepdims=True))
    ce = -log_probability[np.arange(len(labels)), :, labels].mean(axis=0)
    return {"BA": np.mean(recalls, axis=0), "F1": np.mean(f1_values, axis=0), "CE": ce}


def percentile_ci(values: np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    return [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))]


def dataframe_markdown(frame: Any, float_digits: int = 9) -> str:
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        return str(value).replace("|", "\\|")
    headers = [str(name) for name in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)
