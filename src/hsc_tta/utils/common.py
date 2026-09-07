from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return data


def config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def require_cpu(device: str = "cpu") -> None:
    if device.lower() != "cpu":
        raise ValueError("CPU phase only: --device must be cpu")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, ""):
        raise RuntimeError("CPU phase requires CUDA_VISIBLE_DEVICES to be empty")

