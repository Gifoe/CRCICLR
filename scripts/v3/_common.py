from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def project_root(value: str | Path) -> Path:
    path = Path(value).resolve()
    if (path / "data").exists() and (path / "repo").exists(): return path
    if (path.parent / "data").exists(): return path.parent
    raise FileNotFoundError(f"cannot resolve project root from {path}")


def load_yaml(path: str | Path) -> dict[str, object]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def config_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
