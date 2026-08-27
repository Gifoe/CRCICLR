from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
P4A = REPO / "experiments" / "persist_eeg_p4a_cross_setting_expansion_v1"
P4B = REPO / "experiments" / "persist_eeg_p4b_identity_reliability_discovery_v1"
P4C = REPO / "experiments" / "persist_eeg_p4c_suppression_safety_validation_v1"
P4A_RUNS = P4A / "runtime" / "runs"

SETTINGS: dict[str, dict[str, str]] = {
    "S1": {"dataset": "OpenBMI", "task": "MI", "backbone": "EEGNet", "evidence": "historical"},
    "S2": {"dataset": "OpenBMI", "task": "MI", "backbone": "EEGConformer", "evidence": "historical"},
    "S3": {"dataset": "WBCIC", "task": "MI", "backbone": "EEGNet", "evidence": "historical"},
    "S4": {"dataset": "WBCIC", "task": "MI", "backbone": "EEGConformer", "evidence": "prospective"},
    "S5": {"dataset": "OpenBMI", "task": "ERP", "backbone": "EEGNet", "evidence": "supplementary_partial"},
    "S6": {"dataset": "OpenBMI", "task": "ERP", "backbone": "EEGConformer", "evidence": "prospective"},
}
METHODS = ("DANN", "MMD", "CORAL")
LAMBDAS = (0.01, 0.1, 1.0)
FOLDS = tuple(range(5))
SEEDS = (0, 1, 2)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 440_041
EPS = 1e-12


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(clean(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_seed(*parts: Any) -> int:
    token = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32 - 1)


def slug(method: str, lam: float) -> str:
    return f"{method.lower()}__lambda-{lam:.2f}"


def source_complete(setting: str, fold: int, seed: int, method: str, lam: float) -> Path:
    return P4A_RUNS / setting / f"fold-{fold}" / f"seed-{seed}" / "source_freeze" / slug(method, lam) / "SOURCE_COMPLETE.json"


def checkpoint_path(setting: str, fold: int, seed: int, method: str, lam: float) -> Path:
    return P4A_RUNS / setting / f"fold-{fold}" / f"seed-{seed}" / "checkpoints" / f"{slug(method, lam)}.pt"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(f"required file missing: {path}")
    return path
