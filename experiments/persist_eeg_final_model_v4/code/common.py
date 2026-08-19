from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = EXPERIMENT_ROOT / "outputs"
PROTOCOL = OUTPUTS / "protocol"
DESIGN = OUTPUTS / "design"
RESEARCH_LOG = OUTPUTS / "research_log"
LEADERBOARD = OUTPUTS / "leaderboard"
ABLATIONS = OUTPUTS / "ablations"
DIAGNOSTICS = OUTPUTS / "diagnostics"
FIGURES = OUTPUTS / "figures"
FINAL_LOCK = OUTPUTS / "final_lock"
V3_ROOT = REPO_ROOT / "experiments" / "persist_eeg_residual_actionability_v3"
V21_ROOT = REPO_ROOT / "experiments" / "persist_eeg_prospective_action_policy_v2_1"
V4_SEED = 20260819
BOOTSTRAP_REPETITIONS = 10_000
OUTER_TEST_USED = False


def ensure_directories() -> None:
    for path in (
        OUTPUTS,
        PROTOCOL,
        DESIGN,
        RESEARCH_LOG,
        LEADERBOARD,
        ABLATIONS,
        DIAGNOSTICS,
        FIGURES,
        FINAL_LOCK,
    ):
        path.mkdir(parents=True, exist_ok=True)


def default_openbmi_cache() -> Path:
    override = os.environ.get("PERSIST_ROUTER_CACHE_ROOT")
    if override:
        return Path(override)
    return Path(
        r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\experiments\persist_eeg_router\outputs\cache"
    )


def default_wbcic_repo() -> Path:
    override = os.environ.get("PERSIST_WBCIC_SOURCE_REPO")
    if override:
        return Path(override)
    return Path(r"D:\nips-temp\TotalP\P1\CRCICLR_WBCIC_EEGNET")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, set):
        return [clean(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-8, 1 - 1e-8)
    return np.log(values / (1 - values))


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(rows)
