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
DIAGNOSTICS = OUTPUTS / "diagnostics"
RESULTS = OUTPUTS / "results"
RESEARCH_LOG = OUTPUTS / "research_log"
FIGURES = OUTPUTS / "figures"
NEXT_STAGE = OUTPUTS / "next_stage"
V21_ROOT = REPO_ROOT / "experiments" / "persist_eeg_prospective_action_policy_v2_1"
V21_CODE = V21_ROOT / "code"
V21_OUTPUTS = V21_ROOT / "outputs"
V2_ROOT = REPO_ROOT / "experiments" / "persist_eeg_prospective_action_policy_v2"
V2_OUTPUTS = V2_ROOT / "outputs"
AUDIT_SEED = 20260819
BOOTSTRAP_REPETITIONS = 10_000
DATASET_ID = "OpenBMI_SSVEP_NEMAR_nm000273_offline"
OUTER_TEST_USED = False


def ensure_directories() -> None:
    for path in (OUTPUTS, PROTOCOL, DIAGNOSTICS, RESULTS, RESEARCH_LOG, FIGURES, NEXT_STAGE):
        path.mkdir(parents=True, exist_ok=True)


def default_cache_root() -> Path:
    override = os.environ.get("PERSIST_ROUTER_CACHE_ROOT")
    if override:
        return Path(override)
    return Path(
        r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\experiments\persist_eeg_router\outputs\cache"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def stable_seed(*parts: object) -> int:
    text = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(values / (1.0 - values))


def pool_mask(frame: pd.DataFrame, pool: str) -> np.ndarray:
    if pool == "all_52_exploratory":
        return np.ones(len(frame), dtype=bool)
    return frame.source_pool.to_numpy(dtype=str) == pool
