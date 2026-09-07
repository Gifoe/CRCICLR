from __future__ import annotations

import hashlib
import json
import math
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
RESEARCH_LOG = OUTPUTS / "research_log"
LEADERBOARD = OUTPUTS / "leaderboard"
ABLATIONS = OUTPUTS / "ablations"
FINAL_CANDIDATE = OUTPUTS / "final_candidate"
CACHE = OUTPUTS / "cache"
V4_ROOT = REPO_ROOT / "experiments" / "persist_eeg_final_model_v4"
V5_SEED = 20260820
BOOTSTRAP_DRAWS = 20_000
OUTER_TEST_USED = False


def ensure_directories() -> None:
    for path in (
        OUTPUTS,
        PROTOCOL,
        DIAGNOSTICS,
        RESEARCH_LOG,
        LEADERBOARD,
        ABLATIONS,
        FINAL_CANDIDATE,
        CACHE,
    ):
        path.mkdir(parents=True, exist_ok=True)


def default_wbcic_repo() -> Path:
    return Path(
        os.environ.get(
            "PERSIST_WBCIC_SOURCE_REPO",
            r"D:\nips-temp\TotalP\P1\CRCICLR_WBCIC_EEGNET",
        )
    )


def default_stage0_repo() -> Path:
    return Path(
        os.environ.get(
            "PERSIST_STAGE0_REPO",
            r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full",
        )
    )


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
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
        return None if not math.isfinite(value) else value
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(clean(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(values) - np.log1p(-values)
