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
BASELINES = OUTPUTS / "baselines"
LEADERBOARD = OUTPUTS / "leaderboard"
ABLATIONS = OUTPUTS / "ablations"
RESEARCH_LOG = OUTPUTS / "research_log"
FINAL_CANDIDATE = OUTPUTS / "final_candidate"
CACHE = OUTPUTS / "cache"
V6_SEED = 20260820
BOOTSTRAP_DRAWS = 20_000
OUTER_TEST_USED = False


def ensure_directories() -> None:
    paths = (
        PROTOCOL,
        DIAGNOSTICS,
        BASELINES,
        LEADERBOARD,
        ABLATIONS,
        RESEARCH_LOG,
        FINAL_CANDIDATE,
        CACHE,
    )
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    for relative in (
        "protocol",
        "baselines",
        "backbones",
        "history_encoder",
        "adapters",
        "persist",
        "prototypes",
        "geometry",
        "fusion",
        "meta_learning",
        "risk",
        "evaluation",
    ):
        package = Path(__file__).parent / relative
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").touch(exist_ok=True)


def stage0_root() -> Path:
    return Path(os.environ.get("PERSIST_STAGE0_REPO", r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full"))


def wbcic_source_root() -> Path:
    return Path(os.environ.get("PERSIST_WBCIC_SOURCE_REPO", r"D:\nips-temp\TotalP\P1\CRCICLR_WBCIC_EEGNET"))


def v5_output_root() -> Path:
    return Path(os.environ.get("PERSIST_V5_OUTPUTS", r"D:\nips-temp\TotalP\P1\CRCICLR_RESIDUAL_V3\experiments\persist_eeg_final_model_v5\outputs"))


def stable_seed(*parts: object) -> int:
    value = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(clean(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def sigmoid(value: np.ndarray) -> np.ndarray:
    array = np.clip(np.asarray(value, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-array))


def logit(value: np.ndarray) -> np.ndarray:
    array = np.clip(np.asarray(value, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(array) - np.log1p(-array)
