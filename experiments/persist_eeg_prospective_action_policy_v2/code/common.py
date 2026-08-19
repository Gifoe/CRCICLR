from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = EXPERIMENT_ROOT / "outputs"
PROTOCOL = OUTPUTS / "protocol"
EXPLORATION = OUTPUTS / "exploration"
RESEARCH_LOG = OUTPUTS / "research_log"
FREEZE = OUTPUTS / "freeze"
HOLDOUT = OUTPUTS / "holdout"
FIGURES = OUTPUTS / "figures"
NEXT_STAGE = OUTPUTS / "next_stage"

SEED = 20260819
SPLIT_SALT = "PERSIST_EEG_POLICY_V2_20260819"
HOLDOUT_THRESHOLD = 0.25
OUTER_TEST_USED = False


def ensure_directories() -> None:
    for path in (OUTPUTS, PROTOCOL, EXPLORATION, RESEARCH_LOG, FREEZE, HOLDOUT, FIGURES, NEXT_STAGE):
        path.mkdir(parents=True, exist_ok=True)


def default_cache_root() -> Path:
    override = os.environ.get("PERSIST_ROUTER_CACHE_ROOT")
    if override:
        return Path(override)
    candidates = (
        Path(r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\experiments\persist_eeg_router\outputs\cache"),
        REPO_ROOT / "experiments" / "persist_eeg_router" / "outputs" / "cache",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_unit(*parts: object) -> float:
    text = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") / 2**64


def stable_seed(*parts: object) -> int:
    return int.from_bytes(
        hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).digest()[:4], "big"
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def require_false_outer(frame: pd.DataFrame, source: str) -> None:
    suspicious = [column for column in frame.columns if "outer" in column.lower()]
    for column in suspicious:
        values = frame[column].dropna()
        if values.astype(str).str.lower().isin(("true", "1", "yes")).any():
            raise RuntimeError(f"Outer-test dependency in {source}:{column}")

