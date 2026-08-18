from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = REPO_ROOT / "experiments" / "persist_eeg_prospective_action_policy_v1"
OUT = EXP_ROOT / "outputs"
PROTOCOL = OUT / "protocol"
DATA = OUT / "data"
DIAGNOSTICS = OUT / "diagnostics"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"
NEXT_STAGE = OUT / "next_stage"
SEED = 20260819


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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
    path.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10g")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32 - 1)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


def router_pilot_root() -> Path:
    candidates: list[Path] = []
    configured = os.environ.get("PERSIST_EEG_PILOT_ROOT")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            REPO_ROOT,
            REPO_ROOT.parent / "persist_eeg_stage0_repo_full",
            Path(r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full"),
        ]
    )
    marker = Path("experiments/persist_eeg_router/outputs/cache/OOF_ROUTER_FEATURES.parquet")
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Legal Router OOF cache not found. Set PERSIST_EEG_PILOT_ROOT to the historical pilot repository."
    )


def assert_no_outer_columns(frame: pd.DataFrame, source: str) -> None:
    for column in frame.columns:
        column_lower = column.lower()
        if column_lower.startswith("outer_") or "outer_test" in column_lower:
            values = frame[column].dropna()
            if values.dtype == bool and bool(values.any()):
                raise RuntimeError(f"Outer-test use detected in {source}:{column}")
            lowered = {str(value).strip().lower() for value in values.unique()}
            forbidden = {"true", "1", "used", "evaluated", "opened"}
            if lowered & forbidden:
                raise RuntimeError(f"Outer-test use detected in {source}:{column}={lowered & forbidden}")


def recursive_outer_true(payload: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            key_lower = str(key).lower()
            is_outer_scope = key_lower.startswith("outer_") or "outer_test" in key_lower
            if is_outer_scope and value is True:
                found.append(path)
            found.extend(recursive_outer_true(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(recursive_outer_true(value, f"{prefix}[{index}]"))
    return found


def subject_balanced_ba(frame: pd.DataFrame, prediction: str, label: str = "outcome_label") -> float:
    from sklearn.metrics import balanced_accuracy_score

    values: list[float] = []
    for _, group in frame.groupby(["fold_id", "seed_id", "subject_id"], dropna=False, sort=False):
        if group[label].nunique() < 2:
            continue
        values.append(float(balanced_accuracy_score(group[label], group[prediction])))
    return float(np.mean(values)) if values else float("nan")


def package_versions() -> dict[str, Any]:
    import matplotlib
    import scipy
    import sklearn

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required artifacts are missing:\n" + "\n".join(missing))
