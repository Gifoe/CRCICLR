"""Shared utilities for the frozen final closure repair.

The module deliberately imports only the authorized OpenBMI V8_SEARCH code
path.  It contains no WBCIC or internal-holdout loader.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import stats


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
FINAL = REPO / "experiments" / "persist_eeg_persist_net_final_v1"
SOURCE = REPO / "experiments" / "persist_eeg_persist_net_source_only_diagnostic_v1"
PREVIOUS = REPO / "experiments" / "persist_eeg_final_failure_localization_and_aux_v1"
PROTOCOL_PATH = EXP / "CLOSURE_REPAIR_PROTOCOL_FROZEN.json"
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
RUNTIME = EXP / "runtime"

sys.path.insert(0, str(FINAL / "code"))
sys.path.insert(0, str(SOURCE / "code"))
import core  # type: ignore  # noqa: E402
import run_diagnostic as diag  # type: ignore  # noqa: E402


def ensure_dirs() -> None:
    for path in (RESULTS, FIGURES, RUNTIME):
        path.mkdir(parents=True, exist_ok=True)


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_md(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def protocol() -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise RuntimeError("CLOSURE_REPAIR_PROTOCOL_FROZEN.json must exist before any analysis or training")
    payload = read_json(PROTOCOL_PATH)
    if payload.get("repository_base_sha") != "3b519a138fe5074858717c70b611926fd3708f75":
        raise RuntimeError("repair protocol base SHA changed")
    if not payload.get("frozen_before_phase_a_reaggregation") or not payload.get("frozen_before_phase_b_training"):
        raise RuntimeError("repair protocol is not frozen")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    raw = json.dumps(clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def state_sha256(model: torch.nn.Module, prefixes: Sequence[str] | None = None) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if prefixes is not None and not any(name.startswith(prefix) for prefix in prefixes):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def expected_subjects() -> tuple[str, ...]:
    return tuple(map(str, protocol()["dataset"]["subject_pool"]))


def data_experiment() -> Path:
    """Resolve the authorized cache from explicit env or frozen RUN_LOCK provenance."""
    explicit = os.environ.get("PERSIST_DATA_EXPERIMENT", "").strip()
    candidates: list[Path] = [Path(explicit)] if explicit else []
    candidates.append(FINAL)
    lock_path = historical_run_dir(0, 0) / "RUN_LOCK.json"
    if lock_path.is_file():
        lock = read_json(lock_path)
        normalizer = Path(str(lock["normalizer"]))
        candidates.append(normalizer.parents[2])
    for candidate in candidates:
        metadata = candidate / "runtime" / "cache" / "OPENBMI_V8_SEARCH_MI_METADATA.parquet"
        signal = candidate / "runtime" / "cache" / "OPENBMI_V8_SEARCH_MI_RAW.npy"
        if metadata.is_file() and signal.is_file():
            return candidate
    raise FileNotFoundError("authorized 40-subject cache was not found in explicit, repository, or frozen RUN_LOCK locations")


def restore_authorized_labels(data: core.DevelopmentData, source_experiment: Path) -> core.DevelopmentData:
    meta_path = source_experiment / "runtime" / "cache" / "OPENBMI_V8_SEARCH_MI_METADATA.parquet"
    full = pd.read_parquet(meta_path, columns=["subject_id", "session_id", "label"], engine="pyarrow")
    if len(full) != len(data.metadata):
        raise RuntimeError("authorized metadata row count changed")
    observed_order = full.subject_id.astype(str).to_numpy()
    if not np.array_equal(observed_order, data.metadata.subject_id.astype(str).to_numpy()):
        raise RuntimeError("authorized metadata order changed")
    labels = full.label.astype(int).to_numpy()
    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError(f"authorized development labels are invalid: {sorted(set(labels.tolist()))}")
    data.metadata = data.metadata.copy()
    data.metadata["label"] = labels
    return data


def load_authorized_data() -> core.DevelopmentData:
    source_experiment = data_experiment()
    data, audit = diag.load_authorized_s2_data(source_experiment)
    if audit.get("pass") is not True:
        raise RuntimeError(f"authorized data audit failed: {audit}")
    data = restore_authorized_labels(data, source_experiment)
    guard_authorized_data(data)
    return data


def guard_authorized_data(data: core.DevelopmentData) -> None:
    expected = set(expected_subjects())
    observed = set(data.metadata.subject_id.astype(str).unique())
    if observed != expected:
        raise RuntimeError(f"restricted or missing subject detected; expected={sorted(expected)} observed={sorted(observed)}")
    if set(map(str, data.search_subjects)) != expected:
        raise RuntimeError("search subject pool differs from frozen 40-subject pool")
    if len(data.metadata) != 8000:
        raise RuntimeError(f"authorized cache cardinality changed: {len(data.metadata)}")
    if int(data.holdout_count) != 14:
        raise RuntimeError("holdout count provenance changed")


def audit_frozen_tables() -> dict[str, Any]:
    expected = set(expected_subjects())
    files = {
        "source_only_raw": SOURCE / "results" / "source_only_raw.csv",
        "replay_per_subject": SOURCE / "results" / "replay_per_subject.csv",
        "adapted_authoritative_raw": SOURCE / "results" / "adapted_authoritative_raw.csv",
        "mechanism_raw": SOURCE / "results" / "mechanism_raw.csv",
        "per_subject_results": SOURCE / "results" / "per_subject_results.csv",
        "certificate_transfer": PREVIOUS / "results" / "certificate_transfer.csv",
        "functional_persistence": PREVIOUS / "results" / "functional_persistence.csv",
        "reliance_metrics": PREVIOUS / "results" / "reliance_metrics.csv",
        "gradient_conflict": PREVIOUS / "results" / "gradient_conflict.csv",
        "calibration_metrics": PREVIOUS / "results" / "calibration_metrics.csv",
    }
    issues: list[str] = []
    hashes: dict[str, str] = {}
    rows: dict[str, int] = {}
    for name, path in files.items():
        if not path.is_file():
            issues.append(f"missing:{name}:{path}")
            continue
        hashes[name] = sha256_file(path)
        frame = pd.read_csv(path)
        rows[name] = len(frame)
        if "subject_id" in frame:
            observed = set(frame.subject_id.astype(str))
            if not observed.issubset(expected):
                issues.append(f"restricted_subject:{name}:{sorted(observed - expected)}")
        for column in ("internal_holdout_used", "internal_holdout_accessed", "WBCIC_outer_used", "WBCIC_outer_accessed", "outer_test_used"):
            if column in frame and frame[column].fillna(False).astype(bool).any():
                issues.append(f"restricted_flag:{name}:{column}")
    return {"pass": not issues, "issues": issues, "hashes": hashes, "rows": rows}


def subject_bootstrap(values: Iterable[float], seed: int = 9173, draws: int = 10_000) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"n": 0, "mean": math.nan, "median": math.nan, "ci95_l": math.nan, "ci95_u": math.nan}
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci95_l": float(np.quantile(sampled, 0.025)),
        "ci95_u": float(np.quantile(sampled, 0.975)),
        "positive": int((array > 0).sum()),
        "negative": int((array < 0).sum()),
        "tied": int((array == 0).sum()),
    }


def paired_subject_stats(first: pd.Series, second: pd.Series, seed: int = 9173) -> dict[str, Any]:
    paired = pd.concat([first.rename("first"), second.rename("second")], axis=1).dropna()
    out = subject_bootstrap((paired.second - paired.first).to_numpy(), seed=seed)
    out["delta_definition"] = "second-minus-first"
    return out


def _corr_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan, math.nan
    return float(np.corrcoef(x, y)[0, 1]), float(stats.spearmanr(x, y).statistic)


def subject_bootstrap_corr(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    seed: int = 9173,
    draws: int = 10_000,
) -> dict[str, Any]:
    frame = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}).replace([np.inf, -np.inf], np.nan).dropna()
    xa = frame.x.to_numpy()
    ya = frame.y.to_numpy()
    pearson, spearman = _corr_pair(xa, ya)
    if not math.isfinite(pearson) or not math.isfinite(spearman):
        return {
            "n": int(len(frame)), "pearson": pearson, "spearman": spearman,
            "pearson_ci95_l": math.nan, "pearson_ci95_u": math.nan,
            "spearman_ci95_l": math.nan, "spearman_ci95_u": math.nan,
            "label": "NO RELIABLE ASSOCIATION ESTABLISHED",
        }
    rng = np.random.default_rng(seed)
    pearsons = np.empty(draws, dtype=np.float64)
    spearmans = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        indices = rng.integers(0, len(frame), size=len(frame))
        pearsons[draw], spearmans[draw] = _corr_pair(xa[indices], ya[indices])
    pearsons = pearsons[np.isfinite(pearsons)]
    spearmans = spearmans[np.isfinite(spearmans)]
    payload = {
        "n": int(len(frame)), "pearson": pearson, "spearman": spearman,
        "pearson_ci95_l": float(np.quantile(pearsons, 0.025)),
        "pearson_ci95_u": float(np.quantile(pearsons, 0.975)),
        "spearman_ci95_l": float(np.quantile(spearmans, 0.025)),
        "spearman_ci95_u": float(np.quantile(spearmans, 0.975)),
    }
    payload["label"] = association_label(payload)
    return payload


def association_label(payload: Mapping[str, Any]) -> str:
    p = float(payload.get("pearson", math.nan))
    s = float(payload.get("spearman", math.nan))
    pl = float(payload.get("pearson_ci95_l", math.nan))
    pu = float(payload.get("pearson_ci95_u", math.nan))
    sl = float(payload.get("spearman_ci95_l", math.nan))
    su = float(payload.get("spearman_ci95_u", math.nan))
    if not all(math.isfinite(value) for value in (p, s, pl, pu, sl, su)):
        return "NO RELIABLE ASSOCIATION ESTABLISHED"
    if p > 0 and s > 0 and pl > 0 and sl > 0:
        return "SUPPORTED POSITIVE ASSOCIATION"
    if p < 0 and s < 0 and pu < 0 and su < 0:
        return "SUPPORTED NEGATIVE ASSOCIATION"
    if (p > 0 and s > 0) or (p < 0 and s < 0):
        return "DIRECTIONAL BUT UNCERTAIN"
    return "NO RELIABLE ASSOCIATION ESTABLISHED"


def historical_run_dir(fold: int, seed: int) -> Path:
    return FINAL / "runtime" / "runs" / f"fold-{fold}" / f"seed-{seed}"


def certificate_coordinate_map(fold: int, seed: int) -> dict[int, int]:
    run = historical_run_dir(fold, seed)
    values = np.load(run / "certificate" / "PUD_CERTIFICATE.npz", allow_pickle=False)
    basis = np.asarray(values["basis_PUD"], dtype=np.float64)
    mapping: dict[int, int] = {}
    for column in range(basis.shape[1]):
        vector = basis[:, column]
        direction = int(np.argmax(np.abs(vector)))
        expected = np.zeros_like(vector)
        expected[direction] = 1.0
        if not np.allclose(np.abs(vector), expected, atol=1e-6):
            raise RuntimeError(f"PUD basis is not a coordinate basis in fold={fold} seed={seed} column={column}")
        mapping[column] = direction
    return mapping


def compute_normalizer_local(data: core.DevelopmentData, subjects: Sequence[str], path: Path) -> tuple[np.ndarray, np.ndarray]:
    subjects = tuple(core.subject_sort(subjects))
    if path.is_file():
        values = np.load(path, allow_pickle=False)
        if tuple(values["subjects"].astype(str).tolist()) != subjects:
            raise RuntimeError(f"normalizer subject mismatch: {path}")
        return values["mean"].astype(np.float32), values["std"].astype(np.float32)
    indices = core.row_indices(data.metadata, subjects, (1, 2))
    total = np.zeros(data.x.shape[1], dtype=np.float64)
    total_sq = np.zeros(data.x.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(indices), 64):
        block = np.asarray(data.x[indices[start:start + 64]], dtype=np.float64)
        total += block.sum(axis=(0, 2))
        total_sq += np.square(block).sum(axis=(0, 2))
        count += int(block.shape[0] * block.shape[2])
    mean = total / max(count, 1)
    variance = np.maximum(total_sq / max(count, 1) - np.square(mean), 1e-8)
    std = np.sqrt(variance)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mean=mean.astype(np.float32), std=std.astype(np.float32), subjects=np.asarray(subjects))
    return mean.astype(np.float32), std.astype(np.float32)


def append_engineering_log(section: str, body: str) -> None:
    path = EXP / "ENGINEERING_REPAIR_LOG.md"
    if not path.is_file():
        path.write_text("# Engineering repair log\n\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"## {section}\n\n{body.rstrip()}\n\n")
