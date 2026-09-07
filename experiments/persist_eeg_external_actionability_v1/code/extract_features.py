"""Outcome-blind, subject-filtered EEGMMIDB feature extraction.

The extractor never recursively scans the raw root.  It constructs exactly six
official EDF paths for each subject explicitly allowed by the development
scope lock.  Outer-test signals, annotations, and labels are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import welch


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = REPO_ROOT / "experiments" / "persist_eeg_external_actionability_v1"
OUT = EXP_ROOT / "outputs"
SCOPE_LOCK = OUT / "protocol" / "DEVELOPMENT_SCOPE_LOCK.json"
SEALED_SPLIT = OUT / "protocol" / "EXTERNAL_SPLIT_LOCK.json"
TARGET_RUNS = (4, 6, 8, 10, 12, 14)
BANDS = ((4, 8), (8, 12), (12, 16), (16, 20), (20, 30), (30, 40))
EPS = 1e-20


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_subject(value: str) -> str:
    subject = str(value)
    if re.fullmatch(r"S(?:00[1-9]|0[1-9][0-9]|10[0-9])", subject) is None:
        raise RuntimeError(f"DATA_SCOPE_VIOLATION: invalid subject identifier {subject!r}")
    return subject


def edf_path(raw_root: Path, subject: str, run: int) -> Path:
    validate_subject(subject)
    if int(run) not in TARGET_RUNS:
        raise RuntimeError(f"DATA_SCOPE_VIOLATION: unregistered run {run}")
    path = raw_root / subject / f"{subject}R{int(run):02d}.edf"
    expected_parent = (raw_root / subject).resolve()
    if path.parent.resolve() != expected_parent:
        raise RuntimeError("DATA_SCOPE_VIOLATION: path escaped the allowed subject directory")
    return path


def map_event(run: int, description: str) -> int | None:
    label = str(description).strip().upper()
    if label == "T0":
        return None
    if run in {4, 8, 12}:
        return {"T1": 0, "T2": 1}.get(label)
    if run in {6, 10, 14}:
        return {"T1": 2, "T2": 3}.get(label)
    raise RuntimeError(f"Unexpected run {run}")


def spectral_features(epoch: np.ndarray, sampling_rate: float) -> np.ndarray:
    value = np.asarray(epoch, dtype=np.float64)
    value = value - value.mean(axis=1, keepdims=True)
    frequencies, power = welch(
        value, fs=float(sampling_rate), nperseg=256, noverlap=128,
        detrend="linear", scaling="density", axis=-1,
    )
    total_mask = (frequencies >= 4) & (frequencies < 40)
    total = np.trapezoid(power[:, total_mask], frequencies[total_mask], axis=1)
    absolute: list[np.ndarray] = []
    relative: list[np.ndarray] = []
    for low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        band = np.trapezoid(power[:, mask], frequencies[mask], axis=1)
        absolute.append(np.log(np.maximum(band, EPS)))
        relative.append(np.log(np.maximum(band / np.maximum(total, EPS), EPS)))
    return np.concatenate([np.stack(absolute, axis=1), np.stack(relative, axis=1)], axis=1).reshape(-1).astype(np.float32)


def process_subject(task: tuple[str, str, str]) -> dict[str, Any]:
    raw_root_text, cache_root_text, subject = task
    raw_root, cache_root = Path(raw_root_text), Path(cache_root_text)
    import mne

    features: list[np.ndarray] = []
    labels: list[int] = []
    runs: list[int] = []
    trial_index: list[int] = []
    channel_names: list[str] | None = None
    source_files: list[str] = []
    source_sampling_rates: list[float] = []
    for run in TARGET_RUNS:
        path = edf_path(raw_root, subject, run)
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        current_names = [str(name) for name in raw.ch_names]
        if len(current_names) != 64:
            raise RuntimeError(f"Expected 64 channels in {path}, found {len(current_names)}")
        if channel_names is None:
            channel_names = current_names
        elif channel_names != current_names:
            raise RuntimeError(f"Channel order changed for {subject} run {run}")
        sampling_rate = float(raw.info["sfreq"])
        if sampling_rate not in {128.0, 160.0}:
            raise RuntimeError(f"Unexpected EEGMMIDB sampling rate in {path}: {sampling_rate}")
        signal = raw.get_data(picks=list(range(64)))
        count = 0
        for annotation in raw.annotations:
            label = map_event(run, str(annotation["description"]))
            if label is None:
                continue
            start = int(round(float(annotation["onset"]) * sampling_rate))
            end = start + int(round(4.0 * sampling_rate))
            if start < 0 or end > signal.shape[1]:
                continue
            features.append(spectral_features(signal[:, start:end], sampling_rate))
            labels.append(label)
            runs.append(run)
            trial_index.append(count)
            source_sampling_rates.append(sampling_rate)
            count += 1
        if count < 10:
            raise RuntimeError(f"Too few mapped MI trials in {path}: {count}")
        source_files.append(str(path))
        del signal, raw
    matrix = np.stack(features).astype(np.float32)
    if matrix.shape[1] != 768 or not np.isfinite(matrix).all():
        raise RuntimeError(f"Invalid feature matrix for {subject}: {matrix.shape}")
    target = cache_root / f"{subject}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".npz.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=matrix,
            labels=np.asarray(labels, dtype=np.int16),
            runs=np.asarray(runs, dtype=np.int16),
            trial_index=np.asarray(trial_index, dtype=np.int16),
            subject=np.asarray(subject),
            channel_names=np.asarray(channel_names, dtype="U32"),
            sampling_rate=np.asarray(source_sampling_rates, dtype=np.float32),
        )
    os.replace(temporary, target)
    return {
        "subject": subject,
        "n_trials": int(len(matrix)),
        "n_runs": int(len(set(runs))),
        "n_channels": 64,
        "feature_dim": int(matrix.shape[1]),
        "source_sampling_rates_hz": sorted(set(source_sampling_rates)),
        "cache_path": str(target),
        "cache_sha256": sha256(target),
        "source_files": source_files,
    }


def load_scope() -> dict[str, Any]:
    if not SCOPE_LOCK.exists():
        raise RuntimeError("Prospective development scope lock is missing")
    # Intentionally do not open SEALED_SPLIT.  It contains outer IDs and is
    # prohibited at runtime by DEVELOPMENT_SCOPE_LOCK.json.
    payload = json.loads(SCOPE_LOCK.read_text(encoding="utf-8"))
    if payload.get("outer_subject_ids_present") is not False:
        raise RuntimeError("DATA_SCOPE_VIOLATION: development lock contains outer IDs")
    allowed = [validate_subject(item) for item in payload.get("allowed_subjects", [])]
    if len(allowed) != 90 or len(set(allowed)) != 90:
        raise RuntimeError("DATA_SCOPE_VIOLATION: expected exactly 90 development subjects")
    return payload


def inventory(raw_root: Path, workers: int) -> int:
    files = [edf_path(raw_root, f"S{subject:03d}", run) for subject in range(1, 110) for run in TARGET_RUNS]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        write_json(OUT / "protocol" / "RAW_DATASET_INVENTORY.json", {
            "status": "EXTERNAL_AUDIT_INSUFFICIENT_DATA", "missing": missing,
        })
        return 2
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(sha256, path): path for path in files}
        for future in as_completed(futures):
            path = futures[future]
            rows.append({
                "relative_path": str(path.relative_to(raw_root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": future.result(),
            })
    rows.sort(key=lambda row: row["relative_path"])
    tree = hashlib.sha256()
    for row in rows:
        tree.update(f"{row['relative_path']}\t{row['bytes']}\t{row['sha256']}\n".encode("utf-8"))
    payload = {
        "status": "EEGMMIDB_TARGET_FILES_COMPLETE",
        "official_subjects": 109,
        "target_runs": list(TARGET_RUNS),
        "target_edf_count": len(rows),
        "bytes": int(sum(row["bytes"] for row in rows)),
        "tree_sha256": tree.hexdigest(),
        "files": rows,
        "content_interpreted": False,
        "performance_outcome_inspected": False,
    }
    write_json(OUT / "protocol" / "RAW_DATASET_INVENTORY.json", payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "files"}, indent=2))
    return 0


def extract(raw_root: Path, workers: int, resume: bool) -> int:
    scope = load_scope()
    allowed = sorted(scope["allowed_subjects"])
    cache_root = OUT / "cache" / "eegmmidb_features"
    cache_root.mkdir(parents=True, exist_ok=True)
    existing = {path.stem for path in cache_root.glob("S*.npz")}
    if not existing.issubset(set(allowed)):
        write_json(OUT / "protocol" / "DATA_SCOPE_AUDIT.json", {
            "status": "DATA_SCOPE_VIOLATION",
            "reason": "feature cache contains a subject outside development scope",
            "unexpected_cache_subjects": sorted(existing - set(allowed)),
        })
        raise RuntimeError("DATA_SCOPE_VIOLATION")
    tasks = []
    completed: list[dict[str, Any]] = []
    for subject in allowed:
        cache = cache_root / f"{subject}.npz"
        if resume and cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                if str(data["subject"].item()) != subject or data["features"].shape[1] != 768:
                    raise RuntimeError(f"Invalid resumed cache {cache}")
                completed.append({
                    "subject": subject, "n_trials": int(len(data["labels"])),
                    "n_runs": int(len(np.unique(data["runs"]))), "n_channels": 64,
                    "feature_dim": 768, "cache_path": str(cache),
                    "cache_sha256": sha256(cache), "status": "resumed",
                })
        else:
            tasks.append((str(raw_root), str(cache_root), subject))
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(process_subject, task): task[2] for task in tasks}
        for future in as_completed(futures):
            subject = futures[future]
            row = future.result()
            completed.append(row)
            print(f"[features] {subject} trials={row['n_trials']}", flush=True)
    completed.sort(key=lambda row: row["subject"])
    accessed = [row["subject"] for row in completed]
    if accessed != allowed:
        raise RuntimeError("DATA_SCOPE_VIOLATION: extracted subject set does not equal allowed scope")
    audit = {
        "status": "DATA_SCOPE_PASS",
        "loader": "explicit subject/run path construction; no recursive raw-root enumeration",
        "sealed_split_opened_by_runtime": False,
        "outer_manifest_rows_materialized": False,
        "outer_event_labels_materialized": False,
        "outer_signals_materialized": False,
        "outer_embeddings_materialized": False,
        "outer_subjects_in_cache": False,
        "allowed_subject_count": len(allowed),
        "materialized_subject_count": len(accessed),
        "materialized_subjects_hash": hashlib.sha256(("\n".join(accessed) + "\n").encode()).hexdigest(),
        "source_edf_open_count": int(sum(row["n_runs"] for row in completed)),
        "expected_source_edf_open_count": 90 * 6,
        "feature_cache_files": len(completed),
        "subjects": [{key: value for key, value in row.items() if key != "source_files"} for row in completed],
    }
    if audit["source_edf_open_count"] != audit["expected_source_edf_open_count"]:
        audit["status"] = "DATA_SCOPE_VIOLATION"
        write_json(OUT / "protocol" / "DATA_SCOPE_AUDIT.json", audit)
        raise RuntimeError("DATA_SCOPE_VIOLATION: unexpected source file count")
    write_json(OUT / "protocol" / "DATA_SCOPE_AUDIT.json", audit)
    print(json.dumps({key: value for key, value in audit.items() if key != "subjects"}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inventory", "extract"))
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "inventory":
        return inventory(args.raw_root, args.workers)
    return extract(args.raw_root, args.workers, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
