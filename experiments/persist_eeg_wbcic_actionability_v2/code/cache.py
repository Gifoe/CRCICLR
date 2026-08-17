"""Build the development-only WBCIC EEG epoch cache.

The sealed outer split is deliberately not named or opened by this module.
Only subject IDs present in ``DEVELOPMENT_SCOPE_LOCK.json`` may be materialized.
Raw BDF files are opened read-only and every derivative is written below the
experiment output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = REPO_ROOT / "experiments" / "persist_eeg_wbcic_actionability_v2"
OUT = EXP_ROOT / "outputs"
PROTOCOL = OUT / "protocol"
CACHE = OUT / "cache" / "wbcic_epochs"
SCOPE_PATH = PROTOCOL / "DEVELOPMENT_SCOPE_LOCK.json"
PREPROCESSING_PATH = PROTOCOL / "PREPROCESSING_PROTOCOL_LOCK.json"
SESSION_PATH = PROTOCOL / "SESSION_PROTOCOL_LOCK.json"
IMPLEMENTATION_ID = "persist_eeg_wbcic_eegnet_actionability_v2_20260817"
SESSIONS = (0, 1, 2)
LABELS = {"left_hand": 0, "right_hand": 1}
SUBJECT_PATTERN = re.compile(r"^sub-(?:[1-9]|[1-4]\d|5[01])$")


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No cache rows were produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def read_scope() -> tuple[list[str], Mapping[str, Any], Mapping[str, Any]]:
    required = (SCOPE_PATH, PREPROCESSING_PATH, SESSION_PATH)
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Required prospective lock missing: {path}")
    scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    preprocessing = json.loads(PREPROCESSING_PATH.read_text(encoding="utf-8"))
    session = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    subjects = [str(value) for value in scope.get("allowed_subjects", [])]
    if (
        len(subjects) != 41
        or len(set(subjects)) != 41
        or not all(SUBJECT_PATTERN.fullmatch(subject) for subject in subjects)
        or scope.get("outer_subject_ids_present") is not False
        or scope.get("runtime_must_not_open") != "OUTER_SPLIT_LOCK.json"
    ):
        raise RuntimeError("DATA_SCOPE_VIOLATION: malformed development scope")
    if scope.get("allowed_subjects_hash") != sha_lines(subjects):
        raise RuntimeError("DATA_SCOPE_VIOLATION: development subject hash mismatch")
    if preprocessing.get("channels_out") != 58 or preprocessing.get("output_sampling_rate_hz") != 250:
        raise RuntimeError("Frozen preprocessing lock is incompatible with this cache builder")
    if session.get("bids_mapping") != {"S1": "ses-0", "S2": "ses-1", "S3": "ses-2"}:
        raise RuntimeError("Frozen session lock is incompatible with this cache builder")
    return subjects, preprocessing, session


def session_paths(raw_root: Path, subject: str, session: int) -> tuple[Path, Path, Path]:
    eeg_dir = raw_root / subject / f"ses-{session}" / "eeg"
    stem = f"{subject}_ses-{session}_task-imagery_run-0"
    return (
        eeg_dir / f"{stem}_eeg.bdf",
        eeg_dir / f"{stem}_events.tsv",
        eeg_dir / f"{stem}_channels.tsv",
    )


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def output_paths(subject: str, session: int, cache_root: Path = CACHE) -> tuple[Path, Path, Path]:
    folder = cache_root / subject
    return (
        folder / f"ses-{session}_epochs.npy",
        folder / f"ses-{session}_labels.npy",
        folder / f"ses-{session}_metadata.json",
    )


def validate_existing(subject: str, session: int, cache_root: Path = CACHE) -> dict[str, Any] | None:
    epochs_path, labels_path, metadata_path = output_paths(subject, session, cache_root)
    present = [path.exists() for path in (epochs_path, labels_path, metadata_path)]
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError(f"Incomplete cache triplet; refusing silent repair: {subject} ses-{session}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    epochs = np.load(epochs_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    if (
        metadata.get("implementation_id") != IMPLEMENTATION_ID
        or metadata.get("subject") != subject
        or int(metadata.get("session", -1)) != session
        or epochs.dtype != np.float16
        or labels.dtype != np.int8
        or epochs.shape != (len(labels), 58, 1000)
        or metadata.get("epochs_sha256") != sha256_file(epochs_path)
        or metadata.get("labels_sha256") != sha256_file(labels_path)
    ):
        raise RuntimeError(f"Existing cache failed validation: {subject} ses-{session}")
    return metadata


def process_session(job: tuple[str, int, str, int, str]) -> dict[str, Any]:
    subject, session, raw_root_text, batch_size, cache_root_text = job
    if not SUBJECT_PATTERN.fullmatch(subject) or session not in SESSIONS:
        raise RuntimeError("DATA_SCOPE_VIOLATION: unsafe cache job")
    cache_root = Path(cache_root_text)
    existing = validate_existing(subject, session, cache_root)
    if existing is not None:
        return existing

    from scipy.signal import butter, resample_poly, sosfiltfilt
    import mne

    raw_root = Path(raw_root_text)
    bdf_path, events_path, channels_path = session_paths(raw_root, subject, session)
    if not all(path.is_file() for path in (bdf_path, events_path, channels_path)):
        raise RuntimeError(f"Missing raw session input: {subject} ses-{session}")
    channels = read_tsv(channels_path)
    eeg_names = [row["name"] for row in channels if row["type"] == "EEG"]
    if len(eeg_names) != 59 or "Pz" not in eeg_names or len(set(eeg_names)) != 59:
        raise RuntimeError(f"Unexpected EEG channel policy: {subject} ses-{session}")
    kept_names = [name for name in eeg_names if name != "Pz"]
    pz_index = eeg_names.index("Pz")

    events = [row for row in read_tsv(events_path) if row.get("trial_type") in LABELS]
    events.sort(key=lambda row: float(row["onset"]))
    if not events or any(abs(float(row["duration"]) - 4.0) > 1e-9 for row in events):
        raise RuntimeError(f"Malformed events: {subject} ses-{session}")
    labels = np.asarray([LABELS[row["trial_type"]] for row in events], dtype=np.int8)
    if set(labels.tolist()) != {0, 1}:
        raise RuntimeError(f"Both task classes are required: {subject} ses-{session}")

    raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose="ERROR")
    if float(raw.info["sfreq"]) != 1000.0:
        raise RuntimeError(f"Unexpected sampling rate: {subject} ses-{session}")
    missing = sorted(set(eeg_names) - set(raw.ch_names))
    if missing:
        raise RuntimeError(f"BDF is missing declared EEG channels: {missing}")
    raw.pick(eeg_names).load_data(verbose="ERROR")
    if list(raw.ch_names) != eeg_names:
        raw.reorder_channels(eeg_names)

    epochs_path, labels_path, metadata_path = output_paths(subject, session, cache_root)
    epochs_path.parent.mkdir(parents=True, exist_ok=True)
    epoch_part = epochs_path.with_suffix(epochs_path.suffix + ".part")
    label_part = labels_path.with_suffix(labels_path.suffix + ".part")
    for path in (epoch_part, label_part):
        if path.exists():
            path.unlink()
    output = np.lib.format.open_memmap(
        epoch_part, mode="w+", dtype=np.float16, shape=(len(events), 58, 1000)
    )
    sos = butter(4, (0.5, 40.0), btype="bandpass", fs=1000.0, output="sos")
    margin_samples = 2000
    window_samples = 8000
    for batch_start in range(0, len(events), max(1, batch_size)):
        batch_end = min(len(events), batch_start + max(1, batch_size))
        windows = np.empty((batch_end - batch_start, 58, window_samples), dtype=np.float32)
        for local_index, event_index in enumerate(range(batch_start, batch_end)):
            onset = int(round(float(events[event_index]["onset"]) * 1000.0))
            start = onset - margin_samples
            stop = start + window_samples
            if start < 0 or stop > raw.n_times:
                raise RuntimeError(
                    f"Event margin exceeds recording: {subject} ses-{session} event={event_index}"
                )
            block = raw.get_data(start=start, stop=stop, units="uV")
            referenced = block - block[pz_index : pz_index + 1]
            windows[local_index] = np.delete(referenced, pz_index, axis=0).astype(np.float32)
        filtered = sosfiltfilt(sos, windows, axis=-1)
        imagery = filtered[..., margin_samples : margin_samples + 4000]
        downsampled = resample_poly(imagery, up=1, down=4, axis=-1)
        if downsampled.shape != (batch_end - batch_start, 58, 1000):
            raise RuntimeError(f"Unexpected resampled shape: {downsampled.shape}")
        output[batch_start:batch_end] = np.clip(downsampled / 20.0, -12.5, 12.5).astype(np.float16)
    output.flush()
    del output
    with label_part.open("wb") as handle:
        np.save(handle, labels, allow_pickle=False)
    os.replace(epoch_part, epochs_path)
    os.replace(label_part, labels_path)
    metadata = {
        "implementation_id": IMPLEMENTATION_ID,
        "subject": subject,
        "session": session,
        "raw_bdf": str(bdf_path.resolve()),
        "raw_bdf_bytes": bdf_path.stat().st_size,
        "n_trials": int(len(labels)),
        "left_hand_trials": int(np.sum(labels == 0)),
        "right_hand_trials": int(np.sum(labels == 1)),
        "shape": [int(value) for value in (len(labels), 58, 1000)],
        "dtype": "float16",
        "sampling_rate_hz": 250,
        "channels": kept_names,
        "channel_count": 58,
        "reference": "Pz subtraction then Pz dropped",
        "bandpass_hz": [0.5, 40.0],
        "event_relative_epoch_seconds": [0.0, 4.0],
        "amplitude_transform": "microvolts / 20 clipped [-12.5,12.5]",
        "epochs_path": str(epochs_path.resolve()),
        "labels_path": str(labels_path.resolve()),
        "epochs_sha256": sha256_file(epochs_path),
        "labels_sha256": sha256_file(labels_path),
    }
    write_json(metadata_path, metadata)
    return metadata


def build(raw_root: Path, workers: int, batch_size: int) -> dict[str, Any]:
    subjects, _, _ = read_scope()
    raw_root = raw_root.resolve()
    if not (raw_root / "participants.tsv").is_file():
        raise RuntimeError("WBCIC_DATA_NOT_FOUND")
    CACHE.mkdir(parents=True, exist_ok=True)
    existing_subject_dirs = {path.name for path in CACHE.iterdir() if path.is_dir()}
    if not existing_subject_dirs.issubset(set(subjects)):
        raise RuntimeError(
            f"DATA_SCOPE_VIOLATION: non-development cache directories exist: "
            f"{sorted(existing_subject_dirs - set(subjects))}"
        )
    jobs = [
        (subject, session, str(raw_root), batch_size, str(CACHE))
        for subject in subjects
        for session in SESSIONS
    ]
    rows: list[dict[str, Any]] = []
    if workers <= 1:
        for index, job in enumerate(jobs, 1):
            row = process_session(job)
            rows.append(row)
            print(f"[cache {index}/{len(jobs)}] {row['subject']} ses-{row['session']} n={row['n_trials']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_session, job): job for job in jobs}
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows.append(row)
                print(f"[cache {index}/{len(jobs)}] {row['subject']} ses-{row['session']} n={row['n_trials']}", flush=True)
    rows.sort(key=lambda row: (row["subject"], int(row["session"])))
    expected_keys = {(subject, session) for subject in subjects for session in SESSIONS}
    observed_keys = {(str(row["subject"]), int(row["session"])) for row in rows}
    materialized = {path.name for path in CACHE.iterdir() if path.is_dir()}
    if observed_keys != expected_keys or materialized != set(subjects):
        raise RuntimeError("DATA_SCOPE_VIOLATION: cache scope is not exactly the development cohort")
    inventory_rows = [
        {
            "subject": row["subject"],
            "session": row["session"],
            "n_trials": row["n_trials"],
            "left_hand_trials": row["left_hand_trials"],
            "right_hand_trials": row["right_hand_trials"],
            "shape": "x".join(map(str, row["shape"])),
            "epochs_bytes": Path(row["epochs_path"]).stat().st_size,
            "epochs_sha256": row["epochs_sha256"],
            "labels_sha256": row["labels_sha256"],
        }
        for row in rows
    ]
    write_csv(PROTOCOL / "CACHE_INVENTORY.csv", inventory_rows)
    audit = {
        "status": "DEVELOPMENT_CACHE_COMPLETE",
        "implementation_id": IMPLEMENTATION_ID,
        "allowed_subject_count": len(subjects),
        "allowed_subjects_hash": sha_lines(subjects),
        "materialized_subject_count": len(materialized),
        "materialized_subjects_hash": sha_lines(sorted(materialized)),
        "session_count": len(rows),
        "trial_count": int(sum(int(row["n_trials"]) for row in rows)),
        "outer_subject_ids_materialized": False,
        "raw_root_enumerated": False,
        "sealed_outer_split_opened": False,
        "cache_root": str(CACHE.resolve()),
    }
    write_json(PROTOCOL / "CACHE_SCOPE_AUDIT.json", audit)
    print(json.dumps(audit, indent=2))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("build", nargs="?")
    parser.add_argument("--raw-root", type=Path, default=Path(r"D:\nips-temp\TotalP\P2\nm000348_v1.0.4_bids"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    build(args.raw_root, max(1, args.workers), max(1, args.batch_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
