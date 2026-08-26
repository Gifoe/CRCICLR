"""Assemble the explicit 41-subject per-session cache into a training memmap.

No raw dataset root is enumerated.  Every input path is constructed from the
authoritative development whitelist copied into this experiment's provenance.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PROVENANCE = EXP / "provenance"
CACHE_ROOT = EXP / "runtime" / "cache"
CELL_ROOT = CACHE_ROOT / "wbcic_epochs"
SIGNAL = CACHE_ROOT / "WBCIC_DEVELOPMENT_MI_RAW.npy"
METADATA = CACHE_ROOT / "WBCIC_DEVELOPMENT_MI_METADATA.parquet"
AUDIT = PROVENANCE / "CONSOLIDATED_CACHE_AUDIT.json"
SUBJECT_HASH = "dae8e7ec00cbcf6dcc8c5b25829f2148fd0b5fdf162f75a0cddc18b096af7db4"


def sha_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    scope = json.loads((PROVENANCE / "DEVELOPMENT_SCOPE_LOCK.json").read_text(encoding="utf-8"))
    cache_scope = json.loads((PROVENANCE / "CACHE_SCOPE_AUDIT.json").read_text(encoding="utf-8"))
    subjects = list(map(str, scope["allowed_subjects"]))
    if len(subjects) != 41 or sha_lines(subjects) != SUBJECT_HASH:
        raise RuntimeError("authoritative development whitelist/hash mismatch")
    if (
        cache_scope.get("status") != "DEVELOPMENT_CACHE_COMPLETE"
        or cache_scope.get("materialized_subjects_hash") != SUBJECT_HASH
        or cache_scope.get("materialized_subject_count") != 41
        or cache_scope.get("session_count") != 123
        or cache_scope.get("outer_subject_ids_materialized") is not False
        or cache_scope.get("raw_root_enumerated") is not False
        or cache_scope.get("sealed_outer_split_opened") is not False
    ):
        raise RuntimeError("per-session cache scope audit failed")

    cells: list[tuple[str, int, Path, Path, int]] = []
    total = 0
    for subject in subjects:
        for session in range(3):
            epochs_path = CELL_ROOT / subject / f"ses-{session}_epochs.npy"
            labels_path = CELL_ROOT / subject / f"ses-{session}_labels.npy"
            metadata_path = CELL_ROOT / subject / f"ses-{session}_metadata.json"
            if not epochs_path.is_file() or not labels_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(f"missing authorized cache cell: {subject} ses-{session}")
            cell_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            x = np.load(epochs_path, mmap_mode="r", allow_pickle=False)
            y = np.load(labels_path, mmap_mode="r", allow_pickle=False)
            if x.ndim != 3 or x.shape[1:] != (58, 1000) or x.dtype != np.float16:
                raise RuntimeError(f"invalid signal cache {epochs_path}: {x.shape} {x.dtype}")
            if y.shape != (len(x),) or not set(np.unique(y).tolist()).issubset({0, 1}):
                raise RuntimeError(f"invalid label cache {labels_path}")
            if cell_meta.get("epochs_sha256") != file_sha256(epochs_path):
                raise RuntimeError(f"signal hash mismatch {epochs_path}")
            if cell_meta.get("labels_sha256") != file_sha256(labels_path):
                raise RuntimeError(f"label hash mismatch {labels_path}")
            cells.append((subject.removeprefix("sub-"), session, epochs_path, labels_path, len(x)))
            total += len(x)
    if len(cells) != 123 or total != 24591:
        raise RuntimeError(f"unexpected development cache cardinality: cells={len(cells)} trials={total}")

    if SIGNAL.is_file() and METADATA.is_file():
        x = np.load(SIGNAL, mmap_mode="r", allow_pickle=False)
        frame = pd.read_parquet(METADATA)
        if x.shape != (24591, 58, 1000) or x.dtype != np.float16 or len(frame) != 24591:
            raise RuntimeError("existing consolidated cache has invalid shape")
        print("CONSOLIDATED_CACHE_ALREADY_COMPLETE")
    else:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        signal_part = SIGNAL.with_name(SIGNAL.name + ".part")
        metadata_part = METADATA.with_name(METADATA.name + ".part")
        if signal_part.exists():
            signal_part.unlink()
        output = np.lib.format.open_memmap(signal_part, mode="w+", dtype=np.float16, shape=(total, 58, 1000))
        rows: list[dict[str, Any]] = []
        cursor = 0
        for cell_index, (subject, session, epochs_path, labels_path, count) in enumerate(cells, start=1):
            x = np.load(epochs_path, mmap_mode="r", allow_pickle=False)
            y = np.load(labels_path, mmap_mode="r", allow_pickle=False).astype(np.int8)
            output[cursor:cursor + count] = x
            rows.extend(
                {
                    "subject_id": subject,
                    "session_id": session,
                    "label": int(label),
                    "trial_in_session": int(index),
                    "authorized_development_subject": True,
                }
                for index, label in enumerate(y)
            )
            cursor += count
            if cell_index == 1 or cell_index % 10 == 0 or cell_index == len(cells):
                output.flush()
                print(f"[assemble {cell_index:03d}/123] rows={cursor}", flush=True)
        del output
        frame = pd.DataFrame(rows)
        frame.to_parquet(metadata_part, index=False, engine="pyarrow")
        os.replace(signal_part, SIGNAL)
        os.replace(metadata_part, METADATA)

    frame = pd.read_parquet(METADATA)
    x = np.load(SIGNAL, mmap_mode="r", allow_pickle=False)
    observed_subjects = list(dict.fromkeys("sub-" + frame.subject_id.astype(str)))
    cells_table = frame.groupby(["subject_id", "session_id", "label"]).size()
    passed = bool(
        x.shape == (24591, 58, 1000)
        and x.dtype == np.float16
        and observed_subjects == subjects
        and set(frame.session_id.astype(int)) == {0, 1, 2}
        and set(frame.label.astype(int)) == {0, 1}
        and len(cells_table) == 41 * 3 * 2
        and int(cells_table.min()) >= 20
        and np.isfinite(np.asarray(x[::197], dtype=np.float32)).all()
    )
    payload = {
        "schema": "WBCIC_DEVELOPMENT_CONSOLIDATED_CACHE_AUDIT_V1",
        "pass": passed,
        "subject_count": 41,
        "subject_hash": sha_lines(observed_subjects),
        "session_count": 123,
        "trial_count": int(len(frame)),
        "signal_shape": list(x.shape),
        "signal_dtype": str(x.dtype),
        "signal_sha256": file_sha256(SIGNAL),
        "metadata_sha256": file_sha256(METADATA),
        "minimum_trials_per_subject_session_class": int(cells_table.min()),
        "maximum_trials_per_subject_session_class": int(cells_table.max()),
        "input_paths_constructed_from_authorized_whitelist_only": True,
        "raw_root_enumerated": False,
        "sealed_WBCIC_outer_accessed": False,
        "sealed_WBCIC_outer_enumerated": False,
        "OpenBMI_accessed": False,
    }
    write_json(AUDIT, payload)
    if not passed or payload["subject_hash"] != SUBJECT_HASH:
        raise RuntimeError(f"consolidated cache audit failed: {payload}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
