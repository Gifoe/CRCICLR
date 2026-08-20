"""Build contiguous authorized-development raw caches and legal history EA matrices."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import CACHE, PROTOCOL, ensure_directories, sha256_file, stage0_root, v6_outputs, wbcic_source_root, write_json


def _natural_subject_key(value: str) -> tuple[int, str]:
    digits = "".join(character for character in str(value) if character.isdigit())
    return (int(digits) if digits else 10**9, str(value))


def _openbmi() -> tuple[np.ndarray, pd.DataFrame, tuple[int, ...], str]:
    root = stage0_root()
    manifest = pd.read_parquet(root / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet")
    frame = manifest.loc[manifest.paradigm.astype(str).str.lower().eq("mi")].copy()
    frame["subject_id"] = frame.subject_id.astype(str)
    frame["session_id"] = frame.session_id.astype(int)
    frame["label"] = frame.event_label.astype(str).map({"left_hand": 0, "right_hand": 1}).astype(int)
    frame["trial_uid"] = "OpenBMI_nm000273_MI:" + frame.trial_id.astype(str)
    frame = frame.sort_values(["subject_id", "session_id", "cache_index"]).reset_index(drop=True)
    path = CACHE / "OPENBMI_RAW_EPOCHS_FLOAT16.npy"
    target = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(frame), 62, 1000))
    for relative, group in frame.groupby("signal_cache_path", sort=False):
        source = np.load(root / str(relative), mmap_mode="r", allow_pickle=False)
        target[group.index.to_numpy(int)] = np.asarray(source[group.cache_index.to_numpy(int)], dtype=np.float16)
    target.flush()
    del target
    metadata = frame[["subject_id", "session_id", "label", "trial_uid"]].copy()
    return np.load(path, mmap_mode="r", allow_pickle=False), metadata, (1,), "OpenBMI_MI_S1_to_S2"


def _wbcic() -> tuple[np.ndarray, pd.DataFrame, tuple[int, ...], str]:
    metadata_path = v6_outputs() / "cache" / "WBCIC_SHARED_FOLD_0_EEGNET_STABLE_ALL_SESSION_METADATA.parquet"
    metadata = pd.read_parquet(metadata_path).copy().reset_index(drop=True)
    if metadata.OUTER_TEST_USED.astype(bool).any() or metadata.subject_id.nunique() != 41:
        raise RuntimeError("Unauthorized or malformed WBCIC development metadata")
    path = CACHE / "WBCIC_RAW_EPOCHS_FLOAT16.npy"
    target = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(metadata), 58, 1000))
    source_root = wbcic_source_root() / "experiments" / "persist_eeg_wbcic_actionability_v2" / "outputs" / "cache" / "wbcic_epochs"
    for (subject, session), group in metadata.groupby(["subject_id", "session_id"], sort=False):
        source = np.load(source_root / str(subject) / f"ses-{int(session) - 1}_epochs.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(source_root / str(subject) / f"ses-{int(session) - 1}_labels.npy", allow_pickle=False).astype(int)
        trial_index = group.trial_index_within_subject_session.to_numpy(int)
        if not np.array_equal(labels[trial_index], group.label.to_numpy(int)):
            raise RuntimeError(f"WBCIC raw/metadata label mismatch {subject} S{session}")
        target[group.index.to_numpy(int)] = np.asarray(source[trial_index], dtype=np.float16)
    target.flush()
    del target
    return np.load(path, mmap_mode="r", allow_pickle=False), metadata[["subject_id", "session_id", "label", "trial_uid"]].copy(), (1, 2), "WBCIC_S1S2_to_S3_authorized_development"


def _alignment(raw: np.ndarray, metadata: pd.DataFrame, history_sessions: tuple[int, ...]) -> tuple[np.ndarray, dict[str, int]]:
    subjects = sorted(metadata.subject_id.astype(str).unique(), key=_natural_subject_key)
    mapping = {subject: index for index, subject in enumerate(subjects)}
    matrices = np.empty((len(subjects), raw.shape[1], raw.shape[1]), dtype=np.float32)
    for subject in subjects:
        mask = metadata.subject_id.astype(str).eq(subject).to_numpy() & metadata.session_id.astype(int).isin(history_sessions).to_numpy()
        indices = np.flatnonzero(mask)
        if len(indices) > 100:
            indices = indices[np.linspace(0, len(indices) - 1, 100).round().astype(int)]
        values = np.asarray(raw[indices, :, ::4], dtype=np.float32)
        values -= values.mean(axis=2, keepdims=True)
        covariance = np.einsum("nct,ndt->ncd", values, values, optimize=True) / values.shape[2]
        trace = np.trace(covariance, axis1=1, axis2=2)
        covariance /= np.maximum(trace[:, None, None], 1e-20)
        reference = covariance.mean(axis=0)
        reference = 0.5 * (reference + reference.T)
        eigenvalue, eigenvector = np.linalg.eigh(reference.astype(np.float64))
        floor = max(float(np.mean(eigenvalue)) * 1e-4, 1e-12)
        inverse_root = (eigenvector * (1.0 / np.sqrt(np.maximum(eigenvalue, floor)))[None, :]) @ eigenvector.T
        matrices[mapping[subject]] = inverse_root.astype(np.float32)
    return matrices, mapping


def _run_one(key: str) -> dict:
    if key == "openbmi":
        raw, metadata, history_sessions, benchmark = _openbmi()
        raw_path = CACHE / "OPENBMI_RAW_EPOCHS_FLOAT16.npy"
        metadata_path = CACHE / "OPENBMI_RAW_METADATA.parquet"
        alignment_path = CACHE / "OPENBMI_HISTORY_EA_MATRICES.npy"
    else:
        raw, metadata, history_sessions, benchmark = _wbcic()
        raw_path = CACHE / "WBCIC_RAW_EPOCHS_FLOAT16.npy"
        metadata_path = CACHE / "WBCIC_RAW_METADATA.parquet"
        alignment_path = CACHE / "WBCIC_HISTORY_EA_MATRICES.npy"
    matrices, mapping = _alignment(raw, metadata, history_sessions)
    metadata["subject_index"] = metadata.subject_id.astype(str).map(mapping).astype(int)
    metadata["target_future_label_used_for_fit"] = False
    metadata["OUTER_TEST_USED"] = False
    metadata.to_parquet(metadata_path, index=False)
    np.save(alignment_path, matrices, allow_pickle=False)
    if not np.isfinite(matrices).all() or metadata.trial_uid.duplicated().any():
        raise RuntimeError(f"{key} raw EA cache integrity failure")
    return {
        "benchmark": benchmark,
        "rows": len(metadata),
        "channels": int(raw.shape[1]),
        "samples": int(raw.shape[2]),
        "subjects": int(metadata.subject_id.nunique()),
        "history_sessions_for_alignment": list(history_sessions),
        "raw_dtype": str(raw.dtype),
        "raw_sha256": sha256_file(raw_path),
        "metadata_sha256": sha256_file(metadata_path),
        "alignment_sha256": sha256_file(alignment_path),
        "future_data_used_for_alignment": False,
        "future_labels_used_for_alignment": False,
        "OUTER_TEST_USED": False,
    }


def run() -> None:
    ensure_directories()
    payload = {key: _run_one(key) for key in ("openbmi", "wbcic")}
    payload["OUTER_TEST_USED"] = False
    write_json(PROTOCOL / "RAW_EA_CACHE_AUDIT.json", payload)
    print(payload, flush=True)


if __name__ == "__main__":
    run()
