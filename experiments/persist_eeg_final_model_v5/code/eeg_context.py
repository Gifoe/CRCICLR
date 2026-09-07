"""Compact label-free trial EEG descriptors for V5 competence models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from common import CACHE, PROTOCOL, default_stage0_repo, default_wbcic_repo, ensure_directories, sha256_file, write_json
from datasets import load_openbmi


SUMMARY_NAMES = ("mean", "std", "q25", "q75")


def _summaries(values: np.ndarray) -> np.ndarray:
    """Summarize a (trial, channel, feature) tensor over channels."""
    return np.concatenate(
        [
            np.mean(values, axis=1),
            np.std(values, axis=1),
            np.quantile(values, 0.25, axis=1),
            np.quantile(values, 0.75, axis=1),
        ],
        axis=1,
    )


def compact_epoch_features(epochs: np.ndarray, sfreq: float = 250.0) -> np.ndarray:
    epochs = np.asarray(epochs, dtype=np.float64)
    frequencies, psd = welch(
        epochs,
        fs=float(sfreq),
        nperseg=min(256, epochs.shape[-1]),
        noverlap=min(128, epochs.shape[-1] // 2),
        axis=-1,
        detrend="constant",
    )
    psd = np.maximum(psd, np.finfo(float).tiny)
    passband = (frequencies >= 4.0) & (frequencies <= 45.0)
    total = np.trapezoid(psd[..., passband], frequencies[passband], axis=-1)
    total = np.maximum(total, np.finfo(float).tiny)
    bands = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))
    band_power = []
    for low, high in bands:
        mask = (frequencies >= low) & (frequencies < high if high < 45 else frequencies <= high)
        band_power.append(np.trapezoid(psd[..., mask], frequencies[mask], axis=-1))
    band_power = np.maximum(np.stack(band_power, axis=-1), np.finfo(float).tiny)
    spectral_probability = psd[..., passband] / np.maximum(
        psd[..., passband].sum(axis=-1, keepdims=True), np.finfo(float).tiny
    )
    spectral_entropy = -np.sum(
        spectral_probability * np.log(np.maximum(spectral_probability, np.finfo(float).tiny)),
        axis=-1,
    )
    spectral_channel = np.concatenate(
        [
            np.log(band_power),
            band_power / total[..., None],
            np.log(total)[..., None],
            spectral_entropy[..., None],
        ],
        axis=-1,
    )

    centered = epochs - epochs.mean(axis=-1, keepdims=True)
    variance = np.var(centered, axis=-1, ddof=1)
    rms = np.sqrt(np.mean(centered**2, axis=-1))
    first = np.diff(centered, axis=-1)
    second = np.diff(first, axis=-1)
    first_var = np.var(first, axis=-1, ddof=1)
    second_var = np.var(second, axis=-1, ddof=1)
    mobility = np.sqrt(first_var / np.maximum(variance, np.finfo(float).tiny))
    complexity = np.sqrt(second_var / np.maximum(first_var, np.finfo(float).tiny)) / np.maximum(
        mobility, np.finfo(float).tiny
    )
    temporal_channel = np.stack(
        [
            np.log(np.maximum(variance, np.finfo(float).tiny)),
            np.log(np.maximum(rms, np.finfo(float).tiny)),
            mobility,
            complexity,
            np.nan_to_num(skew(centered, axis=-1, bias=False)),
            np.nan_to_num(kurtosis(centered, axis=-1, bias=False)),
        ],
        axis=-1,
    )

    covariance = np.einsum("bct,bdt->bcd", centered, centered, optimize=True) / centered.shape[-1]
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), np.finfo(float).tiny)
    normalized_eigenvalues = eigenvalues / eigenvalues.sum(axis=1, keepdims=True)
    effective_rank = np.exp(-np.sum(normalized_eigenvalues * np.log(normalized_eigenvalues), axis=1))
    eigen_summary = np.column_stack(
        [
            np.log(eigenvalues[:, -1]),
            np.log(eigenvalues[:, -3:].mean(axis=1)),
            np.log(np.median(eigenvalues, axis=1)),
            np.log(eigenvalues[:, :3].mean(axis=1)),
            np.log(eigenvalues.sum(axis=1)),
            np.log(eigenvalues[:, -1] / eigenvalues[:, 0]),
            effective_rank,
        ]
    )
    result = np.concatenate(
        [_summaries(spectral_channel), _summaries(temporal_channel), eigen_summary], axis=1
    )
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite compact EEG features")
    return result.astype(np.float32)


def extract_wbcic() -> Path:
    ensure_directories()
    all_sessions_path = CACHE / "WBCIC_DEV_ALL_SESSION_EXPERTS.parquet"
    frame = pd.read_parquet(all_sessions_path)
    frame = frame.loc[frame.session_id.astype(int).eq(2)].copy()
    source = (
        default_wbcic_repo()
        / "experiments"
        / "persist_eeg_wbcic_actionability_v2"
        / "outputs"
        / "cache"
        / "wbcic_epochs"
    )
    chunks, uids = [], []
    for subject, group in frame.groupby("subject_id", sort=False):
        group = group.sort_values("trial_index_within_subject_session")
        path = source / str(subject) / "ses-2_epochs.npy"
        epochs = np.load(path, mmap_mode="r", allow_pickle=False)
        indices = group.trial_index_within_subject_session.to_numpy(int)
        if indices.min() != 0 or indices.max() >= len(epochs) or len(np.unique(indices)) != len(indices):
            raise RuntimeError(f"WBCIC epoch alignment failed for {subject}")
        chunks.append(compact_epoch_features(np.asarray(epochs[indices]), sfreq=250.0))
        uids.extend(group.trial_uid.astype(str).tolist())
        print(f"[WBCIC compact context] subject={subject} trials={len(group)}", flush=True)
    values = np.concatenate(chunks).astype(np.float32)
    values_path = CACHE / "WBCIC_S3_COMPACT_EEG_CONTEXT.npy"
    metadata_path = CACHE / "WBCIC_S3_COMPACT_EEG_CONTEXT_METADATA.parquet"
    np.save(values_path, values, allow_pickle=False)
    pd.DataFrame({"trial_uid": uids, "OUTER_TEST_USED": False}).to_parquet(metadata_path, index=False)
    write_json(
        PROTOCOL / "WBCIC_EEG_CONTEXT_AUDIT.json",
        {
            "status": "WBCIC_COMPACT_EEG_CONTEXT_COMPLETE",
            "trials": len(values),
            "dimensions": values.shape[1],
            "values_sha256": sha256_file(values_path),
            "metadata_sha256": sha256_file(metadata_path),
            "labels_used": False,
            "target_batch_adaptation": False,
            "outer_subject_ids_loaded": False,
            "OUTER_TEST_USED": False,
        },
    )
    return values_path


def extract_openbmi() -> Path:
    ensure_directories()
    data = load_openbmi()
    stage0 = default_stage0_repo() / "outputs" / "persist_eeg_stage0"
    feature_root = stage0 / "features" / "handcrafted_base"
    indices = data.metadata.manifest_index.to_numpy(int)
    spectral = np.asarray(np.load(feature_root / "spectral.npy", mmap_mode="r")[indices])
    temporal = np.asarray(np.load(feature_root / "temporal.npy", mmap_mode="r")[indices])
    spectral = spectral.reshape(len(indices), 62, -1)
    temporal = temporal.reshape(len(indices), 62, -1)
    values = np.concatenate([_summaries(spectral), _summaries(temporal)], axis=1).astype(np.float32)
    values_path = CACHE / "OPENBMI_COMPACT_EEG_CONTEXT.npy"
    metadata_path = CACHE / "OPENBMI_COMPACT_EEG_CONTEXT_METADATA.parquet"
    np.save(values_path, values, allow_pickle=False)
    pd.DataFrame({"trial_uid": data.trial_uid, "OUTER_TEST_USED": False}).to_parquet(metadata_path, index=False)
    write_json(
        PROTOCOL / "OPENBMI_EEG_CONTEXT_AUDIT.json",
        {
            "status": "OPENBMI_COMPACT_EEG_CONTEXT_COMPLETE",
            "trials": len(values),
            "dimensions": values.shape[1],
            "values_sha256": sha256_file(values_path),
            "metadata_sha256": sha256_file(metadata_path),
            "labels_used": False,
            "target_batch_adaptation": False,
            "OUTER_TEST_USED": False,
        },
    )
    return values_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("both", "wbcic", "openbmi"), default="both")
    args = parser.parse_args()
    if args.dataset in {"both", "wbcic"}:
        extract_wbcic()
    if args.dataset in {"both", "openbmi"}:
        extract_openbmi()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
