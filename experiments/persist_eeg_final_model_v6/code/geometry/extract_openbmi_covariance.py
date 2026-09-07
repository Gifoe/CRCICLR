"""Build motor-channel filter-bank spectral covariance for OpenBMI MI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import CACHE, PROTOCOL, ensure_directories, sha256_file, stage0_root, write_json


CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8", "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10", "FC3", "FC4",
    "C5", "C1", "C2", "C6", "CP3", "CPz", "CP4", "P1", "P2", "POz", "FT9",
    "FTT9h", "TTP7h", "TP7", "TPP9h", "FT10", "FTT10h", "TPP8h", "TP8", "TPP10h",
    "F9", "F10", "AF7", "AF3", "AF4", "AF8", "PO3", "PO4",
)
MOTOR_CHANNELS = (
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz",
    "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
)
BANDS = ((4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40))


def run(device_name: str, force: bool) -> None:
    ensure_directories()
    root = stage0_root()
    old = root / "outputs" / "persist_eeg_stage0"
    manifest_path = old / "manifests" / "openbmi_trials.parquet"
    audit_path = old / "manifests" / "openbmi_recordings_audit.parquet"
    manifest = pd.read_parquet(manifest_path)
    metadata = manifest.loc[manifest.paradigm.astype(str).eq("mi")].copy().reset_index(drop=True)
    audit = pd.read_parquet(audit_path)
    audit_channels = json.loads(audit.channel_names.iloc[0])
    if list(audit_channels[:62]) != list(CHANNELS) or len(metadata) != 10_800:
        raise RuntimeError("OpenBMI raw geometry source mismatch")
    output_path = CACHE / "OPENBMI_MI_MOTOR20_FB_COVARIANCE.npy"
    metadata_path = CACHE / "OPENBMI_MI_MOTOR20_FB_METADATA.parquet"
    if not force and output_path.is_file() and metadata_path.is_file():
        existing = np.load(output_path, mmap_mode="r", allow_pickle=False)
        if existing.shape == (10_800, len(BANDS), len(MOTOR_CHANNELS), len(MOTOR_CHANNELS)):
            print("[geometry cache] already complete", flush=True)
            return
    indices = [CHANNELS.index(name) for name in MOTOR_CHANNELS]
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(metadata), len(BANDS), len(indices), len(indices)),
    )
    frequencies = torch.fft.rfftfreq(1000, d=1.0 / 250.0, device=device)
    window = torch.hann_window(1000, periodic=False, device=device).view(1, 1, -1)
    for shard_index, (relative, rows) in enumerate(metadata.groupby("signal_cache_path", sort=False), start=1):
        path = root / str(relative)
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        cache_indices = rows.cache_index.to_numpy(int)
        x = torch.as_tensor(np.asarray(raw[cache_indices][:, indices], dtype=np.float32), device=device)
        x = (x - x.mean(dim=-1, keepdim=True)) * window
        spectrum = torch.fft.rfft(x, dim=-1)
        band_values = []
        for lower, upper in BANDS:
            mask = (frequencies >= lower) & (frequencies < upper)
            value = spectrum[:, :, mask]
            covariance = torch.einsum("bcf,bdf->bcd", value, value.conj()).real
            trace = torch.diagonal(covariance, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1e-12)
            covariance = covariance / trace[:, None, None]
            band_values.append(covariance.float().cpu().numpy())
        output[rows.index.to_numpy(int)] = np.stack(band_values, axis=1)
        if shard_index % 18 == 0:
            output.flush()
            print(f"[geometry cache] shards={shard_index}/108", flush=True)
    output.flush()
    del output
    metadata["subject_id"] = metadata.subject_id.astype(str)
    metadata["session_id"] = metadata.session_id.astype(int)
    metadata["label"] = metadata.event_label.astype(str).map({"left_hand": 0, "right_hand": 1}).astype(int)
    metadata["trial_uid"] = "OpenBMI_nm000273_MI:" + metadata.trial_id.astype(str)
    metadata["OUTER_TEST_USED"] = False
    metadata.to_parquet(metadata_path, index=False)
    values = np.load(output_path, mmap_mode="r", allow_pickle=False)
    if values.shape != (10_800, 9, 20, 20) or not np.isfinite(values).all():
        raise RuntimeError("Nonfinite OpenBMI geometry cache")
    write_json(
        PROTOCOL / "OPENBMI_GEOMETRY_CACHE_AUDIT.json",
        {
            "status": "COMPLETE",
            "rows": len(metadata),
            "shape": list(values.shape),
            "channels": list(MOTOR_CHANNELS),
            "bands_hz": [list(item) for item in BANDS],
            "covariance": "Hann-windowed real cross-spectrum, trace normalized",
            "source_manifest_sha256": sha256_file(manifest_path),
            "cache_sha256": sha256_file(output_path),
            "metadata_sha256": sha256_file(metadata_path),
            "future_labels_used_to_construct_covariance": False,
            "OUTER_TEST_USED": False,
        },
    )
    print("[geometry cache] complete", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.device, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
