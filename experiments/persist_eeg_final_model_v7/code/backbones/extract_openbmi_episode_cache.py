"""Extract fold-compatible MI-specific representations for V7.

The five encoders are the frozen V6 outcome-fold checkpoints.  No target
future label is used here; labels are copied only for later legal episode
construction and scoring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from backbones.openbmi import build
from common import CACHE, PROTOCOL, ensure_directories, sha256_file, stage0_root, v6_outputs, write_json


OPENBMI_BEST_EPOCHS = (54, 54, 25, 47, 44)


def _normalizer(fold: int) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = (
        stage0_root() / "outputs" / "persist_eeg_p2p3" / "backbone" / "checkpoints"
        / "eegnet" / f"fold-{fold}" / "seed-0" / "trajectory"
        / f"epoch-{OPENBMI_BEST_EPOCHS[fold]:03d}.pt"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return np.asarray(payload["channel_mean"], dtype=np.float32), np.asarray(payload["channel_std"], dtype=np.float32)


def run() -> None:
    ensure_directories()
    manifest_path = stage0_root() / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet"
    manifest = pd.read_parquet(manifest_path)
    manifest = manifest.loc[manifest.paradigm.astype(str).str.lower().eq("mi")].copy()
    manifest["subject_id"] = manifest.subject_id.astype(str)
    manifest["session_id"] = manifest.session_id.astype(int)
    manifest["label"] = manifest.event_label.astype(str).map({"left_hand": 0, "right_hand": 1}).astype(int)
    manifest["trial_uid"] = "OpenBMI_nm000273_MI:" + manifest.trial_id.astype(str)
    manifest = manifest.sort_values(["subject_id", "session_id", "cache_index"]).reset_index(drop=True)
    if len(manifest) != 10_800 or manifest.trial_uid.duplicated().any():
        raise RuntimeError("OpenBMI MI manifest coverage failure")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = stage0_root()
    audit = []
    for fold in range(5):
        checkpoint = v6_outputs() / "cache" / f"OPENBMI_MI_SPECIFIC_BACKBONE_FOLD_{fold}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("OUTER_TEST_USED") is not False:
            raise RuntimeError("V6 checkpoint outer flag is not false")
        model = build(payload["configuration"]).to(device)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        mean_np, std_np = _normalizer(fold)
        mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)[None, :, None]
        std = torch.clamp(torch.as_tensor(std_np, dtype=torch.float32, device=device)[None, :, None], min=1e-8)
        features = np.empty((len(manifest), 64), dtype=np.float32)
        logits = np.empty(len(manifest), dtype=np.float32)
        cursor = 0
        for relative, group in manifest.groupby("signal_cache_path", sort=False):
            source = np.load(root / str(relative), mmap_mode="r", allow_pickle=False)
            indices = group.cache_index.to_numpy(dtype=int)
            values = np.asarray(source[indices], dtype=np.float32)
            for start in range(0, len(values), 128):
                stop = min(start + 128, len(values))
                x = torch.as_tensor(values[start:stop], dtype=torch.float32, device=device)
                x = (x - mean) / std
                with torch.inference_mode():
                    feature = model.forward_features(x)
                    output = model.head(feature)
                count = stop - start
                features[cursor:cursor + count] = feature.cpu().numpy()
                logits[cursor:cursor + count] = (output[:, 1] - output[:, 0]).cpu().numpy()
                cursor += count
        if cursor != len(manifest) or not np.isfinite(features).all() or not np.isfinite(logits).all():
            raise RuntimeError(f"OpenBMI extraction failure fold {fold}")
        prefix = CACHE / f"OPENBMI_MI_SPECIFIC_FOLD_{fold}"
        feature_path = prefix.with_name(prefix.name + "_FEATURES.npy")
        logit_path = prefix.with_name(prefix.name + "_LOGITS.npy")
        metadata_path = prefix.with_name(prefix.name + "_METADATA.parquet")
        np.save(feature_path, features, allow_pickle=False)
        np.save(logit_path, logits, allow_pickle=False)
        metadata = manifest[["subject_id", "session_id", "label", "trial_uid", "trial_id"]].copy()
        metadata["outer_fold"] = fold
        metadata["target_future_label_used_for_fit"] = False
        metadata["OUTER_TEST_USED"] = False
        metadata.to_parquet(metadata_path, index=False)
        audit.append({
            "fold": fold,
            "rows": len(metadata),
            "feature_dimension": 64,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "features_sha256": sha256_file(feature_path),
            "logits_sha256": sha256_file(logit_path),
            "metadata_sha256": sha256_file(metadata_path),
            "labels_used_for_extraction": False,
            "OUTER_TEST_USED": False,
        })
        print(f"[OpenBMI V7 cache] fold={fold} rows={cursor}", flush=True)
    write_json(PROTOCOL / "OPENBMI_EPISODE_CACHE_AUDIT.json", {"folds": audit, "OUTER_TEST_USED": False})


if __name__ == "__main__":
    run()
