from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common as c


def load_s1_s2_memmap() -> tuple[np.memmap, pd.DataFrame, Path]:
    """Open the development memmap but read signal pages only for S1/S2 later."""
    root = c.STAGE0.P3.locate_authorized_cache()
    metadata = pd.read_parquet(
        root / "WBCIC_DEVELOPMENT_MI_METADATA.parquet",
        columns=["subject_id", "session_id", "label"],
        engine="pyarrow",
    )
    metadata["subject_id"] = metadata.subject_id.astype(str)
    metadata["session_id"] = metadata.session_id.astype(int)
    metadata["label"] = metadata.label.astype(int)
    expected_subjects = set(c.target_fold_map())
    if set(metadata.subject_id.unique()) != expected_subjects:
        raise RuntimeError("cache metadata does not match the locked 41-subject pool")
    if set(metadata.session_id.unique()) != {0, 1, 2} or set(metadata.label.unique()) != {0, 1}:
        raise RuntimeError("cache metadata session/label audit failed")
    raw_path = root / "WBCIC_DEVELOPMENT_MI_RAW.npy"
    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    if raw.shape != (len(metadata), 58, 1000) or raw.dtype != np.float16:
        raise RuntimeError(f"unexpected authorized cache shape/dtype: {raw.shape} {raw.dtype}")
    return raw, metadata.reset_index(drop=True), raw_path


def main() -> None:
    c.ensure_dirs()
    lock = c.verify_feature_lock(require_committed=True)
    if lock["feature_definition_changes_after_S3_association"] is not False:
        raise RuntimeError("the feature lock is not prospective")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Stage-0.5 feature extraction is authorized only on the server GPU")

    raw_memmap, metadata, raw_path = load_s1_s2_memmap()
    rows: list[dict] = []
    signal_rows_read: list[int] = []
    lr = float(lock["adapter"]["learning_rate"])

    for fold in c.FOLDS:
        target_subjects = c.roles(fold)["outcome"]
        mask = metadata.subject_id.isin(target_subjects).to_numpy(copy=True)
        mask &= metadata.session_id.isin([0, 1]).to_numpy()
        global_indices = np.flatnonzero(mask).astype(np.int64)
        selected_metadata = metadata.iloc[global_indices].reset_index(drop=True)
        if set(selected_metadata.session_id.unique()) != {0, 1}:
            raise RuntimeError(f"fold {fold} did not select exactly S1/S2")
        if 2 in set(selected_metadata.session_id.unique()):
            raise RuntimeError("S3 signal selection is forbidden")
        # This is the only signal-array read: the explicitly selected S1/S2 rows.
        selected_array = np.asarray(raw_memmap[global_indices], dtype=np.float16).copy()
        if not np.isfinite(selected_array[:: max(1, len(selected_array) // 31)]).all():
            raise RuntimeError(f"non-finite S1/S2 signal in fold {fold}")
        signal_rows_read.extend(global_indices.tolist())
        raw = torch.from_numpy(selected_array).to(device)
        local_indices = np.arange(len(selected_metadata), dtype=np.int64)

        for backbone in c.BACKBONES:
            for seed in c.SEEDS:
                model, mean, std, _ = c.STAGE0.load_anchor(backbone, fold, seed, device)
                extracted = c.STAGE0.extract(
                    model,
                    raw,
                    selected_metadata,
                    local_indices,
                    mean,
                    std,
                    batch_size=512,
                )
                initial_weight = model.head.weight.detach().float().cpu().numpy()
                initial_bias = model.head.bias.detach().float().cpu().numpy()

                for subject in target_subjects:
                    subject_mask = extracted["subjects"].astype(str) == str(subject)
                    features = extracted["features"][subject_mask]
                    labels = extracted["labels"][subject_mask]
                    sessions = extracted["sessions"][subject_mask]
                    anchor_logits = extracted["logits"][subject_mask]
                    if set(np.unique(sessions)) != {0, 1}:
                        raise RuntimeError(f"subject {subject} feature input was not S1/S2-only")
                    s1_all = np.flatnonzero(sessions == 0)
                    relative_train, relative_validation = c.STAGE0.chronological_class_split(labels[s1_all])
                    train_idx = s1_all[relative_train]
                    validation_idx = s1_all[relative_validation]
                    s2_idx = np.flatnonzero(sessions == 1)
                    adapted = c.STAGE0.adapt_head(
                        features,
                        labels,
                        train_idx,
                        validation_idx,
                        initial_weight,
                        initial_bias,
                        lr,
                        c.STAGE0.stable_seed("SCAA-frozen-head", backbone, fold, seed, subject, lr),
                    )
                    adapted_logits = adapted["logits"]

                    anchor_margin_s1 = c.centered_correct_margin(anchor_logits[validation_idx], labels[validation_idx])
                    anchor_margin_s2 = c.centered_correct_margin(anchor_logits[s2_idx], labels[s2_idx])
                    adapted_margin_s1 = c.centered_correct_margin(adapted_logits[validation_idx], labels[validation_idx])
                    adapted_margin_s2 = c.centered_correct_margin(adapted_logits[s2_idx], labels[s2_idx])
                    effect_s1 = adapted_margin_s1 - anchor_margin_s1
                    effect_s2 = adapted_margin_s2 - anchor_margin_s2

                    precision = c.certificate_precision(
                        labels[s2_idx],
                        anchor_logits[s2_idx].argmax(axis=1),
                        adapted_logits[s2_idx].argmax(axis=1),
                        c.stable_seed("SCAA-stage05-certificate-bootstrap", backbone, fold, seed, subject),
                    )
                    rows.append(
                        {
                            "backbone": backbone,
                            "fold": fold,
                            "seed": seed,
                            "subject_id": str(subject),
                            "feature_input_sessions": "S1_validation,S2",
                            "s3_signal_rows_read": 0,
                            "n_s1_validation": int(len(validation_idx)),
                            "n_s2": int(len(s2_idx)),
                            "adaptation_effect_stability": c.class_conditioned_shift_stability(
                                effect_s1,
                                labels[validation_idx],
                                effect_s2,
                                labels[s2_idx],
                            ),
                            "decision_stability": c.class_conditioned_shift_stability(
                                anchor_margin_s1,
                                labels[validation_idx],
                                anchor_margin_s2,
                                labels[s2_idx],
                            ),
                            "representation_stability": c.class_conditioned_representation_stability(
                                features[validation_idx],
                                labels[validation_idx],
                                features[s2_idx],
                                labels[s2_idx],
                            ),
                            **precision,
                            "s1_parameter_relative_change": float(adapted["parameter_relative_change"]),
                            "s1_anchor_confidence": float(
                                c.STAGE0.softmax_np(anchor_logits[validation_idx]).max(axis=1).mean()
                            ),
                        }
                    )
                del model, extracted
                torch.cuda.empty_cache()
                print(f"[stage05-features] {backbone} fold={fold} seed={seed}", flush=True)

        del raw, selected_array
        torch.cuda.empty_cache()
        gc.collect()

    seed_frame = pd.DataFrame(rows)
    seed_frame["_subject_sort"] = seed_frame.subject_id.astype(int)
    seed_frame = seed_frame.sort_values(["backbone", "_subject_sort", "seed"]).drop(columns="_subject_sort")
    if len(seed_frame) != 41 * 2 * 3:
        raise RuntimeError(f"expected 246 seed feature rows, found {len(seed_frame)}")
    if seed_frame.groupby(["backbone", "subject_id"]).seed.nunique().ne(3).any():
        raise RuntimeError("incomplete matched-seed feature coverage")
    if seed_frame.s3_signal_rows_read.ne(0).any():
        raise RuntimeError("S3 signal contamination")

    numeric = [
        "n_s1_validation",
        "n_s2",
        "adaptation_effect_stability",
        "decision_stability",
        "representation_stability",
        "raw_delta2",
        "certificate_se",
        "certificate_snr",
        "certificate_lcb90",
        "s1_parameter_relative_change",
        "s1_anchor_confidence",
    ]
    features = seed_frame.groupby(["backbone", "fold", "subject_id"], as_index=False)[numeric].mean()
    features["identity_I"] = np.nan
    features["identity_status"] = "UNAVAILABLE_NO_LEGAL_TARGET_LEVEL_FROZEN_SCORE"
    features["feature_input_sessions"] = "S1_validation,S2"
    features["s3_feature_input_accessed"] = False
    features["seed_count"] = 3
    features["_subject_sort"] = features.subject_id.astype(int)
    features = features.sort_values(["backbone", "_subject_sort"]).drop(columns="_subject_sort")

    stage0 = pd.read_csv(c.STAGE0_RESULTS / "PER_SUBJECT_UTILITY.csv", dtype={"subject_id": str})
    stage0 = stage0[stage0.scope.isin(c.BACKBONES)][["backbone", "subject_id", "Delta_S2_BA"]]
    check = features.merge(stage0, on=["backbone", "subject_id"], how="left", validate="one_to_one")
    max_difference = float(np.max(np.abs(check.raw_delta2 - check.Delta_S2_BA)))
    if max_difference > 1e-7:
        raise RuntimeError(f"recomputed S2 certificate differs from Stage-0 by {max_difference}")

    c.write_csv(c.RUNTIME / "PER_SEED_FEATURES.csv", seed_frame)
    c.write_csv(c.RESULTS / "PER_SUBJECT_FEATURES.csv", features)
    execution = {
        "schema": "PERSIST_EEG_SCAA_RELIABILITY_STAGE05_FEATURE_EXECUTION_V1",
        "complete": True,
        "device": str(device),
        "signal_memmap": raw_path.name,
        "signal_sessions_read": [0, 1],
        "S3_signal_rows_read": 0,
        "signal_row_read_count_with_fold_repetition": len(signal_rows_read),
        "unique_signal_rows_read": len(set(signal_rows_read)),
        "feature_rows": len(features),
        "seed_feature_rows": len(seed_frame),
        "stage0_delta2_max_abs_difference": max_difference,
        "feature_protocol_lock_sha256": c.sha256(c.PROTOCOL / "RELIABILITY_FEATURE_PROTOCOL_LOCK.json"),
        "per_subject_features_sha256": c.sha256(c.RESULTS / "PER_SUBJECT_FEATURES.csv"),
    }
    c.write_json(c.RUNTIME / "FEATURE_EXTRACTION_EXECUTION.json", execution)
    print("SCAA_RELIABILITY_STAGE05_FEATURES_COMPLETE")


if __name__ == "__main__":
    main()

