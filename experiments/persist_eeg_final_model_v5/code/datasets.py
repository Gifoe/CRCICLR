from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import V4_ROOT, default_stage0_repo, default_wbcic_repo, logit, sigmoid


OPENBMI_RUNS = tuple(f"fold-{fold}_seed-{seed}" for fold in range(3) for seed in range(2))
WBCIC_EXPERTS = ("EEGNet_STABLE", "EEGNet_STD", "DeepConvNet", "EEGConformer", "TeCh")


@dataclass
class V5Dataset:
    dataset_id: str
    trial_uid: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    labels: np.ndarray
    expert_logits: np.ndarray
    expert_mask: np.ndarray
    expert_names: list[str]
    static_probability: np.ndarray
    static_prediction: np.ndarray
    current_probability: np.ndarray
    current_prediction: np.ndarray
    folds: list[dict[str, Any]]
    metadata: pd.DataFrame


def _load_openbmi_current_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = V4_ROOT / "outputs" / "diagnostics" / "FINAL_ABLATION_OOF_PREDICTIONS.csv"
    frame = pd.read_csv(path)
    if frame.OUTER_TEST_USED.astype(bool).any():
        raise RuntimeError("V4 OpenBMI predictions claim outer use")
    static = frame.loc[frame.method_id.eq("A0_STATIC_B_STRONG")].copy()
    current = frame.loc[frame.method_id.eq("A1_DYNAMIC_KEEP_FINAL")].copy()
    if len(static) != 10_400 or len(current) != 10_400:
        raise RuntimeError("V4 OpenBMI baseline prediction coverage mismatch")
    return static, current


def load_openbmi() -> V5Dataset:
    cache = (
        default_stage0_repo()
        / "experiments"
        / "persist_eeg_router"
        / "outputs"
        / "cache"
    )
    rows = pd.read_parquet(cache / "OOF_BASE_LOGITS.parquet")
    rows["run_id"] = "fold-" + rows.fold.astype(str) + "_seed-" + rows.seed.astype(str)
    rows["margin"] = rows.keep_logit_1 - rows.keep_logit_0
    if len(rows) != 40_800 or set(rows.run_id) != set(OPENBMI_RUNS):
        raise RuntimeError("OpenBMI V4 expert cache mismatch")
    run_index = {name: index for index, name in enumerate(OPENBMI_RUNS)}
    metadata_rows: list[dict[str, Any]] = []
    logits_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    for manifest_index, group in rows.groupby("manifest_index", sort=True):
        values = np.full(len(OPENBMI_RUNS), np.nan, dtype=float)
        for item in group.itertuples(index=False):
            values[run_index[str(item.run_id)]] = float(item.margin)
        mask = np.isfinite(values)
        if mask.sum() not in (2, 4, 6):
            raise RuntimeError(f"Malformed OpenBMI expert coverage: {manifest_index}")
        metadata_rows.append(
            {
                "manifest_index": int(manifest_index),
                "trial_uid": f"OpenBMI_SSVEP_NEMAR_nm000273_offline:{int(manifest_index)}",
                "subject_id": str(group.subject.iloc[0]),
                "session_id": str(group.session.iloc[0]),
                "label": int(group.label.iloc[0]),
                "OUTER_TEST_USED": False,
            }
        )
        logits_rows.append(values)
        mask_rows.append(mask)
    metadata = pd.DataFrame(metadata_rows)
    expert_logits = np.stack(logits_rows)
    expert_mask = np.stack(mask_rows)
    static_probability = np.nanmean(sigmoid(expert_logits), axis=1)
    static_prediction = (np.nanmean(expert_logits, axis=1) >= 0).astype(int)
    static_v4, current_v4 = _load_openbmi_current_predictions()
    static_v4 = static_v4.set_index("trial_uid").loc[metadata.trial_uid].reset_index()
    current_v4 = current_v4.set_index("trial_uid").loc[metadata.trial_uid].reset_index()
    if not np.array_equal(current_v4.label.to_numpy(int), metadata.label.to_numpy(int)):
        raise RuntimeError("OpenBMI V4 prediction alignment failed")
    # Preserve the exact V4 static definition (logit mean), not probability mean.
    static_probability = static_v4.probability.to_numpy(float)
    static_prediction = static_v4.prediction.to_numpy(int)
    current_probability = current_v4.probability.to_numpy(float)
    current_prediction = current_v4.prediction.to_numpy(int)
    fold_path = (
        V4_ROOT.parent
        / "persist_eeg_residual_actionability_v3"
        / "outputs"
        / "protocol"
        / "GROUPED_NESTED_CV.json"
    )
    payload = json.loads(fold_path.read_text(encoding="utf-8"))
    folds = [
        {
            "outer_fold": int(item["outer_fold"]),
            "train_subjects": list(map(str, item["model_training_subjects"])),
            "calibration_subjects": list(map(str, item["calibration_subjects"])),
            "test_subjects": list(map(str, item["heldout_subjects"])),
        }
        for item in payload["folds"]
    ]
    if len(metadata) != 10_400 or metadata.subject_id.nunique() != 52:
        raise RuntimeError("OpenBMI unique-trial coverage mismatch")
    return V5Dataset(
        dataset_id="OpenBMI_MI_NEMAR_nm000273_offline",
        trial_uid=metadata.trial_uid.to_numpy(str),
        subjects=metadata.subject_id.to_numpy(str),
        sessions=metadata.session_id.to_numpy(str),
        labels=metadata.label.to_numpy(int),
        expert_logits=expert_logits,
        expert_mask=expert_mask,
        expert_names=list(OPENBMI_RUNS),
        static_probability=static_probability,
        static_prediction=static_prediction,
        current_probability=current_probability,
        current_prediction=current_prediction,
        folds=folds,
        metadata=metadata,
    )


def load_wbcic() -> V5Dataset:
    table_path = V4_ROOT / "outputs" / "cache" / "WBCIC_DEV_KEEP_EXPERTS.parquet"
    frame = pd.read_parquet(table_path).sort_values(
        ["outer_fold", "subject_id", "trial_index_within_subject_session"]
    ).reset_index(drop=True)
    expert_logits = frame[[f"margin_{name}" for name in WBCIC_EXPERTS]].to_numpy(float)
    expert_mask = np.isfinite(expert_logits)
    if not expert_mask.all():
        raise RuntimeError("WBCIC V4 expert table has missing values")
    static_probability = sigmoid(expert_logits).mean(axis=1)
    static_prediction = (static_probability >= 0.5).astype(int)
    pred_path = V4_ROOT / "outputs" / "diagnostics" / "WBCIC_DEV_KEEP_SEARCH_OOF_PREDICTIONS.csv"
    predictions = pd.read_csv(pred_path)
    if predictions.OUTER_TEST_USED.astype(bool).any():
        raise RuntimeError("V4 WBCIC predictions claim outer use")
    current = predictions.loc[predictions.method_id.eq("W1_RAW_LINEAR")].copy()
    if len(current) != len(frame):
        raise RuntimeError("WBCIC current baseline prediction coverage mismatch")
    current = current.set_index("trial_uid").loc[frame.trial_uid].reset_index()
    if not np.array_equal(current.label.to_numpy(int), frame.label.to_numpy(int)):
        raise RuntimeError("WBCIC V4 current prediction alignment failed")

    scope_path = (
        default_wbcic_repo()
        / "experiments"
        / "persist_eeg_wbcic_actionability_v2"
        / "outputs"
        / "protocol"
        / "DEVELOPMENT_SCOPE_LOCK.json"
    )
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    allowed = list(map(str, scope["allowed_subjects"]))
    if (
        len(allowed) != 41
        or scope.get("outer_subject_ids_present") is not False
        or set(frame.subject_id.astype(str)) != set(allowed)
    ):
        raise RuntimeError("WBCIC development scope violation")
    folds = []
    for fold in range(5):
        test = list(map(str, scope["folds"][f"F{fold}"]))
        calibration = list(map(str, scope["folds"][f"F{(fold + 1) % 5}"]))
        train = sorted(set(allowed) - set(test) - set(calibration))
        folds.append(
            {
                "outer_fold": fold,
                "train_subjects": train,
                "calibration_subjects": calibration,
                "test_subjects": test,
            }
        )
    metadata = frame[
        ["trial_uid", "subject_id", "session_id", "label", "outer_fold", "trial_index_within_subject_session"]
    ].copy()
    metadata["OUTER_TEST_USED"] = False
    if len(frame) != 8_195 or frame.subject_id.nunique() != 41 or frame.trial_uid.duplicated().any():
        raise RuntimeError("WBCIC S3 coverage mismatch")
    return V5Dataset(
        dataset_id="WBCIC_NEMAR_nm000348_authorized_development_S3",
        trial_uid=frame.trial_uid.to_numpy(str),
        subjects=frame.subject_id.to_numpy(str),
        sessions=np.full(len(frame), "S3", dtype="U2"),
        labels=frame.label.to_numpy(int),
        expert_logits=expert_logits,
        expert_mask=expert_mask,
        expert_names=list(WBCIC_EXPERTS),
        static_probability=static_probability,
        static_prediction=static_prediction,
        current_probability=current.probability.to_numpy(float),
        current_prediction=current.prediction.to_numpy(int),
        folds=folds,
        metadata=metadata,
    )
