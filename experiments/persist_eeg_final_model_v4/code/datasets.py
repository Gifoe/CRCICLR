from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import V3_ROOT, logit, sigmoid


OPENBMI_KEYS = ["fold", "seed", "router_fold", "manifest_index", "subject", "session", "label"]
OPENBMI_RUNS = tuple(f"fold-{fold}_seed-{seed}" for fold in range(3) for seed in range(2))
ACTION_NAMES = ("AMPLIFY", "GEOMETRY", "ERASE")


@dataclass
class ExpertDataset:
    dataset_id: str
    trial_uid: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    labels: np.ndarray
    base_logits: np.ndarray
    keep_run_logits: np.ndarray
    keep_run_mask: np.ndarray
    action_run_logits: np.ndarray
    action_logits: np.ndarray
    persist_context: np.ndarray
    persist_names: list[str]
    folds: list[dict[str, Any]]
    metadata: pd.DataFrame

    @property
    def base_probability(self) -> np.ndarray:
        return sigmoid(self.base_logits)

    @property
    def base_prediction(self) -> np.ndarray:
        return (self.base_logits >= 0).astype(int)


def _read_openbmi_rows(cache_root: Path) -> pd.DataFrame:
    paths = {
        "base": cache_root / "OOF_BASE_LOGITS.parquet",
        "action": cache_root / "OOF_COUNTERFACTUAL_LOGITS.parquet",
        "geometry": cache_root / "OOF_GEOMETRY_FEATURES.parquet",
        "persist": cache_root / "OOF_ROUTER_FEATURES.parquet",
    }
    if not all(path.is_file() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.is_file()]
        raise RuntimeError(f"OpenBMI frozen cache is incomplete: {missing}")
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    for name, frame in frames.items():
        if frame.duplicated(OPENBMI_KEYS).any():
            raise RuntimeError(f"Duplicate OpenBMI key in {name}")
        if len(frame) != 40_800:
            raise RuntimeError(f"Unexpected OpenBMI rows in {name}: {len(frame)}")
    merged = frames["base"].merge(
        frames["action"], on=OPENBMI_KEYS, validate="one_to_one"
    ).merge(frames["geometry"], on=OPENBMI_KEYS, validate="one_to_one").merge(
        frames["persist"], on=OPENBMI_KEYS, validate="one_to_one"
    )
    merged["run_id"] = (
        "fold-" + merged.fold.astype(str) + "_seed-" + merged.seed.astype(str)
    )
    if set(merged.run_id.unique()) != set(OPENBMI_RUNS):
        raise RuntimeError("Unexpected OpenBMI frozen run identities")
    merged["keep_margin"] = merged.keep_logit_1 - merged.keep_logit_0
    merged["amplify_margin"] = merged.amplify_logit_1 - merged.amplify_logit_0
    merged["geometry_margin"] = merged.geometry_logit_1 - merged.geometry_logit_0
    merged["erase_margin"] = merged.erase_logit_1 - merged.erase_logit_0
    return merged.sort_values(["manifest_index", "fold", "seed"]).reset_index(drop=True)


def _openbmi_folds() -> list[dict[str, Any]]:
    path = V3_ROOT / "outputs" / "protocol" / "GROUPED_NESTED_CV.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = []
    for item in payload["folds"]:
        folds.append(
            {
                "outer_fold": int(item["outer_fold"]),
                "train_subjects": [str(value) for value in item["model_training_subjects"]],
                "calibration_subjects": [str(value) for value in item["calibration_subjects"]],
                "test_subjects": [str(value) for value in item["heldout_subjects"]],
            }
        )
    return folds


def load_openbmi(cache_root: Path) -> ExpertDataset:
    rows = _read_openbmi_rows(cache_root)
    router_columns = [
        column
        for column in rows.columns
        if column
        not in set(OPENBMI_KEYS)
        | {
            "keep_logit_0",
            "keep_logit_1",
            "erase_logit_0",
            "erase_logit_1",
            "amplify_logit_0",
            "amplify_logit_1",
            "geometry_logit_0",
            "geometry_logit_1",
            "run_id",
            "keep_margin",
            "amplify_margin",
            "geometry_margin",
            "erase_margin",
        }
        and pd.api.types.is_numeric_dtype(rows[column])
    ]
    forbidden = ("label", "outcome", "correct", "rescue", "harm", "target")
    offenders = [name for name in router_columns if any(token in name.lower() for token in forbidden)]
    if offenders:
        raise RuntimeError(f"Outcome-dependent OpenBMI context features: {offenders}")

    records: list[dict[str, Any]] = []
    keep_rows: list[np.ndarray] = []
    keep_masks: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    context_rows: list[np.ndarray] = []
    persist_names = [f"persist_mean_{name}" for name in router_columns] + [
        f"persist_std_{name}" for name in router_columns
    ]
    run_position = {name: index for index, name in enumerate(OPENBMI_RUNS)}
    for manifest, group in rows.groupby("manifest_index", sort=True):
        for column in ("subject", "session", "label"):
            if group[column].nunique() != 1:
                raise RuntimeError(f"OpenBMI manifest {manifest} maps to multiple {column} values")
        keep = np.full(len(OPENBMI_RUNS), np.nan, dtype=np.float64)
        action = np.full((len(ACTION_NAMES), len(OPENBMI_RUNS)), np.nan, dtype=np.float64)
        for item in group.itertuples(index=False):
            position = run_position[str(item.run_id)]
            keep[position] = float(item.keep_margin)
            action[0, position] = float(item.amplify_margin)
            action[1, position] = float(item.geometry_margin)
            action[2, position] = float(item.erase_margin)
        mask = np.isfinite(keep)
        if mask.sum() not in (2, 4, 6) or not np.array_equal(np.isfinite(action).all(axis=0), mask):
            raise RuntimeError(f"Malformed OpenBMI expert coverage for manifest {manifest}")
        context = np.concatenate(
            [
                group[router_columns].to_numpy(dtype=float).mean(axis=0),
                group[router_columns].to_numpy(dtype=float).std(axis=0),
            ]
        )
        records.append(
            {
                "manifest_index": int(manifest),
                "trial_uid": f"OpenBMI_SSVEP_NEMAR_nm000273_offline:{int(manifest)}",
                "subject_id": str(group.subject.iloc[0]),
                "session_id": str(group.session.iloc[0]),
                "label": int(group.label.iloc[0]),
                "n_runs": int(mask.sum()),
                "router_folds": "|".join(map(str, sorted(group.router_fold.astype(int).unique()))),
                "OUTER_TEST_USED": False,
            }
        )
        keep_rows.append(keep)
        keep_masks.append(mask)
        action_rows.append(action)
        context_rows.append(context)
    metadata = pd.DataFrame(records)
    keep_run_logits = np.stack(keep_rows)
    keep_run_mask = np.stack(keep_masks)
    action_run_logits = np.stack(action_rows)
    base_logits = np.nanmean(keep_run_logits, axis=1)
    action_logits = np.nanmean(action_run_logits, axis=2)
    persist_context = np.stack(context_rows)
    if len(metadata) != 10_400 or metadata.subject_id.nunique() != 52:
        raise RuntimeError("OpenBMI unique-trial coverage failed")
    return ExpertDataset(
        dataset_id="OpenBMI_SSVEP_NEMAR_nm000273_offline",
        trial_uid=metadata.trial_uid.to_numpy(dtype=str),
        subjects=metadata.subject_id.to_numpy(dtype=str),
        sessions=metadata.session_id.to_numpy(dtype=str),
        labels=metadata.label.to_numpy(dtype=int),
        base_logits=base_logits,
        keep_run_logits=keep_run_logits,
        keep_run_mask=keep_run_mask,
        action_run_logits=action_run_logits,
        action_logits=action_logits,
        persist_context=persist_context,
        persist_names=persist_names,
        folds=_openbmi_folds(),
        metadata=metadata,
    )


def openbmi_keep_pool(data: ExpertDataset) -> tuple[np.ndarray, np.ndarray, list[str]]:
    keep = data.keep_run_logits
    mask = data.keep_run_mask
    n = len(data.labels)
    candidates = [data.base_logits]
    candidate_mask = [np.ones(n, dtype=bool)]
    names = ["B6_ALL_RUN_LOGIT_MEAN"]
    probabilities = sigmoid(keep)
    b4 = logit(np.nanmean(probabilities, axis=1))
    candidates.append(b4)
    candidate_mask.append(np.ones(n, dtype=bool))
    names.append("B4_ALL_RUN_PROBABILITY_MEAN")
    vote = np.nanmean(np.where(mask, (keep >= 0).astype(float), np.nan), axis=1)
    b2 = logit(np.clip(vote, 1e-4, 1 - 1e-4))
    candidates.append(b2)
    candidate_mask.append(np.ones(n, dtype=bool))
    names.append("B2_ALL_RUN_HARD_MAJORITY")
    for position, run in enumerate(OPENBMI_RUNS):
        candidates.append(np.nan_to_num(keep[:, position], nan=0.0))
        candidate_mask.append(mask[:, position])
        names.append(f"INDIVIDUAL_{run}_KEEP")
    for position, run in enumerate(OPENBMI_RUNS):
        denominator = mask.sum(axis=1) - mask[:, position].astype(int)
        numerator = np.nansum(keep, axis=1) - np.where(mask[:, position], keep[:, position], 0.0)
        valid = mask[:, position] & (denominator > 0)
        value = np.divide(numerator, denominator, out=np.zeros(n), where=denominator > 0)
        candidates.append(value)
        candidate_mask.append(valid)
        names.append(f"LEAVE_{run}_OUT_KEEP_LOGIT_MEAN")
    return np.column_stack(candidates), np.column_stack(candidate_mask), names


def openbmi_full_pool(
    data: ExpertDataset, *, safe_alphas: tuple[float, ...] = (0.25, 0.5)
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    logits, mask, names = openbmi_keep_pool(data)
    families = ["KEEP"] * len(names)
    extra, extra_names, extra_families = [], [], []
    for action_index, action in enumerate(ACTION_NAMES):
        action_logit = data.action_logits[:, action_index]
        extra.append(action_logit)
        extra_names.append(f"ALL_{action}")
        extra_families.append(action)
        for alpha in safe_alphas:
            extra.append(data.base_logits + alpha * (action_logit - data.base_logits))
            extra_names.append(f"B6_TO_{action}_ALPHA_{alpha:g}")
            extra_families.append(f"SOFT_{action}")
    extra_array = np.column_stack(extra)
    return (
        np.column_stack([logits, extra_array]),
        np.column_stack([mask, np.ones_like(extra_array, dtype=bool)]),
        names + extra_names,
        families + extra_families,
    )


def load_wbcic_development(expert_table: Path, wbcic_repo: Path) -> ExpertDataset:
    frame = pd.read_parquet(expert_table).sort_values(
        ["outer_fold", "subject_id", "trial_index_within_subject_session"]
    ).reset_index(drop=True)
    expert_names = ("EEGNet_STABLE", "EEGNet_STD", "DeepConvNet", "EEGConformer", "TeCh")
    required = {f"margin_{name}" for name in expert_names}
    if not required.issubset(frame.columns):
        raise RuntimeError("WBCIC expert table is missing frozen experts")
    inventory = pd.read_csv(
        wbcic_repo
        / "experiments"
        / "persist_eeg_wbcic_actionability_v2"
        / "outputs"
        / "protocol"
        / "CACHE_INVENTORY.csv"
    )
    expected_s3_trials = int(inventory.loc[inventory.session.astype(int).eq(2), "n_trials"].sum())
    if len(frame) != expected_s3_trials or frame.subject_id.nunique() != 41 or frame.trial_uid.duplicated().any():
        raise RuntimeError("WBCIC development expert table coverage mismatch")
    keep = np.full((len(frame), 6), np.nan, dtype=float)
    for index, expert in enumerate(expert_names):
        keep[:, index] = frame[f"margin_{expert}"].to_numpy(dtype=float)
    mask = np.isfinite(keep)
    base = frame.margin_EEGNet_STABLE.to_numpy(dtype=float)

    scope_path = (
        wbcic_repo
        / "experiments"
        / "persist_eeg_wbcic_actionability_v2"
        / "outputs"
        / "protocol"
        / "DEVELOPMENT_SCOPE_LOCK.json"
    )
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    allowed = [str(value) for value in scope["allowed_subjects"]]
    if set(frame.subject_id.astype(str)) != set(allowed) or scope.get("outer_subject_ids_present") is not False:
        raise RuntimeError("WBCIC expert table exceeds authorized development scope")
    folds = []
    for fold in range(5):
        test = [str(value) for value in scope["folds"][f"F{fold}"]]
        calibration = [str(value) for value in scope["folds"][f"F{(fold + 1) % 5}"]]
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
    return ExpertDataset(
        dataset_id="WBCIC_NEMAR_nm000348_authorized_development_S3",
        trial_uid=frame.trial_uid.to_numpy(dtype=str),
        subjects=frame.subject_id.to_numpy(dtype=str),
        sessions=np.full(len(frame), "S3", dtype=str),
        labels=frame.label.to_numpy(dtype=int),
        base_logits=base,
        keep_run_logits=keep,
        keep_run_mask=mask,
        action_run_logits=np.empty((len(frame), 0, 6), dtype=float),
        action_logits=np.empty((len(frame), 0), dtype=float),
        persist_context=np.empty((len(frame), 0), dtype=float),
        persist_names=[],
        folds=folds,
        metadata=metadata,
    )
