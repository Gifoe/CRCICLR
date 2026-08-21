from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common import CACHE, load_config, sha256_lines, stable_uint64, subject_code, subject_sort, write_json


def stage0_root() -> Path:
    return Path(os.environ.get(
        "PERSIST_STAGE0_REPO",
        r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full",
    ))


MANIFEST_COLUMNS = [
    "subject_id", "session_id", "paradigm", "trial_id", "event_label",
    "sampling_rate", "n_channels", "n_times", "signal_cache_path", "cache_index",
]


@dataclass(frozen=True)
class DevelopmentSplit:
    fold: int
    model_fit_subjects: tuple[str, ...]
    calibration_subjects: tuple[str, ...]
    outcome_subjects: tuple[str, ...]
    original_train_subjects: tuple[str, ...]
    source_json_sha256: str

    @property
    def allowed_subjects(self) -> tuple[str, ...]:
        return self.original_train_subjects + self.outcome_subjects

    def payload(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "model_fit_subjects": list(self.model_fit_subjects),
            "calibration_subjects": list(self.calibration_subjects),
            "outcome_subjects": list(self.outcome_subjects),
            "original_train_subjects": list(self.original_train_subjects),
            "model_fit_subjects_sha256": sha256_lines(self.model_fit_subjects),
            "calibration_subjects_sha256": sha256_lines(self.calibration_subjects),
            "outcome_subjects_sha256": sha256_lines(self.outcome_subjects),
            "allowed_subjects_sha256": sha256_lines(self.allowed_subjects),
            "source_split_json_sha256": self.source_json_sha256,
            "outer_split_field_read": False,
            "outer_test_used": False,
        }


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_development_split(fold: int) -> DevelopmentSplit:
    config = load_config()
    if int(fold) not in set(map(int, config["development_folds"])):
        raise ValueError(f"Fold {fold} is not development-authorized")
    path = stage0_root() / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json"
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    folds = payload["openbmi"]["folds"]
    record = next(item for item in folds if int(item["fold"]) == int(fold))
    # Deliberately access only the two development fields. The outer field is
    # neither read nor hashed by this experiment.
    original_train = tuple(subject_sort(record["train_subjects"]))
    outcome = tuple(subject_sort(record["validation_subjects"]))
    if len(original_train) != 34 or len(outcome) != 9 or set(original_train) & set(outcome):
        raise RuntimeError(f"Malformed development-only split for fold {fold}")
    ordered = sorted(
        original_train,
        key=lambda subject: stable_uint64("invariance-rescue-v1", fold, subject),
    )
    calibration_count = int(config["calibration_subject_count"])
    calibration = tuple(subject_sort(ordered[:calibration_count]))
    model_fit = tuple(subject_sort(ordered[calibration_count:]))
    if len(model_fit) != int(config["model_fit_subject_count"]):
        raise RuntimeError("Nested development split count mismatch")
    return DevelopmentSplit(
        fold=int(fold),
        model_fit_subjects=model_fit,
        calibration_subjects=calibration,
        outcome_subjects=outcome,
        original_train_subjects=original_train,
        source_json_sha256=_file_sha256(path),
    )


def persist_split_manifests() -> list[dict[str, Any]]:
    config = load_config()
    rows = []
    for fold in config["development_folds"]:
        split = load_development_split(int(fold))
        payload = split.payload()
        write_json(CACHE.parent / "protocol" / f"DEVELOPMENT_SPLIT_FOLD_{fold}.json", payload)
        rows.append(payload)
    return rows


def load_manifest(
    split: DevelopmentSplit,
    subjects: Sequence[str] | None = None,
) -> pd.DataFrame:
    path = stage0_root() / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet"
    selected_subjects = tuple(map(str, subjects)) if subjects is not None else split.allowed_subjects
    if not set(selected_subjects).issubset(set(split.allowed_subjects)):
        raise RuntimeError("Manifest request exceeds development-authorized subjects")
    # Arrow predicate pushdown prevents unauthorized subjects from being
    # materialized in this process.
    frame = pd.read_parquet(
        path,
        columns=MANIFEST_COLUMNS,
        filters=[
            ("paradigm", "==", "mi"),
            ("subject_id", "in", list(selected_subjects)),
        ],
    )
    frame["subject_id"] = frame.subject_id.astype(str)
    frame["session_id"] = frame.session_id.astype(int)
    frame["label"] = frame.event_label.astype(str).map({"left_hand": 0, "right_hand": 1})
    if frame.label.isna().any():
        raise RuntimeError("Unexpected OpenBMI MI event labels")
    frame["label"] = frame.label.astype(int)
    frame["subject_code"] = frame.subject_id.map(subject_code).astype(int)
    frame["trial_uid"] = "OpenBMI_nm000273_MI:" + frame.trial_id.astype(str)
    frame = frame.sort_values(["subject_code", "session_id", "signal_cache_path", "cache_index"]).reset_index(drop=True)
    per_cell = frame.groupby(["subject_id", "session_id", "label"]).size()
    expected_rows = len(selected_subjects) * 2 * 100
    if (
        len(frame) != expected_rows
        or set(frame.subject_id) != set(selected_subjects)
        or set(frame.session_id) != {1, 2}
        or set(per_cell.tolist()) != {50}
        or frame.trial_uid.duplicated().any()
        or set(frame.n_channels.astype(int)) != {62}
        or set(frame.n_times.astype(int)) != {1000}
        or set(frame.sampling_rate.astype(float)) != {250.0}
    ):
        raise RuntimeError(f"Development MI manifest coverage failure fold={split.fold}")
    frame["manifest_position"] = np.arange(len(frame), dtype=np.int64)
    frame["OUTER_TEST_USED"] = False
    return frame


def select_frame(
    manifest: pd.DataFrame,
    subjects: Sequence[str],
    sessions: Sequence[int],
) -> pd.DataFrame:
    mask = (
        manifest.subject_id.isin(set(map(str, subjects)))
        & manifest.session_id.isin(set(map(int, sessions)))
    )
    return manifest.loc[mask].copy().reset_index(drop=True)


def normalizer(fold: int, manifest: pd.DataFrame, subjects: Sequence[str]) -> tuple[np.ndarray, np.ndarray, Path]:
    path = CACHE / f"NORMALIZER_FOLD_{fold}.npz"
    expected_hash = sha256_lines(subject_sort(subjects))
    if path.exists():
        saved = np.load(path, allow_pickle=False)
        if str(saved["subjects_sha256"].item()) != expected_hash:
            raise RuntimeError("Cached normalizer subject hash mismatch")
        return saved["mean"].astype(np.float32), saved["std"].astype(np.float32), path
    selected = manifest[manifest.subject_id.isin(set(map(str, subjects)))]
    total = np.zeros(62, dtype=np.float64)
    square = np.zeros(62, dtype=np.float64)
    count = 0
    root = stage0_root()
    for relative, group in selected.groupby("signal_cache_path", sort=True):
        array = np.load(root / str(relative), mmap_mode="r", allow_pickle=False)
        indices = group.cache_index.to_numpy(dtype=np.int64)
        values = np.asarray(array[indices], dtype=np.float64)
        total += values.sum(axis=(0, 2))
        square += np.square(values).sum(axis=(0, 2))
        count += int(values.shape[0] * values.shape[2])
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 1e-8)
    std = np.sqrt(variance)
    np.savez_compressed(
        path,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        subjects_sha256=np.asarray(expected_hash),
        rows=np.asarray(len(selected), dtype=np.int64),
        outer_test_used=np.asarray(False),
    )
    return mean.astype(np.float32), std.astype(np.float32), path


class SignalAccessor:
    def __init__(self, frame: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> None:
        self.paths = frame.signal_cache_path.astype(str).to_numpy()
        self.cache_indices = frame.cache_index.to_numpy(dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float32)[:, None]
        self.std = np.maximum(np.asarray(std, dtype=np.float32)[:, None], 1e-6)
        self.arrays: dict[str, np.ndarray] = {}

    def get(self, index: int) -> torch.Tensor:
        relative = self.paths[int(index)]
        if relative not in self.arrays:
            self.arrays[relative] = np.load(stage0_root() / relative, mmap_mode="r", allow_pickle=False)
        value = np.asarray(self.arrays[relative][self.cache_indices[int(index)]], dtype=np.float32)
        return torch.from_numpy((value - self.mean) / self.std)


class MIDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        mean: np.ndarray,
        std: np.ndarray,
        domain_map: Mapping[str, int] | None,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.labels = self.frame.label.to_numpy(dtype=np.int64)
        self.positions = self.frame.manifest_position.to_numpy(dtype=np.int64)
        self.subjects = self.frame.subject_id.astype(str).to_numpy()
        self.domains = np.asarray(
            [int(domain_map.get(subject, -1)) if domain_map is not None else -1 for subject in self.subjects],
            dtype=np.int64,
        )
        self.accessor = SignalAccessor(self.frame, mean, std)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        return (
            self.accessor.get(index),
            torch.tensor(self.labels[index], dtype=torch.long),
            torch.tensor(self.domains[index], dtype=torch.long),
            torch.tensor(self.positions[index], dtype=torch.long),
        )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=torch.Generator().manual_seed(int(seed)),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(drop_last),
    )


def domain_map(subjects: Iterable[str]) -> dict[str, int]:
    return {subject: index for index, subject in enumerate(subject_sort(subjects))}
