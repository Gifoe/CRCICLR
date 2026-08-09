from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class ConfirmatoryEpisode:
    dataset: str
    subject_id: str
    context_indices: np.ndarray
    future_indices: np.ndarray
    channel_names: tuple[str, ...]
    label_map: dict[str, int]
    raw_source_hash: str

    def validate(self) -> None:
        if set(map(int, self.context_indices)) & set(map(int, self.future_indices)):
            raise ValueError(f"U/V overlap for {self.subject_id}")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError(f"duplicate channel names for {self.subject_id}")
        if not self.raw_source_hash:
            raise ValueError("raw source hash is required")


class ConfirmatoryDatasetAdapter(abc.ABC):
    """Dataset boundary that keeps acquisition and labels outside method logic."""

    @abc.abstractmethod
    def dataset_fingerprint(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def subjects(self) -> list[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def iter_episodes(self) -> Iterator[ConfirmatoryEpisode]:
        raise NotImplementedError

    @abc.abstractmethod
    def load_signal(self, subject_id: str, indices: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def load_labels(self, subject_id: str, indices: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def validate_manifest(path: str | Path, expected_freeze_hash: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"dataset", "dataset_hash", "license_approved", "pretraining_overlap_audited",
                "calibration_subjects", "test_subjects", "method_freeze_sha256"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"confirmation manifest missing {sorted(missing)}")
    if payload["method_freeze_sha256"] != expected_freeze_hash:
        raise ValueError("confirmation manifest does not reference the frozen method")
    if not payload["license_approved"] or not payload["pretraining_overlap_audited"]:
        raise PermissionError("license and pretraining-overlap audits must be approved before execution")
    calibration, test = set(payload["calibration_subjects"]), set(payload["test_subjects"])
    if not calibration or not test or calibration & test:
        raise ValueError("confirmatory calibration/test subjects must be nonempty and disjoint")
    return payload


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
