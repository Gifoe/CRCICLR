from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import CACHE, stage0_root, v6_outputs, wbcic_source_root


@dataclass
class FoldDataset:
    benchmark: str
    fold: int
    embeddings: np.ndarray
    logits: np.ndarray
    metadata: pd.DataFrame
    model_fit_subjects: tuple[str, ...]
    discovery_subjects: tuple[str, ...]
    outcome_subjects: tuple[str, ...]
    history_sessions: tuple[int, ...]
    future_session: int

    @property
    def nonoutcome_subjects(self) -> tuple[str, ...]:
        return tuple(self.model_fit_subjects) + tuple(self.discovery_subjects)

    def mask(self, subjects, sessions) -> np.ndarray:
        return (
            self.metadata.subject_id.astype(str).isin(list(map(str, subjects))).to_numpy()
            & self.metadata.session_id.astype(int).isin(list(map(int, sessions))).to_numpy()
        )


def _openbmi_split() -> dict:
    path = stage0_root() / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json"
    payload = json.loads(path.read_text(encoding="utf-8-sig"))["openbmi"]
    if payload.get("split_unit") != "subject" or len(payload.get("subjects", [])) != 54:
        raise RuntimeError("Malformed OpenBMI split lock")
    return payload


def load_openbmi_fold(fold: int) -> FoldDataset:
    if fold not in range(5):
        raise ValueError(fold)
    prefix = CACHE / f"OPENBMI_MI_SPECIFIC_FOLD_{fold}"
    feature_path = prefix.with_name(prefix.name + "_FEATURES.npy")
    logit_path = prefix.with_name(prefix.name + "_LOGITS.npy")
    metadata_path = prefix.with_name(prefix.name + "_METADATA.parquet")
    if not all(path.is_file() for path in (feature_path, logit_path, metadata_path)):
        raise FileNotFoundError(f"OpenBMI V7 episode cache missing for fold {fold}")
    embeddings = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    logits = np.load(logit_path, mmap_mode="r", allow_pickle=False)
    metadata = pd.read_parquet(metadata_path)
    per_cell = metadata.groupby(["subject_id", "session_id", "label"]).size()
    if (
        embeddings.shape[0] != 10_800
        or embeddings.shape[1] != 64
        or logits.shape != (10_800,)
        or len(metadata) != 10_800
        or metadata.subject_id.nunique() != 54
        or set(metadata.session_id.astype(int)) != {1, 2}
        or set(per_cell.tolist()) != {50}
        or metadata.trial_uid.duplicated().any()
        or not np.isfinite(embeddings).all()
        or not np.isfinite(logits).all()
    ):
        raise RuntimeError(f"Malformed OpenBMI V7 cache fold {fold}")
    split = _openbmi_split()["folds"][fold]
    model_fit = tuple(map(str, split["train_subjects"]))
    discovery = tuple(map(str, split["validation_subjects"]))
    outcome = tuple(map(str, split["outer_test_subjects"]))
    if set(model_fit) & set(discovery) or set(model_fit) & set(outcome) or set(discovery) & set(outcome):
        raise RuntimeError("OpenBMI subject-role leakage")
    if set(model_fit) | set(discovery) | set(outcome) != set(metadata.subject_id.astype(str)):
        raise RuntimeError("OpenBMI roles do not cover the cache")
    return FoldDataset(
        benchmark="OpenBMI_MI_S1_to_S2",
        fold=fold,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        logits=np.asarray(logits, dtype=np.float32),
        metadata=metadata,
        model_fit_subjects=model_fit,
        discovery_subjects=discovery,
        outcome_subjects=outcome,
        history_sessions=(1,),
        future_session=2,
    )


def _wbcic_cache_paths(fold: int) -> tuple[Path, Path, Path]:
    prefix = v6_outputs() / "cache" / f"WBCIC_SHARED_FOLD_{fold}_EEGNET_STABLE_ALL_SESSION"
    return (
        prefix.with_name(prefix.name + "_EMBEDDINGS.npy"),
        prefix.with_name(prefix.name + "_LOGITS.npy"),
        prefix.with_name(prefix.name + "_METADATA.parquet"),
    )


def load_wbcic_fold(fold: int) -> FoldDataset:
    if fold not in range(5):
        raise ValueError(fold)
    embedding_path, logits_path, metadata_path = _wbcic_cache_paths(fold)
    if not all(path.is_file() for path in (embedding_path, logits_path, metadata_path)):
        raise FileNotFoundError(f"WBCIC V6 cache missing for fold {fold}")
    embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    raw_logits = np.load(logits_path, mmap_mode="r", allow_pickle=False)
    metadata = pd.read_parquet(metadata_path)
    logits = raw_logits[:, 1] - raw_logits[:, 0]
    if (
        embeddings.shape != (24_591, 32)
        or raw_logits.shape != (24_591, 2)
        or len(metadata) != 24_591
        or metadata.subject_id.nunique() != 41
        or set(metadata.session_id.astype(int)) != {1, 2, 3}
        or metadata.trial_uid.duplicated().any()
        or metadata.OUTER_TEST_USED.astype(bool).any()
    ):
        raise RuntimeError(f"Malformed WBCIC cache fold {fold}")
    scope_path = (
        wbcic_source_root() / "experiments" / "persist_eeg_wbcic_actionability_v2"
        / "outputs" / "protocol" / "DEVELOPMENT_SCOPE_LOCK.json"
    )
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    if scope.get("outer_subject_ids_present") is not False or scope.get("runtime_must_not_open") != "OUTER_SPLIT_LOCK.json":
        raise RuntimeError("WBCIC outer lock violation")
    roles = scope["audit_roles"][str(fold)]
    model_fit = tuple(map(str, roles["model_fit"]))
    discovery = tuple(map(str, roles["discovery_decision"]))
    outcome = tuple(map(str, roles["outcome"]))
    if set(model_fit) & set(discovery) or set(model_fit) & set(outcome) or set(discovery) & set(outcome):
        raise RuntimeError("WBCIC subject-role leakage")
    if set(model_fit) | set(discovery) | set(outcome) != set(metadata.subject_id.astype(str)):
        raise RuntimeError("WBCIC roles do not cover the authorized development cache")
    return FoldDataset(
        benchmark="WBCIC_S1S2_to_S3_authorized_development",
        fold=fold,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        logits=np.asarray(logits, dtype=np.float32),
        metadata=metadata,
        model_fit_subjects=model_fit,
        discovery_subjects=discovery,
        outcome_subjects=outcome,
        history_sessions=(1, 2),
        future_session=3,
    )


def load_fold(benchmark: str, fold: int) -> FoldDataset:
    key = benchmark.lower()
    if key in {"openbmi", "openbmi_mi"}:
        return load_openbmi_fold(fold)
    if key == "wbcic":
        return load_wbcic_fold(fold)
    raise ValueError(benchmark)
