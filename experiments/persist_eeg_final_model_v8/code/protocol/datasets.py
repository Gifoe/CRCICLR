from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import CACHE, PROTOCOL, stage0_root, v6_outputs, v7_outputs, wbcic_source_root


@dataclass(frozen=True)
class BenchmarkProtocol:
    key: str
    name: str
    history_sessions: tuple[int, ...]
    future_session: int
    search_subjects: tuple[str, ...]
    holdout_subjects: tuple[str, ...]


@dataclass
class FeatureFold:
    protocol: BenchmarkProtocol
    source_fold: int
    features: np.ndarray
    logits: np.ndarray
    metadata: pd.DataFrame
    meta_subjects: tuple[str, ...]
    search_outcome_subjects: tuple[str, ...]
    holdout_outcome_subjects: tuple[str, ...]


def _split_payload() -> dict:
    path = PROTOCOL / "V8_SEARCH_SPLIT.json"
    if not path.is_file():
        raise FileNotFoundError("Run protocol/bootstrap_v8.py before loading V8 data")
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_protocol(benchmark: str) -> BenchmarkProtocol:
    key = benchmark.lower()
    if key not in {"openbmi", "wbcic"}:
        raise ValueError(benchmark)
    payload = _split_payload()[key]
    return BenchmarkProtocol(
        key=key,
        name=payload["benchmark"],
        history_sessions=tuple(map(int, payload["history_sessions"])),
        future_session=int(payload["future_session"]),
        search_subjects=tuple(map(str, payload["V8_SEARCH"])),
        holdout_subjects=tuple(map(str, payload["V8_INTERNAL_HOLDOUT"])),
    )


def _source_roles(benchmark: str, fold: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if fold not in range(5):
        raise ValueError(fold)
    if benchmark == "openbmi":
        path = stage0_root() / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json"
        payload = json.loads(path.read_text(encoding="utf-8-sig"))["openbmi"]
        row = payload["folds"][fold]
        train = tuple(map(str, row["train_subjects"] + row["validation_subjects"]))
        outcome = tuple(map(str, row["outer_test_subjects"]))
    else:
        path = (
            wbcic_source_root() / "experiments" / "persist_eeg_wbcic_actionability_v2"
            / "outputs" / "protocol" / "DEVELOPMENT_SCOPE_LOCK.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("outer_subject_ids_present") is not False:
            raise RuntimeError("WBCIC authorized development scope is malformed")
        row = payload["audit_roles"][str(fold)]
        train = tuple(map(str, row["model_fit"] + row["discovery_decision"]))
        outcome = tuple(map(str, row["outcome"]))
    if set(train) & set(outcome):
        raise RuntimeError("Source subject-role overlap")
    return train, outcome


def _feature_paths(benchmark: str, fold: int, family: str) -> tuple[Path, Path, Path]:
    family_key = family.upper()
    prefix = "OPENBMI" if benchmark == "openbmi" else "WBCIC"
    if family_key == "MI_SPECIFIC" and benchmark == "openbmi":
        root = v7_outputs() / "cache" / f"OPENBMI_MI_SPECIFIC_FOLD_{fold}"
        return (
            root.with_name(root.name + "_FEATURES.npy"),
            root.with_name(root.name + "_LOGITS.npy"),
            root.with_name(root.name + "_METADATA.parquet"),
        )
    if family_key == "MI_SPECIFIC" and benchmark == "wbcic":
        root = v6_outputs() / "cache" / f"WBCIC_SHARED_FOLD_{fold}_EEGNET_STABLE_ALL_SESSION"
        return (
            root.with_name(root.name + "_EMBEDDINGS.npy"),
            root.with_name(root.name + "_LOGITS.npy"),
            root.with_name(root.name + "_METADATA.parquet"),
        )
    root = v7_outputs() / "cache" / f"{prefix}_{family_key}_FOLD_{fold}"
    return (
        root.with_name(root.name + "_FEATURES.npy"),
        root.with_name(root.name + "_LOGITS.npy"),
        root.with_name(root.name + "_METADATA.parquet"),
    )


def load_feature_fold(benchmark: str, fold: int, family: str = "CONFORMER_NORM") -> FeatureFold:
    protocol = benchmark_protocol(benchmark)
    train, outcome = _source_roles(protocol.key, fold)
    feature_path, logit_path, _ = _feature_paths(protocol.key, fold, family)
    prefix = "OPENBMI" if protocol.key == "openbmi" else "WBCIC"
    metadata_path = CACHE / f"{prefix}_SEARCH_ROWS_FOLD_{fold}.parquet"
    if not all(path.is_file() for path in (feature_path, logit_path, metadata_path)):
        raise FileNotFoundError((feature_path, logit_path, metadata_path))
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    raw_logits = np.load(logit_path, mmap_mode="r", allow_pickle=False)
    metadata = pd.read_parquet(metadata_path)
    if raw_logits.ndim == 2 and raw_logits.shape[1] == 2:
        logits = raw_logits[:, 1] - raw_logits[:, 0]
    elif raw_logits.ndim == 1:
        logits = raw_logits
    else:
        raise RuntimeError(f"Unexpected logit shape {raw_logits.shape}")
    if (
        int(metadata.source_rows_total.iloc[0]) != len(features)
        or len(features) != len(logits)
        or metadata.trial_uid.duplicated().any()
        or metadata.OUTER_TEST_USED.astype(bool).any()
        or set(metadata.subject_id.astype(str)) != set(protocol.search_subjects)
    ):
        raise RuntimeError(f"Malformed authorized cache for {protocol.key} fold {fold}")
    search = set(protocol.search_subjects)
    holdout = set(protocol.holdout_subjects)
    if search & holdout or search != set(metadata.subject_id.astype(str)):
        raise RuntimeError("V8 subject partition mismatch")
    meta_subjects = tuple(subject for subject in train if subject in search)
    search_outcome = tuple(subject for subject in outcome if subject in search)
    holdout_outcome = tuple(subject for subject in outcome if subject in holdout)
    return FeatureFold(
        protocol=protocol,
        source_fold=fold,
        features=features,
        logits=logits,
        metadata=metadata,
        meta_subjects=meta_subjects,
        search_outcome_subjects=search_outcome,
        holdout_outcome_subjects=holdout_outcome,
    )


def assert_search_only(subjects: tuple[str, ...] | list[str], benchmark: str) -> None:
    protocol = benchmark_protocol(benchmark)
    illegal = set(map(str, subjects)) & set(protocol.holdout_subjects)
    if illegal:
        raise RuntimeError(
            "V8 internal holdout access attempted during search. "
            f"Blocked {len(illegal)} subject IDs."
        )


def internal_holdout_enabled() -> bool:
    return os.environ.get("PERSIST_V8_ENABLE_INTERNAL_HOLDOUT", "") == "FROZEN_ONCE_20260821"


def baseline_predictions(benchmark: str) -> tuple[pd.DataFrame, str]:
    key = benchmark.lower()
    prefix = "OPENBMI" if key == "openbmi" else "WBCIC"
    path = CACHE / f"{prefix}_V7_LOCKED_GENERIC_SEARCH.parquet"
    method = "ANCHOR_BLEND__CONFORMER_NORM_FIXED_HEAD"
    frame = pd.read_parquet(path)
    frame = frame.loc[frame.method_id.astype(str).eq(method)].copy()
    if frame.empty or frame.trial_uid.duplicated().any() or frame.OUTER_TEST_USED.astype(bool).any():
        raise RuntimeError(f"Malformed locked V7 strongest-generic predictions: {path}")
    return frame, method
