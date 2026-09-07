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

from common import CACHE, stage0_root, wbcic_source_root


@dataclass
class FoldDataset:
    benchmark: str
    fold: int
    embeddings: np.ndarray
    metadata: pd.DataFrame
    model_fit_subjects: tuple[str, ...]
    discovery_subjects: tuple[str, ...]
    outcome_subjects: tuple[str, ...]
    history_sessions: tuple[int, ...]
    future_session: int
    backbone_logits: np.ndarray | None = None

    def mask(self, subjects: tuple[str, ...] | list[str], sessions: tuple[int, ...] | list[int]) -> np.ndarray:
        return self.metadata.subject_id.astype(str).isin(list(map(str, subjects))).to_numpy() & self.metadata.session_id.astype(int).isin(list(map(int, sessions))).to_numpy()


OPENBMI_BEST_EPOCHS = (54, 54, 25, 47, 44)


def _openbmi_split() -> dict:
    path = stage0_root() / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json"
    payload = json.loads(path.read_text(encoding="utf-8-sig"))["openbmi"]
    if (
        payload.get("split_unit") != "subject"
        or int(payload.get("outer_folds", 0)) != 5
        or len(payload.get("subjects", [])) != 54
    ):
        raise RuntimeError("Malformed OpenBMI split lock")
    return payload


def load_openbmi_fold(fold: int) -> FoldDataset:
    if fold not in range(5):
        raise ValueError(fold)
    root = stage0_root() / "outputs" / "persist_eeg_p2p3" / "p3" / "seed_0" / f"fold-{fold}"
    selection = json.loads((root / "CHECKPOINT_SELECTION.json").read_text(encoding="utf-8"))
    epoch = int(selection["best_epoch"])
    if epoch != OPENBMI_BEST_EPOCHS[fold] or selection.get("test_data_used") is not False:
        raise RuntimeError(f"OpenBMI fold {fold} checkpoint selection mismatch")
    directory = root / "embeddings" / f"epoch-{epoch:03d}"
    metadata_all = pd.read_parquet(directory / "metadata.parquet")
    embeddings_all = np.load(directory / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    if len(metadata_all) != len(embeddings_all) or embeddings_all.shape != (235_439, 128):
        raise RuntimeError(f"Malformed OpenBMI embedding cache for fold {fold}")
    mask = metadata_all.paradigm.astype(str).str.lower().eq("mi").to_numpy()
    metadata = metadata_all.loc[mask].copy().reset_index(drop=True)
    embeddings = np.asarray(embeddings_all[mask], dtype=np.float32)
    mapping = {"left_hand": 0, "right_hand": 1}
    if set(metadata.event_label.astype(str)) != set(mapping):
        raise RuntimeError("Unexpected OpenBMI MI labels")
    metadata["label"] = metadata.event_label.astype(str).map(mapping).astype(int)
    metadata["subject_id"] = metadata.subject_id.astype(str)
    metadata["session_id"] = metadata.session_id.astype(int)
    metadata["trial_uid"] = "OpenBMI_nm000273_MI:" + metadata.trial_id.astype(str)
    metadata["target_future_label_used_for_fit"] = False
    metadata["OUTER_TEST_USED"] = False
    per_cell = metadata.groupby(["subject_id", "session_id", "label"]).size()
    if (
        len(metadata) != 10_800
        or metadata.subject_id.nunique() != 54
        or set(metadata.session_id) != {1, 2}
        or set(per_cell.tolist()) != {50}
        or metadata.trial_uid.duplicated().any()
        or not np.isfinite(embeddings).all()
    ):
        raise RuntimeError(f"OpenBMI MI protocol coverage failure fold {fold}")
    split = _openbmi_split()["folds"][fold]
    model_fit = tuple(map(str, split["train_subjects"]))
    discovery = tuple(map(str, split["validation_subjects"]))
    outcome = tuple(map(str, split["outer_test_subjects"]))
    if set(model_fit) | set(discovery) != set(map(str, split["outer_train_subjects"])):
        raise RuntimeError("OpenBMI train/discovery roles do not reconstruct outer-train")
    if set(model_fit) & set(discovery) or set(model_fit) & set(outcome) or set(discovery) & set(outcome):
        raise RuntimeError("OpenBMI subject role leakage")
    if set(model_fit) | set(discovery) | set(outcome) != set(metadata.subject_id):
        raise RuntimeError("OpenBMI fold roles are not exhaustive")
    return FoldDataset(
        benchmark="OpenBMI_MI_S1_to_S2",
        fold=fold,
        embeddings=embeddings,
        metadata=metadata,
        model_fit_subjects=model_fit,
        discovery_subjects=discovery,
        outcome_subjects=outcome,
        history_sessions=(1,),
        future_session=2,
    )


def _wbcic_cache_paths(fold: int) -> tuple[Path, Path, Path]:
    prefix = CACHE / f"WBCIC_SHARED_FOLD_{fold}_EEGNET_STABLE_ALL_SESSION"
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
        raise FileNotFoundError(f"WBCIC fold-compatible all-session cache missing for fold {fold}")
    embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    logits = np.load(logits_path, mmap_mode="r", allow_pickle=False)
    metadata = pd.read_parquet(metadata_path)
    if (
        embeddings.shape != (24_591, 32)
        or logits.shape != (24_591, 2)
        or len(metadata) != 24_591
        or metadata.subject_id.nunique() != 41
        or set(metadata.session_id.astype(int)) != {1, 2, 3}
        or metadata.trial_uid.duplicated().any()
        or metadata.OUTER_TEST_USED.astype(bool).any()
    ):
        raise RuntimeError(f"Malformed WBCIC V6 cache fold {fold}")
    scope_path = wbcic_source_root() / "experiments" / "persist_eeg_wbcic_actionability_v2" / "outputs" / "protocol" / "DEVELOPMENT_SCOPE_LOCK.json"
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    if scope.get("outer_subject_ids_present") is not False or scope.get("runtime_must_not_open") != "OUTER_SPLIT_LOCK.json":
        raise RuntimeError("WBCIC outer lock violation")
    roles = scope["audit_roles"][str(fold)]
    model_fit = tuple(map(str, roles["model_fit"]))
    discovery = tuple(map(str, roles["discovery_decision"]))
    outcome = tuple(map(str, roles["outcome"]))
    if set(model_fit) & set(discovery) or set(model_fit) & set(outcome) or set(discovery) & set(outcome):
        raise RuntimeError("WBCIC subject role leakage")
    if set(model_fit) | set(discovery) | set(outcome) != set(metadata.subject_id.astype(str)):
        raise RuntimeError("WBCIC fold roles are not exhaustive")
    return FoldDataset(
        benchmark="WBCIC_S1S2_to_S3_authorized_development",
        fold=fold,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        backbone_logits=np.asarray(logits, dtype=np.float32),
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
