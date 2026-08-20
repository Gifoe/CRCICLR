"""Extract fold-compatible WBCIC EEGNet features for S1/S2/S3.

The script deliberately reads only the frozen 41-subject development cache.
It never opens the sealed outer split or enumerates the raw WBCIC root.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import CACHE, PROTOCOL, ensure_directories, sha256_file, wbcic_source_root, write_json


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("persist_v6_wbcic_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _number(subject: str) -> int:
    return int(str(subject).split("-", 1)[1])


def _paths(fold: int) -> tuple[Path, Path, Path]:
    prefix = CACHE / f"WBCIC_SHARED_FOLD_{fold}_EEGNET_STABLE_ALL_SESSION"
    return (
        prefix.with_name(prefix.name + "_EMBEDDINGS.npy"),
        prefix.with_name(prefix.name + "_LOGITS.npy"),
        prefix.with_name(prefix.name + "_METADATA.parquet"),
    )


def run(source: Path, folds: list[int], device_name: str, workers: int, batch_size: int, force: bool) -> None:
    ensure_directories()
    reference = source / "experiments" / "persist_eeg_wbcic_actionability_v2"
    scope_path = reference / "outputs" / "protocol" / "DEVELOPMENT_SCOPE_LOCK.json"
    frozen_path = reference / "outputs" / "protocol" / "REPRESENTATION_FROZEN.json"
    cache_audit_path = reference / "outputs" / "protocol" / "CACHE_SCOPE_AUDIT.json"
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    cache_audit = json.loads(cache_audit_path.read_text(encoding="utf-8"))
    allowed = sorted(map(str, scope.get("allowed_subjects", [])), key=_number)
    if (
        len(allowed) != 41
        or scope.get("outer_subject_ids_present") is not False
        or scope.get("runtime_must_not_open") != "OUTER_SPLIT_LOCK.json"
        or cache_audit.get("outer_subject_ids_materialized") is not False
        or cache_audit.get("status") != "DEVELOPMENT_CACHE_COMPLETE"
    ):
        raise RuntimeError("DATA_SCOPE_VIOLATION")
    core = _load_module(reference / "code" / "core.py")
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    for fold in folds:
        if fold not in range(5):
            raise ValueError(f"Invalid fold {fold}")
        embedding_path, logits_path, metadata_path = _paths(fold)
        if not force and all(path.is_file() for path in (embedding_path, logits_path, metadata_path)):
            metadata = pd.read_parquet(metadata_path)
            if len(metadata) == 24_591 and metadata.trial_uid.nunique() == 24_591:
                print(f"[extract] fold={fold} already complete", flush=True)
                continue
        records = [item for item in frozen["competence_checkpoint_set"] if int(item["fold"]) == fold]
        if len(records) != 1:
            raise RuntimeError(f"Missing frozen checkpoint for fold {fold}")
        checkpoint = source / Path(records[0]["checkpoint"])
        if sha256_file(checkpoint) != records[0]["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch for fold {fold}")
        model, payload = core.load_model(checkpoint, device)
        trained = set(map(str, payload.get("train_subjects", [])))
        expected = set(map(str, scope["audit_roles"][str(fold)]["model_fit"])) | set(
            map(str, scope["audit_roles"][str(fold)]["discovery_decision"])
        )
        forbidden = set(map(str, scope["audit_roles"][str(fold)]["outcome"]))
        if trained != expected or trained & forbidden:
            raise RuntimeError(f"Fold {fold} checkpoint role mismatch")
        arrays = core.infer(model, allowed, [0, 1, 2], device, workers, batch_size=batch_size)
        subject = arrays["subjects"][arrays["subject_index"].astype(int)].astype(str)
        session = arrays["session"].astype(int) + 1
        metadata = pd.DataFrame(
            {
                "subject_id": subject,
                "session_id": session,
                "label": arrays["labels"].astype(int),
                "fold_representation": fold,
            }
        )
        metadata["trial_index_within_subject_session"] = metadata.groupby(
            ["subject_id", "session_id"], sort=False
        ).cumcount()
        metadata["trial_uid"] = (
            "WBCIC_nm000348_dev:"
            + metadata.subject_id.astype(str)
            + ":S"
            + metadata.session_id.astype(str)
            + ":"
            + metadata.trial_index_within_subject_session.astype(str)
        )
        metadata["history_role"] = np.where(metadata.session_id.lt(3), "target_history", "future_evaluation")
        metadata["target_future_label_used_for_fit"] = False
        metadata["OUTER_TEST_USED"] = False
        if (
            len(metadata) != 24_591
            or metadata.subject_id.nunique() != 41
            or metadata.trial_uid.duplicated().any()
            or set(metadata.session_id) != {1, 2, 3}
            or arrays["embeddings"].shape != (24_591, 32)
            or arrays["logits"].shape != (24_591, 2)
            or not np.isfinite(arrays["embeddings"]).all()
            or not np.isfinite(arrays["logits"]).all()
        ):
            raise RuntimeError(f"Malformed fold {fold} extraction")
        np.save(embedding_path, arrays["embeddings"].astype(np.float32), allow_pickle=False)
        np.save(logits_path, arrays["logits"].astype(np.float32), allow_pickle=False)
        metadata.to_parquet(metadata_path, index=False)
        print(f"[extract] fold={fold} rows={len(metadata)} device={device}", flush=True)
        del model, arrays
        if device.type == "cuda":
            torch.cuda.empty_cache()
    reports = []
    for fold in range(5):
        embedding_path, logits_path, metadata_path = _paths(fold)
        if not all(path.is_file() for path in (embedding_path, logits_path, metadata_path)):
            continue
        metadata = pd.read_parquet(metadata_path)
        reports.append(
            {
                "fold": fold,
                "rows": len(metadata),
                "subjects": int(metadata.subject_id.nunique()),
                "sessions": sorted(map(int, metadata.session_id.unique())),
                "embedding_sha256": sha256_file(embedding_path),
                "logits_sha256": sha256_file(logits_path),
                "metadata_sha256": sha256_file(metadata_path),
                "heldout_subject_checkpoint_overlap": 0,
            }
        )
    write_json(
        PROTOCOL / "WBCIC_SHARED_REPRESENTATION_AUDIT.json",
        {
            "status": "COMPLETE" if len(reports) == 5 else "PARTIAL",
            "folds": reports,
            "scope_sha256": sha256_file(scope_path),
            "frozen_representation_sha256": sha256_file(frozen_path),
            "target_future_labels_used_to_extract": False,
            "sealed_outer_split_opened": False,
            "outer_subject_ids_loaded": False,
            "OUTER_TEST_USED": False,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=wbcic_source_root())
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.source, args.folds, args.device, max(0, args.workers), args.batch_size, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
