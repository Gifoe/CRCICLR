"""Extract a compatible EEGNet representation space separately for each fold."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import CACHE, PROTOCOL, default_wbcic_repo, ensure_directories, sha256_file, write_json


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _subject_number(value: str) -> int:
    return int(str(value).split("-", 1)[1])


def run(wbcic_repo: Path, device_name: str, workers: int) -> None:
    ensure_directories()
    reference = wbcic_repo / "experiments" / "persist_eeg_wbcic_actionability_v2"
    scope_path = reference / "outputs" / "protocol" / "DEVELOPMENT_SCOPE_LOCK.json"
    frozen_path = reference / "outputs" / "protocol" / "REPRESENTATION_FROZEN.json"
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    allowed = sorted(map(str, scope.get("allowed_subjects", [])), key=_subject_number)
    if (
        len(allowed) != 41
        or scope.get("outer_subject_ids_present") is not False
        or scope.get("runtime_must_not_open") != "OUTER_SPLIT_LOCK.json"
    ):
        raise RuntimeError("DATA_SCOPE_VIOLATION")
    core = _load_module(reference / "code" / "core.py", "v5_shared_embedding_core")
    device = torch.device(
        device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    records = []
    for fold in range(5):
        matches = [item for item in frozen["competence_checkpoint_set"] if int(item["fold"]) == fold]
        if len(matches) != 1:
            raise RuntimeError(f"Missing stable checkpoint for fold {fold}")
        checkpoint = wbcic_repo / Path(matches[0]["checkpoint"])
        if sha256_file(checkpoint) != matches[0]["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch for fold {fold}")
        model, payload = core.load_model(checkpoint, device)
        arrays = core.infer(model, allowed, [2], device, workers, batch_size=256)
        subjects = arrays["subjects"][arrays["subject_index"].astype(int)].astype(str)
        sessions = arrays["session"].astype(int)
        labels = arrays["labels"].astype(int)
        metadata = pd.DataFrame({"subject_id": subjects, "session_id": sessions, "label": labels})
        metadata["trial_index_within_subject_session"] = metadata.groupby(
            ["subject_id", "session_id"], sort=False
        ).cumcount()
        metadata["trial_uid"] = (
            "WBCIC_nm000348_dev:"
            + metadata.subject_id
            + ":S3:"
            + metadata.trial_index_within_subject_session.astype(str)
        )
        metadata["fold_representation"] = fold
        metadata["OUTER_TEST_USED"] = False
        values_path = CACHE / f"WBCIC_SHARED_FOLD_{fold}_EEGNET_STABLE_EMBEDDINGS.npy"
        metadata_path = CACHE / f"WBCIC_SHARED_FOLD_{fold}_EEGNET_STABLE_METADATA.parquet"
        np.save(values_path, arrays["embeddings"].astype(np.float32), allow_pickle=False)
        metadata.to_parquet(metadata_path, index=False)
        test_subjects = set(map(str, scope["folds"][f"F{fold}"]))
        trained_subjects = set(map(str, payload.get("train_subjects", [])))
        if trained_subjects & test_subjects:
            raise RuntimeError(f"Fold {fold} checkpoint trained on evaluation subjects")
        records.append(
            {
                "fold": fold,
                "rows": len(metadata),
                "dimensions": int(arrays["embeddings"].shape[1]),
                "checkpoint_sha256": sha256_file(checkpoint),
                "values_sha256": sha256_file(values_path),
                "metadata_sha256": sha256_file(metadata_path),
                "checkpoint_train_subjects": len(trained_subjects),
                "heldout_subject_overlap": 0,
            }
        )
        print(f"[shared embedding] fold={fold} rows={len(metadata)}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(
        PROTOCOL / "WBCIC_SHARED_REPRESENTATION_AUDIT.json",
        {
            "status": "WBCIC_SHARED_FOLD_REPRESENTATIONS_COMPLETE",
            "scope_sha256": sha256_file(scope_path),
            "frozen_protocol_sha256": sha256_file(frozen_path),
            "folds": records,
            "target_labels_used_to_extract_embeddings": False,
            "outer_split_lock_opened": False,
            "outer_subject_ids_loaded": False,
            "OUTER_TEST_USED": False,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wbcic-repo", type=Path, default=default_wbcic_repo())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    run(args.wbcic_repo, args.device, max(0, args.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
