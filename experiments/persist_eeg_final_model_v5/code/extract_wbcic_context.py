"""Build legal all-session WBCIC expert/context tables on the GPU server.

Only the 41 subjects listed in DEVELOPMENT_SCOPE_LOCK are materialized.  The
sealed outer split file is deliberately never opened.  Fold-specific models are
the same frozen checkpoints used by V4, so every target subject remains OOF.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from common import CACHE, PROTOCOL, default_wbcic_repo, ensure_directories, sha256_file, write_json


EXPERTS = ("EEGNet_STABLE", "EEGNet_STD", "DeepConvNet", "EEGConformer", "TeCh")


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


def _checkpoint_record(payload: dict[str, Any], fold: int) -> dict[str, Any]:
    matches = [item for item in payload["competence_checkpoint_set"] if int(item["fold"]) == fold]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one frozen checkpoint for fold {fold}")
    return matches[0]


def build(wbcic_repo: Path, device_name: str, workers: int) -> Path:
    ensure_directories()
    reference = wbcic_repo / "experiments" / "persist_eeg_wbcic_actionability_v2"
    multi = wbcic_repo / "experiments" / "persist_eeg_multibackbone_final_closure"
    scope_path = reference / "outputs" / "protocol" / "DEVELOPMENT_SCOPE_LOCK.json"
    inventory_path = reference / "outputs" / "protocol" / "CACHE_INVENTORY.csv"
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    allowed = list(map(str, scope.get("allowed_subjects", [])))
    if (
        len(allowed) != 41
        or scope.get("outer_subject_ids_present") is not False
        or scope.get("runtime_must_not_open") != "OUTER_SPLIT_LOCK.json"
    ):
        raise RuntimeError("DATA_SCOPE_VIOLATION")
    inventory = pd.read_csv(inventory_path)
    if set(inventory.subject.astype(str)) != set(allowed):
        raise RuntimeError("Development cache inventory exceeds authorized scope")
    expected_rows = int(inventory.n_trials.sum())

    eeg_core = _load_module(reference / "code" / "core.py", "v5_wbcic_eeg_core")
    _load_module(multi / "code" / "models.py", "models")
    multi_common = _load_module(multi / "code" / "common.py", "v5_wbcic_multi_common")
    device = torch.device(
        device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    frozen_eeg = json.loads(
        (reference / "outputs" / "protocol" / "REPRESENTATION_FROZEN.json").read_text(
            encoding="utf-8"
        )
    )
    frozen_multi = {
        backbone: json.loads(
            (multi / "outputs" / "protocol" / f"BACKBONE_{backbone.upper()}_FROZEN.json").read_text(
                encoding="utf-8"
            )
        )
        for backbone in ("DeepConvNet", "EEGConformer", "TeCh")
    }

    frames: list[pd.DataFrame] = []
    embedding_chunks: dict[str, list[np.ndarray]] = {name: [] for name in EXPERTS}
    checkpoint_audit: list[dict[str, Any]] = []
    for fold in range(5):
        subjects = sorted(map(str, scope["folds"][f"F{fold}"]), key=_subject_number)
        if not set(subjects).issubset(set(allowed)):
            raise RuntimeError("Non-development subject requested")
        arrays_by_expert: dict[str, dict[str, np.ndarray]] = {}

        stable_record = _checkpoint_record(frozen_eeg, fold)
        stable_path = wbcic_repo / Path(stable_record["checkpoint"])
        if sha256_file(stable_path) != stable_record["checkpoint_sha256"]:
            raise RuntimeError("EEGNet stable checkpoint hash mismatch")
        model, _ = eeg_core.load_model(stable_path, device)
        arrays_by_expert["EEGNet_STABLE"] = eeg_core.infer(
            model, subjects, [0, 1, 2], device, workers, batch_size=256
        )
        checkpoint_audit.append(
            {"expert": "EEGNet_STABLE", "fold": fold, "sha256": sha256_file(stable_path)}
        )
        del model

        std_path = reference / "outputs" / "model" / "competence" / f"EEGNET_STD_fold-{fold}.pt"
        model, payload = eeg_core.load_model(std_path, device)
        if payload.get("config", {}).get("id") != "EEGNET_STD":
            raise RuntimeError("Unexpected EEGNet standard checkpoint")
        arrays_by_expert["EEGNet_STD"] = eeg_core.infer(
            model, subjects, [0, 1, 2], device, workers, batch_size=256
        )
        checkpoint_audit.append(
            {"expert": "EEGNet_STD", "fold": fold, "sha256": sha256_file(std_path)}
        )
        del model

        for backbone in ("DeepConvNet", "EEGConformer", "TeCh"):
            record = _checkpoint_record(frozen_multi[backbone], fold)
            checkpoint = wbcic_repo / Path(record["checkpoint"])
            if sha256_file(checkpoint) != record["checkpoint_sha256"]:
                raise RuntimeError(f"{backbone} checkpoint hash mismatch")
            model, _ = multi_common.load_model(checkpoint, device)
            arrays_by_expert[backbone] = multi_common.infer(
                model, subjects, [0, 1, 2], device, workers, batch_size=256
            )
            checkpoint_audit.append(
                {"expert": backbone, "fold": fold, "sha256": sha256_file(checkpoint)}
            )
            del model

        reference_arrays = arrays_by_expert["EEGNet_STABLE"]
        labels = reference_arrays["labels"].astype(int)
        actual_subjects = reference_arrays["subjects"][
            reference_arrays["subject_index"].astype(int)
        ].astype(str)
        sessions = reference_arrays["session"].astype(int)
        frame = pd.DataFrame(
            {
                "outer_fold": fold,
                "subject_id": actual_subjects,
                "session_id": sessions,
                "label": labels,
            }
        )
        frame["trial_index_within_subject_session"] = frame.groupby(
            ["subject_id", "session_id"], sort=False
        ).cumcount()
        frame["trial_uid"] = (
            "WBCIC_nm000348_dev:"
            + frame.subject_id
            + ":S"
            + (frame.session_id + 1).astype(str)
            + ":"
            + frame.trial_index_within_subject_session.astype(str)
        )
        for expert in EXPERTS:
            arrays = arrays_by_expert[expert]
            candidate_subjects = arrays["subjects"][arrays["subject_index"].astype(int)].astype(str)
            if (
                not np.array_equal(arrays["labels"].astype(int), labels)
                or not np.array_equal(candidate_subjects, actual_subjects)
                or not np.array_equal(arrays["session"].astype(int), sessions)
            ):
                raise RuntimeError(f"Expert alignment failed: {expert} fold {fold}")
            frame[f"margin_{expert}"] = arrays["logits"][:, 1] - arrays["logits"][:, 0]
            embeddings = arrays.get("embeddings")
            if embeddings is not None:
                embedding_chunks[expert].append(np.asarray(embeddings, dtype=np.float32))
        frame["OUTER_TEST_USED"] = False
        frames.append(frame)
        print(
            f"[all-session experts] fold={fold} subjects={len(subjects)} rows={len(frame)}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = pd.concat(frames, ignore_index=True)
    if (
        len(output) != expected_rows
        or output.subject_id.nunique() != 41
        or set(output.subject_id) != set(allowed)
        or output.trial_uid.duplicated().any()
        or output.OUTER_TEST_USED.astype(bool).any()
    ):
        raise RuntimeError("All-session expert table failed scope/coverage audit")
    table_path = CACHE / "WBCIC_DEV_ALL_SESSION_EXPERTS.parquet"
    output.to_parquet(table_path, index=False)
    embedding_records = {}
    for expert, chunks in embedding_chunks.items():
        if not chunks:
            continue
        values = np.concatenate(chunks).astype(np.float32)
        if len(values) != len(output):
            raise RuntimeError(f"Embedding alignment failed for {expert}")
        path = CACHE / f"WBCIC_DEV_ALL_SESSION_{expert}_EMBEDDINGS.npy"
        np.save(path, values, allow_pickle=False)
        embedding_records[expert] = {
            "path": str(path),
            "shape": list(values.shape),
            "sha256": sha256_file(path),
        }

    audit = {
        "status": "WBCIC_DEVELOPMENT_ALL_SESSION_EXPERTS_COMPLETE",
        "rows": len(output),
        "expected_rows_from_development_inventory": expected_rows,
        "subjects": output.subject_id.nunique(),
        "sessions": sorted(output.session_id.unique().astype(int).tolist()),
        "expert_roster": list(EXPERTS),
        "table_sha256": sha256_file(table_path),
        "scope_sha256": sha256_file(scope_path),
        "inventory_sha256": sha256_file(inventory_path),
        "embeddings": embedding_records,
        "checkpoint_audit": checkpoint_audit,
        "target_S3_labels_used_for_reliability": False,
        "outer_split_lock_opened": False,
        "outer_subject_ids_loaded": False,
        "OUTER_TEST_USED": False,
    }
    write_json(PROTOCOL / "WBCIC_ALL_SESSION_EXPERT_AUDIT.json", audit)
    print(json.dumps({key: audit[key] for key in ("status", "rows", "subjects", "sessions")}, indent=2))
    return table_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wbcic-repo", type=Path, default=default_wbcic_repo())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    build(args.wbcic_repo, args.device, max(0, args.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
