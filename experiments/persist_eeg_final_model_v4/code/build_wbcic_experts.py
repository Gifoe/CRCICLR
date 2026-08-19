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

from common import OUTPUTS, default_wbcic_repo, ensure_directories, sha256_file, write_json


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
        raise RuntimeError(f"Expected one frozen competence checkpoint for fold {fold}")
    return matches[0]


def build(wbcic_repo: Path, device_name: str, workers: int) -> Path:
    ensure_directories()
    reference = wbcic_repo / "experiments" / "persist_eeg_wbcic_actionability_v2"
    multi = wbcic_repo / "experiments" / "persist_eeg_multibackbone_final_closure"
    scope_path = reference / "outputs" / "protocol" / "DEVELOPMENT_SCOPE_LOCK.json"
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    allowed = [str(value) for value in scope["allowed_subjects"]]
    if len(allowed) != 41 or scope.get("outer_subject_ids_present") is not False:
        raise RuntimeError("WBCIC development scope is not legal")
    if scope.get("runtime_must_not_open") != "OUTER_SPLIT_LOCK.json":
        raise RuntimeError("WBCIC outer denylist is not frozen")
    inventory_path = reference / "outputs" / "protocol" / "CACHE_INVENTORY.csv"
    inventory = pd.read_csv(inventory_path)
    expected_s3_trials = int(inventory.loc[inventory.session.astype(int).eq(2), "n_trials"].sum())

    eeg_code = reference / "code"
    eeg_core = _load_module(eeg_code / "core.py", "v4_wbcic_eeg_core")
    multi_code = multi / "code"
    # multi/common.py prospectively imports its sibling models.py as `models`.
    # This standalone builder does not import the V4 models package.
    _load_module(multi_code / "models.py", "models")
    multi_common = _load_module(multi_code / "common.py", "v4_wbcic_multi_common")
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    eeg_frozen = json.loads(
        (reference / "outputs" / "protocol" / "REPRESENTATION_FROZEN.json").read_text(encoding="utf-8")
    )
    multi_frozen = {
        backbone: json.loads(
            (multi / "outputs" / "protocol" / f"BACKBONE_{backbone.upper()}_FROZEN.json").read_text(
                encoding="utf-8"
            )
        )
        for backbone in ("DeepConvNet", "EEGConformer", "TeCh")
    }
    checkpoint_audit: list[dict[str, Any]] = []
    fold_frames = []
    for fold in range(5):
        subjects = sorted([str(value) for value in scope["folds"][f"F{fold}"]], key=_subject_number)
        if not set(subjects).issubset(set(allowed)):
            raise RuntimeError("Non-development WBCIC subject requested")
        arrays_by_expert: dict[str, dict[str, np.ndarray]] = {}

        stable_record = _checkpoint_record(eeg_frozen, fold)
        stable_path = wbcic_repo / Path(stable_record["checkpoint"])
        if sha256_file(stable_path) != stable_record["checkpoint_sha256"]:
            raise RuntimeError("EEGNet stable checkpoint hash mismatch")
        model, _ = eeg_core.load_model(stable_path, device)
        arrays_by_expert["EEGNet_STABLE"] = eeg_core.infer(
            model, subjects, [2], device, workers, batch_size=256
        )
        checkpoint_audit.append(
            {"expert": "EEGNet_STABLE", "fold": fold, "path": str(stable_path), "sha256": sha256_file(stable_path)}
        )
        del model

        std_path = reference / "outputs" / "model" / "competence" / f"EEGNET_STD_fold-{fold}.pt"
        model, payload = eeg_core.load_model(std_path, device)
        if payload.get("config", {}).get("id") != "EEGNET_STD":
            raise RuntimeError("EEGNet standard checkpoint is not the frozen candidate")
        arrays_by_expert["EEGNet_STD"] = eeg_core.infer(model, subjects, [2], device, workers, batch_size=256)
        checkpoint_audit.append(
            {"expert": "EEGNet_STD", "fold": fold, "path": str(std_path), "sha256": sha256_file(std_path)}
        )
        del model

        for backbone in ("DeepConvNet", "EEGConformer", "TeCh"):
            record = _checkpoint_record(multi_frozen[backbone], fold)
            checkpoint = wbcic_repo / Path(record["checkpoint"])
            if sha256_file(checkpoint) != record["checkpoint_sha256"]:
                raise RuntimeError(f"{backbone} checkpoint hash mismatch")
            model, _ = multi_common.load_model(checkpoint, device)
            arrays_by_expert[backbone] = multi_common.infer(
                model, subjects, [2], device, workers, batch_size=256
            )
            checkpoint_audit.append(
                {"expert": backbone, "fold": fold, "path": str(checkpoint), "sha256": sha256_file(checkpoint)}
            )
            del model

        reference_arrays = arrays_by_expert["EEGNet_STABLE"]
        labels = reference_arrays["labels"].astype(int)
        actual_subjects = reference_arrays["subjects"][reference_arrays["subject_index"].astype(int)].astype(str)
        sessions = reference_arrays["session"].astype(int)
        if set(actual_subjects) != set(subjects) or np.any(sessions != 2):
            raise RuntimeError("WBCIC inference scope/session mismatch")
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
            + ":S3:"
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
                raise RuntimeError(f"WBCIC expert alignment failed: {expert} fold {fold}")
            frame[f"margin_{expert}"] = arrays["logits"][:, 1] - arrays["logits"][:, 0]
        frame["OUTER_TEST_USED"] = False
        fold_frames.append(frame)
        print(f"[WBCIC experts] fold={fold} subjects={len(subjects)} trials={len(frame)}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = pd.concat(fold_frames, ignore_index=True)
    if (
        len(output) != expected_s3_trials
        or output.subject_id.nunique() != 41
        or output.trial_uid.duplicated().any()
        or set(output.subject_id) != set(allowed)
        or output.OUTER_TEST_USED.astype(bool).any()
    ):
        raise RuntimeError("WBCIC development expert table failed coverage audit")
    output_path = OUTPUTS / "cache" / "WBCIC_DEV_KEEP_EXPERTS.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    audit = {
        "status": "WBCIC_DEVELOPMENT_EXPERT_TABLE_COMPLETE",
        "rows": len(output),
        "expected_S3_rows_from_frozen_cache_inventory": expected_s3_trials,
        "subjects": output.subject_id.nunique(),
        "sessions": sorted(output.session_id.unique().astype(int).tolist()),
        "expert_roster": list(EXPERTS),
        "expert_count": len(EXPERTS),
        "all_experts_task_competent": True,
        "development_scope_sha256": sha256_file(scope_path),
        "development_cache_inventory_sha256": sha256_file(inventory_path),
        "output_sha256": sha256_file(output_path),
        "checkpoint_audit": checkpoint_audit,
        "outer_split_lock_opened": False,
        "raw_root_enumerated": False,
        "outer_subject_ids_loaded": False,
        "OUTER_TEST_USED": False,
    }
    write_json(OUTPUTS / "protocol" / "WBCIC_EXPERT_TABLE_AUDIT.json", audit)
    print(json.dumps({key: audit[key] for key in ("status", "rows", "subjects", "expert_roster")}, indent=2))
    return output_path


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
