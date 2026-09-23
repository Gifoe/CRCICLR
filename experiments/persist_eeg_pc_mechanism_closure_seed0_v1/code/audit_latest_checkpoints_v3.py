"""Source-recipe-specific correction to the historical endpoint audit.

The seven-backbone EEGNet search runs to its configured epoch maximum;
EEGConformer may legally stop earlier. Both retain latest.pt. Earlier v1/v2
availability records are preserved and explicitly superseded.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
sys.path.insert(0, str(ROOT / "experiments" / "persist_eeg_protected_complement_coupling_seed0_v1" / "code"))
import run_coupling as C  # noqa: E402


def main() -> None:
    folder = EXP / "gate"
    previous_path = folder / "TRAINING_CHECKPOINT_AVAILABILITY_V2.json"
    destination = folder / "TRAINING_CHECKPOINT_AVAILABILITY_V3.json"
    if destination.exists():
        raise RuntimeError("v3 availability output already exists; no overwrite")
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    if previous.get("status") != "COMPLETE" or len(previous.get("cells", [])) != 20:
        raise RuntimeError("prior checkpoint audit incomplete")
    rows = []
    for old in previous["cells"]:
        model, task, fold = old["model"], old["task"], int(old["fold"])
        record, selected_path, _, _ = C.UP.provenance_row(model, task, fold)
        latest_path = Path(old["latest_path"])
        if not latest_path.is_file() or C.digest(latest_path) != old.get("latest_sha256"):
            raise RuntimeError(f"latest checkpoint changed: {model}/{task}/{fold}")
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        source = "SEVEN_BACKBONE_SEARCH" if model == "EEGNet" else "EEGCONFORMER_EARLY_STOPPING"
        expected_epoch = (record["recipe"]["epochs"] if model == "EEGNet"
                          else record["epochs_completed"])
        expected_best = (record["best_inner_validation_BA"] if model == "EEGNet"
                         else record["inner_validation_BA"])
        checks = dict(old["latest_checks"])
        checks["final_epoch"] = latest.get("epoch") == expected_epoch
        checks["best_validation"] = abs(float(latest["best"]) - float(expected_best)) < 1e-10
        checks["selected_checkpoint_hash"] = C.digest(selected_path) == old["historical_best_sha256"]
        verified = all(checks.values())
        rows.append({
            "model": model, "task": task, "fold": fold, "seed": 0,
            "source_training_recipe": source,
            "historical_selected_checkpoint_sha256": old["historical_best_sha256"],
            "historical_latest_checkpoint_sha256": old["latest_sha256"],
            "historical_last_completed_epoch": latest["epoch"],
            "historical_selected_epoch": record["selected_epoch"],
            "historical_early_stopped": bool(record.get("early_stopped", False)),
            "historical_endpoint_verified": verified,
            "intermediate_10_25_50_75_verified": False,
            "replay_required_for_intermediate_schedule": True,
            "checks": checks,
        })
        print("HISTORICAL_ENDPOINT", model, task, fold, verified, flush=True)
    C.write_json(destination, {
        "status": "COMPLETE",
        "schema": "PERSIST_EEG_PC_MECHANISM_CLOSURE_CHECKPOINT_AVAILABILITY_V3",
        "supersedes_v2_sha256": C.digest(previous_path),
        "historical_endpoint_verified_cells": sum(row["historical_endpoint_verified"] for row in rows),
        "intermediate_schedule_verified_cells": 0,
        "replay_required_for_intermediate_schedule": True,
        "final_heldout_accessed": False,
        "cells": rows,
    })
    print("HISTORICAL_ENDPOINT_AUDIT_COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    main()
