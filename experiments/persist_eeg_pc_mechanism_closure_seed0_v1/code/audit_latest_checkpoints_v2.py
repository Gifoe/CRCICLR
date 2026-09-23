"""Correct the checkpoint inventory: verify historical latest.pt endpoints.

This preserves the earlier name-filter inventory as evidence and writes a new
availability record. It never loads task data or final-heldout subjects.
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

OUT = EXP / "gate"


def exact_state(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[key].cpu(), b[key].cpu()) for key in a)


def main() -> None:
    prior = OUT / "TRAINING_CHECKPOINT_AVAILABILITY.json"
    destination = OUT / "TRAINING_CHECKPOINT_AVAILABILITY_V2.json"
    if destination.exists():
        raise RuntimeError("v2 availability output already exists; no overwrite")
    gate = json.loads((OUT / "GATE_AUDIT.json").read_text(encoding="utf-8"))
    if gate.get("status") != "PASSED" or gate.get("original_cells_complete") != 20:
        raise RuntimeError("integrity gate did not pass")
    rows = json.loads(prior.read_text(encoding="utf-8"))
    if len(rows) != 20:
        raise RuntimeError("prior availability inventory lacks 20 cells")
    result = []
    for row in rows:
        model, task, fold = row["model"], row["task"], int(row["fold"])
        record, selected_path, _, _ = C.UP.provenance_row(model, task, fold)
        latest_path = selected_path.with_name("latest.pt")
        evidence = {
            "model": model, "task": task, "fold": fold, "seed": 0,
            "historical_best_sha256": C.digest(selected_path),
            "latest_path": str(latest_path),
            "latest_exists": latest_path.is_file(),
            "historical_final_epoch_verified": False,
            "historical_best_state_matches_latest_best_state": False,
            "intermediate_10_25_50_75_verified": False,
            "replay_required_for_intermediate_schedule": True,
        }
        if latest_path.is_file():
            latest = torch.load(latest_path, map_location="cpu", weights_only=False)
            selected = torch.load(selected_path, map_location="cpu", weights_only=False)
            checks = {
                "invariant": latest.get("invariant") == record.get("invariant_sha256") == selected.get("invariant"),
                "selected_epoch": latest.get("best_epoch") == record.get("selected_epoch") == selected.get("selected_epoch"),
                "final_epoch": latest.get("epoch") == record.get("recipe", {}).get("epochs"),
                "history_length": len(latest.get("history", [])) == latest.get("epoch"),
                "history_identity": latest.get("history") == record.get("history"),
                "best_validation": abs(float(latest.get("best", float("nan"))) -
                                       float(record.get("best_inner_validation_BA", float("nan")))) < 1e-10,
                "best_state_identity": exact_state(latest["best_state"], selected["state_dict"]),
            }
            evidence.update({
                "latest_sha256": C.digest(latest_path),
                "latest_epoch": latest.get("epoch"),
                "selected_epoch": record.get("selected_epoch"),
                "latest_checks": checks,
                "historical_final_epoch_verified": all(checks.values()),
                "historical_best_state_matches_latest_best_state": checks["best_state_identity"],
            })
        result.append(evidence)
        print("LATEST_AUDIT", model, task, fold, evidence["historical_final_epoch_verified"], flush=True)
    C.write_json(destination, {
        "status": "COMPLETE", "schema": "PERSIST_EEG_PC_MECHANISM_CLOSURE_CHECKPOINT_AVAILABILITY_V2",
        "supersedes_name_filter_only_inventory_sha256": C.digest(prior),
        "historical_final_epoch_verified_cells": sum(r["historical_final_epoch_verified"] for r in result),
        "intermediate_schedule_verified_cells": 0,
        "replay_required_for_intermediate_schedule": True,
        "final_heldout_accessed": False,
        "cells": result,
    })
    print("LATEST_CHECKPOINT_AUDIT_COMPLETE", len(result), flush=True)


if __name__ == "__main__":
    main()
