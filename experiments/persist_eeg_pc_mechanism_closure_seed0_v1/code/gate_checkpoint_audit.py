"""Read-only upstream integrity gate and training-checkpoint availability audit.

No final-heldout loader is called. An integrity failure writes FAIL_CLOSED evidence
and prohibits downstream mechanism-closure cells.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
COUPLING = ROOT / "experiments" / "persist_eeg_protected_complement_coupling_seed0_v1"
sys.path.insert(0, str(COUPLING / "code"))
import run_coupling as C  # noqa: E402

ORIGINAL_SHA = "4aab984969f4ba43128ed738a38ac138d94ee1c9b1252662af91b0c44e3f8eee"
PROTOCOL_SHA = "3ddb5006bf7159b00a0c80fed31856849ccdaa69ecf5060b0161c8673327e15c"
REPAIR_SHA = "a3008191adff61b6009f63b8dfc9150713f23792212e3a9b1605f04f42f9f9c8"
PUBLISH_SHA = "295e8bda5041d039e6a87226308bbaa638b2b5f50f12b8c5590cf4ea018f9ab0"
OUT = EXP / "gate"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_csv(path: Path, expected_rows: int, metric: str) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == expected_rows, f"{path.name} row count mismatch")
    require(all(float(row[metric]) < 1e-5 for row in rows), f"{path.name} reconstruction failed")
    require(all(row.get("final_heldout_accessed", "False") == "False" for row in rows),
            f"{path.name} final-heldout flag")


def main() -> None:
    if C.digest(COUPLING / "protocol" / "PROVENANCE.json") != PROTOCOL_SHA:
        raise RuntimeError("frozen coupling protocol hash mismatch")
    if C.digest(COUPLING / "code" / "run_coupling.py") != ORIGINAL_SHA:
        raise RuntimeError("frozen coupling implementation hash mismatch")
    if C.digest(COUPLING / "code" / "repair_error_predictability_v2.py") != REPAIR_SHA:
        raise RuntimeError("separately provenanced prediction repair hash mismatch")
    publish_path = COUPLING / "outputs_compact" / "PUBLISH_VALIDATION.json"
    if C.digest(publish_path) != PUBLISH_SHA:
        raise RuntimeError("reviewed coupling publication manifest hash mismatch")
    publish = json.loads(publish_path.read_text(encoding="utf-8"))
    require(publish.get("original_cells_valid") == 20 and publish.get("repaired_cells_valid") == 20,
            "reviewed coupling cell count mismatch")
    require(publish.get("criterion_C_confirmatory_eligible") is False and
            publish.get("final_heldout_accessed") is False, "reviewed publication scientific flags mismatch")
    require(publish.get("criterion_A_qualifying_layer_summaries", 0) > 0,
            "no unchanged upstream mechanism progression evidence")
    full_dirs = {"original_full_output_manifest": COUPLING / "outputs",
                 "repaired_full_output_manifest": COUPLING / "outputs_protocol_repair_v2"}
    for key, folder in full_dirs.items():
        for name, metadata in publish[key].items():
            path = folder / name
            require(path.is_file() and C.digest(path) == metadata["sha256"],
                    f"upstream full output hash mismatch: {key}/{name}")
    check_csv(COUPLING / "outputs" / "FINAL_PC_DECOMPOSITION_AUDIT.csv", 20, "max_abs_z_error")
    check_csv(COUPLING / "outputs" / "LAYER_PC_DECOMPOSITION_AUDIT.csv", 120, "max_abs_reconstruction")
    checkpoint_rows = []
    stage_count = 0
    for model in C.MODELS:
        for task in C.TASKS:
            for fold in C.FOLDS:
                row = json.loads(C.cell_path(model, task, fold).read_text(encoding="utf-8"))
                require(row.get("status") == "COMPLETE" and row.get("protocol_sha256") == PROTOCOL_SHA and
                        row.get("implementation_sha256") == ORIGINAL_SHA and
                        row.get("final_heldout_accessed") is False and
                        row.get("final_decomposition_max_abs", 1) < 1e-5,
                        f"upstream coupling cell invalid: {model}/{task}/{fold}")
                require(len(row.get("stages", [])) == (5 if model == "EEGNet" else 7),
                        "upstream stage count mismatch")
                for stage in row["stages"]:
                    require(stage["reconstruction_max_abs"] < 1e-5 and
                            len({r["draw"] for r in stage["random_subject_effects"]}) == 100 and
                            len({r["draw"] for r in stage["random_surrogate"]}) == 20,
                            f"upstream stage audit failed: {model}/{task}/{fold}/{stage['stage']}")
                    stage_count += 1
                record, best, stored, _ = C.UP.provenance_row(model, task, fold)
                require(C.digest(best) == row["checkpoint_sha256"], "historical checkpoint mismatch")
                # This is only an availability inventory. Nearby files are not
                # declared historical epoch checkpoints without identity proof.
                nearby = sorted(p.name for p in best.parent.iterdir()
                                if p.is_file() and p.suffix.lower() in {".pt", ".pth", ".ckpt"}
                                and p != best and ("epoch" in p.stem.lower() or "ep_" in p.stem.lower()))
                checkpoint_rows.append({
                    "model": model, "task": task, "fold": fold, "seed": 0,
                    "historical_best_path": str(best),
                    "historical_best_sha256": C.digest(best),
                    "record_best_epoch": record.get("best_epoch"),
                    "record_epoch": record.get("epoch"),
                    "record_keys": sorted(record),
                    "nearby_epoch_named_files_unverified": nearby,
                    "verified_intermediate_checkpoint_count": 0,
                    "availability_status": "UNVERIFIED_CANDIDATES" if nearby else "REPLAY_REQUIRED_IF_TRAJECTORY_REQUESTED",
                })
    require(stage_count == 120, "expected 120 upstream stage cells")
    gate = {
        "status": "PASSED", "schema": "PERSIST_EEG_PC_MECHANISM_CLOSURE_GATE_V1",
        "scope": "EEGNet/EEGConformer x OpenBMI_MI/OpenBMI_SSVEP x seed0 x folds0-4",
        "upstream_protocol_sha256": PROTOCOL_SHA,
        "upstream_implementation_sha256": ORIGINAL_SHA,
        "upstream_repair_sha256": REPAIR_SHA,
        "upstream_reviewed_manifest_sha256": PUBLISH_SHA,
        "original_cells_complete": 20, "repair_cells_complete": 20,
        "layer_stage_cells_reconstructed": stage_count,
        "original_future_error_prediction_valid": False,
        "corrected_prediction_confirmatory_eligible": False,
        "upstream_criterion_A_qualifying_layer_summaries": publish["criterion_A_qualifying_layer_summaries"],
        "final_heldout_accessed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    C.write_json(OUT / "GATE_AUDIT.json", gate)
    C.write_json(OUT / "TRAINING_CHECKPOINT_AVAILABILITY.json", checkpoint_rows)
    print("MECHANISM_CLOSURE_GATE_PASSED", stage_count, len(checkpoint_rows), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        C.write_json(OUT / "FAIL_CLOSED.json", {
            "status": "FAIL_CLOSED", "reason": f"{type(exc).__name__}: {exc}",
            "final_heldout_accessed": False,
        })
        traceback.print_exc()
        raise
