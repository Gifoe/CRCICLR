"""Enforce the global all-runs source freeze before any outer label access."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd

import common


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    target = common.RUNTIME / "GLOBAL_SOURCE_FREEZE.json"
    if target.exists():
        marker = common.read_json(target)
        if marker.get("pass") is not True or len(marker.get("runs", {})) != 30:
            raise RuntimeError("invalid existing global source freeze")
        print("[global source freeze cached]")
        return
    manifests = {}
    ledger = []
    for backbone in common.BACKBONES:
        for fold in range(5):
            for seed in range(3):
                context = common.unit_dir(backbone, fold, seed)
                if (context / "OUTCOME_COMPLETE.json").exists():
                    raise RuntimeError("outcome artifact exists before global source freeze")
                path = context / "SOURCE_COMPLETE.json"
                if not path.exists():
                    raise RuntimeError(f"source incomplete: {backbone} fold={fold} seed={seed}")
                marker = common.read_json(path)
                if marker.get("pass") is not True or marker.get("outcome_subjects_or_labels_loaded") is not False:
                    raise RuntimeError(f"source marker failed: {path}")
                for name, expected in marker["source_artifacts"].items():
                    if sha256(context / name) != expected:
                        raise RuntimeError(f"artifact hash mismatch before freeze: {context / name}")
                frame = pd.read_csv(context / "source_direction_cells.csv")
                if len(frame) != 8 or frame.direction_SHA.nunique() != 8 or frame.outcome_loaded.any():
                    raise RuntimeError(f"source cells invalid: {context}")
                run_id = f"{backbone}|fold-{fold}|seed-{seed}"
                manifests[run_id] = {
                    "source_complete_sha256": sha256(path),
                    "checkpoint_sha256": marker["source_artifacts"]["checkpoint.pt"],
                    "directions_sha256": marker["source_artifacts"]["DIRECTIONS_FROZEN.npz"],
                    "source_cells_sha256": marker["source_artifacts"]["source_direction_cells.csv"],
                }
                training = common.read_json(context / "training.json")
                ledger.append(
                    {
                        "backbone": backbone,
                        "fold": fold,
                        "seed": seed,
                        "best_epoch": training["best_epoch"],
                        "epochs_executed": training["epochs_executed"],
                        "fit_validation_BA": training["best_validation_BA"],
                        "fit_validation_NLL": training["best_validation_NLL"],
                        "elapsed_seconds": training["elapsed_seconds"],
                        "checkpoint_SHA": training["checkpoint_sha256"],
                        "pseudo_target_used_for_training_or_selection": False,
                        "outcome_used": False,
                    }
                )
    if len(manifests) != 30:
        raise RuntimeError("global source freeze requires 30 runs")
    common.write_csv(common.RESULTS / "training_ledger.csv", pd.DataFrame(ledger))
    common.write_json(
        target,
        {
            "schema": "PERSIST_EEG_GLOBAL_SOURCE_FREEZE_V1",
            "pass": True,
            "run_count": 30,
            "direction_cell_count": 240,
            "runs": manifests,
            "all_models_directions_diagnostics_and_U_pseudo_frozen": True,
            "all_source_artifact_hashes_verified": True,
            "outer_outcome_labels_loaded_before_freeze": False,
            "outer_outcome_evaluation_authorized": True,
            "frozen_at_unix": time.time(),
        },
    )
    print("GLOBAL_SOURCE_FREEZE_PASS")


if __name__ == "__main__":
    main()
