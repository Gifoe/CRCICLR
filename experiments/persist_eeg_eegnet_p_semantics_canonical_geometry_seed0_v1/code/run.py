"""Frozen EEGNet P-semantics audit, executed one fold and phase at a time.

The geometry phase reads only inner-train EEG. Held-role readers are not called
until the geometry artifact exists and its hashes have been reverified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import data_geometry as dg


def write_new_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")


def run_geometry(fold: int) -> None:
    output = dg.RUNTIME / "geometry" / f"fold{fold}_seed0"
    if output.exists():
        raise FileExistsError(f"geometry already exists; refusing to overwrite: {output}")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    een, pw, up = dg.upstream()
    train, context = dg.load_train(fold, een, up)
    record, checkpoint_path, model, head, device = dg.checkpoint(fold, train, context, een)
    original_sha = dg.model_sha(model)
    stages = pw.Stages(model, head, "EEGNet", device)
    features, logits = dg.extract(train, stages)
    geometry, provenance = dg.fit_geometry(fold, train, features, context, een, pw)
    if dg.model_sha(model) != original_sha:
        raise RuntimeError("frozen EEGNet parameter or BatchNorm buffer changed")
    if any(not np.isfinite(value).all() for value in features.values()):
        raise RuntimeError("nonfinite stage activation")
    if not np.isfinite(logits).all():
        raise RuntimeError("nonfinite native logits")
    provenance.update({
        "schema": "P_SEMANTICS_EEGNET_INNER_TRAIN_GEOMETRY_V1",
        "checkpoint_sha256": dg.file_sha(checkpoint_path),
        "checkpoint_record_sha256": dg.array_sha(np.frombuffer(json.dumps(record, sort_keys=True).encode(), dtype=np.uint8)),
        "model_state_sha256_before_after": original_sha,
        "canonical_checkpoint_selected_epoch": record["selected_epoch"],
        "stage_names": list(dg.STAGES),
        "train_capped_source_rows": context["capped_source_rows"],
        "train_capped_future_rows": context["capped_future_rows"],
        "train_source_normalizer_full_raw_rows": context["raw_source_rows"],
        "train_future_full_raw_rows": context["raw_future_rows"],
        "train_feature_sha256": {key: dg.array_sha(value) for key, value in features.items()},
        "train_logits_sha256": dg.array_sha(logits),
        "checkpoint_validation_eeg_reads": 0,
        "outer_development_eeg_reads": 0,
        "final_heldout_eeg_reads": 0,
    })
    output.mkdir(parents=True, exist_ok=False)
    for stage, entry in geometry.items():
        np.savez(output / f"{stage}.npz", mu=entry["mu"], q=entry["q"])
        actual = np.load(output / f"{stage}.npz")
        if dg.array_sha(actual["mu"], actual["q"]) != entry["sha256"]:
            raise RuntimeError(f"geometry serialization mismatch {stage}")
    write_new_json(output / "GEOMETRY_PROVENANCE.json", provenance)
    print(json.dumps({"fold": fold, "checkpoint_sha256": provenance["checkpoint_sha256"],
                      "selected_P_rank": provenance["final_P_rank"],
                      "stage_geometry_sha256": provenance["stage_geometry_sha256"],
                      "final_heldout_eeg_reads": 0}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("geometry",))
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    if args.phase == "geometry":
        run_geometry(args.fold)
