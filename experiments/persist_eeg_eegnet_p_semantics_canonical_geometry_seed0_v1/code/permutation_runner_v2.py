"""One early-stage V2 permutation audit with exact frozen-source checks."""
from __future__ import annotations

import argparse
import json

import torch

import analysis_runner as ar
import data_geometry as dg
import permutations_v2


def run(fold: int, stage: str) -> None:
    if stage == "embedding_64d":
        raise RuntimeError("embedding V1 output is preserved; V2 early stages only")
    target = dg.RUNTIME / "analysis" / f"fold{fold}_seed0" / stage / "permutations"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    een, pw, up = dg.upstream()
    train, context = dg.load_train(fold, een, up)
    _, ckpt, model, head, device = dg.checkpoint(fold, train, context, een)
    before = dg.model_sha(model)
    geometry, provenance = ar.verify_geometry(fold, stage, dg.file_sha(ckpt), context)
    held = {role: ar.cap_held(dg.load_held(role, context, een, up), fold, een)
            for role in ("CHECKPOINT_VALIDATION", "OUTER_DEVELOPMENT")}
    stages = pw.Stages(model, head, "EEGNet", device)
    populations = {"TRAIN_GEOMETRY": (train, ar.extract_stage(train, stages, stage))}
    populations.update({role: (pop, ar.extract_stage(pop, stages, stage)) for role, pop in held.items()})
    rows = permutations_v2.control_rows(fold, stage, geometry, populations, context["sessions"], 200)
    if dg.model_sha(model) != before:
        raise RuntimeError("frozen neural parameter/BatchNorm state changed")
    target.mkdir(parents=True, exist_ok=False)
    path = target / "PERMUTATION_CONTROL_SUMMARY.csv"
    ar.write_rows(path, rows)
    audit = {"schema": "P_SEMANTICS_PERMUTATION_CONTROL_V2", "fold": fold, "stage": stage,
             "checkpoint_sha256": dg.file_sha(ckpt),
             "geometry_provenance_sha256": dg.file_sha(
                 dg.RUNTIME / "geometry" / f"fold{fold}_seed0" / "GEOMETRY_PROVENANCE.json"),
             "stage_geometry_sha256": provenance["stage_geometry_sha256"][stage],
             "session_identity_permutations": 200,
             "within_cell_trial_label_permutations": 200,
             "label_permutation_stage_scope": "all four stages after V2 early-stage extension",
             "random_subspace_draws": 20, "model_state_sha256_before_after": before,
             "outer_development_fit_rows": 0, "final_heldout_eeg_reads": 0,
             "files": {path.name: {"sha256": dg.file_sha(path), "rows": len(rows)}}}
    with (target / "AUDIT.json").open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"fold": fold, "stage": stage, "rows": len(rows),
                      "sha256": dg.file_sha(path), "final_heldout_eeg_reads": 0}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--stage", choices=[stage for stage in dg.STAGES if stage != "embedding_64d"], required=True)
    args = parser.parse_args()
    run(args.fold, args.stage)
