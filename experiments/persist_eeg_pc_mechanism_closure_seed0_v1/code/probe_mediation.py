"""One-cell TRAIN-only engineering probe of adjacent mediation, not a result run."""
from __future__ import annotations

import json
import os
import sys
import traceback
import argparse
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
sys.path.insert(0, str(ROOT / "experiments" / "persist_eeg_protected_complement_coupling_seed0_v1" / "code"))
import run_coupling as C  # noqa: E402
from mediation_core_v4 import LONG_RANGE_STAGES, adjacent_mediation, long_range_mediation, verify_adjacent_forward  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=C.MODELS, default="EEGConformer")
    parser.add_argument("--task", choices=C.TASKS, default="OpenBMI_MI")
    parser.add_argument("--fold", type=int, choices=C.FOLDS, default=0)
    args = parser.parse_args()
    model, task, fold = args.model, args.task, args.fold
    slug = f"{model}_{task}_fold{fold}_canonical_final_v5"
    target = EXP / "probe" / f"{slug}.json"
    failure = EXP / "probe" / f"{slug}_FAIL_CLOSED.json"
    if target.exists() or failure.exists():
        raise RuntimeError("probe evidence already exists; no overwrite or duplicate run")
    try:
        gate_path = EXP / "gate" / "GATE_AUDIT.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate["status"] != "PASSED" or gate["original_cells_complete"] != 20 or gate["layer_stage_cells_reconstructed"] != 120:
            raise RuntimeError("upstream gate is not fully passed")
        if gate["final_heldout_accessed"]:
            raise RuntimeError("final-heldout gate failure")
        source = C.prior_record(model, task, fold)
        protocol_sha = C.assert_lock(model, task, fold, source)
        if protocol_sha != gate["upstream_protocol_sha256"]:
            raise RuntimeError("protocol SHA differs from upstream gate")
        historical_cell = json.loads(C.cell_path(model, task, fold).read_text(encoding="utf-8"))
        if historical_cell.get("status") != "COMPLETE" or historical_cell.get("implementation_sha256") != gate["upstream_implementation_sha256"]:
            raise RuntimeError("upstream cell/hash differs from gate")
        record, stored, checkpoint, data = source["record"], source["stored"], source["checkpoint"], source["data"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net, head = C.UP.helper(model).build_model({
            "Model": model, "Task": task, "fold": fold, "seed": 0,
            "channels": int(record.get("channels") or 62),
            "samples": int(record.get("samples") or 1000),
            "classes": int(record["classes"]),
            "checkpoint_path": str(checkpoint),
            "recipe_name": record.get("recipe", {}).get("name"),
            "trainable_parameters": int(record.get("trainable_parameters", record.get("parameters", 0))),
        }, device)
        runner = C.PW.Stages(net, head, model, device)
        tx, ty, _, _ = C.BR.trial_train(data, model, task, fold)
        C.PW.stage_check(runner, tx)
        adjacent_errors = verify_adjacent_forward(runner, tx)
        centroid = C.PW.train_centroids(runner, data, model, task, fold)
        capped = C.UP.capped_train(data, task, model, fold)
        bh, _, _ = C.UP.hook_representations(net, head, capped[0], model, device)
        spec = C.UP.helper(model).spectrum(bh, *capped[1:], task, model, fold)
        dims = np.asarray(stored["protected_blocks"], int)
        if C.UP.array_sha(spec["mean"], spec["basis"], spec["scale"], spec["directions"]) != source["hashes"]["basis_sha256"]:
            raise RuntimeError("canonical basis SHA mismatch")
        canonical = C.PW.canonical(centroid["h"], spec)
        q_by_stage: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for stage in runner.names:
            if stage == "classifier_input":
                q, mu = C.final_q(spec, dims)
            else:
                q, mu = C.raw_q(C.PW.PathFit(centroid["acts"][stage]), canonical[:, dims])
            stage_file = C.RUNTIME / "stage_cache" / model / task / f"fold{fold}" / f"{stage}.json"
            frozen_stage = json.loads(stage_file.read_text(encoding="utf-8"))
            if frozen_stage.get("mapping_sha256") != C.AR.array_digest(q, mu):
                raise RuntimeError(f"TRAIN-only P mapping changed at {stage}")
            q_by_stage[stage] = (q, mu)
        # The probe is deliberately limited to two TRAIN trials. The class
        # distinction checks donor-margin handling; it is not a scientific
        # donor-selection rule or an outer-development evaluation.
        i = 0
        other = np.flatnonzero(ty != ty[i])
        if not len(other):
            raise RuntimeError("probe TRAIN data lacks a second class")
        j = int(other[0])
        pair_x = tx[[i, j]]
        final_left = np.ascontiguousarray(((spec["basis"] / spec["scale"][None, :]) @ spec["directions"][:, dims]).astype(np.float32))
        final_right = np.ascontiguousarray(C.UP.raw_base(spec)[dims].astype(np.float32))
        th, tz, _ = C.UP.hook_representations(net, head, pair_x, model, device)
        tq, _, _, _ = C.BR.decompose(th, tz, spec, dims, head)
        canonical_p = tq[:, dims] @ final_right
        projected_p = ((th - spec["mean"]) @ final_left) @ final_right
        final_projection_error = float(np.max(np.abs(canonical_p - projected_p)))
        if final_projection_error >= 1e-5:
            raise RuntimeError(f"amended final projector differs from frozen canonical P: {final_projection_error}")
        results = []
        for stage in runner.names[:-1]:
            following = runner.names[runner.names.index(stage) + 1]
            activation = C.PW.outer_stage(runner, pair_x, stage)["a"]
            q, mu = q_by_stage[stage]
            q_next, mu_next = q_by_stage[following]
            metrics = adjacent_mediation(
                runner, stage, activation[[0]], activation[[1]],
                tuple(centroid["shapes"][stage]), q, mu, q_next, mu_next,
                ty[[i]], ty[[j]],
                (final_left, final_right) if following == "classifier_input" else None,
            )
            if not all(np.isfinite(value).all() for value in metrics.values()):
                raise RuntimeError(f"nonfinite mediation metric at {stage}")
            results.append({
                "stage": stage, "next_stage": following,
                "P_total_norm": float(metrics["P_total_norm"][0]),
                "P_to_P_norm": float(metrics["P_to_P_norm"][0]),
                "P_to_C_norm": float(metrics["P_to_C_norm"][0]),
                "C_total_norm": float(metrics["C_total_norm"][0]),
                "C_to_P_norm": float(metrics["C_to_P_norm"][0]),
                "C_to_C_norm": float(metrics["C_to_C_norm"][0]),
            })
            print("MEDIATION_PROBE_STAGE", stage, "OK", flush=True)
        long_range = []
        final_q, final_mu = q_by_stage["classifier_input"]
        for stage in LONG_RANGE_STAGES[model]:
            activation = C.PW.outer_stage(runner, pair_x, stage)["a"]
            q, mu = q_by_stage[stage]
            metrics = long_range_mediation(
                runner, stage, activation[[0]], activation[[1]],
                tuple(centroid["shapes"][stage]), q, mu, final_q, final_mu,
                ty[[i]], ty[[j]], (final_left, final_right),
            )
            if not all(np.isfinite(value).all() for value in metrics.values()):
                raise RuntimeError(f"nonfinite long-range mediation metric at {stage}")
            long_range.append({
                "stage": stage, "final_stage": "classifier_input",
                "P_total_norm": float(metrics["P_total_norm"][0]),
                "P_to_P_norm": float(metrics["P_to_P_norm"][0]),
                "P_to_C_norm": float(metrics["P_to_C_norm"][0]),
                "C_total_norm": float(metrics["C_total_norm"][0]),
                "C_to_P_norm": float(metrics["C_to_P_norm"][0]),
                "C_to_C_norm": float(metrics["C_to_C_norm"][0]),
            })
            print("LONG_RANGE_PROBE_STAGE", stage, "OK", flush=True)
        C.write_json(target, {
            "status": "PROBE_PASSED", "schema": "PC_MECHANISM_CLOSURE_CANONICAL_FINAL_ENGINEERING_PROBE_V5",
            "model": model, "task": task, "fold": fold, "seed": 0,
            "data_scope": "TRAIN_ONLY_TWO_TRIAL_ENGINEERING_CHECK",
            "upstream_gate_sha256": C.digest(gate_path),
            "upstream_protocol_sha256": protocol_sha,
            "upstream_cell_sha256": C.digest(C.cell_path(model, task, fold)),
            "mediation_core_sha256": C.digest(Path(__file__).with_name("mediation_core_v4.py")),
            "final_projector_mode": "FROZEN_CANONICAL_OBLIQUE",
            "intermediate_projector_mode": "TRAIN_ONLY_ORTHOGONAL_PATHWAY",
            "canonical_final_P_max_abs": final_projection_error,
            "implementation_sha256": C.digest(Path(__file__)),
            "adjacent_forward_max_abs": adjacent_errors,
            "transitions": results,
            "long_range": long_range,
            "final_heldout_accessed": False,
        })
        print("MEDIATION_PROBE_PASSED", len(results), len(long_range), flush=True)
    except Exception as error:
        C.write_json(failure, {
            "status": "FAIL_CLOSED", "reason": f"{type(error).__name__}: {error}",
            "implementation_sha256": C.digest(Path(__file__)),
            "final_heldout_accessed": False,
        })
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
