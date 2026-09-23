"""Production first-cell mediation phase; not a complete mechanism cell.

Writes independently checkable TRAIN/outer subject-level rows for every
adjacent and preregistered long-range transition. Directional controls and
training replay are separate required phases; this phase cannot mark a cell
COMPLETE or be used as a final scientific result on its own.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
RUNTIME = Path(os.environ.get("MECHANISM_RUNTIME", str(ROOT.parent / "pc_mechanism_closure_runtime")))
sys.path.insert(0, str(ROOT / "experiments" / "persist_eeg_protected_complement_coupling_seed0_v1" / "code"))
import run_coupling as C  # noqa: E402
from cell_data_v1 import materialize_stage, stage_norms  # noqa: E402
from mediation_core_v4 import LONG_RANGE_STAGES, adjacent_mediation, long_range_mediation, verify_adjacent_forward  # noqa: E402
from sampling_core import trial_view  # noqa: E402

AMENDMENT_SHA = "aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474"
ANALYSIS_SHA = "ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88"
ORIGINAL_IMPL_SHA = "4aab984969f4ba43128ed738a38ac138d94ee1c9b1252662af91b0c44e3f8eee"


def verified_context(model: str, task: str, fold: int):
    gate_path = EXP / "gate" / "GATE_AUDIT.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (gate.get("status") != "PASSED" or gate.get("original_cells_complete") != 20
            or gate.get("layer_stage_cells_reconstructed") != 120 or gate.get("final_heldout_accessed")):
        raise RuntimeError("upstream Phase 2 gate is not PASSED")
    if C.digest(EXP / "protocol" / "FINAL_PROJECTOR_AMENDMENT.md") != AMENDMENT_SHA:
        raise RuntimeError("authorized final projector amendment hash mismatch")
    if C.digest(EXP / "protocol" / "MECHANISM_CLOSURE_ANALYSIS_LOCK.md") != ANALYSIS_SHA:
        raise RuntimeError("frozen mechanism analysis lock hash mismatch")
    source = C.prior_record(model, task, fold)
    protocol_sha = C.assert_lock(model, task, fold, source)
    if protocol_sha != gate["upstream_protocol_sha256"]:
        raise RuntimeError("upstream protocol differs from Phase 2 gate")
    historical = json.loads(C.cell_path(model, task, fold).read_text(encoding="utf-8"))
    if (historical.get("status") != "COMPLETE" or historical.get("protocol_sha256") != protocol_sha
            or historical.get("implementation_sha256") != ORIGINAL_IMPL_SHA
            or historical.get("final_heldout_accessed") is not False):
        raise RuntimeError("frozen coupling cell is invalid")
    if source["hashes"]["checkpoint_sha256"] != historical["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA differs from original coupling cell")
    return gate_path, gate, source, protocol_sha


def subject_session_rows(metrics: dict[str, np.ndarray], recipient_global: np.ndarray,
                         subjects: np.ndarray, sessions: np.ndarray, *,
                         model: str, task: str, fold: int, split: str, stage: str,
                         successor: str, kind: str, protocol_sha: str) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for position, row in enumerate(recipient_global):
        groups[(str(subjects[row]), int(sessions[row]))].append(position)
    rows = []
    for (subject, session), positions in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        values = {key: float(np.mean(np.asarray(value)[positions])) for key, value in metrics.items()}
        if any(not np.isfinite(value) for value in values.values()):
            raise RuntimeError(f"nonfinite subject-level mediation metric: {stage}/{subject}/{session}")
        rows.append({
            "model": model, "task": task, "fold": fold, "seed": 0, "split": split,
            "stage": stage, "successor": successor, "transition_kind": kind,
            "subject": subject, "session": session, "pair_count": len(positions),
            "projector_mode_successor": "FROZEN_CANONICAL_OBLIQUE" if successor == "classifier_input" else "TRAIN_ONLY_ORTHOGONAL_PATHWAY",
            "upstream_protocol_sha256": protocol_sha, "upstream_implementation_sha256": ORIGINAL_IMPL_SHA,
            **values,
        })
    return rows


def measured_pairs(runner, stage: str, selection, view, q, mu, q_next, mu_next, shape,
                   final_projector, *, long_range: bool) -> dict[str, np.ndarray]:
    callback = long_range_mediation if long_range else adjacent_mediation
    results: dict[str, list[np.ndarray]] = defaultdict(list)
    for start in range(0, len(selection.recipient_local), 8):
        stop = min(start + 8, len(selection.recipient_local))
        recipient_idx = selection.recipient_local[start:stop]
        donor_idx = selection.donor_local[start:stop]
        labels = view.labels[selection.recipient_global[start:stop]]
        donor_labels = selection.donor_label[start:stop]
        args = (runner, stage, selection.activations[recipient_idx], selection.activations[donor_idx],
                tuple(shape), q, mu, q_next, mu_next, labels, donor_labels)
        part = callback(*args, final_projector) if long_range else callback(*args, final_projector)
        for key, values in part.items():
            if not np.isfinite(values).all():
                raise RuntimeError(f"nonfinite {stage} {key} mediation value")
            results[key].append(np.asarray(values, np.float32))
    return {key: np.concatenate(values) for key, values in results.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("EEGNet", "EEGConformer"), required=True)
    parser.add_argument("--task", choices=("OpenBMI_MI", "OpenBMI_SSVEP"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    model, task, fold = args.model, args.task, args.fold
    cell_dir = RUNTIME / "cells" / model.lower() / task.lower() / f"fold{fold}_seed0"
    manifest = cell_dir / "MEDIATION_V1.json"
    failure = cell_dir / "MEDIATION_V1_FAIL_CLOSED.json"
    if manifest.exists() or failure.exists():
        raise RuntimeError("mediation cell phase already has terminal evidence; no overwrite")
    try:
        gate_path, gate, source, protocol_sha = verified_context(model, task, fold)
        record, stored, checkpoint, data = source["record"], source["stored"], source["checkpoint"], source["data"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net, head = C.UP.helper(model).build_model({
            "Model": model, "Task": task, "fold": fold, "seed": 0,
            "channels": int(record.get("channels") or 62), "samples": int(record.get("samples") or 1000),
            "classes": int(record["classes"]), "checkpoint_path": str(checkpoint),
            "recipe_name": record.get("recipe", {}).get("name"),
            "trainable_parameters": int(record.get("trainable_parameters", record.get("parameters", 0))),
        }, device)
        runner = C.PW.Stages(net, head, model, device)
        tx, _, _, _ = C.BR.trial_train(data, model, task, fold)
        C.PW.stage_check(runner, tx)
        forward_errors = verify_adjacent_forward(runner, tx)
        centroid = C.PW.train_centroids(runner, data, model, task, fold)
        capped = C.UP.capped_train(data, task, model, fold)
        bh, _, _ = C.UP.hook_representations(net, head, capped[0], model, device)
        spec = C.UP.helper(model).spectrum(bh, *capped[1:], task, model, fold)
        dims = np.asarray(stored["protected_blocks"], dtype=int)
        if C.UP.array_sha(spec["mean"], spec["basis"], spec["scale"], spec["directions"]) != source["hashes"]["basis_sha256"]:
            raise RuntimeError("frozen canonical basis SHA mismatch")
        if int(spec["rank"]) != int(stored["rank"]):
            raise RuntimeError("frozen canonical rank mismatch")
        canonical = C.PW.canonical(centroid["h"], spec)
        mapping = {}
        for stage in runner.names:
            if stage == "classifier_input":
                q, mu = C.final_q(spec, dims)
            else:
                q, mu = C.raw_q(C.PW.PathFit(centroid["acts"][stage]), canonical[:, dims])
            cache = C.RUNTIME / "stage_cache" / model / task / f"fold{fold}" / f"{stage}.json"
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if (cached.get("mapping_sha256") != C.AR.array_digest(q, mu)
                    or cached.get("protocol_sha256") != protocol_sha
                    or cached.get("implementation_sha256") != ORIGINAL_IMPL_SHA):
                raise RuntimeError(f"frozen TRAIN mapping/cache mismatch: {stage}")
            mapping[stage] = (q, mu)
        final_left = np.ascontiguousarray(((spec["basis"] / spec["scale"][None, :]) @ spec["directions"][:, dims]).astype(np.float32))
        final_right = np.ascontiguousarray(C.UP.raw_base(spec)[dims].astype(np.float32))
        th, tz, _ = C.UP.hook_representations(net, head, tx[:8], model, device)
        tq, _, _, _ = C.BR.decompose(th, tz, spec, dims, head)
        canonical_error = float(np.max(np.abs((tq[:, dims] @ final_right) - ((th - spec["mean"]) @ final_left) @ final_right)))
        if canonical_error >= 1e-5:
            raise RuntimeError(f"canonical final P disagrees with frozen projector: {canonical_error}")
        views = {split: trial_view(data, split) for split in ("TRAIN", "OUTER_DEVELOPMENT")}
        norms = {split: stage_norms(runner, view, tuple(runner.names)) for split, view in views.items()}
        print("MEDIATION_NORMS_COMPLETE", model, task, fold, flush=True)
        stage_files = []
        for stage in runner.names[:-1]:
            destination = cell_dir / "mediation_v1" / f"{stage}.json"
            if destination.exists():
                cached = json.loads(destination.read_text(encoding="utf-8"))
                if (cached.get("status") != "STAGE_COMPLETE" or cached.get("implementation_sha256") != C.digest(Path(__file__))):
                    raise RuntimeError(f"stale phase-2 stage evidence: {stage}")
                stage_files.append(destination)
                print("MEDIATION_STAGE_CACHED", stage, flush=True)
                continue
            successor = runner.names[runner.names.index(stage) + 1]
            q, mu = mapping[stage]
            q_next, mu_next = mapping[successor]
            final_projector = (final_left, final_right) if successor == "classifier_input" else None
            stage_rows = []
            coverages = {}
            for split, view in views.items():
                selected = materialize_stage(runner, view, model=model, task=task, fold=fold,
                                             stage=stage, precomputed_norms=norms[split][stage])
                coverages[split] = list(selected.coverage)
                effects = measured_pairs(runner, stage, selected, view, q, mu, q_next, mu_next,
                                         centroid["shapes"][stage], final_projector, long_range=False)
                stage_rows.extend(subject_session_rows(effects, selected.recipient_global, view.subjects,
                                                       view.sessions, model=model, task=task, fold=fold,
                                                       split=split, stage=stage, successor=successor,
                                                       kind="ADJACENT", protocol_sha=protocol_sha))
                if stage in LONG_RANGE_STAGES[model]:
                    qf, muf = mapping["classifier_input"]
                    long_effects = measured_pairs(runner, stage, selected, view, q, mu, qf, muf,
                                                  centroid["shapes"][stage], (final_left, final_right),
                                                  long_range=True)
                    stage_rows.extend(subject_session_rows(long_effects, selected.recipient_global, view.subjects,
                                                           view.sessions, model=model, task=task, fold=fold,
                                                           split=split, stage=stage, successor="classifier_input",
                                                           kind="LONG_RANGE", protocol_sha=protocol_sha))
            C.write_json(destination, {
                "status": "STAGE_COMPLETE", "schema": "PC_MECHANISM_MEDIATION_STAGE_V1",
                "model": model, "task": task, "fold": fold, "stage": stage,
                "upstream_gate_sha256": C.digest(gate_path),
                "upstream_protocol_sha256": protocol_sha,
                "upstream_implementation_sha256": ORIGINAL_IMPL_SHA,
                "analysis_lock_sha256": ANALYSIS_SHA, "projector_amendment_sha256": AMENDMENT_SHA,
                "current_mapping_sha256": C.AR.array_digest(q, mu),
                "successor_mapping_sha256": C.AR.array_digest(q_next, mu_next),
                "subject_session_rows": stage_rows, "coverage": coverages,
                "implementation_sha256": C.digest(Path(__file__)), "final_heldout_accessed": False,
            })
            stage_files.append(destination)
            print("MEDIATION_STAGE_COMPLETE", model, task, fold, stage, len(stage_rows), flush=True)
        C.write_json(manifest, {
            "status": "MEDIATION_PHASE_COMPLETE_PENDING_OTHER_REQUIRED_ANALYSES",
            "model": model, "task": task, "fold": fold, "seed": 0,
            "stage_count": len(stage_files), "stage_sha256": {p.stem: C.digest(p) for p in stage_files},
            "native_forward_max_abs": forward_errors, "canonical_final_P_max_abs": canonical_error,
            "upstream_gate_sha256": C.digest(gate_path),
            "upstream_protocol_sha256": protocol_sha,
            "upstream_implementation_sha256": ORIGINAL_IMPL_SHA,
            "implementation_sha256": C.digest(Path(__file__)), "final_heldout_accessed": False,
        })
        print("MEDIATION_PHASE_COMPLETE_PENDING_OTHER_REQUIRED_ANALYSES", model, task, fold, flush=True)
    except Exception as error:
        C.write_json(failure, {
            "status": "FAIL_CLOSED", "reason": f"{type(error).__name__}: {error}",
            "implementation_sha256": C.digest(Path(__file__)), "final_heldout_accessed": False,
        })
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
