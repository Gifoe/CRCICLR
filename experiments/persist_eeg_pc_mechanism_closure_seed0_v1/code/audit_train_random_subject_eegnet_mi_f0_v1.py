"""Recover missing TRAIN subject/session random-control rows for one frozen stage.

Each draw is independently restartable and is checked against the completed
directional-v4 control. This does not rerank pairs or change the 20 controls.
"""
from __future__ import annotations

import argparse
import hashlib
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
from cell_data_v1 import materialize_stage  # noqa: E402
from directional_core_v3 import fit_directions  # noqa: E402
from run_directional_cell_v4 import (  # noqa: E402
    evaluated_rows, matrix_and_pairs, unique_recipient_rows,
)
from run_mediation_cell_v1 import (  # noqa: E402
    verified_context, AMENDMENT_SHA, ANALYSIS_SHA, ORIGINAL_IMPL_SHA,
)
from sampling_core import trial_view  # noqa: E402

MODEL, TASK, FOLD = "EEGNet", "OpenBMI_MI", 0
STAGE = "depth_point_elu_pool2"


def main() -> None:
    global TASK, FOLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("OpenBMI_MI", "OpenBMI_SSVEP"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--draw", type=int, choices=range(20), required=True)
    args = parser.parse_args()
    TASK, FOLD, draw = args.task, args.fold, args.draw
    cell = RUNTIME / "cells" / MODEL.lower() / TASK.lower() / f"fold{FOLD}_seed0"
    result = cell / "train_random_subject_v1" / f"draw_{draw:02d}.json"
    failure = cell / "train_random_subject_v1" / f"draw_{draw:02d}.FAIL_CLOSED.json"
    if result.exists() or failure.exists():
        raise RuntimeError("random-subject draw has terminal evidence")
    try:
        gate_path, _, source, protocol_sha = verified_context(MODEL, TASK, FOLD)
        stage_path = cell / "directional_v4" / f"{STAGE}.json"
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        manifest = json.loads((cell / "DIRECTIONAL_V4.json").read_text(encoding="utf-8"))
        if (stage.get("status") != "STAGE_COMPLETE" or
                manifest.get("status") != "DIRECTIONAL_PHASE_COMPLETE_PENDING_OTHER_REQUIRED_ANALYSES" or
                C.digest(stage_path) != manifest["stage_sha256"][STAGE] or
                stage.get("analysis_lock_sha256") != ANALYSIS_SHA or
                stage.get("projector_amendment_sha256") != AMENDMENT_SHA or
                stage.get("upstream_protocol_sha256") != protocol_sha or
                stage.get("upstream_implementation_sha256") != ORIGINAL_IMPL_SHA or
                stage.get("final_heldout_accessed") is not False or
                len(stage["random_controls"]) != 20):
            raise RuntimeError("frozen directional stage/lock evidence invalid")
        control = stage["random_controls"][draw]
        subset = np.asarray(source["randoms"][draw], dtype=int)
        subset_sha = hashlib.sha256(np.ascontiguousarray(subset).tobytes()).hexdigest()
        if (int(control["draw"]) != draw or control["final_subset"] != subset.tolist() or
                control["final_subset_sha256"] != subset_sha):
            raise RuntimeError("frozen random-subspace identity mismatch")
        record, stored, checkpoint, data = (source[key] for key in
                                            ("record", "stored", "checkpoint", "data"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net, head = C.UP.helper(MODEL).build_model({
            "Model": MODEL, "Task": TASK, "fold": FOLD, "seed": 0,
            "channels": int(record.get("channels") or 62),
            "samples": int(record.get("samples") or 1000),
            "classes": int(record["classes"]), "checkpoint_path": str(checkpoint),
            "recipe_name": record.get("recipe", {}).get("name"),
            "trainable_parameters": int(record.get("trainable_parameters", record.get("parameters", 0))),
        }, device)
        runner = C.PW.Stages(net, head, MODEL, device)
        centroid = C.PW.train_centroids(runner, data, MODEL, TASK, FOLD)
        capped = C.UP.capped_train(data, TASK, MODEL, FOLD)
        h, z, hz = C.UP.hook_representations(net, head, capped[0], MODEL, device)
        if np.max(np.abs(z - hz)) >= 1e-5:
            raise RuntimeError("native historical classifier mismatch")
        spec = C.UP.helper(MODEL).spectrum(h, *capped[1:], TASK, MODEL, FOLD)
        if (C.UP.array_sha(spec["mean"], spec["basis"], spec["scale"], spec["directions"])
                != source["hashes"]["basis_sha256"] or
                int(spec["rank"]) != int(stored["rank"])):
            raise RuntimeError("historical canonical basis/rank mismatch")
        canonical = C.PW.canonical(centroid["h"], spec)
        fit = C.PW.PathFit(centroid["acts"][STAGE])
        qr, mur = C.raw_q(fit, canonical[:, subset])
        if control["mapping_sha256"] != C.AR.array_digest(qr, mur):
            raise RuntimeError("frozen TRAIN random mapping mismatch")
        view = trial_view(data, "TRAIN")
        selection = materialize_stage(runner, view, model=MODEL, task=TASK,
                                      fold=FOLD, stage=STAGE)
        train_a = selection.activations[unique_recipient_rows(selection)]
        axes = fit_directions(train_a, qr, mur, c_limit=16, device=device)
        if (axes.p_rank != int(control["p_rank"]) or
                axes.c_rank != int(control["c_rank"])):
            raise RuntimeError("frozen random direction ranks mismatch")
        top = tuple(tuple(map(int, pair)) for pair in control["train_top10"])
        if len(top) != 10 or len(set(top)) != 10:
            raise RuntimeError("invalid frozen random top-10 pair list")
        shape = tuple(centroid["shapes"][STAGE])
        matrix, effects, identity = matrix_and_pairs(
            runner, STAGE, shape, selection, view, axes, pair_subset=top)
        rows = evaluated_rows(effects, identity, top, split="TRAIN", train_means=matrix)
        groups: dict[tuple[str, int], list[float]] = defaultdict(list)
        for row in rows:
            groups[(str(row["subject"]), int(row["session"]))].append(
                float(row["interaction_norm"]))
        if len(rows) != 520 or len(groups) != 52 or any(len(values) != 10 for values in groups.values()):
            raise RuntimeError("TRAIN random subject/session top-10 coverage invalid")
        compact = [
            {"split": "TRAIN", "subject": subject, "session": session,
             "random_top10_interaction_norm": float(np.mean(values)), "direction_count": len(values)}
            for (subject, session), values in sorted(groups.items())
        ]
        if not np.isfinite([row["random_top10_interaction_norm"] for row in compact]).all():
            raise RuntimeError("nonfinite TRAIN random subject/session value")
        C.write_json(result, {
            "status": "TRAIN_RANDOM_SUBJECT_DRAW_COMPLETE", "model": MODEL,
            "task": TASK, "fold": FOLD, "seed": 0, "stage": STAGE,
            "draw": draw, "final_subset_sha256": subset_sha,
            "mapping_sha256": C.AR.array_digest(qr, mur),
            "frozen_train_top10": [list(pair) for pair in top],
            "train_recipient_count": int(len(unique_recipient_rows(selection))),
            "train_subject_session_rows": compact,
            "directional_stage_sha256": C.digest(stage_path),
            "upstream_gate_sha256": C.digest(gate_path),
            "upstream_protocol_sha256": protocol_sha,
            "upstream_implementation_sha256": ORIGINAL_IMPL_SHA,
            "projector_amendment_sha256": AMENDMENT_SHA,
            "analysis_lock_sha256": ANALYSIS_SHA,
            "implementation_sha256": C.digest(Path(__file__)),
            "outer_development_accessed": False,
            "final_heldout_accessed": False,
        })
        print("TRAIN_RANDOM_SUBJECT_DRAW_COMPLETE", draw, len(compact), flush=True)
    except Exception as exc:
        C.write_json(failure, {
            "status": "FAIL_CLOSED", "draw": draw,
            "reason": f"{type(exc).__name__}: {exc}",
            "implementation_sha256": C.digest(Path(__file__)),
            "final_heldout_accessed": False,
        })
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
