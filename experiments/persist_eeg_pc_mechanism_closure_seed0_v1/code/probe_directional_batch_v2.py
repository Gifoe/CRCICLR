"""TRAIN-only v2 batch-size equivalence and timing probe."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
sys.path.insert(0, str(ROOT / "experiments" / "persist_eeg_protected_complement_coupling_seed0_v1" / "code"))
import run_coupling as C  # noqa: E402
from cell_data_v1 import materialize_stage, stage_norms  # noqa: E402
from directional_core_v2 import fit_directions, mixed_difference  # noqa: E402
from run_mediation_cell_v2 import verified_context  # noqa: E402
from sampling_core import trial_view  # noqa: E402


def main() -> None:
    result = EXP / "probe" / "EEGNet_OpenBMI_MI_fold0_directional_batch_probe_v2.json"
    if result.exists():
        raise RuntimeError("directional probe evidence already exists")
    model, task, fold = "EEGNet", "OpenBMI_MI", 0
    _, _, source, protocol_sha = verified_context(model, task, fold)
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
    centroid = C.PW.train_centroids(runner, data, model, task, fold)
    capped = C.UP.capped_train(data, task, model, fold)
    bh, _, _ = C.UP.hook_representations(net, head, capped[0], model, device)
    spec = C.UP.helper(model).spectrum(bh, *capped[1:], task, model, fold)
    canonical = C.PW.canonical(centroid["h"], spec)
    dims = np.asarray(stored["protected_blocks"], dtype=int)
    view = trial_view(data, "TRAIN")
    checks = []
    for stage in (runner.names[0], runner.names[-1]):
        norms = stage_norms(runner, view, (stage,))[stage]
        selection = materialize_stage(runner, view, model=model, task=task, fold=fold,
                                      stage=stage, precomputed_norms=norms)
        rows = np.unique(selection.recipient_local)
        train_a = selection.activations[rows]
        if stage == "classifier_input":
            q, mu = C.final_q(spec, dims)
            left = np.ascontiguousarray(((spec["basis"] / spec["scale"][None, :]) @ spec["directions"][:, dims]).astype(np.float32))
            right = np.ascontiguousarray(C.UP.raw_base(spec)[dims].astype(np.float32))
            projector = (left, right)
        else:
            q, mu = C.raw_q(C.PW.PathFit(centroid["acts"][stage]), canonical[:, dims])
            projector = None
        axes = fit_directions(train_a, q, mu, c_limit=16, final_projector=projector, device=device)
        probe_rows = rows[:64]
        x = selection.activations[probe_rows]
        original = selection.row_indices[probe_rows]
        labels = view.labels[original]
        logits = selection.logits[probe_rows]
        started = time.perf_counter()
        small = mixed_difference(runner, stage, tuple(centroid["shapes"][stage]),
                                 x, labels, logits, axes, (0, 0), epsilon=0.5, batch_size=16)
        seconds16 = time.perf_counter() - started
        started = time.perf_counter()
        effect = mixed_difference(runner, stage, tuple(centroid["shapes"][stage]),
                                  x, labels, logits, axes, (0, 0), epsilon=0.5, batch_size=64)
        seconds64 = time.perf_counter() - started
        differences = {key: float(np.max(np.abs(effect[key] - small[key]))) for key in effect}
        if any(not np.isfinite(value).all() for value in effect.values()):
            raise RuntimeError("nonfinite fixed directional intervention")
        if max(differences.values()) >= 1e-5:
            raise RuntimeError("forward batch size changed numerical result beyond tolerance")
        checks.append({"stage": stage, "projector_mode": axes.projector_mode,
                       "p_rank": axes.p_rank, "c_rank": axes.c_rank,
                       "train_recipient_count": len(rows), "probe_trials": len(probe_rows),
                       "seconds_batch16": seconds16, "seconds_batch64": seconds64,
                       "max_abs_batch_difference": differences,
                       "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                       "effect_norm": effect["norm"].tolist(),
                       "mapping_sha256": C.AR.array_digest(q, mu)})
    C.write_json(result, {"status": "ENGINEERING_BATCH_PROBE_PASSED", "data_scope": "TRAIN_ONLY",
                          "model": model, "task": task, "fold": fold,
                          "upstream_protocol_sha256": protocol_sha,
                          "checks": checks, "implementation_sha256": C.digest(Path(__file__)),
                          "final_heldout_accessed": False})
    print("DIRECTIONAL_BATCH_PROBE_PASSED", len(checks), flush=True)


if __name__ == "__main__":
    main()
