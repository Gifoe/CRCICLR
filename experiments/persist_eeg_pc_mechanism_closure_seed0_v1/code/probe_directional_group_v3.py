"""TRAIN-only equivalence/throughput probe for grouping fixed P×C pairs."""
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
from directional_core_v3 import fit_directions, mixed_difference, mixed_differences_grouped  # noqa: E402
from run_mediation_cell_v2 import verified_context  # noqa: E402
from sampling_core import trial_view  # noqa: E402


def main() -> None:
    result = EXP / "probe" / "EEGNet_OpenBMI_MI_fold0_directional_group_probe_v3.json"
    if result.exists():
        raise RuntimeError("group probe evidence already exists")
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
    stage = runner.names[0]
    norms = stage_norms(runner, view, (stage,))[stage]
    selection = materialize_stage(runner, view, model=model, task=task, fold=fold,
                                  stage=stage, precomputed_norms=norms)
    rows = np.unique(selection.recipient_local)
    train_a = selection.activations[rows]
    q, mu = C.raw_q(C.PW.PathFit(centroid["acts"][stage]), canonical[:, dims])
    axes = fit_directions(train_a, q, mu, c_limit=16, device=device)
    pairs = tuple(np.ndindex((axes.p_rank, axes.c_rank)))[:4]
    if len(pairs) != 4:
        raise RuntimeError("need four frozen direction pairs")
    probe_rows = rows[:64]
    x = selection.activations[probe_rows]
    labels = view.labels[selection.row_indices[probe_rows]]
    logits = selection.logits[probe_rows]
    started = time.perf_counter()
    separate = {pair: mixed_difference(runner, stage, tuple(centroid["shapes"][stage]),
                                       x, labels, logits, axes, pair, batch_size=16)
                for pair in pairs}
    separate_seconds = time.perf_counter() - started
    started = time.perf_counter()
    grouped = mixed_differences_grouped(runner, stage, tuple(centroid["shapes"][stage]),
                                        x, labels, logits, axes, pairs, trial_batch=16)
    grouped_seconds = time.perf_counter() - started
    max_abs = max(float(np.max(np.abs(separate[pair][key] - grouped[pair][key])))
                  for pair in pairs for key in separate[pair])
    if not np.isfinite(max_abs) or max_abs >= 1e-5:
        raise RuntimeError(f"grouped native forward changed interaction: {max_abs}")
    C.write_json(result, {
        "status": "ENGINEERING_GROUP_PROBE_PASSED", "data_scope": "TRAIN_ONLY",
        "stage": stage, "model": model, "task": task, "fold": fold,
        "pair_count": len(pairs), "trial_count": len(x), "trial_batch": 16,
        "separate_seconds": separate_seconds, "grouped_seconds": grouped_seconds,
        "speedup": separate_seconds / grouped_seconds,
        "max_abs_metric_difference": max_abs,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "upstream_protocol_sha256": protocol_sha,
        "implementation_sha256": C.digest(Path(__file__)), "final_heldout_accessed": False,
    })
    print("DIRECTIONAL_GROUP_PROBE_PASSED", separate_seconds, grouped_seconds, max_abs, flush=True)


if __name__ == "__main__":
    main()
