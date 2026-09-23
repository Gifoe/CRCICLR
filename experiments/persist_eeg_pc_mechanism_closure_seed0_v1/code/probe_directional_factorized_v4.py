"""TRAIN-only test of EEGNet spatial-prefix factorization against native forwards."""
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
from directional_core_v3 import fit_directions, mixed_difference  # noqa: E402
from directional_factorized_eegnet_v4 import factorized_mixed_difference, precompute_spatial_base  # noqa: E402
from run_mediation_cell_v2 import verified_context  # noqa: E402
from sampling_core import trial_view  # noqa: E402


def main() -> None:
    result = EXP / "probe" / "EEGNet_OpenBMI_MI_fold0_directional_factorized_probe_v7.json"
    if result.exists():
        raise RuntimeError("factorization probe evidence already exists")
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
    probe_rows = rows[:64]
    x = selection.activations[probe_rows]
    labels = view.labels[selection.row_indices[probe_rows]]
    logits = selection.logits[probe_rows]
    started = time.perf_counter()
    original = {pair: mixed_difference(runner, stage, tuple(centroid["shapes"][stage]),
                                       x, labels, logits, axes, pair, batch_size=64)
                for pair in pairs}
    original_seconds = time.perf_counter() - started
    started = time.perf_counter()
    factorized = {pair: factorized_mixed_difference(runner, stage, tuple(centroid["shapes"][stage]),
                                                    x, labels, logits, axes, pair, batch_size=64)
                  for pair in pairs}
    factorized_seconds = time.perf_counter() - started
    started = time.perf_counter()
    spatial_base = precompute_spatial_base(runner, tuple(centroid["shapes"][stage]), x, batch_size=64)
    cached = {pair: factorized_mixed_difference(runner, stage, tuple(centroid["shapes"][stage]),
                                                x, labels, logits, axes, pair, batch_size=64,
                                                spatial_base=spatial_base)
              for pair in pairs}
    cached_seconds_including_precompute = time.perf_counter() - started
    differences = {f"{p}_{c}_{key}": float(np.max(np.abs(original[(p, c)][key] - factorized[(p, c)][key])))
                   for p, c in pairs for key in original[(p, c)]}
    maximum = max(differences.values())
    cached_differences = {f"{p}_{c}_{key}": float(np.max(np.abs(original[(p, c)][key] - cached[(p, c)][key])))
                          for p, c in pairs for key in original[(p, c)]}
    cached_maximum = max(cached_differences.values())
    status = ("ENGINEERING_FACTORIZATION_EQUIVALENT" if np.isfinite(maximum)
              and maximum < 1e-5 and np.isfinite(cached_maximum) and cached_maximum < 1e-5
              else "ENGINEERING_FACTORIZATION_NOT_EQUIVALENT")
    C.write_json(result, {
        "status": status, "data_scope": "TRAIN_ONLY", "stage": stage,
        "model": model, "task": task, "fold": fold, "pair_count": len(pairs),
        "trial_count": len(x), "batch_size": 64, "original_seconds": original_seconds,
        "factorized_seconds": factorized_seconds, "speedup": original_seconds / factorized_seconds,
        "cached_seconds_including_precompute": cached_seconds_including_precompute,
        "cached_speedup": original_seconds / cached_seconds_including_precompute,
        "cached_metric_max_abs_differences": cached_differences,
        "cached_max_abs_metric_difference": cached_maximum,
        "metric_max_abs_differences": differences, "max_abs_metric_difference": maximum,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "upstream_protocol_sha256": protocol_sha,
        "implementation_sha256": C.digest(Path(__file__)), "final_heldout_accessed": False,
    })
    print(status, original_seconds, factorized_seconds, cached_seconds_including_precompute,
          maximum, cached_maximum, flush=True)


if __name__ == "__main__":
    main()
