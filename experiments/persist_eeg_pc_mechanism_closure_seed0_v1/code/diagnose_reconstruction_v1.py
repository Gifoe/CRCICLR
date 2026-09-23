"""TRAIN-only diagnosis of first production mediation numerical failure."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
sys.path.insert(0, str(ROOT / "experiments" / "persist_eeg_protected_complement_coupling_seed0_v1" / "code"))
import run_coupling as C  # noqa: E402
from cell_data_v1 import materialize_stage, stage_norms  # noqa: E402
from sampling_core import trial_view  # noqa: E402


def main() -> None:
    path = EXP / "probe" / "EEGNet_OpenBMI_MI_fold0_reconstruction_failure_diagnostic_v1.json"
    if path.exists():
        raise RuntimeError("diagnostic already exists; no overwrite")
    model, task, fold, stage = "EEGNet", "OpenBMI_MI", 0, "temporal_bn"
    source = C.prior_record(model, task, fold)
    protocol_sha = C.assert_lock(model, task, fold, source)
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
    dims = np.asarray(stored["protected_blocks"], int)
    q, mu = C.raw_q(C.PW.PathFit(centroid["acts"][stage]), canonical[:, dims])
    view = trial_view(data, "TRAIN")
    norms = stage_norms(runner, view, (stage,))[stage]
    selected = materialize_stage(runner, view, model=model, task=task, fold=fold,
                                 stage=stage, precomputed_norms=norms)
    indexes = selected.recipient_local[:8]
    ai = torch.from_numpy(np.ascontiguousarray(selected.activations[indexes])).to(device)
    qc = torch.from_numpy(q).to(device)
    mc = torch.from_numpy(mu).to(device)
    centered = ai - mc
    p = (centered @ qc) @ qc.T
    c = centered - p
    reconstructed = mc + p + c
    np64 = ai.cpu().numpy().astype(np.float64)
    q64 = q.astype(np.float64)
    mu64 = mu.astype(np.float64)
    p64 = ((np64 - mu64) @ q64) @ q64.T
    c64 = np64 - mu64 - p64
    C.write_json(path, {
        "status": "DIAGNOSTIC_ONLY", "data_scope": "TRAIN_ONLY_FIRST_EIGHT_FROZEN_PAIRS",
        "upstream_protocol_sha256": protocol_sha,
        "mapping_sha256": C.AR.array_digest(q, mu),
        "row_indices": selected.recipient_global[:8],
        "torch_float32_reconstruction_max_abs": float((reconstructed - ai).abs().max().item()),
        "numpy_float64_reconstruction_max_abs": float(np.max(np.abs(np64 - (mu64 + p64 + c64)))),
        "activation_max_abs": float(ai.abs().max().item()),
        "mean_max_abs": float(mc.abs().max().item()),
        "P_max_abs": float(p.abs().max().item()),
        "C_max_abs": float(c.abs().max().item()),
        "implementation_sha256": C.digest(Path(__file__)), "final_heldout_accessed": False,
    })
    print("RECONSTRUCTION_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
