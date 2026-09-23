"""TRAIN-only all-pair numerical precision audit after v1 failure."""
from __future__ import annotations

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
    path = EXP / "probe" / "EEGNet_OpenBMI_MI_fold0_reconstruction_failure_diagnostic_v2.json"
    if path.exists():
        raise RuntimeError("diagnostic evidence already exists")
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
    a = selected.activations[selected.recipient_local]
    qf = torch.from_numpy(q).to(device)
    muf = torch.from_numpy(mu).to(device)
    error32 = []
    error_double_reassembly = []
    error_full_double = []
    amplitude = []
    for start in range(0, len(a), 8):
        ai = torch.from_numpy(np.ascontiguousarray(a[start:start+8])).to(device)
        centered = ai - muf
        p = (centered @ qf) @ qf.T
        c = centered - p
        error32.extend((muf + p + c - ai).abs().flatten(1).max(1).values.cpu().tolist())
        error_double_reassembly.extend((muf.double() + p.double() + c.double() - ai.double()).abs().flatten(1).max(1).values.cpu().tolist())
        x64, q64, mu64 = ai.double(), qf.double(), muf.double()
        p64 = ((x64 - mu64) @ q64) @ q64.T
        c64 = (x64 - mu64) - p64
        error_full_double.extend((mu64 + p64 + c64 - x64).abs().flatten(1).max(1).values.cpu().tolist())
        amplitude.extend(ai.abs().flatten(1).max(1).values.cpu().tolist())
    e32 = np.asarray(error32)
    worst = int(e32.argmax())
    C.write_json(path, {
        "status": "DIAGNOSTIC_ONLY", "data_scope": "TRAIN_ONLY_ALL_FROZEN_RECIPIENT_PAIRS",
        "upstream_protocol_sha256": protocol_sha,
        "mapping_sha256": C.AR.array_digest(q, mu),
        "pair_count": len(a), "above_threshold_1e_minus_5": int((e32 >= 1e-5).sum()),
        "torch_float32_reconstruction_max_abs": float(e32.max()),
        "double_reassembly_of_float32_components_max_abs": float(max(error_double_reassembly)),
        "full_double_projection_reconstruction_max_abs": float(max(error_full_double)),
        "worst_pair_position": worst, "worst_recipient_global_index": int(selected.recipient_global[worst]),
        "worst_activation_max_abs": float(amplitude[worst]),
        "implementation_sha256": C.digest(Path(__file__)), "final_heldout_accessed": False,
    })
    print("RECONSTRUCTION_DIAGNOSTIC_V2_COMPLETE", len(a), e32.max(), flush=True)


if __name__ == "__main__":
    main()
