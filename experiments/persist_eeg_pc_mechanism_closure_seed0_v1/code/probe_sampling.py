"""TRAIN-only engineering probe of the frozen recipient/donor selector.

Uses upstream capped TRAIN trials solely to keep this probe small. This is
not a full-cell scientific sampling run and writes no outer-development result.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
sys.path.insert(0, str(ROOT / "experiments" / "persist_eeg_protected_complement_coupling_seed0_v1" / "code"))
import run_coupling as C  # noqa: E402
from sampling_core_v1 import select_pairs, trial_view  # noqa: E402


def main() -> None:
    model, task, fold, stage = "EEGNet", "OpenBMI_MI", 0, "temporal_bn"
    out = EXP / "probe" / "EEGNet_OpenBMI_MI_fold0_sampling_v1.json"
    failure = out.with_name("EEGNet_OpenBMI_MI_fold0_sampling_v1_FAIL_CLOSED.json")
    if out.exists() or failure.exists():
        raise RuntimeError("sampling probe already has evidence; no overwrite")
    try:
        gate_file = EXP / "gate" / "GATE_AUDIT.json"
        gate = json.loads(gate_file.read_text(encoding="utf-8"))
        if gate.get("status") != "PASSED" or gate.get("original_cells_complete") != 20 or gate.get("final_heldout_accessed"):
            raise RuntimeError("upstream gate is not fully passed")
        source = C.prior_record(model, task, fold)
        protocol_sha = C.assert_lock(model, task, fold, source)
        if protocol_sha != gate["upstream_protocol_sha256"]:
            raise RuntimeError("upstream protocol hash mismatch")
        historical_cell = json.loads(C.cell_path(model, task, fold).read_text(encoding="utf-8"))
        if historical_cell.get("status") != "COMPLETE" or historical_cell.get("implementation_sha256") != gate["upstream_implementation_sha256"]:
            raise RuntimeError("upstream cell/hash mismatch")
        record, checkpoint, data = source["record"], source["checkpoint"], source["data"]
        view = trial_view(data, "TRAIN")
        if len(view.x) <= 0 or np.any(view.sessions < 0):
            raise RuntimeError("invalid legal TRAIN view")
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
        tx, ty, ts, tse = C.BR.trial_train(data, model, task, fold)
        a = C.PW.outer_stage(runner, tx, stage)["a"]
        plan = select_pairs(a, ty, ts, tse, model=model, task=task, fold=fold, stage=stage, split="TRAIN")
        if not len(plan.recipient):
            raise RuntimeError("no legal TRAIN donor pairs")
        if not (np.all(ts[plan.recipient] == ts[plan.donor]) and
                np.all(tse[plan.recipient] == tse[plan.donor]) and
                np.all(ty[plan.recipient] != ty[plan.donor])):
            raise RuntimeError("recipient/donor eligibility failure")
        again = select_pairs(a, ty, ts, tse, model=model, task=task, fold=fold, stage=stage, split="TRAIN")
        if not (np.array_equal(plan.recipient, again.recipient) and np.array_equal(plan.donor, again.donor)):
            raise RuntimeError("nondeterministic pair plan")
        C.write_json(out, {
            "status": "PROBE_PASSED", "schema": "PC_MECHANISM_SAMPLING_TRAIN_ENGINEERING_V1",
            "model": model, "task": task, "fold": fold, "stage": stage,
            "data_scope": "UPSTREAM_CAPPED_TRAIN_ENGINEERING_ONLY",
            "full_legal_train_rows": int(len(view.x)),
            "probe_capped_train_rows": int(len(tx)),
            "probe_pairs": int(len(plan.recipient)),
            "coverage": list(plan.coverage),
            "upstream_gate_sha256": C.digest(gate_file),
            "upstream_protocol_sha256": protocol_sha,
            "upstream_cell_sha256": C.digest(C.cell_path(model, task, fold)),
            "sampling_core_sha256": C.digest(Path(__file__).with_name("sampling_core_v1.py")),
            "implementation_sha256": C.digest(Path(__file__)),
            "final_heldout_accessed": False,
        })
        print("SAMPLING_PROBE_PASSED", len(tx), len(plan.recipient), flush=True)
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
