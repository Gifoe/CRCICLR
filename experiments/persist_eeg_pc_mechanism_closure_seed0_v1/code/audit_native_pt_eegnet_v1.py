"""Checkpoint-native TRAIN-only PERSIST discovery on an EEGNet replica epoch.

P_t is newly discovered at this checkpoint and is never substituted for the
frozen historical final Protected object. This phase intentionally does not
read any outer-development outcome or final-heldout material.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
RUNTIME = Path(os.environ.get("MECHANISM_RUNTIME", str(ROOT.parent / "pc_mechanism_closure_runtime")))
sys.path.insert(0, str(ROOT / "experiments" / "persist_eeg_protected_complement_coupling_seed0_v1" / "code"))
import run_coupling as C  # noqa: E402
os.environ.setdefault("SEVEN_RUNTIME", str(ROOT.parent / "seven_backbone_fourtask_3seed_runtime"))
os.environ.setdefault("OFFICIAL_BACKBONE_ROOT", str(Path(os.environ["SEVEN_RUNTIME"]) / "official"))
sys.path.insert(0, str(C.UP.SEVEN_CODE))
import run_search as S  # noqa: E402
from run_mediation_cell_v1 import verified_context, ORIGINAL_IMPL_SHA, ANALYSIS_SHA  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("OpenBMI_MI", "OpenBMI_SSVEP"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()
    task, fold, epoch = args.task, args.fold, args.epoch
    cell = RUNTIME / "cells" / "eegnet" / task.lower() / f"fold{fold}_seed0"
    result = cell / "checkpoint_native_pt_v1" / f"epoch_{epoch:03d}.json"
    failure = cell / "checkpoint_native_pt_v1" / f"epoch_{epoch:03d}.FAIL_CLOSED.json"
    if result.exists() or failure.exists():
        raise RuntimeError("checkpoint-native P_t terminal evidence already exists")
    try:
        gate_path, _, source, protocol_sha = verified_context("EEGNet", task, fold)
        endpoint_path = cell / "REPLICA_ENDPOINT_AUDIT_V1.json"
        replay_path = cell / "replay_v2" / "REPLAY_AUDIT.json"
        endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        if (endpoint.get("status") != "REPLICA_ENDPOINTS_AUDITED_PENDING_NATIVE_P_AND_MECHANISM" or
                endpoint.get("trajectory_provenance") != replay.get("trajectory_provenance") or
                replay.get("status") != "REPLAY_COMPLETE" or
                endpoint.get("final_heldout_accessed") is not False or
                replay.get("final_heldout_accessed") is not False or
                endpoint.get("upstream_gate_sha256") != C.digest(gate_path)):
            raise RuntimeError("replay/endpoint evidence mismatch")
        scheduled = {int(item["epoch"]): item for item in endpoint["schedule"]}
        if epoch not in scheduled:
            raise RuntimeError("epoch is outside frozen checkpoint schedule")
        checkpoint_path = cell / "replay_v2" / f"epoch_{epoch:03d}.pt"
        checkpoint_sha = C.digest(checkpoint_path)
        if (checkpoint_sha != replay["epoch_checkpoint_sha256"][str(epoch)] or
                checkpoint_sha != scheduled[epoch]["checkpoint_sha256"]):
            raise RuntimeError("replay checkpoint SHA mismatch")
        record, data = source["record"], source["data"]
        frozen_recipe = S._frozen_recipe(task, "EEGNet")
        if frozen_recipe != record["recipe"] or data["split_sha256"] != record["split_sha256"]:
            raise RuntimeError("frozen recipe or split mismatch")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(payload["epoch"]) != epoch or payload["invariant"] != record["invariant_sha256"]:
            raise RuntimeError("replay checkpoint invariant mismatch")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = S.build_model("EEGNet", dataset="OpenBMI", channels=int(record["channels"]),
                              samples=int(record["samples"]), classes=int(record["classes"]),
                              tech_recipe=None).to(device)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        x, y, subjects, sessions = C.UP.capped_train(data, task, "EEGNet", fold)
        h, direct_logits, head_logits = C.UP.hook_representations(model, model.head, x, "EEGNet", device)
        if (h.shape[1] != int(model.head.in_features) or
                np.max(np.abs(direct_logits - head_logits)) >= 1e-5):
            raise RuntimeError("checkpoint-native representation/head mismatch")
        helper = C.UP.helper("EEGNet")
        spec = helper.spectrum(h, y, subjects, sessions, task, "EEGNet", fold)
        spec["classes"] = int(record["classes"])
        protected, assignments = helper.select_protected(
            h, y, subjects, sessions, spec, "EEGNet", task, fold)
        selected = np.asarray(protected, np.int64)
        q = helper.canonical(h, spec).astype(np.float32)
        erased = helper.erase(h, spec, selected)
        changed_logits = C.UP.head_logits(model.head, erased, device)
        finite = C.UP.subject_finite(direct_logits, changed_logits, y, subjects)
        if len(selected):
            residual = C.UP.residual(h, q, spec)
            local = C.UP.local_energy(model.head, q, residual, spec,
                                      selected, subjects, device)
        else:
            local = {subject: 0.0 for subject in finite}
        if not all(np.isfinite(value) for value in local.values()):
            raise RuntimeError("nonfinite checkpoint-native local dependence")
        selected_assignments = [row for row in assignments if row["protected"]]
        C.write_json(result, {
            "status": "NATIVE_CHECKPOINT_PT_COMPLETE_PENDING_BACKTRACE_AND_MECHANISM",
            "model": "EEGNet", "task": task, "fold": fold, "seed": 0,
            "epoch": epoch, "schedule_tags": scheduled[epoch]["schedule_tags"],
            "trajectory_provenance": replay["trajectory_provenance"],
            "definition": "CHECKPOINT_NATIVE_TRAIN_ONLY_PERSIST_NOT_FROZEN_FINAL_P",
            "checkpoint_sha256": checkpoint_sha,
            "train_capped_rows": len(h), "train_biological_subjects": len(set(subjects.astype(str))),
            "native_forward_max_abs": float(np.max(np.abs(direct_logits - head_logits))),
            "active_rank": spec["rank"],
            "numerical_rank_lower_bound": spec["numerical_rank_lower_bound"],
            "protected_rank": len(protected), "protected_dimensions": protected,
            "persistence_supported_blocks": sum(bool(row["persistence_supported"])
                                                for row in assignments),
            "selected_block_count": len(selected_assignments),
            "selected_absolute_CE_harm_mean":
                float(np.mean([row["absolute_CE_harm"] for row in selected_assignments]))
                if selected_assignments else None,
            "selected_excess_CE_harm_mean":
                float(np.mean([row["excess_CE_harm"] for row in selected_assignments]))
                if selected_assignments else None,
            "protected_assignment": assignments,
            "native_finite_subject_rows": finite,
            "native_finite_subject_equal_centered_logit_rms":
                float(np.mean([row["centered_logit_rms"] for row in finite.values()])),
            "local_subject_rows": local,
            "local_subject_equal_energy": float(np.mean(list(local.values()))),
            "native_spec": spec,
            "replay_audit_sha256": C.digest(replay_path),
            "endpoint_audit_sha256": C.digest(endpoint_path),
            "upstream_gate_sha256": C.digest(gate_path),
            "upstream_protocol_sha256": protocol_sha,
            "upstream_implementation_sha256": ORIGINAL_IMPL_SHA,
            "analysis_lock_sha256": ANALYSIS_SHA,
            "implementation_sha256": C.digest(Path(__file__)),
            "outer_development_accessed": False,
            "final_heldout_accessed": False,
        })
        print("NATIVE_PT_COMPLETE", task, fold, epoch, "rank", len(protected), flush=True)
    except Exception as exc:
        C.write_json(failure, {"status": "FAIL_CLOSED", "reason": f"{type(exc).__name__}: {exc}",
                               "implementation_sha256": C.digest(Path(__file__)),
                               "outer_development_accessed": False,
                               "final_heldout_accessed": False})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
