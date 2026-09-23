"""Replica-checkpoint P/C mechanism on frozen cap-8 TRAIN pairs only.

This samples the prespecified depth-point to embedding transition for P->C/C->P
mediation. It does not assert that the replica is the historical training path.
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
os.environ.setdefault("SEVEN_RUNTIME", str(ROOT.parent / "seven_backbone_fourtask_3seed_runtime"))
os.environ.setdefault("OFFICIAL_BACKBONE_ROOT", str(Path(os.environ["SEVEN_RUNTIME"]) / "official"))
sys.path.insert(0, str(C.UP.SEVEN_CODE))
import run_search as S  # noqa: E402
from cell_data_v1 import materialize_stage  # noqa: E402
from mediation_core_v4 import adjacent_mediation, verify_adjacent_forward  # noqa: E402
from sampling_core import trial_view  # noqa: E402
from run_mediation_cell_v1 import verified_context, ORIGINAL_IMPL_SHA, ANALYSIS_SHA  # noqa: E402

STAGE = "depth_point_elu_pool2"
SUCCESSOR = "embedding_64d"
MEDIATION_KEYS = ("P_to_C_rep", "C_to_P_rep", "P_to_C_norm", "C_to_P_norm",
                  "P_to_C_true_margin", "C_to_P_true_margin",
                  "P_to_P_rep", "C_to_C_rep", "P_total_norm", "C_total_norm")
FUNCTIONAL_KEYS = ("P_effect_norm", "C_effect_norm", "interaction_norm",
                   "interaction_main_ratio", "P_margin_sign_flip", "C_margin_sign_flip",
                   "P_prediction_flip", "C_prediction_flip")


def summarize(measured: list[dict[str, object]], subjects: np.ndarray,
              sessions: np.ndarray) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row, subject, session in zip(measured, subjects, sessions):
        groups[(str(subject), int(session))].append(row)
    rows = []
    for (subject, session), part in sorted(groups.items()):
        metrics = {}
        for key in (*FUNCTIONAL_KEYS, *MEDIATION_KEYS):
            values = np.asarray([float(row[key]) for row in part], np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError(f"nonfinite {key} for {subject}/{session}")
            metrics[key] = float(values.mean())
        metrics["PC_prediction_flip_disagreement_rate"] = float(np.mean([
            row["P_prediction_flip"] != row["C_prediction_flip"] for row in part]))
        rows.append({"subject": subject, "session": session, "pair_count": len(part), **metrics})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("OpenBMI_MI", "OpenBMI_SSVEP"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()
    epoch, task, fold = args.epoch, args.task, args.fold
    cell = RUNTIME / "cells" / "eegnet" / task.lower() / f"fold{fold}_seed0"
    result = cell / "checkpoint_mechanism_v1" / f"epoch_{epoch:03d}.json"
    failure = cell / "checkpoint_mechanism_v1" / f"epoch_{epoch:03d}.FAIL_CLOSED.json"
    if result.exists() or failure.exists():
        raise RuntimeError("checkpoint mechanism already has terminal evidence")
    try:
        gate_path, _, source, protocol_sha = verified_context("EEGNet", task, fold)
        record, data = source["record"], source["data"]
        endpoint_path = cell / "REPLICA_ENDPOINT_AUDIT_V1.json"
        replay_path = cell / "replay_v2" / "REPLAY_AUDIT.json"
        backtrace_path = cell / "FINAL_P_BACKTRACE_V1.json"
        pt_path = cell / "checkpoint_native_pt_v3" / f"epoch_{epoch:03d}.json"
        endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        backtrace = json.loads(backtrace_path.read_text(encoding="utf-8"))
        pt = json.loads(pt_path.read_text(encoding="utf-8"))
        if (replay.get("status") != "REPLAY_COMPLETE" or
                replay.get("trajectory_provenance") != "RETRAINED_REPLICA_TRAJECTORY" or
                endpoint.get("status") != "REPLICA_ENDPOINTS_AUDITED_PENDING_NATIVE_P_AND_MECHANISM" or
                backtrace.get("status") != "FINAL_P_BACKTRACE_COMPLETE_REPLICA_ONLY" or
                pt.get("status") != "NATIVE_CHECKPOINT_PT_COMPLETE_PENDING_BACKTRACE_AND_MECHANISM" or
                pt.get("final_heldout_accessed") is not False or
                backtrace.get("final_heldout_accessed") is not False):
            raise RuntimeError("checkpoint provenance/evidence mismatch")
        schedule = {int(row["epoch"]): row for row in endpoint["schedule"]}
        if epoch not in schedule:
            raise RuntimeError("epoch outside locked checkpoint schedule")
        checkpoint_path = cell / "replay_v2" / f"epoch_{epoch:03d}.pt"
        checkpoint_sha = C.digest(checkpoint_path)
        if (checkpoint_sha != schedule[epoch]["checkpoint_sha256"] or
                checkpoint_sha != replay["epoch_checkpoint_sha256"][str(epoch)] or
                checkpoint_sha != pt["checkpoint_sha256"]):
            raise RuntimeError("replica checkpoint SHA mismatch")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(payload["epoch"]) != epoch or payload["invariant"] != record["invariant_sha256"]:
            raise RuntimeError("replica checkpoint invariant mismatch")
        search_data = S.load_search_fold(task, fold)
        if (search_data["split_sha256"] != record["split_sha256"] or
                search_data["normalizer"] != record["normalizer"] or
                search_data["classes"] != int(record["classes"])):
            raise RuntimeError("frozen data identity mismatch")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = S.build_model("EEGNet", dataset=search_data["dataset"],
                            channels=int(search_data["channels"]),
                            samples=int(search_data["samples"]),
                            classes=int(search_data["classes"]), tech_recipe=None).to(device)
        net.load_state_dict(payload["model"], strict=True)
        net.eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        runner = C.PW.Stages(net, net.head, "EEGNet", device)
        view = trial_view(data, "TRAIN")
        native_forward = verify_adjacent_forward(runner, view.x)
        capped = C.UP.capped_train(data, task, "EEGNet", fold)
        h_capped, direct, head_copy = C.UP.hook_representations(net, net.head, capped[0], "EEGNet", device)
        if np.max(np.abs(direct - head_copy)) >= 1e-5:
            raise RuntimeError("checkpoint-native head mismatch")
        spec = C.UP.helper("EEGNet").spectrum(h_capped, *capped[1:], task, "EEGNet", fold)
        dims = np.asarray(pt["protected_dimensions"], dtype=int)
        if int(spec["rank"]) != int(pt["active_rank"]) or len(dims) != int(pt["protected_rank"]):
            raise RuntimeError("checkpoint-native P_t rank mismatch")
        for key in ("mean", "basis", "scale", "directions"):
            if not np.allclose(np.asarray(spec[key]), np.asarray(pt["native_spec"][key]), atol=1e-7, rtol=1e-7):
                raise RuntimeError(f"checkpoint-native P_t {key} mismatch")
        centroid = C.PW.train_centroids(runner, data, "EEGNet", task, fold)
        canonical_centroids = C.PW.canonical(centroid["h"], spec)
        q_stage, mu_stage = C.raw_q(C.PW.PathFit(centroid["acts"][STAGE]), canonical_centroids[:, dims])
        q_next, mu_next = C.raw_q(C.PW.PathFit(centroid["acts"][SUCCESSOR]), canonical_centroids[:, dims])
        selection = materialize_stage(runner, view, model="EEGNet", task=task,
                                      fold=fold, stage=STAGE)
        pairs = list(zip(selection.recipient_local.tolist(), selection.donor_local.tolist()))
        labels = view.labels[selection.row_indices]
        subjects = view.subjects[selection.row_indices]
        sessions = view.sessions[selection.row_indices]
        functional = C.functional_pairs(runner, STAGE, centroid["shapes"][STAGE],
                                        selection.activations, selection.logits,
                                        labels, subjects, pairs, q_stage, mu_stage,
                                        q_stage, sessions, local=False)
        if len(functional) != len(pairs):
            raise RuntimeError("functional pair count mismatch")
        recipient_labels = view.labels[selection.recipient_global]
        measured: list[dict[str, object]] = []
        for start in range(0, len(pairs), 8):
            stop = min(start + 8, len(pairs))
            a = selection.activations[selection.recipient_local[start:stop]]
            b = selection.activations[selection.donor_local[start:stop]]
            mediation = adjacent_mediation(runner, STAGE, a, b,
                                            tuple(centroid["shapes"][STAGE]),
                                            q_stage, mu_stage, q_next, mu_next,
                                            recipient_labels[start:stop],
                                            selection.donor_label[start:stop])
            for offset in range(stop - start):
                row = {key: functional[start + offset][key] for key in FUNCTIONAL_KEYS}
                row.update({key: float(mediation[key][offset]) for key in MEDIATION_KEYS})
                measured.append(row)
        subject_rows = summarize(measured, view.subjects[selection.recipient_global],
                                 view.sessions[selection.recipient_global])
        if len(subject_rows) != len(C.PW.natural(view.subjects)) * 2:
            raise RuntimeError("subject-session mechanism coverage incomplete")
        recipient_unique = np.unique(selection.recipient_global)
        h_rec, z_rec, z_head = C.UP.hook_representations(
            net, net.head, view.x[recipient_unique], "EEGNet", device)
        if np.max(np.abs(z_rec - z_head)) >= 1e-5:
            raise RuntimeError("recipient native head mismatch")
        erased = C.UP.helper("EEGNet").erase(h_rec, spec, dims)
        changed = C.UP.head_logits(net.head, erased, device)
        dependence = C.UP.subject_finite(z_rec, changed, view.labels[recipient_unique],
                                         view.subjects[recipient_unique])
        mean_dep = float(np.mean([row["centered_logit_rms"] for row in dependence.values()]))
        if not np.isfinite(mean_dep):
            raise RuntimeError("nonfinite final checkpoint P_t dependence")
        historical, historical_head = C.UP.helper("EEGNet").build_model({
            "Model": "EEGNet", "Task": task, "fold": fold, "seed": 0,
            "channels": int(record.get("channels") or 62),
            "samples": int(record.get("samples") or 1000),
            "classes": int(record["classes"]),
            "checkpoint_path": str(source["checkpoint"]),
            "recipe_name": record.get("recipe", {}).get("name"),
            "trainable_parameters": int(record.get("trainable_parameters", record.get("parameters", 0))),
        }, device)
        h_historical, historical_z, historical_head_z = C.UP.hook_representations(
            historical, historical_head, capped[0], "EEGNet", device)
        if np.max(np.abs(historical_z - historical_head_z)) >= 1e-5:
            raise RuntimeError("historical native head mismatch")
        historical_spec = C.UP.helper("EEGNet").spectrum(
            h_historical, *capped[1:], task, "EEGNet", fold)
        if (C.UP.array_sha(historical_spec["mean"], historical_spec["basis"],
                           historical_spec["scale"], historical_spec["directions"])
                != source["hashes"]["basis_sha256"]):
            raise RuntimeError("frozen historical final-P basis mismatch")
        historical_dims = np.asarray(source["stored"]["protected_blocks"], dtype=int)
        erased_final = C.UP.helper("EEGNet").erase(h_rec, historical_spec, historical_dims)
        changed_final = C.UP.head_logits(net.head, erased_final, device)
        historical_final_dependence = C.UP.subject_finite(
            z_rec, changed_final, view.labels[recipient_unique],
            view.subjects[recipient_unique])
        mean_final_dep = float(np.mean([
            row["centered_logit_rms"] for row in historical_final_dependence.values()]))
        if not np.isfinite(mean_final_dep):
            raise RuntimeError("nonfinite fixed historical final-P dependence")
        overall = {key: float(np.mean([row[key] for row in subject_rows]))
                   for key in (*FUNCTIONAL_KEYS, *MEDIATION_KEYS,
                               "PC_prediction_flip_disagreement_rate")}
        C.write_json(result, {
            "status": "CHECKPOINT_MECHANISM_COMPLETE_REPLICA_ONLY",
            "model": "EEGNet", "task": task, "fold": fold, "seed": 0,
            "epoch": epoch, "schedule_tags": schedule[epoch]["schedule_tags"],
            "trajectory_provenance": replay["trajectory_provenance"],
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_native_P_t_rank": int(pt["protected_rank"]),
            "checkpoint_native_P_t_dimensions": dims.tolist(),
            "transition": f"{STAGE}->{SUCCESSOR}",
            "transition_mapping": "TRAIN_ONLY_ORTHOGONAL_P_T_PATHWAYS",
            "final_projector_mode": "CHECKPOINT_NATIVE_CANONICAL_OBLIQUE_P_T",
            "trial_cap_per_subject_session_class": 8,
            "pair_count": len(pairs), "subject_session_count": len(subject_rows),
            "pair_coverage": selection.coverage,
            "native_adjacent_forward_max_abs": max(native_forward.values()),
            "native_P_t_dependence_equal_subject_centered_logit_rms": mean_dep,
            "native_P_t_subject_finite_dependence": dependence,
            "fixed_historical_final_P_dependence_equal_subject_centered_logit_rms": mean_final_dep,
            "fixed_historical_final_P_subject_finite_dependence": historical_final_dependence,
            "fixed_historical_final_P_checkpoint_sha256": source["hashes"]["checkpoint_sha256"],
            "fixed_historical_final_P_basis_sha256": source["hashes"]["basis_sha256"],
            "subject_session_mechanism": subject_rows,
            "equal_subject_session_means": overall,
            "checkpoint_native_pt_audit_sha256": C.digest(pt_path),
            "final_p_backtrace_audit_sha256": C.digest(backtrace_path),
            "replay_audit_sha256": C.digest(replay_path),
            "upstream_gate_sha256": C.digest(gate_path),
            "upstream_protocol_sha256": protocol_sha,
            "upstream_implementation_sha256": ORIGINAL_IMPL_SHA,
            "analysis_lock_sha256": ANALYSIS_SHA,
            "implementation_sha256": C.digest(Path(__file__)),
            "outer_development_accessed": False,
            "final_heldout_accessed": False,
        })
        print("CHECKPOINT_MECHANISM_V1_COMPLETE", task, fold, epoch, len(pairs), flush=True)
    except Exception as exc:
        C.write_json(failure, {"status": "FAIL_CLOSED", "reason": f"{type(exc).__name__}: {exc}",
                               "implementation_sha256": C.digest(Path(__file__)),
                               "outer_development_accessed": False,
                               "final_heldout_accessed": False})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
