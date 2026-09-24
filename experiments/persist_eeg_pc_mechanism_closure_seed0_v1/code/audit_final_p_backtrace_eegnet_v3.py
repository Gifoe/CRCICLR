"""TRAIN-only recovery audit for the frozen historical final Protected code.

Replica checkpoints are not the historical training trajectory. This version
implements the frozen checkpoint schedule with duplicate epochs deduplicated,
as required by the analysis lock; the previous v2 validator's fixed-six unique
epoch assumption is preserved as failed evidence for this cell.
"""
from __future__ import annotations

import argparse
import json
import math
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

DIRECTIONS = (("within_s1", 1, 1), ("within_s2", 2, 2),
              ("s1_to_s2", 1, 2), ("s2_to_s1", 2, 1))
RIDGE_ALPHA = 1.0
CROSS_FITS = 5


def frozen_schedule_epochs(final_epoch: int, best_epoch: int) -> list[int]:
    """Return rounded 10/25/50/75/best/final epochs with repeat deduplication."""
    if final_epoch < 1 or best_epoch < 1 or best_epoch > final_epoch:
        raise RuntimeError("invalid replay endpoint epochs")
    raw = [math.floor(final_epoch * fraction + 0.5)
           for fraction in (0.10, 0.25, 0.50, 0.75)]
    raw.extend((best_epoch, final_epoch))
    return sorted({max(1, min(final_epoch, int(epoch))) for epoch in raw})


def primal_ridge(fit_x: np.ndarray, fit_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    """Exact feature-space form of pathway ridge with fit-only standardization."""
    x = np.asarray(fit_x, np.float64)
    y = np.asarray(fit_y, np.float64)
    t = np.asarray(test_x, np.float64)
    mu = x.mean(0)
    sd = np.maximum(x.std(0), 1e-6)
    x = (x - mu) / sd
    t = (t - mu) / sd
    d = x.shape[1]
    weight = np.linalg.solve(x.T @ x + RIDGE_ALPHA * d * np.eye(d), x.T @ y)
    return (t @ weight).astype(np.float32)


def verify_dual_identity() -> float:
    rng = np.random.default_rng(8216)
    fit_x = rng.normal(size=(24, 7)).astype(np.float32)
    fit_y = rng.normal(size=(24, 3)).astype(np.float32)
    test_x = rng.normal(size=(9, 7)).astype(np.float32)
    mu = fit_x.astype(np.float64).mean(0)
    sd = np.maximum(fit_x.astype(np.float64).std(0), 1e-6)
    x = (fit_x - mu) / sd
    t = (test_x - mu) / sd
    dual = (t @ x.T / x.shape[1]) @ np.linalg.solve(
        x @ x.T / x.shape[1] + RIDGE_ALPHA * np.eye(len(x)), fit_y)
    error = float(np.max(np.abs(primal_ridge(fit_x, fit_y, test_x) - dual)))
    if error >= 1e-6:
        raise RuntimeError(f"primal/dual ridge identity failure: {error}")
    return error


def scores(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    yy = np.asarray(target, np.float64)
    pp = np.asarray(prediction, np.float64)
    mse = float(np.mean((yy - pp) ** 2))
    variance = float(np.mean((yy - yy.mean(0, keepdims=True)) ** 2))
    r2 = float(1.0 - mse / max(variance, 1e-12))
    corr = []
    for j in range(yy.shape[1]):
        if np.std(yy[:, j]) > 1e-12 and np.std(pp[:, j]) > 1e-12:
            corr.append(float(np.corrcoef(yy[:, j], pp[:, j])[0, 1]))
    return {"r2": r2, "normalized_mse": float(mse / max(variance, 1e-12)),
            "coordinate_correlation": float(np.mean(corr)) if corr else None}


def subject_crossfit(h: np.ndarray, target: np.ndarray, subjects: np.ndarray,
                     sessions: np.ndarray) -> list[dict[str, object]]:
    unique = C.PW.natural(subjects)
    rows = []
    for direction, src, dst in DIRECTIONS:
        for split in range(CROSS_FITS):
            held = np.asarray([subject for i, subject in enumerate(unique) if i % CROSS_FITS == split])
            fit = (sessions == src) & ~np.isin(subjects, held)
            test = (sessions == dst) & np.isin(subjects, held)
            if fit.sum() < 4 or not test.any():
                raise RuntimeError(f"missing TRAIN rows for {direction} crossfit {split}")
            predicted = primal_ridge(h[fit], target[fit], h[test])
            for subject in C.PW.natural(subjects[test]):
                mask = subjects[test] == subject
                rows.append({"subject": str(subject), "direction": direction,
                             "crossfit": split, "fit_trials": int(fit.sum()),
                             "test_trials": int(mask.sum()),
                             **scores(target[test][mask], predicted[mask])})
    if len(rows) != len(unique) * len(DIRECTIONS):
        raise RuntimeError("subject-direction coverage incomplete")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("OpenBMI_MI", "OpenBMI_SSVEP"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    task, fold = args.task, args.fold
    cell = RUNTIME / "cells" / "eegnet" / task.lower() / f"fold{fold}_seed0"
    result = cell / "FINAL_P_BACKTRACE_V2.json"
    failure = cell / "FINAL_P_BACKTRACE_V2.FAIL_CLOSED.json"
    if result.exists() or failure.exists():
        raise RuntimeError("final-P backtrace v2 already has terminal evidence")
    try:
        dual_error = verify_dual_identity()
        gate_path, _, source, protocol_sha = verified_context("EEGNet", task, fold)
        record, stored, checkpoint, data = (source[key] for key in ("record", "stored", "checkpoint", "data"))
        replay_path = cell / "replay_v2" / "REPLAY_AUDIT.json"
        endpoint_path = cell / "REPLICA_ENDPOINT_AUDIT_V1.json"
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
        if (replay.get("status") != "REPLAY_COMPLETE" or
                replay.get("trajectory_provenance") != "RETRAINED_REPLICA_TRAJECTORY" or
                replay.get("final_heldout_accessed") is not False or
                endpoint.get("status") != "REPLICA_ENDPOINTS_AUDITED_PENDING_NATIVE_P_AND_MECHANISM" or
                endpoint.get("final_heldout_accessed") is not False or
                endpoint.get("outer_development_used_for_selection") is not False or
                endpoint.get("replay_audit_sha256") != C.digest(replay_path)):
            raise RuntimeError("replay/endpoint provenance mismatch")
        final_epoch = int(replay["epochs_saved"])
        expected_epochs = frozen_schedule_epochs(final_epoch, int(replay["best_epoch"]))
        schedule_rows = endpoint["schedule"]
        schedule = {int(row["epoch"]): row for row in schedule_rows}
        schedule_epochs = sorted(schedule)
        required_roles = {"10pct", "25pct", "50pct", "75pct", "best_legal", "final"}
        actual_roles = {tag for row in schedule_rows for tag in row["schedule_tags"]}
        if (len(schedule_rows) != len(schedule) or schedule_epochs != expected_epochs or
                not required_roles.issubset(actual_roles)):
            raise RuntimeError("endpoint-derived checkpoint schedule invalid under frozen rounded/deduplicated rule")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x, y, subjects, sessions = C.UP.capped_train(data, task, "EEGNet", fold)
        if sorted(np.unique(sessions).tolist()) != [1, 2]:
            raise RuntimeError("TRAIN source/future sessions not 1/2")
        historical, head = C.UP.helper("EEGNet").build_model({
            "Model": "EEGNet", "Task": task, "fold": fold, "seed": 0,
            "channels": int(record.get("channels") or 62),
            "samples": int(record.get("samples") or 1000),
            "classes": int(record["classes"]), "checkpoint_path": str(checkpoint),
            "recipe_name": record.get("recipe", {}).get("name"),
            "trainable_parameters": int(record.get("trainable_parameters", record.get("parameters", 0))),
        }, device)
        h_final, native_logits, head_logits = C.UP.hook_representations(
            historical, head, x, "EEGNet", device)
        if np.max(np.abs(native_logits - head_logits)) >= 1e-5:
            raise RuntimeError("historical native head mismatch")
        spec = C.UP.helper("EEGNet").spectrum(h_final, y, subjects, sessions, task, "EEGNet", fold)
        if (C.UP.array_sha(spec["mean"], spec["basis"], spec["scale"], spec["directions"])
                != source["hashes"]["basis_sha256"] or int(spec["rank"]) != int(stored["rank"])):
            raise RuntimeError("frozen historical canonical basis/rank mismatch")
        dims = np.asarray(stored["protected_blocks"], dtype=int)
        final_q = C.UP.helper("EEGNet").canonical(h_final, spec)[:, dims].astype(np.float32)
        del historical, head, h_final, native_logits, head_logits
        search_data = S.load_search_fold(task, fold)
        if (search_data["split_sha256"] != record["split_sha256"] or
                search_data["normalizer"] != record["normalizer"] or
                search_data["classes"] != int(record["classes"])):
            raise RuntimeError("frozen SEARCH split/normalizer/classes mismatch")
        output = []
        for epoch in schedule_epochs:
            pt_path = cell / "checkpoint_native_pt_v3" / f"epoch_{epoch:03d}.json"
            pt = json.loads(pt_path.read_text(encoding="utf-8"))
            if (pt.get("status") != "NATIVE_CHECKPOINT_PT_COMPLETE_PENDING_BACKTRACE_AND_MECHANISM" or
                    pt.get("model") != "EEGNet" or pt.get("task") != task or int(pt.get("fold", -1)) != fold or
                    pt.get("final_heldout_accessed") is not False or pt.get("outer_development_accessed") is not False or
                    pt.get("trajectory_provenance") != replay["trajectory_provenance"] or
                    pt.get("checkpoint_sha256") != schedule[epoch]["checkpoint_sha256"]):
                raise RuntimeError(f"checkpoint-native P_t invalid at epoch {epoch}")
            checkpoint_path = cell / "replay_v2" / f"epoch_{epoch:03d}.pt"
            checkpoint_sha = C.digest(checkpoint_path)
            if checkpoint_sha != replay["epoch_checkpoint_sha256"][str(epoch)]:
                raise RuntimeError(f"replica checkpoint SHA mismatch at epoch {epoch}")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if int(payload["epoch"]) != epoch or payload["invariant"] != record["invariant_sha256"]:
                raise RuntimeError(f"replica checkpoint invariant mismatch at epoch {epoch}")
            net = S.build_model("EEGNet", dataset=search_data["dataset"],
                                channels=int(search_data["channels"]),
                                samples=int(search_data["samples"]),
                                classes=int(search_data["classes"]), tech_recipe=None).to(device)
            net.load_state_dict(payload["model"], strict=True)
            net.eval()
            h, direct, head_copy = C.UP.hook_representations(net, net.head, x, "EEGNet", device)
            forward_error = float(np.max(np.abs(direct - head_copy)))
            if forward_error >= 1e-5 or len(h) != len(final_q):
                raise RuntimeError(f"checkpoint representation mismatch at epoch {epoch}")
            rows = subject_crossfit(h, final_q, subjects, sessions)
            summary = {direction: {
                "subject_count": len([r for r in rows if r["direction"] == direction]),
                "equal_subject_mean_r2": float(np.mean([r["r2"] for r in rows if r["direction"] == direction])),
                "equal_subject_mean_normalized_mse": float(np.mean([r["normalized_mse"] for r in rows if r["direction"] == direction])),
            } for direction, _, _ in DIRECTIONS}
            output.append({"epoch": epoch, "schedule_tags": schedule[epoch]["schedule_tags"],
                           "replica_checkpoint_sha256": checkpoint_sha,
                           "native_pt_audit_sha256": C.digest(pt_path),
                           "native_pt_rank": int(pt["protected_rank"]),
                           "native_forward_max_abs": forward_error,
                           "subject_session_recovery": rows, "summary": summary})
            del net, h, direct, head_copy, payload
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if any(not np.isfinite(row["summary"][direction]["equal_subject_mean_r2"])
               for row in output for direction, _, _ in DIRECTIONS):
            raise RuntimeError("nonfinite final-P backtrace summary")
        C.write_json(result, {
            "status": "FINAL_P_BACKTRACE_COMPLETE_REPLICA_ONLY", "model": "EEGNet",
            "task": task, "fold": fold, "seed": 0,
            "trajectory_provenance": replay["trajectory_provenance"],
            "historical_final_P_definition": "FROZEN_CANONICAL_OBLIQUE_FINAL_PROTECTED_COORDINATES",
            "historical_final_checkpoint_sha256": source["hashes"]["checkpoint_sha256"],
            "historical_final_basis_sha256": source["hashes"]["basis_sha256"],
            "historical_final_protected_blocks": dims.tolist(),
            "target_rows": len(final_q), "target_subjects": len(C.PW.natural(subjects)),
            "ridge_alpha": RIDGE_ALPHA, "subject_crossfit_folds": CROSS_FITS,
            "ridge_solver": "PRIMAL_EXACT_KERNEL_EQUIVALENT_FIT_ONLY_STANDARDIZATION",
            "primal_dual_identity_max_abs": dual_error,
            "checkpoint_schedule_epochs": schedule_epochs,
            "checkpoint_schedule_rule": "ROUNDED_10_25_50_75_BEST_FINAL_DEDUPLICATED",
            "checkpoints": output,
            "replay_audit_sha256": C.digest(replay_path),
            "endpoint_audit_sha256": C.digest(endpoint_path),
            "upstream_gate_sha256": C.digest(gate_path),
            "upstream_protocol_sha256": protocol_sha,
            "upstream_implementation_sha256": ORIGINAL_IMPL_SHA,
            "analysis_lock_sha256": ANALYSIS_SHA,
            "implementation_sha256": C.digest(Path(__file__)),
            "outer_development_accessed": False, "final_heldout_accessed": False,
        })
        print("FINAL_P_BACKTRACE_V2_COMPLETE", task, fold, len(output), flush=True)
    except Exception as exc:
        C.write_json(failure, {"status": "FAIL_CLOSED", "reason": f"{type(exc).__name__}: {exc}",
                               "implementation_sha256": C.digest(Path(__file__)),
                               "outer_development_accessed": False,
                               "final_heldout_accessed": False})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
