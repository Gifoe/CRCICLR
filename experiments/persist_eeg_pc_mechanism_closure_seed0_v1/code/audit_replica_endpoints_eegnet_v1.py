"""Read-only, fixed-schedule audit of an independently replayed EEGNet trajectory.

This produces checkpoint identity and outer-development descriptive metrics,
not checkpoint-native P_t or a complete mechanism cell. Outer outcomes are
never used to select epochs, directions, or hyperparameters.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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


def fixed_schedule(final_epoch: int, best_epoch: int) -> list[dict[str, object]]:
    if final_epoch < 1 or not 1 <= best_epoch <= final_epoch:
        raise ValueError("invalid replay epoch range")
    requested = [("10pct", .10), ("25pct", .25), ("50pct", .50), ("75pct", .75)]
    pairs = [(name, min(final_epoch, max(1, math.floor(final_epoch * fraction + .5))))
             for name, fraction in requested]
    pairs.extend((("best_legal", best_epoch), ("final", final_epoch)))
    by_epoch: dict[int, list[str]] = {}
    for name, epoch in pairs:
        by_epoch.setdefault(epoch, []).append(name)
    return [{"epoch": epoch, "schedule_tags": tags} for epoch, tags in sorted(by_epoch.items())]


def metrics(model, x: torch.Tensor, labels: np.ndarray, subjects: np.ndarray,
            batch_size: int) -> dict[str, object]:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            outputs.append(model(x[start:start + batch_size]).float().cpu().numpy())
    logits = np.concatenate(outputs)
    if logits.shape != (len(labels), int(logits.shape[1])) or not np.isfinite(logits).all():
        raise RuntimeError("invalid checkpoint outer logits")
    summary = S._subject_metrics(labels, logits, subjects)
    per_trial_ce = F.cross_entropy(torch.from_numpy(logits),
                                   torch.as_tensor(labels, dtype=torch.long),
                                   reduction="none").numpy()
    subject_ids = subjects.astype(str)
    subject_ce = {subject: float(per_trial_ce[subject_ids == subject].mean())
                  for subject in sorted(set(subject_ids))}
    return {"subject_equal_BA": summary["subject_equal_BA"],
            "subject_equal_macro_F1": summary["subject_equal_macro_F1"],
            "subject_equal_CE": float(np.mean(list(subject_ce.values()))),
            "subject_count": len(subject_ce),
            "subject_rows": [{**row, "CE": subject_ce[row["subject"]]}
                             for row in summary["subjects"]]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("OpenBMI_MI", "OpenBMI_SSVEP"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    task, fold = args.task, args.fold
    cell = RUNTIME / "cells" / "eegnet" / task.lower() / f"fold{fold}_seed0"
    result = cell / "REPLICA_ENDPOINT_AUDIT_V1.json"
    failure = cell / "REPLICA_ENDPOINT_AUDIT_V1_FAIL_CLOSED.json"
    if result.exists() or failure.exists():
        raise RuntimeError("terminal endpoint-audit evidence already exists; no overwrite")
    try:
        gate_path, _, source, protocol_sha = verified_context("EEGNet", task, fold)
        replay_path = cell / "replay_v2" / "REPLAY_AUDIT.json"
        replay = __import__("json").loads(replay_path.read_text(encoding="utf-8"))
        if (replay.get("status") != "REPLAY_COMPLETE" or
                replay.get("trajectory_provenance") not in
                ("EXACT_HISTORICAL_TRAJECTORY", "RETRAINED_REPLICA_TRAJECTORY") or
                replay.get("final_heldout_accessed") is not False or
                replay.get("upstream_gate_sha256") != C.digest(gate_path)):
            raise RuntimeError("replay provenance is incomplete or mismatched")
        record = source["record"]
        data = S.load_search_fold(task, fold)
        recipe = S._frozen_recipe(task, "EEGNet")
        if (data["split_sha256"] != record["split_sha256"] or
                data["normalizer"] != record["normalizer"] or
                recipe != record["recipe"]):
            raise RuntimeError("frozen split, normalizer, or recipe mismatch")
        final_epoch = int(replay["epochs_saved"])
        if final_epoch != recipe["epochs"]:
            raise RuntimeError("replay did not save all frozen epochs")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.from_numpy(S._resample("EEGNet", data["outer_x"])).to(device)
        schedule = fixed_schedule(final_epoch, int(replay["best_epoch"]))
        rows = []
        for item in schedule:
            epoch = int(item["epoch"])
            checkpoint_path = cell / "replay_v2" / f"epoch_{epoch:03d}.pt"
            checkpoint_sha = C.digest(checkpoint_path)
            if checkpoint_sha != replay["epoch_checkpoint_sha256"][str(epoch)]:
                raise RuntimeError(f"replay epoch {epoch} SHA mismatch")
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if int(saved["epoch"]) != epoch or saved["invariant"] != record["invariant_sha256"]:
                raise RuntimeError(f"replay epoch {epoch} identity mismatch")
            model = S.build_model("EEGNet", dataset=data["dataset"],
                                  channels=data["channels"], samples=data["samples"],
                                  classes=data["classes"], tech_recipe=None).to(device)
            model.load_state_dict(saved["model"], strict=True)
            outer = metrics(model, x, data["outer_y"], data["outer_subjects"],
                            recipe["batch_size"])
            rows.append({"epoch": epoch, "schedule_tags": item["schedule_tags"],
                         "checkpoint_sha256": checkpoint_sha,
                         "inner_validation_subject_equal_BA":
                             saved["history"][epoch - 1]["inner_validation"]["subject_equal_BA"],
                         "outer_development": outer})
            del saved, model
            print("REPLICA_ENDPOINT_AUDITED", task, fold, epoch, flush=True)
        C.write_json(result, {
            "status": "REPLICA_ENDPOINTS_AUDITED_PENDING_NATIVE_P_AND_MECHANISM",
            "model": "EEGNet", "task": task, "fold": fold, "seed": 0,
            "trajectory_provenance": replay["trajectory_provenance"],
            "schedule": rows, "replay_audit_sha256": C.digest(replay_path),
            "upstream_gate_sha256": C.digest(gate_path),
            "upstream_protocol_sha256": protocol_sha,
            "upstream_implementation_sha256": ORIGINAL_IMPL_SHA,
            "analysis_lock_sha256": ANALYSIS_SHA,
            "implementation_sha256": C.digest(Path(__file__)),
            "outer_development_used_for_selection": False,
            "final_heldout_accessed": False,
        })
    except Exception as exc:
        C.write_json(failure, {"status": "FAIL_CLOSED", "reason": f"{type(exc).__name__}: {exc}",
                               "implementation_sha256": C.digest(Path(__file__)),
                               "final_heldout_accessed": False})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
