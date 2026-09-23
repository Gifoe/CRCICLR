"""Independent, exact-recipe EEGNet seed-0 training trajectory replay v2.

Original selected/latest checkpoints are read-only. This script uses the
frozen seven-backbone SEARCH recipe and saves each replay epoch under the
mechanism runtime. Endpoint identity is audited before any historical-path
claim; a mismatch yields RETRAINED_REPLICA_TRAJECTORY only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
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
from run_mediation_cell_v1 import verified_context, ORIGINAL_IMPL_SHA  # noqa: E402


def _parameter_difference(left: dict, right: dict) -> dict[str, object]:
    if set(left) != set(right):
        return {"same_keys": False, "all_equal": False, "max_abs": None}
    all_equal = True
    max_abs = 0.0
    for key in sorted(left):
        a, b = left[key].detach().cpu(), right[key].detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype:
            return {"same_keys": True, "all_equal": False, "max_abs": None, "mismatched_key": key}
        all_equal &= bool(torch.equal(a, b))
        if a.numel():
            max_abs = max(max_abs, float((a.to(torch.float64) - b.to(torch.float64)).abs().max().item()))
    return {"same_keys": True, "all_equal": all_equal, "max_abs": max_abs}


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    x, y = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("OpenBMI_MI", "OpenBMI_SSVEP"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    model_name, task, fold, seed = "EEGNet", args.task, args.fold, 0
    replay_dir = RUNTIME / "cells" / "eegnet" / task.lower() / f"fold{fold}_seed0" / "replay_v2"
    result_path = replay_dir / "REPLAY_AUDIT.json"
    failure_path = replay_dir / "REPLAY_FAIL_CLOSED.json"
    lock_path = replay_dir / "RUNNING.lock"
    if result_path.exists() or failure_path.exists() or lock_path.exists() or replay_dir.exists():
        raise RuntimeError("replay evidence/directory already exists; no duplicate or overwrite")
    try:
        gate_path, _, source, protocol_sha = verified_context(model_name, task, fold)
        record = source["record"]
        if record.get("model") != model_name or record.get("task") != task or record.get("seed") != seed:
            raise RuntimeError("historical checkpoint record identity mismatch")
        if C.digest(source["checkpoint"]) != source["hashes"]["checkpoint_sha256"]:
            raise RuntimeError("historical selected checkpoint hash mismatch")
        source_freeze = S._freeze_hash()
        if source_freeze != record["protocol_freeze_sha256"]:
            raise RuntimeError("historical seven-backbone freeze hash mismatch")
        data = S.load_search_fold(task, fold)
        recipe = S._frozen_recipe(task, model_name)
        if (data["split_sha256"] != record["split_sha256"]
                or data["normalizer"] != record["normalizer"] or recipe != record["recipe"]):
            raise RuntimeError("baseline data/split/normalizer/recipe mismatch")
        replay_dir.mkdir(parents=True, exist_ok=False)
        lock_path.mkdir(exist_ok=False)
        (lock_path / "pid.txt").write_text(str(os.getpid()), encoding="utf-8")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        started = time.perf_counter()
        S._set_seed(seed)
        model = S.build_model(model_name, dataset=data["dataset"], channels=data["channels"],
                              samples=data["samples"], classes=data["classes"], tech_recipe=None).to(device)
        initial_hash = S._state_sha(model)
        if initial_hash != record["initial_state_sha256"]:
            raise RuntimeError("baseline initialization hash mismatch")
        S._set_seed(seed + 100000)
        train_x = torch.from_numpy(S._resample(model_name, data["train_x"])).to(device, non_blocking=True)
        val_x = torch.from_numpy(S._resample(model_name, data["val_x"])).to(device, non_blocking=True)
        outer_x = torch.from_numpy(S._resample(model_name, data["outer_x"])).to(device, non_blocking=True)
        train_y = torch.as_tensor(data["train_y"], dtype=torch.long, device=device)
        weight = None
        if data["weighted_cross_entropy"]:
            counts = np.bincount(data["train_y"], minlength=data["classes"])
            weight = torch.as_tensor(len(data["train_y"]) / (data["classes"] * counts),
                                     dtype=torch.float32, device=device)
        invariant = hashlib.sha256(json.dumps({
            "task": task, "model": model_name, "fold": fold, "seed": seed,
            "initial_state": initial_hash, "split": data["split_sha256"],
            "normalizer": data["normalizer"], "recipe": recipe, "freeze": source_freeze,
            "adapter": S.MODEL_NATIVE_RESAMPLING[model_name],
        }, sort_keys=True).encode()).hexdigest()
        if invariant != record["invariant_sha256"]:
            raise RuntimeError("baseline training invariant hash mismatch")
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"],
                                      weight_decay=recipe["weight_decay"])
        history: list[dict] = []
        best, best_epoch, best_state = -float("inf"), 0, None
        for epoch in range(1, recipe["epochs"] + 1):
            tick = time.perf_counter()
            model.train()
            losses = []
            for indices in S._batch_order(len(train_y), task=task, model=model_name, fold=fold,
                                          seed=seed, epoch=epoch, size=recipe["batch_size"]):
                idx = torch.as_tensor(indices, dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(train_x.index_select(0, idx)), train_y.index_select(0, idx), weight=weight)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite baseline replay CE at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip"])
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            inner = S._evaluate(model, val_x, data["val_y"], data["val_subjects"], recipe["batch_size"])
            selected = epoch >= recipe["min_epoch"] and inner["subject_equal_BA"] > best + 1e-12
            if selected:
                best, best_epoch = float(inner["subject_equal_BA"]), epoch
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            history.append({"epoch": epoch, "cross_entropy": float(np.mean(losses)),
                            "inner_validation": inner, "selected": selected})
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            checkpoint = S._cpu_checkpoint({
                "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": S._rng_state(), "best": best, "best_epoch": best_epoch,
                "best_state": best_state, "history": history, "invariant": invariant,
            })
            S._torch(replay_dir / f"epoch_{epoch:03d}.pt", checkpoint)
            del checkpoint
            print("REPLAY_EPOCH", model_name, task, fold, epoch,
                  "inner_BA", round(inner["subject_equal_BA"], 6),
                  "seconds", round(time.perf_counter() - tick, 2), flush=True)
        if best_state is None:
            raise RuntimeError("baseline replay has no eligible selected epoch")
        historical_best = torch.load(source["checkpoint"], map_location="cpu", weights_only=False)
        historical_latest_path = Path(record["checkpoint_path"]).with_name("latest.pt")
        if not historical_latest_path.is_file():
            raise RuntimeError("historical latest endpoint absent after v3 audit")
        historical_latest = torch.load(historical_latest_path, map_location="cpu", weights_only=False)
        best_comparison = _parameter_difference(best_state, historical_best["state_dict"])
        final_comparison = _parameter_difference(model.state_dict(), historical_latest["model"])
        history_equal = history == record["history"]
        replay_best_model = S.build_model(model_name, dataset=data["dataset"], channels=data["channels"],
                                          samples=data["samples"], classes=data["classes"], tech_recipe=None).to(device)
        replay_best_model.load_state_dict(best_state, strict=True)
        replay_logits = []
        historical_logits = []
        historical_model = S.build_model(model_name, dataset=data["dataset"], channels=data["channels"],
                                         samples=data["samples"], classes=data["classes"], tech_recipe=None).to(device)
        historical_model.load_state_dict(historical_best["state_dict"], strict=True)
        with torch.inference_mode():
            for start in range(0, len(data["outer_y"]), recipe["batch_size"]):
                x = outer_x[start:start + recipe["batch_size"]]
                replay_best_model.eval(); historical_model.eval()
                replay_logits.append(replay_best_model(x).float().cpu().numpy())
                historical_logits.append(historical_model(x).float().cpu().numpy())
        rz, hz = np.concatenate(replay_logits), np.concatenate(historical_logits)
        replay_outer = S._subject_metrics(data["outer_y"], rz, data["outer_subjects"])
        historical_outer = S._subject_metrics(data["outer_y"], hz, data["outer_subjects"])
        exact = (best_comparison["all_equal"] and final_comparison["all_equal"] and history_equal
                 and best_epoch == int(record["selected_epoch"])
                 and float(best) == float(record["best_inner_validation_BA"]))
        status = "EXACT_HISTORICAL_TRAJECTORY" if exact else "RETRAINED_REPLICA_TRAJECTORY"
        audit = {
            "status": "REPLAY_COMPLETE", "trajectory_provenance": status,
            "model": model_name, "task": task, "fold": fold, "seed": seed,
            "epochs_saved": recipe["epochs"], "best_epoch": best_epoch,
            "historical_best_epoch": int(record["selected_epoch"]),
            "best_validation_BA": best, "historical_best_validation_BA": float(record["best_inner_validation_BA"]),
            "history_exact_equal": history_equal,
            "best_parameter_comparison": best_comparison, "final_parameter_comparison": final_comparison,
            "outer_development_replay_BA": replay_outer["subject_equal_BA"],
            "outer_development_historical_BA": historical_outer["subject_equal_BA"],
            "outer_development_logits_correlation": _correlation(rz, hz),
            "selected_checkpoint_sha256": C.digest(source["checkpoint"]),
            "historical_latest_sha256": C.digest(historical_latest_path),
            "epoch_checkpoint_sha256": {str(epoch): C.digest(replay_dir / f"epoch_{epoch:03d}.pt")
                                        for epoch in range(1, recipe["epochs"] + 1)},
            "baseline_training_source_sha256": C.digest(Path(S.__file__)),
            "baseline_protocol_freeze_sha256": source_freeze,
            "upstream_gate_sha256": C.digest(gate_path),
            "upstream_protocol_sha256": protocol_sha,
            "upstream_implementation_sha256": ORIGINAL_IMPL_SHA,
            "implementation_sha256": C.digest(Path(__file__)),
            "elapsed_seconds": time.perf_counter() - started,
            "final_heldout_accessed": False,
        }
        C.write_json(result_path, audit)
        (lock_path / "pid.txt").unlink()
        lock_path.rmdir()
        print("REPLAY_COMPLETE", status, model_name, task, fold, flush=True)
    except Exception as error:
        C.write_json(failure_path, {"status": "FAIL_CLOSED", "reason": f"{type(error).__name__}: {error}",
                                    "implementation_sha256": C.digest(Path(__file__)),
                                    "final_heldout_accessed": False})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
