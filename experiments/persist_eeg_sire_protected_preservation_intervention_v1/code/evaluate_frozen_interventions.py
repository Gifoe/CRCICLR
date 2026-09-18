#!/usr/bin/env python3
"""Heldout inference after every frozen Stage-2 checkpoint is audited."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common
from run_stage2_interventions import arms_for, expected_stage2, selected_coordinates


def verify_preheldout_freeze(targets: pd.DataFrame) -> dict:
    path = common.PROTOCOL / "PRE_HOLDOUT_FREEZE.json"
    if not path.is_file():
        raise RuntimeError("intervention heldout evaluation prohibited: PRE_HOLDOUT_FREEZE.json is absent")
    freeze = json.loads(path.read_text())
    if freeze.get("status") != "FROZEN_BEFORE_INTERVENTION_HELDOUT_INFERENCE":
        raise RuntimeError("invalid pre-heldout freeze status")
    for raw_path, expected_hash in freeze["files_sha256"].items():
        actual_path = Path(raw_path)
        if not actual_path.is_file() or common.sha256_file(actual_path) != expected_hash:
            raise RuntimeError(f"frozen artifact drift: {actual_path}")
    expected = expected_stage2(targets)
    if int(freeze["stage2_checkpoints"]) != len(expected):
        raise RuntimeError("frozen Stage-2 checkpoint count mismatch")
    return freeze


def drift_for_model(prd, model, task: str, fold_record: dict, fold: int, seed: int,
                    target: dict, randoms: pd.DataFrame, arm: str, device: torch.device) -> dict:
    bundle = common.stage2_bundle(prd, task, fold_record)
    normalizer_path, mean_np, std_np, _ = common.normalizer(prd, task, fold)
    raw = prd.base_global.RawGPUCache(bundle, device)
    target_arrays = common.target_arrays(task, fold, seed)
    cached = common.coordinate_cache(task, fold, seed)
    if target_arrays["metadata"]["selection_hash"] != target["selection_hash"] or cached["metadata"]["target_selection_hash"] != target["selection_hash"]:
        raise RuntimeError("representation-drift target/cache mismatch")
    q0 = torch.as_tensor(cached["q0"], dtype=torch.float32, device=device)
    mean = torch.as_tensor(target_arrays["mean"], dtype=torch.float32, device=device)
    whitener = torch.as_tensor(target_arrays["whitener"], dtype=torch.float32, device=device)
    directions = torch.as_tensor(target_arrays["directions"], dtype=torch.float32, device=device)
    protected = torch.as_tensor(target_arrays["protected"], dtype=torch.long, device=device)
    active = torch.as_tensor(target_arrays["active"], dtype=torch.long, device=device)
    selected = torch.as_tensor(selected_coordinates(randoms, task, fold, seed, arm, target_arrays["protected"]), dtype=torch.long, device=device)
    # Report every fixed target in every arm.  In particular this makes the
    # required Rj-preserve versus PRD-only drift sanity check possible without
    # changing its coordinate system after training.
    random_targets = {
        rid: torch.as_tensor(selected_coordinates(randoms, task, fold, seed, f"RANDOM_PRESERVE_R{rid}", target_arrays["protected"]), dtype=torch.long, device=device)
        for rid in common.RANDOM_CONTROL_IDS
    } if len(protected) else {}
    p_sum = nonp_sum = global_sum = target_sum = 0.0
    p_count = nonp_count = global_count = target_count = 0
    random_sums = {rid: 0.0 for rid in random_targets}
    random_counts = {rid: 0 for rid in random_targets}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(raw.x), 128):
            ids = np.arange(start, min(start + 128, len(raw.x)), dtype=np.int64)
            x, _ = raw.batch(ids, mean_np, std_np)
            h = model(x)[1].float()
            q = ((h - mean) @ whitener) @ directions
            diff = (q - q0.index_select(0, torch.as_tensor(ids, dtype=torch.long, device=device))).square()
            global_sum += float(diff.sum().cpu()); global_count += diff.numel()
            if len(protected):
                p_sum += float(diff.index_select(1, protected).sum().cpu()); p_count += diff.shape[0] * len(protected)
                nonp = torch.as_tensor(np.setdiff1d(active.cpu().numpy(), protected.cpu().numpy()), dtype=torch.long, device=device)
                nonp_sum += float(diff.index_select(1, nonp).sum().cpu()); nonp_count += diff.shape[0] * len(nonp)
            if len(selected):
                target_sum += float(diff.index_select(1, selected).sum().cpu()); target_count += diff.shape[0] * len(selected)
            for rid, coordinates in random_targets.items():
                random_sums[rid] += float(diff.index_select(1, coordinates).sum().cpu())
                random_counts[rid] += diff.shape[0] * len(coordinates)
    row = {"task": task, "fold": fold, "seed": seed, "arm": arm, "condition": arm,
           "scope": "Stage-2 source-session bundle, inner-train plus inner-validation subjects",
           "protected_rank": int(len(protected)), "active_rank": int(len(active)),
           "P_drift": p_sum / p_count if p_count else np.nan,
           "nonP_drift": nonp_sum / nonp_count if nonp_count else np.nan,
           "global_active_drift": global_sum / global_count if global_count else np.nan,
           "target_drift": target_sum / target_count if target_count else np.nan,
           "target_rank": int(len(selected)), "normalizer_sha256": common.sha256_file(normalizer_path)}
    for rid in common.RANDOM_CONTROL_IDS:
        row[f"R{rid}_drift"] = random_sums[rid] / random_counts[rid] if rid in random_counts else np.nan
    del raw, bundle
    gc.collect(); torch.cuda.empty_cache()
    return row


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    targets = pd.read_csv(common.PROTOCOL / "PROTECTED_TARGETS.csv")
    references = pd.read_csv(common.PROTOCOL / "REFERENCE_CHECKPOINT_AUDIT.csv")
    randoms = pd.read_csv(common.PROTOCOL / "RANDOM_TARGET_MANIFEST.csv")
    stage_audit = pd.read_csv(common.PROTOCOL / "STAGE2_MATCHING_AUDIT.csv")
    verify_preheldout_freeze(targets)
    if len(stage_audit) != len(expected_stage2(targets)):
        raise RuntimeError("checkpoint audit coverage absent")
    prd, runtime, _peeh = common.load_sources()
    _, fold_map, _ = prd.base_global.load_folds()
    device = torch.device("cuda")
    # CE reference rows were already replayed before target construction.
    session_rows = pd.read_csv(common.RUNTIME / "reference_replay_session_cells.csv").to_dict("records")
    for row in session_rows:
        row["arm"] = "CE_REFERENCE"
    drift_rows = []
    random_arm_rows = []
    for target_row in targets.itertuples(index=False):
        task, fold, seed = str(target_row.task), int(target_row.fold), int(target_row.seed)
        target = target_row._asdict()
        dataset = prd.base_global.TASKS[task]["dataset"]
        fold_record = next(item for item in fold_map[dataset] if int(item["fold_id"]) == fold)
        eval_bundle, subjects, sessions = common.evaluation_bundle(prd, runtime, task)
        eval_raw = prd.base_global.RawGPUCache(eval_bundle, device)
        normalizer_path, mean, std, _ = common.normalizer(prd, task, fold)
        # Reference drift uses the same frozen theta0 coordinate cache and therefore should be numerically zero.
        reference = references[(references.task == task) & (references.fold == fold) & (references.seed == seed)]
        if len(reference) != 1:
            raise RuntimeError("reference missing during final evaluation")
        ref_model = common.load_reference_model(prd, task, Path(reference.checkpoint.iloc[0]), device).eval()
        drift_rows.append(drift_for_model(prd, ref_model, task, fold_record, fold, seed, target, randoms, "CE_REFERENCE", device))
        del ref_model
        gc.collect(); torch.cuda.empty_cache()
        for arm in arms_for(int(target_row.protected_rank)):
            checkpoint = common.checkpoint_file(task, fold, seed, arm)
            audit = stage_audit[(stage_audit.task == task) & (stage_audit.fold == fold) & (stage_audit.seed == seed) & (stage_audit.arm == arm)]
            if len(audit) != 1 or common.sha256_file(checkpoint) != audit.checkpoint_sha256.iloc[0]:
                raise RuntimeError("Stage-2 checkpoint changed after pre-heldout freeze")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = prd.base_global.build_model("LiteBN_BASELINE", task)
            model.load_state_dict(payload["state_dict"], strict=True)
            model = model.to(device).eval()
            current = common.evaluate_sessions(prd, model, eval_bundle, eval_raw, mean, std, subjects, sessions)
            for item in current:
                item.update({"task": task, "fold": fold, "seed": seed, "condition": arm, "arm": arm,
                             "checkpoint": str(checkpoint), "checkpoint_sha256": audit.checkpoint_sha256.iloc[0],
                             "normalizer_sha256": common.sha256_file(normalizer_path), "protected_rank": int(target_row.protected_rank)})
            session_rows.extend(current)
            drift = drift_for_model(prd, model, task, fold_record, fold, seed, target, randoms, arm, device)
            drift.update({"checkpoint": str(checkpoint), "checkpoint_sha256": audit.checkpoint_sha256.iloc[0]})
            drift_rows.append(drift)
            if arm.startswith("RANDOM_PRESERVE_R"):
                random_arm_rows.extend(current)
            print(f"FROZEN_HELDOUT_EVAL task={task} f{fold} s{seed} arm={arm}", flush=True)
            del model
            gc.collect(); torch.cuda.empty_cache()
        del eval_raw, eval_bundle
        gc.collect(); torch.cuda.empty_cache()
        common.atomic_csv(common.OUT / "SUBJECT_SESSION_RESULTS.csv", session_rows)
        common.atomic_csv(common.OUT / "REPRESENTATION_DRIFT.csv", drift_rows)
    common.atomic_csv(common.OUT / "RANDOM_ARM_RESULTS.csv", random_arm_rows)
    print("FROZEN_INTERVENTION_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
