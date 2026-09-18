#!/usr/bin/env python3
"""Run matched fixed-budget Stage-2 PRD/preservation interventions.

This program never builds an evaluation bundle.  It writes epoch-20 final
checkpoints only, then freezes their audits.  Heldout inference lives in the
separate evaluator and is refused until PRE_HOLDOUT_FREEZE.json exists.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import common


def seed_all(value: int) -> None:
    # The protocol's SHA-derived seed spans torch's 63-bit range; NumPy's
    # legacy global RNG accepts only 32 bits.  This deterministic reduction is
    # applied identically for every matched arm.
    random.seed(value); np.random.seed(value % (2**32)); torch.manual_seed(value); torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def load_frozen_protocols():
    required = [common.PROTOCOL / name for name in ("INTERVENTION_SPEC.json", "REFERENCE_CHECKPOINT_AUDIT.csv", "PROTECTED_TARGETS.csv", "RANDOM_TARGET_MANIFEST.csv", "STAGE2_MATCHING_AUDIT.csv")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required target-stage protocols missing: " + "; ".join(missing))
    spec = json.loads((common.PROTOCOL / "INTERVENTION_SPEC.json").read_text())
    if (float(spec["PRD_lambda"]), float(spec["preserve_lambda"]), int(spec["stage2_epochs"])) != (common.PRD_LAMBDA, common.PRESERVE_LAMBDA, common.STAGE2_EPOCHS):
        raise RuntimeError("intervention spec is not the predeclared fixed protocol")
    return spec, pd.read_csv(common.PROTOCOL / "REFERENCE_CHECKPOINT_AUDIT.csv"), pd.read_csv(common.PROTOCOL / "PROTECTED_TARGETS.csv"), pd.read_csv(common.PROTOCOL / "RANDOM_TARGET_MANIFEST.csv"), pd.read_csv(common.PROTOCOL / "STAGE2_MATCHING_AUDIT.csv")


def selected_coordinates(randoms: pd.DataFrame, task: str, fold: int, seed: int, arm: str, protected: np.ndarray) -> np.ndarray:
    if arm == "PROTECTED_PRESERVE":
        return protected
    if arm.startswith("RANDOM_PRESERVE_R"):
        random_id = int(arm.rsplit("R", 1)[1])
        value = randoms[(randoms.task == task) & (randoms.fold == fold) & (randoms.seed == seed) & (randoms.random_id == random_id)]
        if len(value) != 1 or not bool(value.used_primary_control.iloc[0]):
            raise RuntimeError("frozen random target unavailable")
        return np.asarray(json.loads(value.coordinate_ids.iloc[0]), dtype=np.int64)
    return np.empty(0, dtype=np.int64)


def arms_for(protected_rank: int) -> tuple[str, ...]:
    return ("PRD_ONLY",) if not protected_rank else ("PRD_ONLY", "PROTECTED_PRESERVE", "RANDOM_PRESERVE_R0", "RANDOM_PRESERVE_R1", "RANDOM_PRESERVE_R2")


def run_arm(prd, task: str, fold: int, seed: int, reference: pd.Series, target: dict, randoms: pd.DataFrame,
            bundle, cache, manifest: dict, device: torch.device, arm: str, expected_audit: pd.Series) -> dict:
    output = common.checkpoint_file(task, fold, seed, arm)
    if output.is_file():
        previous = torch.load(output, map_location="cpu", weights_only=False)
        keys = ("reference_checkpoint_sha256", "manifest_hash", "initial_model_hash", "initial_BN_buffer_hash", "arm")
        expected = (str(reference.checkpoint_sha256), str(manifest["manifest_hash"]), str(expected_audit.initial_model_hash), str(expected_audit.initial_BN_buffer_hash), arm)
        if tuple(str(previous.get(key)) for key in keys) != expected:
            raise RuntimeError(f"stale Stage-2 checkpoint: {output}")
        return {"checkpoint": str(output), "checkpoint_sha256": common.sha256_file(output), "reused": True, **previous["audit"]}

    target_arrays = common.target_arrays(task, fold, seed)
    coordinate_cache = common.coordinate_cache(task, fold, seed)
    if target_arrays["metadata"]["selection_hash"] != target["selection_hash"] or coordinate_cache["metadata"]["target_selection_hash"] != target["selection_hash"]:
        raise RuntimeError("target/cache freeze mismatch")
    selected = selected_coordinates(randoms, task, fold, seed, arm, target_arrays["protected"])
    if arm != "PRD_ONLY" and len(selected) != int(target["protected_rank"]):
        raise RuntimeError("preservation target is not equal-rank")
    q0 = torch.as_tensor(coordinate_cache["q0"], dtype=torch.float32, device=device)
    mean = torch.as_tensor(target_arrays["mean"], dtype=torch.float32, device=device)
    whitener = torch.as_tensor(target_arrays["whitener"], dtype=torch.float32, device=device)
    directions = torch.as_tensor(target_arrays["directions"], dtype=torch.float32, device=device)
    selected_t = torch.as_tensor(selected, dtype=torch.long, device=device)

    # All arms reconstruct the same initial state, fresh optimizer and RNG state.
    rng_seed = int(expected_audit.dropout_rng_seed)
    seed_all(rng_seed)
    checkpoint = Path(reference.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    initial = payload["state_dict"]
    model = prd.base_global.build_model("LiteBN_BASELINE", task).to(device)
    model.load_state_dict(initial, strict=True)
    initial_hash = common.state_hash(model.state_dict())
    initial_bn = common.bn_hash(model.state_dict())
    if initial_hash != expected_audit.initial_model_hash or initial_bn != expected_audit.initial_BN_buffer_hash:
        raise RuntimeError("Stage-2 initial model/BN state does not match frozen audit")
    optimizer = torch.optim.AdamW(model.parameters(), lr=common.LR, weight_decay=common.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    started = time.perf_counter()
    history = []
    model.train()
    for epoch, episodes in enumerate(manifest["episodes"], 1):
        ce_values, prd_values, preserve_values, total_values = [], [], [], []
        for episode in episodes:
            indices = np.asarray(episode["support_indices"] + episode["query_indices"], dtype=np.int64)
            x, y = cache.batch(indices)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, h = model(x)
                ce = F.cross_entropy(logits, y)
                support_slots = torch.as_tensor(episode["support_slots"], device=device)
                query_slots = torch.as_tensor(episode["query_slots"], device=device)
                prd_value, _ = prd.prd_loss(h[:64], y[:64], support_slots, h[64:], y[64:], query_slots, num_classes=2)
                preserve = torch.zeros((), device=device)
                if arm != "PRD_ONLY":
                    q = ((h.float() - mean) @ whitener) @ directions
                    preserve = (q.index_select(1, selected_t) - q0.index_select(0, torch.as_tensor(indices, dtype=torch.long, device=device)).index_select(1, selected_t)).square().mean()
                loss = ce + common.PRD_LAMBDA * prd_value + common.PRESERVE_LAMBDA * preserve
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), common.CLIP)
            scaler.step(optimizer); scaler.update()
            ce_values.append(float(ce.detach().cpu())); prd_values.append(float(prd_value.detach().cpu()))
            preserve_values.append(float(preserve.detach().cpu())); total_values.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "CE": float(np.mean(ce_values)), "raw_PRD": float(np.mean(prd_values)),
                        "preserve": float(np.mean(preserve_values)), "total": float(np.mean(total_values)),
                        "updates": len(episodes)})
        print(f"STAGE2 task={task} f{fold} s{seed} arm={arm} epoch={epoch:02d} CE={history[-1]['CE']:.5f} P={history[-1]['preserve']:.6f}", flush=True)
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    final_bn = common.bn_hash(final_state)
    audit = {"task": task, "fold": fold, "seed": seed, "arm": arm,
             "reference_checkpoint_sha256": str(reference.checkpoint_sha256), "manifest_hash": str(manifest["manifest_hash"]),
             "initial_model_hash": initial_hash, "initial_BN_buffer_hash": initial_bn, "final_model_hash": common.state_hash(final_state),
             "final_BN_buffer_hash": final_bn, "optimizer_config_hash": str(expected_audit.optimizer_config_hash),
             "dropout_rng_seed": rng_seed, "update_count": common.STAGE2_EPOCHS * common.STEPS_PER_EPOCH,
             "epochs": common.STAGE2_EPOCHS, "PRD_lambda": common.PRD_LAMBDA, "preserve_lambda": common.PRESERVE_LAMBDA,
             "preserve_rank": int(len(selected)), "target_hash": common.sha_obj(selected.tolist()),
             "elapsed_seconds": time.perf_counter() - started, "no_early_stopping": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": final_state, "arm": arm, "epoch": common.STAGE2_EPOCHS, "history": history, "audit": audit,
                "reference_checkpoint_sha256": str(reference.checkpoint_sha256), "manifest_hash": str(manifest["manifest_hash"]),
                "initial_model_hash": initial_hash, "initial_BN_buffer_hash": initial_bn}, output)
    del model, optimizer, scaler
    gc.collect(); torch.cuda.empty_cache()
    return {"checkpoint": str(output), "checkpoint_sha256": common.sha256_file(output), "reused": False, **audit}


def expected_stage2(targets: pd.DataFrame) -> list[tuple[str, int, int, str]]:
    result = []
    for row in targets.itertuples(index=False):
        for arm in arms_for(int(row.protected_rank)):
            result.append((str(row.task), int(row.fold), int(row.seed), arm))
    return result


def train_all() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    _, references, targets, randoms, matching = load_frozen_protocols()
    if len(references) != 30 or len(targets) != 30:
        raise RuntimeError("incomplete frozen reference/target tables")
    prd, _runtime, _peeh = common.load_sources()
    _, fold_map, _ = prd.base_global.load_folds()
    device = torch.device("cuda")
    run_rows = []
    for target_row in targets.itertuples(index=False):
        task, fold, seed = str(target_row.task), int(target_row.fold), int(target_row.seed)
        reference = references[(references.task == task) & (references.fold == fold) & (references.seed == seed)]
        if len(reference) != 1:
            raise RuntimeError("missing frozen reference")
        reference = reference.iloc[0]
        dataset = prd.base_global.TASKS[task]["dataset"]
        fold_record = next(item for item in fold_map[dataset] if int(item["fold_id"]) == fold)
        bundle = common.stage2_bundle(prd, task, fold_record)
        norm_path, mean, std, _ = common.normalizer(prd, task, fold)
        raw = prd.base_global.RawGPUCache(bundle, device)
        cache = prd.base_runner.NormalizedCache(raw, mean, std)
        manifest = json.loads(common.stage_manifest_file(task, fold, seed).read_text())
        if len(manifest["episodes"]) != common.STAGE2_EPOCHS or any(len(x) != common.STEPS_PER_EPOCH for x in manifest["episodes"]):
            raise RuntimeError("fixed Stage-2 batch manifest invalid")
        target = target_row._asdict()
        for arm in arms_for(int(target_row.protected_rank)):
            expected = matching[(matching.task == task) & (matching.fold == fold) & (matching.seed == seed) & (matching.arm == arm)]
            if len(expected) != 1 or not bool(expected.match_preflight.iloc[0]):
                raise RuntimeError("missing matching preflight row")
            record = run_arm(prd, task, fold, seed, reference, target, randoms, bundle, cache, manifest, device, arm, expected.iloc[0])
            record.update({"normalizer": str(norm_path), "normalizer_sha256": common.sha256_file(norm_path),
                           "protected_rank": int(target_row.protected_rank), "protected_target_estimable": bool(target_row.protected_target_estimable)})
            run_rows.append(record)
            common.atomic_csv(common.OUT / "RUN_MANIFEST.csv", run_rows)
        del cache, raw, bundle
        gc.collect(); torch.cuda.empty_cache()
    expected = expected_stage2(targets)
    actual = {(str(row["task"]), int(row["fold"]), int(row["seed"]), str(row["arm"])) for row in run_rows}
    if set(expected) != actual:
        raise RuntimeError(f"Stage-2 checkpoint coverage mismatch expected={len(expected)} actual={len(actual)}")
    audit_rows = []
    for row in run_rows:
        checkpoint = Path(row["checkpoint"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        audit = payload["audit"]
        expected = matching[(matching.task == audit["task"]) & (matching.fold == audit["fold"]) &
                            (matching.seed == audit["seed"]) & (matching.arm == audit["arm"])]
        if len(expected) != 1:
            raise RuntimeError("matching preflight row disappeared")
        expected = expected.iloc[0]
        matches = (audit["reference_checkpoint_sha256"] == expected.reference_hash and
                   audit["manifest_hash"] == expected.batch_manifest_hash and
                   audit["initial_model_hash"] == expected.initial_model_hash and
                   audit["initial_BN_buffer_hash"] == expected.initial_BN_buffer_hash and
                   audit["optimizer_config_hash"] == expected.optimizer_config_hash and
                   audit["dropout_rng_seed"] == int(expected.dropout_rng_seed) and
                   audit["update_count"] == int(expected.updates) and
                   audit["epochs"] == common.STAGE2_EPOCHS and audit["no_early_stopping"])
        audit_rows.append({**audit, "checkpoint": str(checkpoint), "checkpoint_sha256": common.sha256_file(checkpoint),
                           "matches_preflight": bool(matches),
                           "checkpoint_frozen": True})
    if not all(row["matches_preflight"] for row in audit_rows):
        raise RuntimeError("Stage-2 matching audit failed")
    common.atomic_csv(common.PROTOCOL / "STAGE2_MATCHING_AUDIT.csv", audit_rows)
    print(f"STAGE2_ALL_CHECKPOINTS_FROZEN cells={len(audit_rows)}", flush=True)


def freeze() -> None:
    required = [common.PROTOCOL / name for name in ("INTERVENTION_SPEC.json", "REFERENCE_CHECKPOINT_AUDIT.csv", "PROTECTED_TARGETS.csv", "RANDOM_TARGET_MANIFEST.csv", "STAGE2_MATCHING_AUDIT.csv")]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("required pre-holdout protocol file missing")
    targets = pd.read_csv(common.PROTOCOL / "PROTECTED_TARGETS.csv")
    audit = pd.read_csv(common.PROTOCOL / "STAGE2_MATCHING_AUDIT.csv")
    expected = expected_stage2(targets)
    present = {(str(row.task), int(row.fold), int(row.seed), str(row.arm)) for row in audit.itertuples(index=False)}
    if set(expected) != present or not audit.matches_preflight.all() or not audit.checkpoint_frozen.all():
        raise RuntimeError("cannot freeze: Stage-2 checkpoint audit incomplete")
    coordinate_files = [common.target_file(str(row.task), int(row.fold), int(row.seed)) for row in targets.itertuples(index=False)]
    cache_files = [common.coordinate_cache_file(str(row.task), int(row.fold), int(row.seed)) for row in targets.itertuples(index=False)]
    manifest_files = [common.stage_manifest_file(str(row.task), int(row.fold), int(row.seed)) for row in targets.itertuples(index=False)]
    if not all(path.is_file() for path in coordinate_files + cache_files + manifest_files):
        raise RuntimeError("target/cache/manifest coverage incomplete")
    freeze_files = required + coordinate_files + cache_files + manifest_files + [Path(path) for path in audit.checkpoint]
    hashes = {str(path): common.sha256_file(path) for path in freeze_files}
    spec = json.loads((common.PROTOCOL / "INTERVENTION_SPEC.json").read_text())
    common.atomic_json(common.PROTOCOL / "PRE_HOLDOUT_FREEZE.json", {
        "status": "FROZEN_BEFORE_INTERVENTION_HELDOUT_INFERENCE", "files_sha256": hashes,
        "PRD_lambda": spec["PRD_lambda"], "preserve_lambda": spec["preserve_lambda"],
        "stage2_epochs": spec["stage2_epochs"], "primary_random_ids": spec["primary_random_ids"],
        "reference_cells": 30, "stage2_checkpoints": len(audit),
        "estimable_cells": int(targets.protected_target_estimable.sum()),
        "nonestimable_cells": int((~targets.protected_target_estimable).sum()),
        "heldout_opened_for_intervention": False,
    })
    print("PRE_HOLDOUT_FREEZE_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "freeze"), required=True)
    args = parser.parse_args()
    {"train": train_all, "freeze": freeze}[args.stage]()


if __name__ == "__main__":
    main()
