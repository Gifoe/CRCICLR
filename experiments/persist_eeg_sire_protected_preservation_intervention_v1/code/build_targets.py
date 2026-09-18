#!/usr/bin/env python3
"""Audit CE references, replay them, and freeze train-only P/R targets."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import common


def payload_record(row, checkpoint: Path, normalizer_path: Path, model) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    return {
        "task": str(row.task), "fold": int(row.fold), "seed": int(row.seed),
        "checkpoint": str(checkpoint), "checkpoint_sha256": common.sha256_file(checkpoint),
        "normalizer": str(normalizer_path), "normalizer_sha256": common.sha256_file(normalizer_path),
        "selected_epoch": int(payload["selected_epoch"]), "inner_val_BA": float(payload["inner_val_BA"]),
        "parameter_count": common.parameter_count(model), "model_definition_hash": common.model_definition_hash(model),
        "initial_state_hash": common.state_hash(state), "initial_BN_buffer_hash": common.bn_hash(state),
        "source_experiment": "persist_eeg_litebn_prd_multiseed_v1 matched CE checkpoint",
        "new_CE_reference_training": False,
    }


def audit_and_replay(prd, runtime, references: pd.DataFrame, device: torch.device) -> list[dict]:
    audit = []
    raw_rows = []
    official = pd.read_csv(common.PRD_MULTI / "outputs/MULTISEED_TASK_SUMMARY.csv")
    official = official[official.condition == "CE"].set_index("task")
    for task in common.TASKS:
        bundle, subjects, sessions = common.evaluation_bundle(prd, runtime, task)
        raw = prd.base_global.RawGPUCache(bundle, device)
        for row in references[references.task == task].itertuples(index=False):
            checkpoint = Path(row.checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            normalizer_path, mean, std, _ = common.normalizer(prd, task, int(row.fold))
            model = common.load_reference_model(prd, task, checkpoint, device).eval()
            audit.append(payload_record(row, checkpoint, normalizer_path, model))
            current = common.evaluate_sessions(prd, model, bundle, raw, mean, std, subjects, sessions)
            for result in current:
                result.update({"task": task, "fold": int(row.fold), "seed": int(row.seed),
                               "condition": "CE_REFERENCE", "checkpoint_sha256": common.sha256_file(checkpoint)})
            raw_rows.extend(current)
            print(f"REFERENCE_REPLAY task={task} f{row.fold} s{row.seed}", flush=True)
            del model
            gc.collect(); torch.cuda.empty_cache()
        del raw, bundle
        gc.collect(); torch.cuda.empty_cache()
    if len(audit) != 30:
        raise RuntimeError("incomplete reference audit")
    common.atomic_csv(common.PROTOCOL / "REFERENCE_CHECKPOINT_AUDIT.csv", audit)
    common.atomic_csv(common.RUNTIME / "reference_replay_session_cells.csv", raw_rows)

    frame = pd.DataFrame(raw_rows)
    session = frame.groupby(["task", "subject_id", "session"], as_index=False).agg(
        BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), repetitions=("fold", "count"))
    if not (session.repetitions == 15).all():
        raise RuntimeError("reference replay is missing fold/seed coverage")
    replay_rows = []
    for task, part in session.groupby("task"):
        subject_rows = []
        required = ("S1", "S2") if task == "OpenBMI_MI" else ("S0", "S1", "S2")
        for _, cell in part.groupby("subject_id"):
            values = cell.set_index("session")
            subject_rows.append({"future_BA": values.loc["S2", "BA"],
                                 "future_macro_F1": values.loc["S2", "macro_F1"],
                                 "WS_BA": values.loc[list(required), "BA"].min()})
        aggregate = pd.DataFrame(subject_rows).mean()
        for output_name, official_name in (("future_BA", "future_BA"),
                                           ("future_macro_F1", "future_macro_F1"),
                                           ("WS_BA", "WS_BA")):
            observed = float(aggregate[output_name])
            expected = float(official.loc[task, official_name])
            replay_rows.append({"task": task, "metric": output_name, "replayed": observed,
                                "formal_PRD_artifact": expected, "abs_difference": abs(observed - expected),
                                "pass": bool(abs(observed - expected) < 1e-8)})
    common.atomic_csv(common.OUT / "REFERENCE_REPLAY_AUDIT.csv", replay_rows)
    if not all(row["pass"] for row in replay_rows):
        raise RuntimeError("reference replay does not match the formal PRD artifact")
    return audit


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def target_for_cell(prd, peeh, reference, fold_map, device: torch.device) -> tuple[dict, list[dict], list[dict]]:
    task, fold_id, seed = str(reference.task), int(reference.fold), int(reference.seed)
    dataset = prd.base_global.TASKS[task]["dataset"]
    fold = next(item for item in fold_map[dataset] if int(item["fold_id"]) == fold_id)
    checkpoint = Path(reference.checkpoint)
    normalizer_path, mean, std, _ = common.normalizer(prd, task, fold_id)
    model = common.load_reference_model(prd, task, checkpoint, device).eval()

    # Selection sees only inner-train subjects and the formal two-session pairing.
    selection_full = prd.base_global.build_bundle(task, fold["inner_train_subjects"])
    selection_bundle = common.subset_bundle(prd.base_global, selection_full, common.persistence_sessions(task))
    selection_raw = prd.base_global.RawGPUCache(selection_bundle, device)
    selection_h = common.infer_embeddings(model, selection_raw, mean, std, device)
    selection_meta = common.metadata(selection_bundle)
    key = ("protected-preservation-v1", task, fold_id, seed, common.sha256_file(checkpoint))
    spectrum = peeh.build_spectrum(selection_meta, selection_h, key)
    protected, block_rows = peeh.select_protected(selection_h, selection_meta, spectrum, key, int(prd.base_global.TASKS[task]["classes"]))
    active = np.arange(len(spectrum["rho"]), dtype=np.int64)
    protected_array = np.asarray(protected, dtype=np.int64)
    target_payload = {
        "task": task, "fold": fold_id, "seed": seed,
        "reference_checkpoint_sha256": common.sha256_file(checkpoint),
        "normalizer_sha256": common.sha256_file(normalizer_path),
        "source_session_definition": list(common.persistence_sessions(task)),
        "selection_scope": "inner_train biological subjects only",
        "active_rank": int(len(active)), "protected_rank": int(len(protected_array)),
        "protected_fraction": float(len(protected_array) / len(active)),
        "protected_block_ids": [int(row["block"]) for row in block_rows if row["protected"]],
        "protected_coordinate_ids": protected_array.tolist(),
        "nonempty": bool(len(protected_array)),
        "protected_target_estimable": bool(len(protected_array)),
        "spectrum_audit": spectrum["audit"], "block_assignments": block_rows,
        "selector_source": str(common.PEEH), "selector_source_sha256": common.sha256_file(common.PEEH),
    }
    target_hash = common.sha_obj({"metadata": target_payload, "mean": spectrum["mean"],
                                  "whitener": spectrum["whitener"], "directions": spectrum["directions"],
                                  "protected": protected_array})
    target_payload["selection_hash"] = target_hash
    save_npz(common.target_file(task, fold_id, seed), mean=spectrum["mean"], whitener=spectrum["whitener"],
             directions=spectrum["directions"], dewhitener=spectrum["dewhitener"], rho=spectrum["rho"],
             protected=protected_array, active=active, metadata=np.asarray(json.dumps(common.clean(target_payload), sort_keys=True)))

    # Cache fixed θ0 coordinates for the entire Stage-2 bundle before any update.
    work_bundle = common.stage2_bundle(prd, task, fold)
    work_raw = prd.base_global.RawGPUCache(work_bundle, device)
    h0 = common.infer_embeddings(model, work_raw, mean, std, device)
    q0 = peeh.coordinates(h0, spectrum).astype(np.float32)
    work_meta = common.metadata(work_bundle)
    cache_meta = {"task": task, "fold": fold_id, "seed": seed, "reference_checkpoint_sha256": common.sha256_file(checkpoint),
                  "target_selection_hash": target_hash, "rows": int(len(q0)), "active_rank": int(q0.shape[1]),
                  "scope": "Stage-2 source-session bundle: inner-train plus inner-validation subjects; fixed θ0 eval coordinates"}
    save_npz(common.coordinate_cache_file(task, fold_id, seed), q0=q0, h0=h0,
             subject_id=work_meta.subject_id.astype("U16").to_numpy(), session_id=work_meta.session_id.to_numpy(np.int64),
             metadata=np.asarray(json.dumps(cache_meta, sort_keys=True)))

    # The exact existing PRD generator is used; only its first 20 predeclared epochs are Stage 2.
    episodes = prd.make_manifest(work_bundle, fold, task)[:common.STAGE2_EPOCHS]
    manifest = {"task": task, "fold": fold_id, "seed": seed, "epochs": common.STAGE2_EPOCHS,
                "steps_per_epoch": common.STEPS_PER_EPOCH, "episodes": episodes,
                "generator": "persist_eeg_litebn_prd_seed0_v1::make_manifest exact deterministic prefix"}
    manifest_hash = common.sha_obj(manifest)
    manifest["manifest_hash"] = manifest_hash
    common.atomic_json(common.stage_manifest_file(task, fold_id, seed), manifest)

    random_rows = []
    pool_path = common.RUNTIME / "manifests" / task / f"random_pool_fold{fold_id}_seed{seed}.json"
    rng = np.random.default_rng(common.stable_seed("preservation-random-target-pool", task, fold_id, seed, target_hash))
    pool = []
    for random_id in range(common.RANDOM_POOL_SIZE):
        coordinates = rng.choice(active, size=len(protected_array), replace=False).astype(int).tolist() if len(protected_array) else []
        pool.append({"random_id": random_id, "coordinate_ids": coordinates,
                     "target_hash": common.sha_obj({"active_rank": len(active), "coordinates": coordinates})})
    common.atomic_json(pool_path, {"task": task, "fold": fold_id, "seed": seed, "active_rank": len(active),
                                   "protected_rank": len(protected_array), "pool": pool,
                                   "pool_hash": common.sha_obj(pool)})
    for item in pool:
        random_rows.append({"task": task, "fold": fold_id, "seed": seed, "random_id": item["random_id"],
                            "rank": len(item["coordinate_ids"]), "coordinate_ids": json.dumps(item["coordinate_ids"]),
                            "overlap_with_P": len(set(item["coordinate_ids"]) & set(protected_array.tolist())),
                            "target_hash": item["target_hash"], "used_primary_control": item["random_id"] in common.RANDOM_CONTROL_IDS,
                            "pool_file": str(pool_path)})
    stage_arms = ["PRD_ONLY"]
    if len(protected_array):
        stage_arms += ["PROTECTED_PRESERVE", "RANDOM_PRESERVE_R0", "RANDOM_PRESERVE_R1", "RANDOM_PRESERVE_R2"]
    match_rows = []
    initial_state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    optimizer_cfg = {"optimizer": "AdamW", "lr": common.LR, "weight_decay": common.WEIGHT_DECAY,
                     "gradient_clip": common.CLIP, "scheduler": None, "epochs": common.STAGE2_EPOCHS}
    for arm in stage_arms:
        match_rows.append({"task": task, "fold": fold_id, "seed": seed, "arm": arm,
                           "reference_hash": common.sha256_file(checkpoint), "batch_manifest_hash": manifest_hash,
                           "initial_model_hash": common.state_hash(initial_state), "initial_BN_buffer_hash": common.bn_hash(initial_state),
                           "optimizer_config_hash": common.sha_obj(optimizer_cfg), "dropout_rng_seed": common.stable_seed("stage2-rng", task, fold_id, seed),
                           "updates": common.STAGE2_EPOCHS * common.STEPS_PER_EPOCH,
                           "target_selection_hash": target_hash, "match_preflight": True})
    row = {key: target_payload[key] for key in ("task", "fold", "seed", "active_rank", "protected_rank", "protected_fraction",
                                                 "protected_block_ids", "protected_coordinate_ids", "nonempty", "protected_target_estimable",
                                                 "source_session_definition", "selection_hash")}
    del model, selection_raw, work_raw, selection_bundle, selection_full, work_bundle
    gc.collect(); torch.cuda.empty_cache()
    return row, random_rows, match_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-reference-replay", action="store_true",
                        help="reuse a completed identical PRD reference replay after a server migration")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    common.PROTOCOL.mkdir(parents=True, exist_ok=True)
    common.RUNTIME.mkdir(parents=True, exist_ok=True)
    common.OUT.mkdir(parents=True, exist_ok=True)
    prd, runtime, peeh = common.load_sources()
    references = common.reference_records()
    device = torch.device("cuda")
    if args.skip_reference_replay:
        audit_path = common.PROTOCOL / "REFERENCE_CHECKPOINT_AUDIT.csv"
        replay_path = common.OUT / "REFERENCE_REPLAY_AUDIT.csv"
        raw_path = common.RUNTIME / "reference_replay_session_cells.csv"
        if not all(path.is_file() for path in (audit_path, replay_path, raw_path)):
            raise RuntimeError("cannot skip replay: the prior completed reference replay artifacts are absent")
        prior_audit = pd.read_csv(audit_path)
        prior_replay = pd.read_csv(replay_path)
        if len(prior_audit) != 30 or len(prior_replay) != 6 or not prior_replay["pass"].all():
            raise RuntimeError("cannot skip replay: prior CE reference replay is incomplete or failed")
        replay_mode = "reused completed identical-runtime CE reference replay; no duplicate neural inference after server migration"
        print("REFERENCE_REPLAY_REUSED_FROM_IDENTICAL_RUNTIME", flush=True)
    else:
        audit_and_replay(prd, runtime, references, device)
        replay_mode = "new CE reference replay before target construction"
    _, fold_map, split_hash = prd.base_global.load_folds()
    targets, randoms, matching = [], [], []
    for reference in references.itertuples(index=False):
        target, random_rows, match_rows = target_for_cell(prd, peeh, reference, fold_map, device)
        targets.append(target); randoms.extend(random_rows); matching.extend(match_rows)
        print(f"TARGET_FROZEN task={reference.task} f{reference.fold} s{reference.seed} rank={target['protected_rank']}", flush=True)
    if len(targets) != 30 or len(randoms) != 3000:
        raise RuntimeError("target/random coverage failure")
    if any(row["rank"] != next(t for t in targets if (t["task"], t["fold"], t["seed"]) == (row["task"], row["fold"], row["seed"]))["protected_rank"] for row in randoms):
        raise RuntimeError("random target rank mismatch")
    common.atomic_csv(common.PROTOCOL / "PROTECTED_TARGETS.csv", targets)
    common.atomic_csv(common.PROTOCOL / "RANDOM_TARGET_MANIFEST.csv", randoms)
    common.atomic_csv(common.PROTOCOL / "STAGE2_MATCHING_AUDIT.csv", matching)
    specification = {"schema": "SIRE_PROTECTED_PRESERVATION_INTERVENTION_V1", "tasks": list(common.TASKS),
                     "folds": list(common.FOLDS), "seeds": list(common.SEEDS), "reference": "existing matched CE selected checkpoint theta0",
                     "PRD_lambda": common.PRD_LAMBDA, "preserve_lambda": common.PRESERVE_LAMBDA,
                     "stage2_epochs": common.STAGE2_EPOCHS, "steps_per_epoch": common.STEPS_PER_EPOCH,
                     "optimizer": "AdamW", "lr": common.LR, "weight_decay": common.WEIGHT_DECAY, "gradient_clip": common.CLIP,
                     "scheduler": None, "early_stopping": False, "random_pool_size": common.RANDOM_POOL_SIZE,
                     "primary_random_ids": list(common.RANDOM_CONTROL_IDS), "random_overlap_rule": "exact PEEH active-coordinate sampling; overlap with P permitted",
                     "selection": "Appendix-L repaired train-only PERSIST selection; fixed theta0 coordinate system", "new_CE_reference_training": False,
                     "heldout_hyperparameter_tuning": False, "selector_retuning": False, "split_sha256": split_hash,
                     "reference_replay_mode": replay_mode}
    common.atomic_json(common.PROTOCOL / "INTERVENTION_SPEC.json", specification)
    common.atomic_json(common.PROTOCOL / "PROTOCOL.json", {**specification,
        "heldout_scope": {"OpenBMI_MI": "14-subject internal-heldout intervention evaluation", "WBCIC_MI": "10-subject true-outer intervention evaluation"},
        "checkpoint_freeze_rule": "all Stage-2 epoch20 final checkpoints must exist before intervention heldout inference",
        "primary": "ProtectedPreserve - mean(RandomPreserve R0,R1,R2)"})
    print("TARGET_BUILD_AND_REFERENCE_REPLAY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
