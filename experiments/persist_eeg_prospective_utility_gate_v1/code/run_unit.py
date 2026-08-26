"""Run one resumable ERM refit in source or frozen-outcome phase."""
from __future__ import annotations

import argparse
import gc
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import common


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_model(backbone: str, initialization_seed: int, path: Path, device: torch.device) -> torch.nn.Module:
    model = common.build_model(backbone, initialization_seed)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device)


def logits_from_features(model: torch.nn.Module, features: np.ndarray) -> np.ndarray:
    weight = model.head.weight.detach().float().cpu().numpy().astype(np.float64)
    bias = model.head.bias.detach().float().cpu().numpy().astype(np.float64)
    return np.asarray(features, dtype=np.float64) @ weight.T + bias


def role_indices(metadata: pd.DataFrame, roles: dict[str, tuple[str, ...]], role: str, sessions: tuple[int, ...]) -> np.ndarray:
    indices = common.row_indices(metadata, roles[role], sessions)
    expected = len(roles[role]) * len(sessions) * 100
    if len(indices) != expected:
        raise RuntimeError(f"{role} row cardinality {len(indices)} != {expected}")
    return indices


def source_phase(backbone: str, fold: int, seed: int) -> None:
    context = common.unit_dir(backbone, fold, seed)
    context.mkdir(parents=True, exist_ok=True)
    complete = context / "SOURCE_COMPLETE.json"
    if complete.is_file() and common.read_json(complete).get("pass") is True:
        print(f"[source cached] {backbone} fold={fold} seed={seed}", flush=True)
        return
    if (common.RUNTIME / "GLOBAL_SOURCE_FREEZE.json").exists():
        raise RuntimeError("cannot create/modify source artifacts after global source freeze")
    if common.read_json(common.RUNTIME / "PREFLIGHT.json").get("pass") is not True:
        raise RuntimeError("preflight did not pass")
    roles = common.frozen_fold(fold)
    # Fold-specific loader scope excludes this fold's outer outcome subjects.
    scoped = common.materialize_subject_scope(common.load_data(label_subjects=roles["source"]), roles["source"])
    train_idx = role_indices(scoped.metadata, roles, "fit_train", (1, 2))
    val_idx = role_indices(scoped.metadata, roles, "fit_validation", (2,))
    pseudo_idx = role_indices(scoped.metadata, roles, "pseudo_target", (2,))
    if set(train_idx) & set(val_idx) or set(train_idx) & set(pseudo_idx) or set(val_idx) & set(pseudo_idx):
        raise RuntimeError("source role row overlap")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    raw = torch.as_tensor(scoped.x, dtype=torch.float32, device=device)
    mean, std = common.compute_normalizer(raw, train_idx)
    np.savez_compressed(context / "normalizer.npz", mean=mean.cpu().numpy(), std=std.cpu().numpy())
    initialization_seed = common.stable_seed("utility-main-init", backbone, fold, seed)
    loader_seed = common.stable_seed("utility-minibatch", backbone, fold, seed)
    unit_protocol = {
        "backbone": backbone,
        "fold": fold,
        "seed": seed,
        "roles": {key: list(value) for key, value in roles.items()},
        "fit_train_rows": len(train_idx),
        "fit_validation_rows": len(val_idx),
        "pseudo_target_rows": len(pseudo_idx),
        "initialization_seed": initialization_seed,
        "loader_seed": loader_seed,
        "source_label_scope": list(roles["source"]),
        "outcome_subjects_or_labels_materialized": False,
        "data_cache": str(scoped.cache_root),
    }
    common.write_json(context / "UNIT_PROTOCOL.json", unit_protocol)
    checkpoint_path = context / "checkpoint.pt"
    training_meta_path = context / "training.json"
    if checkpoint_path.exists() and training_meta_path.exists():
        training = common.read_json(training_meta_path)
        if training.get("checkpoint_sha256") != file_sha256(checkpoint_path):
            raise RuntimeError("cached checkpoint hash mismatch")
        model = checkpoint_model(backbone, initialization_seed, checkpoint_path, device)
    else:
        model, training = common.train_model(
            backbone=backbone,
            method="ERM",
            lam=0.0,
            raw=raw,
            metadata=scoped.metadata,
            train_indices=train_idx,
            validation_indices=val_idx,
            mean=mean,
            std=std,
            initialization_seed=initialization_seed,
            loader_seed=loader_seed,
            bandwidths=[],
        )
        torch.save({"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()}}, checkpoint_path)
        training.update(
            {
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "fit_validation_session": 2,
                "pseudo_target_used_for_training_or_selection": False,
                "outcome_used": False,
            }
        )
        common.write_json(training_meta_path, training)
    checkpoint_sha = file_sha256(checkpoint_path)

    fit_all = common.evaluate_model(model, raw, scoped.metadata, train_idx, mean, std)
    fit_s2_idx = role_indices(scoped.metadata, roles, "fit_train", (2,))
    fit_s2 = common.evaluate_model(model, raw, scoped.metadata, fit_s2_idx, mean, std)
    validation = common.evaluate_model(model, raw, scoped.metadata, val_idx, mean, std)
    center, basis, direction_meta = common.persistent_directions(fit_all["features"], fit_all["subjects"], fit_all["sessions"], 8)
    norms = np.linalg.norm(basis, axis=0)
    if basis.shape != (64, 8) or not np.allclose(norms, 1.0, atol=1e-10):
        raise RuntimeError("direction normalization/count failure")
    directions_path = context / "DIRECTIONS_FROZEN.npz"
    np.savez_compressed(
        directions_path,
        center=center,
        directions=basis,
        pool_index=np.asarray([row["pool_index"] for row in direction_meta], dtype=np.int64),
        persistence=np.asarray([row["persistence"] for row in direction_meta], dtype=np.float64),
        geometry_strength=np.asarray([row["geometry_strength"] for row in direction_meta], dtype=np.float64),
    )
    directions_file_sha = file_sha256(directions_path)
    model_freeze = {
        "schema": "UTILITY_SOURCE_MODEL_DIRECTIONS_FROZEN_V1",
        "backbone": backbone,
        "fold": fold,
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha,
        "directions_file_sha256": directions_file_sha,
        "direction_sha256": [common.array_sha256(basis[:, i]) for i in range(8)],
        "direction_count": 8,
        "construction_scope": "fit_train Sessions 1+2 only",
        "pseudo_labels_loaded_before_model_direction_freeze": False,
        "outcome_subjects_or_labels_loaded": False,
        "frozen_at_unix": time.time(),
    }
    common.write_json(context / "MODEL_DIRECTIONS_FROZEN.json", model_freeze)

    full_identity = common.identity_probe(fit_all["features"], fit_all["subjects"], fit_all["sessions"])["identity_symmetric"]
    source_rows: list[dict[str, Any]] = []
    pseudo_subject_rows: list[pd.DataFrame] = []
    pseudo = common.evaluate_model(model, raw, scoped.metadata, pseudo_idx, mean, std)
    for direction_id, (direction, meta) in enumerate(zip(basis.T, direction_meta)):
        erased_fit_all = common.erase_direction(fit_all["features"], center, direction)
        erased_fit_all_logits = logits_from_features(model, erased_fit_all)
        erased_identity = common.identity_probe(erased_fit_all, fit_all["subjects"], fit_all["sessions"])["identity_symmetric"]
        erased_fit_s2_logits = logits_from_features(model, common.erase_direction(fit_s2["features"], center, direction))
        erased_val_logits = logits_from_features(model, common.erase_direction(validation["features"], center, direction))
        erased_pseudo_logits = logits_from_features(model, common.erase_direction(pseudo["features"], center, direction))
        train_utility = common.per_subject_utility(fit_s2["labels"], fit_s2["logits"], erased_fit_s2_logits, fit_s2["subjects"])
        val_utility = common.per_subject_utility(validation["labels"], validation["logits"], erased_val_logits, validation["subjects"])
        pseudo_utility = common.per_subject_utility(pseudo["labels"], pseudo["logits"], erased_pseudo_logits, pseudo["subjects"])
        psummary = common.utility_summary(
            pseudo_utility,
            "U_pseudo",
            common.stable_seed("pseudo-subject-bootstrap", backbone, fold, seed, direction_id),
        )
        row = {
            "backbone": backbone,
            "fold": fold,
            "seed": seed,
            "run_id": f"{backbone}|fold-{fold}|seed-{seed}",
            "direction_id": direction_id,
            "direction_rank": direction_id + 1,
            "source_pool_index": int(meta["pool_index"]),
            "persistence": float(meta["persistence"]),
            "geometry_strength": float(meta["geometry_strength"]),
            "identity_score": float(full_identity - erased_identity),
            "identity_full": float(full_identity),
            "identity_erased": float(erased_identity),
            "D_finite": common.exact_d_finite(fit_all["logits"], erased_fit_all_logits),
            "C_train_BA_harm": float(-train_utility.U_BA.mean()),
            "C_train_CE_harm": float(-train_utility.U_CE.mean()),
            "C_validation_BA_harm": float(-val_utility.U_BA.mean()),
            "C_validation_CE_harm": float(-val_utility.U_CE.mean()),
            **psummary,
            "model_SHA": checkpoint_sha,
            "direction_SHA": common.array_sha256(direction),
            "direction_construction_SHA": directions_file_sha,
            "pseudo_subject_count": len(pseudo_utility),
            "future_subject_count": 8,
            "outcome_loaded": False,
        }
        source_rows.append(row)
        pseudo_utility.insert(0, "direction_id", direction_id)
        pseudo_utility.insert(0, "seed", seed)
        pseudo_utility.insert(0, "fold", fold)
        pseudo_utility.insert(0, "backbone", backbone)
        pseudo_subject_rows.append(pseudo_utility)
    source_frame = pd.DataFrame(source_rows)
    subject_frame = pd.concat(pseudo_subject_rows, ignore_index=True)
    common.write_csv(context / "source_direction_cells.csv", source_frame)
    common.write_csv(context / "pseudo_subject_utility.csv", subject_frame)
    artifacts = {
        "UNIT_PROTOCOL.json": file_sha256(context / "UNIT_PROTOCOL.json"),
        "checkpoint.pt": checkpoint_sha,
        "training.json": file_sha256(training_meta_path),
        "MODEL_DIRECTIONS_FROZEN.json": file_sha256(context / "MODEL_DIRECTIONS_FROZEN.json"),
        "DIRECTIONS_FROZEN.npz": directions_file_sha,
        "source_direction_cells.csv": file_sha256(context / "source_direction_cells.csv"),
        "pseudo_subject_utility.csv": file_sha256(context / "pseudo_subject_utility.csv"),
    }
    common.write_json(
        complete,
        {
            "pass": True,
            "backbone": backbone,
            "fold": fold,
            "seed": seed,
            "direction_count": len(source_frame),
            "source_artifacts": artifacts,
            "source_artifacts_frozen": True,
            "pseudo_target_used_for_training_or_selection": False,
            "outcome_subjects_or_labels_loaded": False,
            "completed_at_unix": time.time(),
        },
    )
    print(f"[source complete] {backbone} fold={fold} seed={seed} valBA={training['best_validation_BA']:.5f}", flush=True)
    del model, raw, scoped, fit_all, fit_s2, validation, pseudo
    gc.collect()
    torch.cuda.empty_cache()


def outcome_phase(backbone: str, fold: int, seed: int) -> None:
    context = common.unit_dir(backbone, fold, seed)
    complete = context / "OUTCOME_COMPLETE.json"
    if complete.is_file() and common.read_json(complete).get("pass") is True:
        print(f"[outcome cached] {backbone} fold={fold} seed={seed}", flush=True)
        return
    global_path = common.RUNTIME / "GLOBAL_SOURCE_FREEZE.json"
    if not global_path.is_file() or common.read_json(global_path).get("pass") is not True:
        raise RuntimeError("outer outcome evaluation blocked: global source freeze absent")
    global_marker = common.read_json(global_path)
    run_id = f"{backbone}|fold-{fold}|seed-{seed}"
    if run_id not in global_marker["runs"]:
        raise RuntimeError("run absent from global source freeze manifest")
    source_complete = common.read_json(context / "SOURCE_COMPLETE.json")
    for name, expected in source_complete["source_artifacts"].items():
        if file_sha256(context / name) != expected:
            raise RuntimeError(f"source artifact changed after freeze: {name}")
    roles = common.frozen_fold(fold)
    scoped = common.materialize_subject_scope(common.load_data(label_subjects=roles["outcome"]), roles["outcome"])
    outcome_idx = role_indices(scoped.metadata, roles, "outcome", (2,))
    device = torch.device("cuda")
    raw = torch.as_tensor(scoped.x, dtype=torch.float32, device=device)
    normalizer = np.load(context / "normalizer.npz", allow_pickle=False)
    mean = torch.as_tensor(normalizer["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalizer["std"], dtype=torch.float32, device=device)
    unit_protocol = common.read_json(context / "UNIT_PROTOCOL.json")
    model = checkpoint_model(backbone, int(unit_protocol["initialization_seed"]), context / "checkpoint.pt", device)
    frozen = np.load(context / "DIRECTIONS_FROZEN.npz", allow_pickle=False)
    center = frozen["center"]
    basis = frozen["directions"]
    outcome = common.evaluate_model(model, raw, scoped.metadata, outcome_idx, mean, std)
    rows = []
    subject_rows = []
    for direction_id, direction in enumerate(basis.T):
        erased_logits = logits_from_features(model, common.erase_direction(outcome["features"], center, direction))
        utilities = common.per_subject_utility(outcome["labels"], outcome["logits"], erased_logits, outcome["subjects"])
        summary = common.utility_summary(
            utilities,
            "U_future",
            common.stable_seed("future-subject-bootstrap", backbone, fold, seed, direction_id),
        )
        rows.append(
            {
                "backbone": backbone,
                "fold": fold,
                "seed": seed,
                "run_id": run_id,
                "direction_id": direction_id,
                **summary,
                "future_subject_count": len(utilities),
                "direction_SHA": common.array_sha256(direction),
                "global_source_freeze_SHA": file_sha256(global_path),
            }
        )
        utilities.insert(0, "direction_id", direction_id)
        utilities.insert(0, "seed", seed)
        utilities.insert(0, "fold", fold)
        utilities.insert(0, "backbone", backbone)
        subject_rows.append(utilities)
    frame = pd.DataFrame(rows)
    common.write_csv(context / "future_direction_cells.csv", frame)
    common.write_csv(context / "future_subject_utility.csv", pd.concat(subject_rows, ignore_index=True))
    common.write_json(
        complete,
        {
            "pass": True,
            "backbone": backbone,
            "fold": fold,
            "seed": seed,
            "direction_count": len(frame),
            "global_source_freeze_sha256": file_sha256(global_path),
            "same_frozen_direction_sha": True,
            "outer_labels_first_authorized_after_global_source_freeze": True,
            "completed_at_unix": time.time(),
        },
    )
    print(f"[outcome complete] {run_id}", flush=True)
    del model, raw, scoped, outcome
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("source", "outcome"), required=True)
    parser.add_argument("--backbone", choices=common.BACKBONES, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, choices=range(3), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("server GPU is required")
    common.ensure_dirs()
    if args.phase == "source":
        source_phase(args.backbone, args.fold, args.seed)
    else:
        outcome_phase(args.backbone, args.fold, args.seed)


if __name__ == "__main__":
    main()
