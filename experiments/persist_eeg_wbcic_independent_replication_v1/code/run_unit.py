"""Train one matched backbone/fold/seed stress-test unit on the server GPU."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
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


def load_checkpoint(backbone: str, seed: int, path: Path, device: torch.device) -> torch.nn.Module:
    model = common.build_model(backbone, seed)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device)


def train_candidates(
    backbone: str,
    fold: int,
    seed: int,
    raw: torch.Tensor,
    data: common.DevelopmentData,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    initialization_seed: int,
    loader_seed: int,
    bandwidths: list[float],
    context: Path,
) -> list[dict[str, Any]]:
    candidate_dir = context / "candidates"
    checkpoint_dir = context / "checkpoints"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for method, lam in common.configuration_grid():
        slug = common.config_slug(method, lam)
        meta_path = candidate_dir / f"{slug}.json"
        checkpoint_path = checkpoint_dir / f"{slug}.pt"
        if meta_path.is_file() and checkpoint_path.is_file():
            record = common.read_json(meta_path)
            if record.get("candidate_complete") is not True:
                raise RuntimeError(f"incomplete candidate metadata: {meta_path}")
            if record.get("checkpoint_sha256") != file_sha256(checkpoint_path):
                raise RuntimeError(f"candidate checkpoint hash mismatch: {checkpoint_path}")
        else:
            model, record = common.train_model(
                backbone=backbone,
                method=method,
                lam=lam,
                raw=raw,
                metadata=data.metadata,
                train_indices=train_indices,
                validation_indices=validation_indices,
                mean=mean,
                std=std,
                initialization_seed=initialization_seed,
                loader_seed=loader_seed,
                bandwidths=bandwidths,
            )
            torch.save({"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()}}, checkpoint_path)
            record.update(
                {
                    "fold": fold,
                    "seed": seed,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "candidate_complete": True,
                    "selection_basis": "validation/discovery S3 task BA only",
                    "outcome_evaluated": False,
                    "outcome_S3_labels_used_for_training_or_selection": False,
                }
            )
            common.write_json(meta_path, record)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        records.append(record)
        print(
            f"[candidate complete] backbone={backbone} fold={fold} seed={seed} method={method} "
            f"lambda={lam:g} valBA={float(record['best_validation_BA']):.6f} epoch={record['best_epoch']}",
            flush=True,
        )
    return records


def freeze_selection(
    backbone: str,
    fold: int,
    seed: int,
    records: list[dict[str, Any]],
    roles: dict[str, tuple[str, ...]],
    context: Path,
    bandwidths: list[float],
) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    if len(frame) != 10:
        raise RuntimeError(f"expected 10 fixed configurations, found {len(frame)}")
    if frame.initial_shared_state_sha256.nunique() != 1:
        raise RuntimeError("shared main-network initialization SHA differs across methods")
    if frame.epoch0_minibatch_order_sha256.nunique() != 1 or frame.loader_seed.nunique() != 1:
        raise RuntimeError("minibatch order is not matched across methods")
    selected: dict[str, Any] = {}
    for method, group in frame.groupby("method", sort=False):
        chosen = group.sort_values(
            ["best_validation_BA", "lambda", "best_epoch"],
            ascending=[False, True, True],
        ).iloc[0]
        selected[str(method)] = {
            "lambda": float(chosen["lambda"]),
            "best_validation_BA": float(chosen["best_validation_BA"]),
            "best_validation_NLL": float(chosen["best_validation_NLL"]),
            "best_epoch": int(chosen["best_epoch"]),
        }
    payload = {
        "schema": "STRESS_TEST_UNIT_SELECTION_FROZEN_V1",
        "backbone": backbone,
        "fold": fold,
        "seed": seed,
        "configuration_count": len(frame),
        "selected": selected,
        "selection_metric": "validation/discovery S3 mean subject balanced accuracy",
        "tie_break": "smaller lambda, then earlier best epoch",
        "model_fit_subjects": list(roles["model_fit"]),
        "validation_discovery_subjects": list(roles["validation_discovery"]),
        "outcome_subjects_not_loaded_by_selection": True,
        "outcome_S3_labels_used": False,
        "initial_shared_state_sha256": str(frame.initial_shared_state_sha256.iloc[0]),
        "epoch0_minibatch_order_sha256": str(frame.epoch0_minibatch_order_sha256.iloc[0]),
        "loader_seed": int(frame.loader_seed.iloc[0]),
        "mmd_bandwidths": list(map(float, bandwidths)),
        "frozen_at_unix": time.time(),
    }
    path = context / "LAMBDA_SELECTION_FROZEN.json"
    if path.is_file():
        previous = common.read_json(path)
        for key in ("backbone", "fold", "seed", "selected", "initial_shared_state_sha256", "epoch0_minibatch_order_sha256"):
            if previous.get(key) != payload.get(key):
                raise RuntimeError(f"existing frozen selection differs at {key}")
        return previous
    common.write_json(path, payload)
    return payload


def freeze_source_artifacts(
    backbone: str,
    fold: int,
    seed: int,
    initialization_seed: int,
    raw: torch.Tensor,
    data: common.DevelopmentData,
    mean: torch.Tensor,
    std: torch.Tensor,
    roles: dict[str, tuple[str, ...]],
    records: list[dict[str, Any]],
    selection: dict[str, Any],
    context: Path,
    normalizer_path: Path,
) -> dict[str, Any]:
    """Freeze every source-side artifact before the first outcome-S3 access."""
    source_indices = common.row_indices(data.metadata, roles["model_fit"], (0, 1))
    erm_slug = common.config_slug("ERM", 0.0)
    checkpoint_path = context / "checkpoints" / f"{erm_slug}.pt"
    model = load_checkpoint(backbone, initialization_seed, checkpoint_path, raw.device)
    source = common.evaluate_model(model, raw, data.metadata, source_indices, mean, std, batch_size=512)
    center, basis, direction_meta = common.persistent_directions(
        source["features"], source["subjects"], source["sessions"], count=8
    )
    source_dir = context / "source_freeze"
    source_dir.mkdir(parents=True, exist_ok=True)
    basis_path = source_dir / "erm_persistence_basis.npz"
    if basis_path.is_file():
        existing = np.load(basis_path, allow_pickle=False)
        if not np.array_equal(existing["center"], center) or not np.array_equal(existing["basis"], basis):
            raise RuntimeError("existing frozen ERM persistence basis differs")
    else:
        np.savez_compressed(
            basis_path,
            center=center.astype(np.float64),
            basis=basis.astype(np.float64),
            direction_meta_json=np.asarray(json.dumps(common.clean(direction_meta), sort_keys=True)),
            model_fit_subjects=np.asarray(roles["model_fit"]),
            sessions=np.asarray([0, 1], dtype=np.int8),
        )
    identity = common.identity_probe(source["features"], source["subjects"], source["sessions"])
    checkpoint_rows = []
    for record in records:
        checkpoint_rows.append(
            {
                "method": record["method"],
                "lambda": float(record["lambda"]),
                "checkpoint_sha256": record["checkpoint_sha256"],
                "best_epoch": int(record["best_epoch"]),
                "initial_shared_state_sha256": record["initial_shared_state_sha256"],
                "epoch0_minibatch_order_sha256": record["epoch0_minibatch_order_sha256"],
            }
        )
    payload = {
        "schema": "WBCIC_REPLICATION_RUN_SOURCE_FREEZE_V1",
        "pass": True,
        "backbone": backbone,
        "fold": fold,
        "seed": seed,
        "model_fit_subjects": list(roles["model_fit"]),
        "validation_discovery_subjects": list(roles["validation_discovery"]),
        "outcome_subjects_not_loaded": True,
        "outcome_S3_labels_used": False,
        "checkpoint_count": len(checkpoint_rows),
        "checkpoints": checkpoint_rows,
        "selected_lambdas": selection["selected"],
        "normalizer_policy": "identity_transform_after_frozen_uV_div20_clip",
        "normalizer_sha256": file_sha256(normalizer_path),
        "identity_probe_configuration": "symmetric_S1_to_S2_ridge_alpha_1",
        "erm_source_identity": identity,
        "persistence_basis_file": str(basis_path),
        "persistence_basis_file_sha256": file_sha256(basis_path),
        "persistence_basis_array_sha256": common.array_sha256(basis),
        "persistence_center_array_sha256": common.array_sha256(center),
        "direction_count": int(basis.shape[1]),
        "direction_construction": "symmetrized_S1_S2_subject_centroid_cross_covariance_eigendecomposition",
        "selection_file_sha256": file_sha256(context / "LAMBDA_SELECTION_FROZEN.json"),
        "sealed_WBCIC_outer_accessed": False,
        "OpenBMI_holdout_accessed": False,
        "frozen_at_unix": time.time(),
    }
    target = context / "SOURCE_FREEZE_COMPLETE.json"
    if target.is_file():
        previous = common.read_json(target)
        for key, value in common.clean(payload).items():
            if key != "frozen_at_unix" and previous.get(key) != value:
                raise RuntimeError(f"existing source freeze differs at {key}")
        payload = previous
    else:
        common.write_json(target, payload)
    del model, source
    return payload


def evaluate_candidates(
    backbone: str,
    fold: int,
    seed: int,
    initialization_seed: int,
    raw: torch.Tensor,
    data: common.DevelopmentData,
    mean: torch.Tensor,
    std: torch.Tensor,
    roles: dict[str, tuple[str, ...]],
    selection: dict[str, Any],
    context: Path,
) -> None:
    source_freeze_path = context / "SOURCE_FREEZE_COMPLETE.json"
    if not source_freeze_path.is_file() or common.read_json(source_freeze_path).get("pass") is not True:
        raise RuntimeError("outcome evaluation attempted before SOURCE_FREEZE_COMPLETE")
    selection_path = context / "LAMBDA_SELECTION_FROZEN.json"
    if not selection_path.is_file():
        raise RuntimeError("outcome evaluation attempted before lambda selection was frozen")
    selection_sha = file_sha256(selection_path)
    source_indices = common.row_indices(data.metadata, roles["model_fit"], (0, 1))
    outcome_indices = common.row_indices(data.metadata, roles["outcome"], (2,))
    if len(source_indices) == 0 or len(outcome_indices) == 0:
        raise RuntimeError("source/outcome row cardinality changed")
    if set(source_indices) & set(outcome_indices):
        raise RuntimeError("source/outcome row overlap")
    evaluation_dir = context / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    frozen_basis_path = context / "source_freeze" / "erm_persistence_basis.npz"
    frozen = np.load(frozen_basis_path, allow_pickle=False)
    frozen_center = frozen["center"]
    frozen_basis = frozen["basis"]
    frozen_meta = json.loads(str(frozen["direction_meta_json"].item()))
    device = raw.device
    for method, lam in common.configuration_grid():
        slug = common.config_slug(method, lam)
        target = evaluation_dir / slug
        complete_path = target / "EVALUATION_COMPLETE.json"
        if complete_path.is_file() and common.read_json(complete_path).get("pass") is True:
            print(f"[evaluation cached] {backbone} fold={fold} seed={seed} {slug}", flush=True)
            continue
        target.mkdir(parents=True, exist_ok=True)
        checkpoint_path = context / "checkpoints" / f"{slug}.pt"
        model = load_checkpoint(backbone, initialization_seed, checkpoint_path, device)
        source = common.evaluate_model(model, raw, data.metadata, source_indices, mean, std, batch_size=512)
        # This is the first point in the unit at which outcome labels are used.
        outcome = common.evaluate_model(model, raw, data.metadata, outcome_indices, mean, std, batch_size=512)
        primary_mask = np.isin(source["subjects"].astype(str), list(roles["model_fit"]))
        primary_identity = common.identity_probe(
            source["features"][primary_mask], source["subjects"][primary_mask], source["sessions"][primary_mask]
        )
        sensitivity_identity = common.identity_probe(source["features"], source["subjects"], source["sessions"])
        identity_row = {
            "backbone": backbone,
            "method": method,
            "lambda": float(lam),
            "fold": fold,
            "seed": seed,
            **primary_identity,
            **{f"all_source_{key}": value for key, value in sensitivity_identity.items()},
            "primary_scope": "model_fit_domains",
            "probe_rule": "exact_Exp3_ridge_alpha_1_symmetric_cross_session",
        }
        common.write_csv(target / "identity.csv", pd.DataFrame([identity_row]))
        performance = common.per_subject_performance(outcome["labels"], outcome["logits"], outcome["subjects"])
        performance.insert(0, "backbone", backbone)
        performance.insert(1, "method", method)
        performance.insert(2, "lambda", float(lam))
        performance.insert(3, "fold", fold)
        performance.insert(4, "seed", seed)
        performance["session"] = "S3"
        performance["selected_by_source_validation"] = bool(float(selection["selected"][method]["lambda"]) == float(lam))
        common.write_csv(target / "performance.csv", performance)
        directions = None
        if method == "ERM":
            directions = common.direction_audit(
                model=model,
                source=source,
                outcome=outcome,
                primary_identity_subjects=roles["model_fit"],
                backbone=backbone,
                method=method,
                lam=lam,
                fold=fold,
                seed=seed,
                frozen_center=frozen_center,
                frozen_basis=frozen_basis,
                frozen_meta=frozen_meta,
            )
            common.write_csv(target / "directions.csv", directions)
        np.savez_compressed(
            target / "embeddings.npz",
            source_features=source["features"],
            source_logits=source["logits"],
            source_labels=source["labels"],
            source_subjects=source["subjects"],
            source_sessions=source["sessions"],
            source_indices=source["indices"],
            outcome_features=outcome["features"],
            outcome_logits=outcome["logits"],
            outcome_labels=outcome["labels"],
            outcome_subjects=outcome["subjects"],
            outcome_sessions=outcome["sessions"],
            outcome_indices=outcome["indices"],
        )
        payload = {
            "pass": True,
            "backbone": backbone,
            "method": method,
            "lambda": float(lam),
            "fold": fold,
            "seed": seed,
            "selection_file_sha256": selection_sha,
            "selection_frozen_before_outcome_evaluation": True,
            "source_rows": len(source_indices),
            "outcome_rows": len(outcome_indices),
            "outcome_subjects": list(roles["outcome"]),
            "mean_outcome_BA": float(performance.BA.mean()),
            "identity_symmetric": float(primary_identity["identity_symmetric"]),
            "direction_rows": 0 if directions is None else len(directions),
            "restricted_data_accessed": False,
            "sealed_WBCIC_outer_accessed": False,
            "evaluated_at_unix": time.time(),
        }
        common.write_json(complete_path, payload)
        print(
            f"[evaluation complete] {backbone} fold={fold} seed={seed} {slug} "
            f"I={payload['identity_symmetric']:.5f} BA={payload['mean_outcome_BA']:.5f}",
            flush=True,
        )
        del model, source, outcome, directions, performance
        gc.collect()
        torch.cuda.empty_cache()


def run(backbone: str, fold: int, seed: int) -> None:
    if backbone not in common.BACKBONES or fold not in range(5) or seed not in range(3):
        raise ValueError("invalid backbone/fold/seed")
    if not torch.cuda.is_available():
        raise RuntimeError("full stress-test training requires the server GPU")
    common.ensure_dirs()
    context = common.unit_dir(backbone, fold, seed)
    context.mkdir(parents=True, exist_ok=True)
    unit_complete = context / "UNIT_COMPLETE.json"
    if unit_complete.is_file() and common.read_json(unit_complete).get("pass") is True:
        print(f"[unit cached] backbone={backbone} fold={fold} seed={seed}", flush=True)
        return
    data = common.load_data()
    roles = common.frozen_fold(fold)
    train_indices = common.row_indices(data.metadata, roles["model_fit"], (0, 1))
    validation_indices = common.row_indices(data.metadata, roles["validation_discovery"], (2,))
    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise RuntimeError("model-fit/validation-discovery row cardinality failure")
    if set(train_indices) & set(validation_indices):
        raise RuntimeError("model-fit/validation-discovery overlap")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    raw = torch.as_tensor(np.asarray(data.x), dtype=torch.float32).to(device)
    mean, std = common.compute_normalizer(raw, train_indices)
    normalizer_path = context / "normalizer.npz"
    if normalizer_path.is_file():
        existing_normalizer = np.load(normalizer_path, allow_pickle=False)
        if (
            not np.array_equal(existing_normalizer["mean"], mean.cpu().numpy())
            or not np.array_equal(existing_normalizer["std"], std.cpu().numpy())
            or list(existing_normalizer["subjects"].astype(str)) != list(roles["model_fit"])
            or list(existing_normalizer["sessions"].astype(int)) != [0, 1]
        ):
            raise RuntimeError("existing source normalizer differs")
    else:
        np.savez_compressed(
            normalizer_path,
            mean=mean.cpu().numpy(),
            std=std.cpu().numpy(),
            subjects=np.asarray(roles["model_fit"]),
            sessions=np.asarray([0, 1]),
        )
    initialization_seed = common.stable_seed("stress-main-init", backbone, fold, seed)
    loader_seed = common.stable_seed("stress-minibatch", backbone, fold, seed)
    bandwidths = common.determine_mmd_bandwidths(
        backbone, initialization_seed, raw, train_indices, mean, std
    )
    unit_protocol = {
        "backbone": backbone,
        "fold": fold,
        "seed": seed,
        "roles": {key: list(value) for key, value in roles.items()},
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "normalizer": str(normalizer_path),
        "initialization_seed": initialization_seed,
        "loader_seed": loader_seed,
        "mmd_bandwidths": bandwidths,
        "data_cache": str(data.cache_root),
        "data_scope": "authorized_41_subject_WBCIC_development_cache_only",
        "outcome_labels_used": False,
    }
    common.write_json(context / "UNIT_PROTOCOL.json", unit_protocol)
    records = train_candidates(
        backbone,
        fold,
        seed,
        raw,
        data,
        train_indices,
        validation_indices,
        mean,
        std,
        initialization_seed,
        loader_seed,
        bandwidths,
        context,
    )
    selection = freeze_selection(backbone, fold, seed, records, roles, context, bandwidths)
    freeze_source_artifacts(
        backbone,
        fold,
        seed,
        initialization_seed,
        raw,
        data,
        mean,
        std,
        roles,
        records,
        selection,
        context,
        normalizer_path,
    )
    evaluate_candidates(
        backbone,
        fold,
        seed,
        initialization_seed,
        raw,
        data,
        mean,
        std,
        roles,
        selection,
        context,
    )
    expected = 10
    observed = sum(
        (context / "evaluation" / common.config_slug(method, lam) / "EVALUATION_COMPLETE.json").is_file()
        for method, lam in common.configuration_grid()
    )
    if observed != expected:
        raise RuntimeError(f"unit evaluation incomplete: {observed}/{expected}")
    common.write_json(
        unit_complete,
        {
            "pass": True,
            "backbone": backbone,
            "fold": fold,
            "seed": seed,
            "configuration_count": observed,
            "shared_initial_state_sha256": records[0]["initial_shared_state_sha256"],
            "selection_frozen_before_outcome_evaluation": True,
            "restricted_data_accessed": False,
            "completed_at_unix": time.time(),
        },
    )
    print(f"[unit complete] backbone={backbone} fold={fold} seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=common.BACKBONES, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, choices=range(3), required=True)
    args = parser.parse_args()
    run(args.backbone, args.fold, args.seed)


if __name__ == "__main__":
    main()
