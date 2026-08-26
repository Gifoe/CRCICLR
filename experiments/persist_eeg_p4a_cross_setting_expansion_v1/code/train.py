from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

import common


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def outcome_subject_frame(setting: str, fold: int, seed: int, evaluated: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prediction = evaluated["logits"].argmax(1)
    for subject in common.subject_sort(np.unique(evaluated["subjects"].astype(str))):
        mask = evaluated["subjects"].astype(str) == subject
        rows.append(
            {
                "setting_id": setting,
                "fold": fold,
                "seed": seed,
                "subject_id": subject,
                "BA": float(balanced_accuracy_score(evaluated["labels"][mask], prediction[mask])),
                "macro_f1": float(f1_score(evaluated["labels"][mask], prediction[mask], average="macro", zero_division=0)),
                "rows": int(mask.sum()),
                "scope": "ERM_OUTCOME_COMPETENCE_ONLY",
            }
        )
    return pd.DataFrame(rows)


def freeze_guard() -> dict[str, Any]:
    path = common.EXP / "PROTOCOL_FREEZE_COMMIT.json"
    if not path.is_file():
        raise RuntimeError("ERM outcome competence is blocked until the protocol-freeze commit is recorded")
    payload = common.read_json(path)
    if payload.get("pass") is not True or payload.get("recorded_before_new_setting_outcome_access") is not True:
        raise RuntimeError("invalid protocol-freeze record")
    if payload.get("protocol_sha256") != common.file_sha256(common.PROTOCOL_PATH):
        raise RuntimeError("frozen protocol hash changed")
    return payload


def run_configuration(
    setting: str,
    fold: int,
    seed: int,
    method: str,
    lam: float,
    bundle: common.DataBundle,
    raw: torch.Tensor,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    outcome_indices: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    initialization_seed: int,
    loader_seed: int,
    bandwidths: list[float],
    scope_hashes: dict[str, str],
) -> None:
    unit = common.run_dir(setting, fold, seed)
    slug = common.config_slug(method, lam)
    target = unit / "source_freeze" / slug
    complete_path = target / "SOURCE_COMPLETE.json"
    if complete_path.is_file() and common.read_json(complete_path).get("pass") is True:
        print(f"[cached] {setting} fold={fold} seed={seed} {slug}", flush=True)
        return

    model, record = common.train_model(
        setting,
        method,
        lam,
        raw,
        bundle.metadata,
        train_indices,
        validation_indices,
        mean,
        std,
        initialization_seed,
        loader_seed,
        bandwidths,
    )
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = unit / "checkpoints" / f"{slug}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "setting_id": setting, "method": method, "lambda": float(lam)}, checkpoint)
    checkpoint_sha = common.file_sha256(checkpoint)
    record.update({"fold": fold, "seed": seed, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha})
    common.write_json(unit / "candidates" / f"{slug}.json", record)

    source = common.evaluate_model(model, raw, bundle.metadata, train_indices, mean, std, batch_size=512)
    validation = common.evaluate_model(model, raw, bundle.metadata, validation_indices, mean, std, batch_size=512)
    source_identity = common.identity_probe(source["features"], source["subjects"], source["sessions"])
    validation_metrics = common.mean_subject_metrics(validation["labels"], validation["logits"], validation["subjects"])
    validation_ce = float(common.numpy_cross_entropy(validation["logits"], validation["labels"]).mean())
    save_npz(
        target / "embeddings.npz",
        source_features=source["features"],
        source_logits=source["logits"],
        source_labels=source["labels"],
        source_subjects=source["subjects"],
        source_sessions=source["sessions"],
        source_indices=source["indices"],
        validation_features=validation["features"],
        validation_logits=validation["logits"],
        validation_labels=validation["labels"],
        validation_subjects=validation["subjects"],
        validation_sessions=validation["sessions"],
        validation_indices=validation["indices"],
    )
    summary = {
        "pass": True,
        "setting_id": setting,
        "fold": fold,
        "seed": seed,
        "method": method,
        "lambda": float(lam),
        "source_identity": source_identity,
        "source_validation_BA": validation_metrics["BA"],
        "source_validation_F1": validation_metrics["macro_f1"],
        "source_validation_CE": validation_ce,
        "checkpoint_sha256": checkpoint_sha,
        "normalizer_sha256": common.file_sha256(unit / "normalizer.npz"),
        "source_scope_hash": scope_hashes["source"],
        "validation_scope_hash": scope_hashes["validation"],
        "training_epoch": record["best_epoch"],
        "selection_metric": "source_validation_mean_subject_BA_then_NLL",
        "outcome_status": "P4B_DIRECTION_UTILITY_SEALED",
        "invariance_outcome_accessed": False,
        "direction_future_utility_accessed": False,
    }

    if method == "ERM":
        evidence, controls, artifact = common.direction_rows(
            setting,
            fold,
            seed,
            model,
            source,
            validation,
            checkpoint_sha,
            summary["normalizer_sha256"],
            scope_hashes,
        )
        common.write_csv(target / "source_evidence.csv", evidence)
        common.write_csv(target / "matched_controls.csv", controls)
        save_npz(
            target / "persistence_basis.npz",
            center=artifact["center"].astype(np.float64),
            basis=artifact["basis"].astype(np.float64),
        )
        freeze = freeze_guard()
        outcome = common.evaluate_model(model, raw, bundle.metadata, outcome_indices, mean, std, batch_size=512)
        competence = outcome_subject_frame(setting, fold, seed, outcome)
        common.write_csv(target / "outcome_competence.csv", competence)
        summary.update(
            {
                "outcome_competence_BA": float(competence.BA.mean()),
                "outcome_competence_F1": float(competence.macro_f1.mean()),
                "outcome_competence_subjects": len(competence),
                "outcome_access_scope": "ERM_COMPETENCE_ONLY",
                "protocol_freeze_commit": freeze["protocol_freeze_commit"],
            }
        )
    common.write_json(complete_path, summary)
    print(f"[complete] {setting} fold={fold} seed={seed} {slug}", flush=True)
    del model, source, validation
    torch.cuda.empty_cache()


def run_setting(setting: str, folds: list[int], seeds: list[int], tier: str) -> None:
    if setting not in {"S4", "S5", "S6"}:
        raise ValueError("train.py only trains new settings S4/S5/S6")
    bundle = common.load_data(setting)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("P4A new setting execution requires the server GPU")
    raw = torch.from_numpy(np.asarray(bundle.x)).to(device=device, non_blocking=False)
    print(f"[{setting}] loaded {tuple(raw.shape)} {raw.dtype} on {torch.cuda.get_device_name(0)}", flush=True)
    if tier == "erm":
        grid = (("ERM", 0.0),)
    elif tier == "grid":
        grid = tuple(item for item in common.METHOD_GRID if item[0] != "ERM")
    else:
        grid = common.METHOD_GRID
    for fold in folds:
        roles = common.roles_for(setting, fold)
        train_indices = common.row_indices(bundle.metadata, roles["model_fit"], bundle.source_sessions)
        validation_indices = common.row_indices(bundle.metadata, roles["validation"], bundle.source_sessions)
        outcome_indices = common.row_indices(bundle.metadata, roles["outcome"], (bundle.future_session,))
        if set(train_indices) & set(validation_indices) or set(train_indices) & set(outcome_indices) or set(validation_indices) & set(outcome_indices):
            raise RuntimeError(f"{setting} fold {fold} row overlap")
        mean, std = common.compute_normalizer(setting, raw, train_indices)
        scope_hashes = {
            "source": common.array_sha256(train_indices.astype(np.int64)),
            "validation": common.array_sha256(validation_indices.astype(np.int64)),
            "outcome": common.array_sha256(outcome_indices.astype(np.int64)),
        }
        for seed in seeds:
            unit = common.run_dir(setting, fold, seed)
            unit.mkdir(parents=True, exist_ok=True)
            normalizer = unit / "normalizer.npz"
            if not normalizer.is_file():
                save_npz(normalizer, mean=mean.detach().cpu().numpy(), std=std.detach().cpu().numpy())
            initialization_seed = common.stable_seed("P4A-init", setting, fold, seed)
            loader_seed = common.stable_seed("P4A-loader", setting, fold, seed)
            bandwidths = common.determine_mmd_bandwidths(setting, initialization_seed, raw, train_indices, mean, std)
            common.write_json(
                unit / "UNIT_PROTOCOL.json",
                {
                    "setting_id": setting,
                    "fold": fold,
                    "seed": seed,
                    "roles": roles,
                    "train_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                    "outcome_rows": len(outcome_indices),
                    "source_sessions": bundle.source_sessions,
                    "future_session": bundle.future_session,
                    "initialization_seed": initialization_seed,
                    "loader_seed": loader_seed,
                    "mmd_bandwidths": bandwidths,
                    "normalizer_sha256": common.file_sha256(normalizer),
                    "scope_hashes": scope_hashes,
                    "outcome_labels_used_for_training_or_selection": False,
                },
            )
            for method, lam in grid:
                run_configuration(
                    setting,
                    fold,
                    seed,
                    method,
                    lam,
                    bundle,
                    raw,
                    train_indices,
                    validation_indices,
                    outcome_indices,
                    mean,
                    std,
                    initialization_seed,
                    loader_seed,
                    bandwidths,
                    scope_hashes,
                )
    del raw
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", nargs="+", choices=["S4", "S5", "S6"], default=["S4", "S5", "S6"])
    parser.add_argument("--folds", nargs="+", type=int, default=list(common.FOLDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(common.SEEDS))
    parser.add_argument("--tier", choices=["erm", "grid", "all"], default="all")
    args = parser.parse_args()
    common.protocol()
    if args.tier in {"erm", "all"}:
        freeze_guard()
    for setting in args.settings:
        run_setting(setting, args.folds, args.seeds, args.tier)
    print("P4A_REQUESTED_NEW_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
