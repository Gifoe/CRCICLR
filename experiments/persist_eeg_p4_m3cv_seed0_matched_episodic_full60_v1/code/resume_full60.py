"""Resume the archived M3CV seed-0 runs to the fixed 60-epoch horizon.

This runner deliberately imports the archived, audited implementation rather
than copying either model or sampler.  The sole behavioral change is that the
previous patience-based termination is disabled after the exact saved states
are restored.  It writes only to the separate ``full60_v1`` runtime.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SOURCE_ROOT = Path(os.environ.get("M3CV_SOURCE_ROOT", "/root/p4_m3cv_seed0_matched_episodic_v1"))
OUTPUT_ROOT = Path(os.environ.get("M3CV_FULL60_ROOT", "/root/p4_m3cv_seed0_matched_episodic_full60_v1"))
SOURCE_COMMIT = "e9c689f7f4147b2419b9cbf122c9c0a9a6a34a54"
EXPECTED_STEPS = (81, 81, 81, 82, 82)
MODELS = ("EEGNet", "SIRE-EEG")


def load_archived_module() -> Any:
    path = SOURCE_ROOT / "code" / "run_m3cv_seed0_matched_episodic.py"
    if not path.is_file():
        raise RuntimeError(f"archived runner missing: {path}")
    spec = importlib.util.spec_from_file_location("m3cv_archived_seed0", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import archived runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_normalizer(fold_id: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Path]:
    path = SOURCE_ROOT / "normalizers" / f"fold{fold_id}_s1_normalizer.npz"
    if not path.is_file():
        raise RuntimeError(f"normalizer missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        mean = payload["mean"].astype(np.float32, copy=False)
        std = payload["std"].astype(np.float32, copy=False)
        metadata = json.loads(str(payload["metadata"].item()))
    semantic = hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest()
    if metadata.get("mean_std_sha256") != semantic:
        raise RuntimeError(f"normalizer content digest mismatch: {path}")
    if mean.shape != (64,) or std.shape != (64,) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise RuntimeError(f"normalizer arrays invalid: {path}")
    return mean, std, metadata, path


def load_manifest(module: Any, fold_id: int) -> tuple[list[list[dict[str, Any]]], Path, str]:
    path = SOURCE_ROOT / "episode_manifests" / f"fold{fold_id}.json"
    if not path.is_file():
        raise RuntimeError(f"manifest missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("epochs")
    if not isinstance(episodes, list) or len(episodes) != 60:
        raise RuntimeError(f"manifest epoch coverage invalid: {path}")
    if any(len(epoch) != EXPECTED_STEPS[fold_id] for epoch in episodes):
        raise RuntimeError(f"manifest steps/epoch drift: {path}")
    return episodes, path, module.sha256(path)


def source_records() -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    path = SOURCE_ROOT / "seed0_training_records.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    records = {(row["model"], int(row["fold"])): row for row in data["records"]}
    if set(records) != {(name, fold) for name in MODELS for fold in range(5)}:
        raise RuntimeError("archived training record coverage invalid")
    return records, data


def audit_checkpoint(module: Any, source: Path, target: Path, name: str, fold: dict[str, Any], manifest_sha: str, normalizer: dict[str, Any], models: dict[str, Any], device: torch.device) -> dict[str, Any]:
    if not source.is_file():
        raise RuntimeError(f"source latest checkpoint missing: {source}")
    saved = torch.load(source, map_location="cpu", weights_only=False)
    required = {"epoch", "history", "best_val_BA", "best_epoch", "best_state", "current_state", "optimizer", "scaler", "rng", "manifest_sha256", "init_sha256", "normalizer_sha256"}
    if required.difference(saved):
        raise RuntimeError(f"incomplete resume state: {source}")
    epoch = int(saved["epoch"])
    history = saved["history"]
    if not (1 <= epoch < 60) or len(history) != epoch or [int(row.get("epoch", -1)) for row in history] != list(range(1, epoch + 1)):
        raise RuntimeError(f"non-continuous history: {source}")
    if saved["best_epoch"] is None or saved["best_state"] is None or not (10 <= int(saved["best_epoch"]) <= epoch):
        raise RuntimeError(f"unrecoverable selected-best state: {source}")
    if not isinstance(saved["optimizer"], dict) or not saved["optimizer"].get("state"):
        raise RuntimeError(f"optimizer state absent: {source}")
    if not isinstance(saved["scaler"], dict) or not isinstance(saved["rng"], dict):
        raise RuntimeError(f"scaler/RNG state absent: {source}")
    module.set_seed(module.SEED)
    instance = module.construct(name, models["EEGNet"]["class_ref"], models["SIRE-EEG"]["class_ref"]).to(device)
    init_sha = module.state_hash(copy.deepcopy(instance.state_dict()))
    del instance
    if saved["init_sha256"] != init_sha or saved["manifest_sha256"] != manifest_sha or saved["normalizer_sha256"] != normalizer["mean_std_sha256"]:
        raise RuntimeError(f"resume provenance mismatch: {source}")
    if target.is_file() and file_sha256(source) != file_sha256(target):
        raise RuntimeError(f"copied resume checkpoint drift: {target}")
    return {
        "model": name,
        "fold": int(fold["fold_id"]),
        "previous_latest_epoch": epoch,
        "resume_start_epoch": epoch + 1,
        "previous_best_epoch": int(saved["best_epoch"]),
        "init_sha256": init_sha,
        "manifest_sha256": manifest_sha,
        "normalizer_semantic_sha256": normalizer["mean_std_sha256"],
        "normalizer_file_sha256": file_sha256(SOURCE_ROOT / "normalizers" / f"fold{fold['fold_id']}_s1_normalizer.npz"),
        "optimizer_restored": True,
        "scaler_restored": True,
        "rng_restored": True,
        "source_latest_sha256": file_sha256(source),
    }


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def restore_rng_after_map_location(value: dict[str, Any]) -> None:
    """Restore the exact saved bytes after ``map_location=cuda``.

    The archived checkpoints correctly save CPU RNG byte tensors.  PyTorch
    maps those tensors to CUDA with the rest of the checkpoint, whereas
    ``torch.set_rng_state`` requires the original CPU tensor.  Moving only the
    state containers back to CPU preserves their bytes and has no sampling
    effect beyond making the exact archived state restorable.
    """
    import random

    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].detach().cpu())
    if "cuda" in value and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.detach().cpu() for state in value["cuda"]])


def run() -> None:
    module = load_archived_module()
    if module.EPOCHS != 60 or module.MIN_EPOCH != 10:
        raise RuntimeError("archived fixed-horizon constants drift")
    old_records, old_data = source_records()
    if old_data.get("eligible_subjects") is None or len(old_data["eligible_subjects"]) != 93:
        raise RuntimeError("archived cohort is not the fixed 93-subject cohort")
    if (OUTPUT_ROOT / "runtime").exists() and not (OUTPUT_ROOT / "runtime" / "checkpoints").is_dir():
        raise RuntimeError(f"unsafe pre-existing runtime directory: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_checkpoints = SOURCE_ROOT / "runtime" / "checkpoints"
    target_checkpoints = OUTPUT_ROOT / "runtime" / "checkpoints"
    if not target_checkpoints.exists():
        shutil.copytree(source_checkpoints, target_checkpoints, copy_function=shutil.copy2)

    stage1, eegnet, sire, recipe = module.load_authoritative()
    models = module.audit_models(eegnet, sire)
    models["EEGNet"]["class_ref"] = eegnet
    models["SIRE-EEG"]["class_ref"] = sire
    eligible = old_data["eligible_subjects"]
    bundle = module.load_bundle(stage1, eligible)
    folds, split_rows = module.splits(eligible)
    if [len(fold["outer_test_subjects"]) for fold in folds] != [19, 19, 19, 18, 18]:
        raise RuntimeError("outer fold sizes drift")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizers: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    manifests: dict[int, list[list[dict[str, Any]]]] = {}
    manifest_paths: dict[int, Path] = {}
    resume_rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        mean, std, norm, _ = load_normalizer(fold_id)
        manifest, manifest_path, manifest_sha = load_manifest(module, fold_id)
        normalizers[fold_id] = (mean, std, norm)
        manifests[fold_id] = manifest
        manifest_paths[fold_id] = manifest_path
        for name in MODELS:
            source_latest = source_checkpoints / name / f"fold{fold_id}_seed0" / "checkpoint_latest.pt"
            target_latest = target_checkpoints / name / f"fold{fold_id}_seed0" / "checkpoint_latest.pt"
            resume_rows.append(audit_checkpoint(module, source_latest, target_latest, name, fold, manifest_sha, norm, models, device))

    # The only behavioral correction: no patience termination.  All remaining
    # authoritative manifest epochs are consumed through epoch 60.
    module.ROOT = OUTPUT_ROOT
    module.EARLY_STOP_PATIENCE = 10_000
    module.restore_rng = restore_rng_after_map_location
    records: list[dict[str, Any]] = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        mean, std, norm = normalizers[fold_id]
        cache = module.BatchCache(bundle, mean, std, device)
        for name in MODELS:
            record = module.train_one(name, fold, manifests[fold_id], manifest_paths[fold_id], bundle, cache, eegnet, sire, models, norm)
            if record["epochs_completed"] != 60 or record["early_stopped"]:
                raise RuntimeError(f"fixed-60 completion failure: {name}/fold{fold_id}")
            record["early_stop_patience"] = None
            record["training_horizon"] = "fixed_60_no_early_stop"
            records.append(record)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    completed = {(item["model"], item["fold"]): item for item in records}
    for row in resume_rows:
        record = completed[(row["model"], row["fold"])]
        row["final_epoch"] = int(record["epochs_completed"])
        row["full60_selected_epoch"] = int(record["selected_epoch"])
        row["full60_inner_val_S2_BA"] = float(record["best_inner_val_S2_subject_BA"])
        row["selected_checkpoint_changed"] = bool(record["selected_epoch"] != row["previous_best_epoch"])

    # Outer data is reached only after every selected checkpoint is frozen.
    subject_rows: list[dict[str, Any]] = []
    outer_fold: dict[tuple[str, int], float] = {}
    for fold in folds:
        fold_id = int(fold["fold_id"])
        mean, std, norm = normalizers[fold_id]
        cache = module.BatchCache(bundle, mean, std, device)
        for name in MODELS:
            record = completed[(name, fold_id)]
            model = module.construct(name, eegnet, sire).to(device)
            model.load_state_dict(torch.load(record["checkpoint_path"], map_location=device, weights_only=False))
            for _, session_id, session_label in module.SESSIONS:
                values = module.eval_session(model, bundle, cache, fold["outer_test_subjects"], session_id)
                if session_label == "S2":
                    outer_fold[(name, fold_id)] = float(np.mean([value["BA"] for value in values.values()]))
                for subject, metric in values.items():
                    subject_rows.append({"model": name, "fold": fold_id, "seed": module.SEED, "checkpoint_sha256": record["checkpoint_sha256"], "selected_epoch": record["selected_epoch"], "normalizer_sha256": norm["mean_std_sha256"], "subject": subject, "session": session_label, "session_id": session_id, **metric})
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary: dict[str, dict[str, float]] = {}
    per_subject: dict[str, dict[str, dict[str, float]]] = {}
    for name in MODELS:
        summary[name], per_subject[name] = module.subject_summary(subject_rows, name, eligible)
    contrast = {metric: module.bootstrap(np.asarray([per_subject["SIRE-EEG"][subject][metric] - per_subject["EEGNet"][subject][metric] for subject in eligible])) for metric in ("future S2 BA", "WS-BA")}

    write_rows(OUTPUT_ROOT / "resume_integrity.csv", resume_rows, list(resume_rows[0]))
    write_rows(OUTPUT_ROOT / "seed0_subject_metrics_full60.csv", subject_rows, ["model", "fold", "seed", "checkpoint_sha256", "selected_epoch", "normalizer_sha256", "subject", "session", "session_id", "BA", "Macro_F1", "accuracy", "trials"])
    fold_rows = [{"seed": module.SEED, "fold": row["fold"], "model": row["model"], "outer_subjects": len(folds[row["fold"]]["outer_test_subjects"]), "previous_stop_epoch": next(item["previous_latest_epoch"] for item in resume_rows if item["model"] == row["model"] and item["fold"] == row["fold"]), "selected_epoch": row["selected_epoch"], "inner_val_S2_BA": row["best_inner_val_S2_subject_BA"], "outer_S2_BA": outer_fold[(row["model"], row["fold"])], "checkpoint_sha256": row["checkpoint_sha256"], "parameter_count": row["parameter_count"]} for row in records]
    write_rows(OUTPUT_ROOT / "seed0_fold_metrics_full60.csv", fold_rows, list(fold_rows[0]))
    summary_rows = [{"row_type": "model", "model": name, "metric": "", "mean": "", "ci95_low": "", "ci95_high": "", "improved": "", "tied": "", "harmed": "", "future_S2_BA": summary[name]["future S2 BA"], "future_S2_Macro_F1": summary[name]["future S2 Macro-F1"], "WS_BA": summary[name]["WS-BA"], "subjects": len(eligible), "bootstrap_resamples": ""} for name in MODELS]
    summary_rows.extend({"row_type": "paired_contrast", "model": "SIRE-EEG minus EEGNet", "metric": metric, "mean": item["mean"], "ci95_low": item["ci_low"], "ci95_high": item["ci_high"], "improved": item["improved"], "tied": item["tied"], "harmed": item["harmed"], "future_S2_BA": "", "future_S2_Macro_F1": "", "WS_BA": "", "subjects": item["subjects"], "bootstrap_resamples": item["resamples"]} for metric, item in contrast.items())
    write_rows(OUTPUT_ROOT / "seed0_summary_full60.csv", summary_rows, list(summary_rows[0]))
    module.write_json(OUTPUT_ROOT / "seed0_training_records_full60.json", {"seed": module.SEED, "source_archival_commit": SOURCE_COMMIT, "training_horizon": "fixed 60 epochs; patience termination disabled", "training_recipe": recipe, "records": records, "resume_integrity": resume_rows})

    correction = ["# Full-60 protocol correction", "", f"Source archival commit: `{SOURCE_COMMIT}`.", "", "The earlier seed-0 execution used a patience-8 termination rule. This follow-up restores the authoritative fixed 60-epoch training horizon while holding the archived cache, subject splits, episode manifests, normalizers, models, seeds, optimizer state, AMP scaler state and RNG state fixed.", "", "Only the archived `checkpoint_latest.pt` states were resumed; no model was reinitialized and no early-stopped run was restarted from epoch 1."]
    write_text(OUTPUT_ROOT / "FULL60_PROTOCOL_CORRECTION.md", correction)
    integrity = ["# Resume integrity audit", "", "Every source checkpoint was copied byte-for-byte to the separate full-60 runtime before resumption. The existing normalizer files were loaded, not refit. `normalizer_semantic_sha256` is the digest stored in the checkpoint; `normalizer_file_sha256` is the physical NPZ-file digest.", "", "| Model | Fold | Previous latest | Resume start | Final | Previous best | Full60 selected | Changed |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    integrity.extend(f"| {row['model']} | {row['fold']} | {row['previous_latest_epoch']} | {row['resume_start_epoch']} | {row['final_epoch']} | {row['previous_best_epoch']} | {row['full60_selected_epoch']} | {'yes' if row['selected_checkpoint_changed'] else 'no'} |" for row in resume_rows)
    write_text(OUTPUT_ROOT / "RESUME_INTEGRITY_AUDIT.md", integrity)
    comparison = ["# Early-stop versus fixed-60 descriptive audit", "", "This is descriptive only; the fixed 60-epoch continuation was prescribed before any new outer evaluation.", "", "| Model | Fold | Previous selected | Full60 selected | Previous inner S2 BA | Full60 inner S2 BA | Changed |", "|---|---:|---:|---:|---:|---:|---|"]
    for row in resume_rows:
        old = old_records[(row["model"], row["fold"])]
        comparison.append(f"| {row['model']} | {row['fold']} | {old['selected_epoch']} | {row['full60_selected_epoch']} | {old['best_inner_val_S2_subject_BA']:.6f} | {row['full60_inner_val_S2_BA']:.6f} | {'yes' if row['selected_checkpoint_changed'] else 'no'} |")
    write_text(OUTPUT_ROOT / "EARLYSTOP_VS_FULL60_AUDIT.md", comparison)
    report = ["# M3CV seed-0 full-60 result summary", "", "Dataset: M3CV / NEMAR `nm000166`.", "", "- Downloaded subjects: 95; eligible subjects: 93; exclusions: `sub-035`, `sub-080`.", "- Task: left- versus right-hand motor execution; S1=`ses-01`, S2=`ses-02`; input: 64 x 1000 at 250 Hz.", "- All ten archived runs were resumed from their stored state and completed at epoch 60. Checkpoint selection uses maximum inner-val S2 subject-equal BA over epochs 10..60, with the earliest exact tie retained.", "", "| Model | Future S2 BA | Future S2 Macro-F1 | WS-BA |", "|---|---:|---:|---:|"]
    report.extend(f"| {name} | {summary[name]['future S2 BA']:.4f} | {summary[name]['future S2 Macro-F1']:.4f} | {summary[name]['WS-BA']:.4f} |" for name in MODELS)
    report.extend(["", "| Contrast | Delta | 95% CI | Improved / tied / harmed |", "|---|---:|---:|---:|"])
    report.extend(f"| {metric} | {item['mean']:+.4f} | [{item['ci_low']:+.4f}, {item['ci_high']:+.4f}] | {item['improved']} / {item['tied']} / {item['harmed']} |" for metric, item in contrast.items())
    report.extend(["", "| Fold | Model | Previous stop | Final | Previous selected | Full60 selected | Inner-val S2 BA | Outer S2 BA |", "|---:|---|---:|---:|---:|---:|---:|---:|"])
    for row in sorted(fold_rows, key=lambda item: (item["fold"], item["model"])):
        prior = old_records[(row["model"], row["fold"])]
        report.append(f"| {row['fold']} | {row['model']} | {row['previous_stop_epoch']} | 60 | {prior['selected_epoch']} | {row['selected_epoch']} | {row['inner_val_S2_BA']:.4f} | {row['outer_S2_BA']:.4f} |")
    write_text(OUTPUT_ROOT / "SEED0_FULL60_RESULT_SUMMARY.md", report)
    print("M3CV_SEED0_FULL60_COMPLETE", flush=True)


if __name__ == "__main__":
    run()
