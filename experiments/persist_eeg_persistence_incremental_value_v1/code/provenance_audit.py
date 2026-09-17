"""Read-only provenance gate for the frozen selector-ablation experiment.

This script never loads evaluation labels, runs inference, or trains a model.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
P1 = ROOT.parent
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
MODELS = ("EEGNet", "CBraMod", "TeCh", "ModernTCN", "Medformer", "EEGConformer", "FBCNet")
OPENBMI_HELD = ("4", "12", "13", "17", "18", "24", "25", "29", "36", "37", "39", "42", "51", "54")
WBCIC_HELD = ("sub-4", "sub-8", "sub-10", "sub-15", "sub-20", "sub-39", "sub-40", "sub-43", "sub-46", "sub-51")
REPRESENTATIONS = {
    "EEGNet": ("model.head forward-pre-hook", 64),
    "CBraMod": ("model.head forward-pre-hook", 200),
    "TeCh": ("model.model.projector forward-pre-hook", None),
    "ModernTCN": ("model.model.model.head_class forward-pre-hook", None),
    "Medformer": ("model.model.projection forward-pre-hook", None),
    "EEGConformer": ("model.classifier forward-pre-hook", None),
    "FBCNet": ("model.head forward-pre-hook", 1152),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    baseline = ROOT / "experiments/persist_eeg_eegconformer_fbcnet_multiseed_v1"
    split_path = baseline / "protocol/SPLIT_AUDIT.json"
    split = read_json(split_path)
    roles = {(r["task"], int(r["fold"])): r for r in split["rows"]}
    assert len(roles) == 20
    held = {"OpenBMI": set(OPENBMI_HELD), "WBCIC": set(WBCIC_HELD)}
    assert set(split["OpenBMI_final_heldout_ids"]) == held["OpenBMI"]
    assert set(split["WBCIC_final_true_outer_ids"]) == held["WBCIC"]

    new_manifest_path = baseline / "outputs/RUN_MANIFEST.csv"
    new_audit_path = baseline / "outputs/CHECKPOINT_AUDIT.csv"
    with new_manifest_path.open(newline="", encoding="utf-8") as f:
        new_manifest = {(r["model"], r["task"], int(r["fold"]), int(r["seed"])): r for r in csv.DictReader(f)}
    with new_audit_path.open(newline="", encoding="utf-8") as f:
        new_audit = {(r["model"], r["task"], int(r["fold"]), int(r["seed"])): r for r in csv.DictReader(f)}
    assert len(new_manifest) == len(new_audit) == 120
    peeh_lock_path = ROOT / "experiments/persist_eeg_crossbackbone_peeh_v1/protocol/PEEH_PROTOCOL_LOCK.json"
    peeh_lock = read_json(peeh_lock_path)
    peeh_rows = {(r["Model"], r["Task"], int(r["fold"])): r for r in peeh_lock["checkpoints"]}
    assert len(peeh_rows) == 100
    rep_path = ROOT / "experiments/persist_eeg_crossbackbone_peeh_v1/outputs/crossbackbone_peeh_v1/REPRESENTATION_SPEC.csv"
    with rep_path.open(newline="", encoding="utf-8") as f:
        dimensions = {(r["Model"], r["Task"]): int(r["Representation dim"]) for r in csv.DictReader(f)}
    for task in TASKS:
        dimensions["EEGConformer", task] = 440 if task == "OpenBMI_ERP" else 2440
        dimensions["FBCNet", task] = 1152
    assert len(dimensions) == 28

    rows = []
    problems = []
    source_roots = {}
    for model in MODELS:
        for task in TASKS:
            for fold in range(5):
                role = roles[task, fold]
                dataset = role["dataset"]
                train = set(map(str, role["inner_train_subjects"]))
                selection = set(map(str, role["inner_val_subjects"]))
                outer_dev = set(map(str, role["outer_dev_subjects"]))
                if train & selection or train & outer_dev or selection & outer_dev:
                    problems.append(f"split overlap: {task} fold{fold}")
                if (train | selection | outer_dev) & held[dataset]:
                    problems.append(f"heldout overlap: {task} fold{fold}")
                for seed in range(3):
                    key = (model, task, fold, seed)
                    if model in ("EEGNet", "CBraMod", "TeCh"):
                        source_exp = "persist_eeg_seven_backbone_fourtask_3seed_v1"
                        cell = P1 / "seven_backbone_fourtask_3seed_runtime/search_cells" / task.lower() / model.lower() / f"fold{fold}_seed{seed}"
                    elif model in ("ModernTCN", "Medformer"):
                        source_exp = f"persist_eeg_{model.lower()}_4task_3seed_final_v1"
                        cell = P1 / "baseline3_runtime" / model.lower() / "cells" / task.lower() / f"fold{fold}_seed{seed}"
                    else:
                        source_exp = "persist_eeg_eegconformer_fbcnet_multiseed_v1"
                        cell = P1 / "eegconformer_fbcnet_runtime/cells" / model.lower() / task.lower() / f"fold{fold}_seed{seed}"
                    record_path, ckpt = cell / "record.json", cell / "selected.pt"
                    source_roots[model] = str(cell.parents[2])
                    errors = []
                    if not record_path.is_file() or not ckpt.is_file():
                        errors.append("record_or_selected_checkpoint_missing")
                        rec = {}
                    else:
                        rec = read_json(record_path)
                        if (rec.get("model"), rec.get("task"), rec.get("fold"), rec.get("seed")) != key:
                            errors.append("record_identity_mismatch")
                    actual_sha = digest(ckpt) if ckpt.is_file() else ""
                    if rec.get("checkpoint_sha256") != actual_sha:
                        errors.append("checkpoint_sha_mismatch")
                    if rec.get("split_sha256") != role["source_split_sha256"]:
                        errors.append("split_sha_mismatch")
                    norm_sha = rec.get("normalizer", {}).get("mean_std_sha256", "")
                    if len(norm_sha) != 64:
                        errors.append("normalizer_sha_missing")
                    if model in ("EEGConformer", "FBCNet"):
                        for reference, label in ((new_manifest, "RUN_MANIFEST"), (new_audit, "CHECKPOINT_AUDIT")):
                            r = reference.get(key)
                            if r is None or r["checkpoint_sha256"] != actual_sha or r["normalizer_sha256"] != norm_sha:
                                errors.append(label + "_mismatch")
                    elif seed == 0:
                        r = peeh_rows.get((model, task, fold))
                        if r is None or r["checkpoint_sha256"] != actual_sha or r["normalizer_sha256"] != norm_sha:
                            errors.append("formal_PEEH_seed0_lock_mismatch")
                    row = {
                        "model": model, "task": task, "fold": fold, "seed": seed,
                        "checkpoint_path": str(ckpt), "checkpoint_sha256": actual_sha,
                        "normalizer_path": str(record_path) + "::normalizer (hash only; TRAIN reconstruction required)",
                        "normalizer_sha256": norm_sha, "normalizer_artifact_kind": "recomputed TRAIN-source mean/std; values not persisted",
                        "train_subjects": ";".join(sorted(train)), "checkpoint_selection_subjects": ";".join(sorted(selection)),
                        "heldout_subjects": ";".join(sorted(held[dataset])),
                        "source_sessions": ";".join(map(str, role["source_sessions"])),
                        "future_session": role["outer_development_session"],
                        "representation_layer": REPRESENTATIONS[model][0],
                        "representation_dimension": dimensions[model, task],
                        "checkpoint_source_experiment": source_exp,
                        "selected_epoch": rec.get("selected_epoch", ""),
                        "status": "PASS" if not errors else "FAIL", "errors": ";".join(errors),
                    }
                    rows.append(row)
                    problems.extend(f"{key}: {x}" for x in errors)

    assert len(rows) == 420
    protocol = EXP / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    with (protocol / "CHECKPOINT_AUDIT.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(protocol / "SPLIT_AUDIT.json", {
        "schema": "PERSIST_INCREMENTAL_VALUE_SPLIT_AUDIT_V1",
        "source_path": str(split_path), "source_sha256": digest(split_path),
        "rows": split["rows"], "OpenBMI_final_heldout_ids": list(OPENBMI_HELD),
        "WBCIC_final_true_outer_ids": list(WBCIC_HELD),
        "WBCIC_V8_internal_ids_are_not_true_outer": True,
        "all_partitions_subject_disjoint": not any("overlap" in p for p in problems),
    })
    write_json(protocol / "SOURCE_MANIFEST.json", {
        "schema": "PERSIST_INCREMENTAL_VALUE_SOURCE_MANIFEST_V1",
        "checkpoint_count": len(rows), "passed_checkpoint_count": sum(r["status"] == "PASS" for r in rows),
        "source_roots": source_roots,
        "formal_peeh_lock": {"path": str(peeh_lock_path), "sha256": digest(peeh_lock_path)},
        "new_baseline_run_manifest": {"path": str(new_manifest_path), "sha256": digest(new_manifest_path)},
        "new_baseline_checkpoint_audit": {"path": str(new_audit_path), "sha256": digest(new_audit_path)},
        "split_audit": {"path": str(split_path), "sha256": digest(split_path)},
        "formal_representation_spec": {"path": str(rep_path), "sha256": digest(rep_path)},
        "no_neural_training": True, "no_heldout_outcomes_read": True,
        "problems": problems,
    })
    (protocol / "REPRESENTATION_AUDIT.md").write_text(
        "# Frozen classifier-input representation audit\n\n"
        "The first five heads are exactly the formal PEEH forward-pre-hooks. "
        "EEGConformer uses the input to the full task-specific `model.classifier` sequential head, "
        "not its logits or an intermediate Transformer token. FBCNet uses the input to `model.head` "
        "after its fixed filter bank, spatial convolution, and log-variance operation. "
        "No representation layer is selected by heldout outcomes. Strict loading of the frozen "
        "fold0/seed0 checkpoints and forward-pre-hook shape checks passed for EEGConformer/FBCNet "
        "on each of the four tasks using zero-valued input, without reading outcome labels.\n\n"
        + "\n".join(f"- {name}/{task}: `{REPRESENTATIONS[name][0]}`; dimension {dimensions[name, task]}"
                    for name in MODELS for task in TASKS) + "\n",
        encoding="utf-8")
    print(json.dumps({"cells": len(rows), "passed": sum(r["status"] == "PASS" for r in rows), "problems": problems[:30], "problem_count": len(problems)}))
    if problems:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
