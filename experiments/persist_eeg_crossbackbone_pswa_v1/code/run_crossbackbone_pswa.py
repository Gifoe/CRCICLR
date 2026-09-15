"""Strict frozen-artifact audit for cross-backbone PSWA.

This runner intentionally does not contain model construction, inference,
spectrum estimation, random sampling, or probe fitting code.  It only consumes
artifacts that were physically persisted by the corrected PEEH run.  Missing
artifacts produce explicit INCOMPLETE records.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable


EXP = Path(__file__).resolve().parents[1]
OUT = EXP / "outputs" / "crossbackbone_pswa_v1"
PEEH_REPO = Path(os.environ.get("PEEH_REPO", r"D:\nips-temp\TotalP\P1\CRCICLR_CROSSBACKBONE_PEEH_WORK"))
PEEH_EXP = PEEH_REPO / "experiments" / "persist_eeg_crossbackbone_peeh_v1"
PEEH_RUNTIME = Path(os.environ.get("PEEH_RUNTIME", r"D:\nips-temp\TotalP\P1\crossbackbone_peeh_runtime"))
SEVEN_RUNTIME = Path(os.environ.get("SEVEN_RUNTIME", r"D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime"))
BASELINE3 = Path(os.environ.get("BASELINE3_RUNTIME", r"D:\nips-temp\TotalP\P1\baseline3_runtime"))

MODELS = ("EEGNet", "CBraMod", "TeCh", "ModernTCN", "Medformer", "SGN")
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
FOLDS = tuple(range(5))
SEEDS = (0, 1, 2)
SESSIONS = {
    "OpenBMI_MI": ("S1", "S2"),
    "OpenBMI_ERP": ("S1", "S2"),
    "OpenBMI_SSVEP": ("S1", "S2"),
    "WBCIC_MI": ("S0", "S1", "S2"),
}
EXPECTED_PROBE = "Appendix-L ridge/linear probe; alpha=0.01; TRAIN-only standardizer"
FROZEN_SUFFIXES = {".npz", ".npy", ".pkl", ".pickle", ".joblib", ".parquet"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields)
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def checkpoint_record(model: str, task: str, fold: int, seed: int) -> Path:
    if model in {"EEGNet", "CBraMod", "TeCh"}:
        model_dir = model.lower().replace("-", "_")
        return SEVEN_RUNTIME / "search_cells" / task.lower() / model_dir / f"fold{fold}_seed{seed}" / "record.json"
    return BASELINE3 / model.lower() / "cells" / task.lower() / f"fold{fold}_seed{seed}" / "record.json"


def peeh_cell(model: str, task: str, fold: int, seed: int) -> Path:
    return PEEH_RUNTIME / "cells" / model.lower() / task.lower() / f"fold{fold}_seed{seed}.json"


def persisted_frozen_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in (PEEH_RUNTIME, PEEH_EXP):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in FROZEN_SUFFIXES:
                rows.append({"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)})
    return rows


def audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = persisted_frozen_candidates()
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for task in TASKS:
            for fold in FOLDS:
                for seed in SEEDS:
                    rec_path = checkpoint_record(model, task, fold, seed)
                    record = read_json(rec_path) if rec_path.is_file() else {}
                    ppath = peeh_cell(model, task, fold, seed)
                    peeh = read_json(ppath) if ppath.is_file() else {}
                    protected_rank = peeh.get("protected_dimensions")
                    protected_present = bool(ppath.is_file() and "protected_blocks" in peeh and "protected_assignment" in peeh)

                    # The corrected PEEH implementation persisted only JSON summaries.
                    # No representation, spectral transform, or random-coordinate file
                    # exists to map here; candidates are included in VALIDATION.json so
                    # an unexpected future artifact cannot be silently ignored.
                    embedding_path = ""
                    spectrum_path = ""
                    random_path = ""
                    if not protected_present:
                        status = "INCOMPLETE"
                        reason = "MISSING_CORRECTED_PEEH_PROTECTED_ASSIGNMENT"
                    elif int(protected_rank or 0) == 0:
                        status = "EMPTY_PROTECTED"
                        reason = "PROTECTED_RANK_ZERO_EXCLUDED_BY_PROTOCOL"
                    else:
                        status = "INCOMPLETE"
                        reason = "MISSING_FROZEN_EMBEDDING_SPECTRAL_TRANSFORM_AND_RANDOM_COORDINATE_FILES"

                    rows.append({
                        "Model": model,
                        "Task": task,
                        "fold": fold,
                        "seed": seed,
                        "checkpoint_record_path": str(rec_path),
                        "checkpoint_present": rec_path.is_file(),
                        "checkpoint_path": record.get("checkpoint_path", str(rec_path.parent / "selected.pt") if rec_path.is_file() else ""),
                        "checkpoint_sha256": record.get("checkpoint_sha256", ""),
                        "split_sha256": record.get("split_sha256", ""),
                        "representation_layer": "final pre-classifier hook used by corrected PEEH" if ppath.is_file() else "",
                        "protected_assignment_path": str(ppath) if ppath.is_file() else "",
                        "protected_assignment_sha256": sha256(ppath) if ppath.is_file() else "",
                        "protected_rank": protected_rank if protected_rank is not None else "",
                        "frozen_embedding_path": embedding_path,
                        "spectral_transform_path": spectrum_path,
                        "random_coordinate_path": random_path,
                        "sessions": ";".join(SESSIONS[task]),
                        "probe_configuration": EXPECTED_PROBE,
                        "status": status,
                        "reason": reason,
                    })

    audit_fields = list(rows[0])
    write_csv(OUT / "ARTIFACT_AUDIT.csv", rows, audit_fields)

    primary: list[dict[str, Any]] = []
    multi: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for task in TASKS:
            seed0 = [r for r in rows if r["Model"] == model and r["Task"] == task and r["seed"] == 0]
            nonempty = [r for r in seed0 if r["protected_rank"] != "" and int(r["protected_rank"]) > 0]
            ranks = [int(r["protected_rank"]) for r in seed0 if r["protected_rank"] != ""]
            complete = [r for r in seed0 if r["status"] == "COMPLETE"]
            base = {
                "Model": model,
                "Task": task,
                "Protected rank": (sum(ranks) / len(ranks)) if ranks else "",
                "Coverage": f"{len(nonempty)}/5",
                "Protected-only WS-BA": "",
                "Random-only WS-BA": "",
                "PSWA mean": "",
                "PSWA median": "",
                "PSWA CI low": "",
                "PSWA CI high": "",
                "Significant": "NOT_ESTIMATED",
                "Status": "COMPLETE" if len(complete) == 5 else "INCOMPLETE",
                "Reason": "" if len(complete) == 5 else "required frozen representation/spectral/random artifacts were not persisted",
            }
            primary.append(base)

            all15 = [r for r in rows if r["Model"] == model and r["Task"] == task]
            multi.append({
                "Model": model, "Task": task, "cells_available": sum(r["status"] == "COMPLETE" for r in all15),
                "cells_required": 15, "Protected-only WS-BA": "", "Random-only WS-BA": "",
                "PSWA mean": "", "PSWA median": "", "PSWA CI low": "", "PSWA CI high": "",
                "Significant": "NOT_ESTIMATED", "Status": "INCOMPLETE",
                "Reason": "no model-task has a complete 15-cell frozen PSWA artifact matrix",
            })
            for session in SESSIONS[task]:
                session_rows.append({
                    "Model": model, "Task": task, "session": session,
                    "Protected BA": "", "Random BA": "", "Advantage pp": "",
                    "Status": "INCOMPLETE", "Reason": "frozen retained-coordinate predictions unavailable",
                })

    primary_fields = list(primary[0])
    multi_fields = list(multi[0])
    session_fields = list(session_rows[0])
    write_csv(OUT / "CROSSBACKBONE_PSWA_SEED0.csv", primary, primary_fields)
    write_csv(OUT / "CROSSBACKBONE_PSWA_MULTISEED.csv", multi, multi_fields)
    write_csv(OUT / "SESSION_SPECIFIC_PSWA.csv", session_rows, session_fields)
    write_csv(OUT / "PROTECTED_ONLY_SESSION_RESULTS.csv", [], ["Model", "Task", "fold", "seed", "subject_id", "session", "BA"])
    write_csv(OUT / "RANDOM_ONLY_SESSION_RESULTS.csv", [], ["Model", "Task", "fold", "seed", "draw", "subject_id", "session", "BA"])
    write_csv(OUT / "SUBJECT_LEVEL_PSWA_SEED0.csv", [], ["Model", "Task", "subject_id", "Protected_WSBA", "Random_WSBA", "PSWA_pp", "folds"])

    nonempty_peeh = sum(r["protected_rank"] != "" and int(r["protected_rank"]) > 0 for r in rows)
    empty_peeh = sum(r["status"] == "EMPTY_PROTECTED" for r in rows)
    missing_assignment = sum(r["reason"] == "MISSING_CORRECTED_PEEH_PROTECTED_ASSIGNMENT" for r in rows)
    validation = {
        "pass": True,
        "scientific_status": "PSWA_NOT_ESTIMABLE_FROM_PERSISTED_FROZEN_ARTIFACTS",
        "artifact_audit_rows": len(rows),
        "expected_artifact_audit_rows": 360,
        "seed0_primary_rows": len(primary),
        "secondary_rows": len(multi),
        "persisted_array_or_table_candidates": candidates,
        "nonempty_protected_cells_without_required_frozen_artifacts": nonempty_peeh,
        "empty_protected_cells": empty_peeh,
        "missing_peeh_assignment_cells": missing_assignment,
        "estimated_pswa_cells": 0,
        "training_performed": False,
        "inference_rerun": False,
        "protected_assignment_recomputed": False,
        "persistence_spectrum_recomputed": False,
        "random_controls_redrawn": False,
        "probe_hyperparameters_tuned": False,
        "evaluation_labels_used_for_selection": False,
        "bootstrap_draws_if_estimable": 20_000,
        "bootstrap_unit": "biological subject",
    }
    write_json(OUT / "VALIDATION.json", validation)

    lines = [
        "# Final cross-backbone PSWA v1 report", "",
        "## Outcome", "",
        "PSWA is **not estimable** under the frozen-artifact rules. No numeric PSWA value was fabricated.", "",
        "The corrected PEEH run persisted checkpoint provenance, Protected ranks/assignments, and subject-level erasure summaries. It did not persist the frozen pre-classifier representation matrices, the canonical spectral transform (mean/basis/scale/directions), or the exact 100 final random-coordinate sets. Retained-subspace probe performance cannot be recovered from erased-subspace BA summaries.", "",
        "## Primary seed-0 coverage", "",
        "|Model|Task|Protected rank (mean over recorded folds)|Protected coverage|Status|", "|---|---|---:|---:|---|",
    ]
    for row in primary:
        rank = f"{row['Protected rank']:.2f}" if row["Protected rank"] != "" else "NA"
        lines.append(f"|{row['Model']}|{row['Task']}|{rank}|{row['Coverage']}|{row['Status']}|")
    lines += [
        "", "## Integrity conclusion", "",
        "- Training performed: NO", "- Inference rerun: NO", "- Protected assignments recomputed: NO",
        "- Persistence spectrum recomputed: NO", "- Random controls redrawn: NO",
        "- Numeric PSWA estimates: 0", "",
        "A future PSWA run requires a separately authorized artifact-recovery amendment that deterministically regenerates and then freezes the exact representations, spectral transforms, and random coordinate sets. That amendment is outside this locked analysis.",
    ]
    (OUT / "FINAL_CROSSBACKBONE_PSWA_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("CROSSBACKBONE_PSWA_FROZEN_ARTIFACT_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    audit()
