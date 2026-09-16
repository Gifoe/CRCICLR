"""Read-only completion inventory for the six retained PERSIST-EEG baselines.

This audit never opens EEG arrays or heldout labels. It checks checkpoint
identity/manifest metadata and existing compact evaluation artifacts only.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
OUT = EXP / "outputs"
PROTOCOL = EXP / "protocol"
P1 = Path(r"D:\nips-temp\TotalP\P1")
SEVEN = P1 / "seven_backbone_fourtask_3seed_runtime" / "search_cells"
BASELINE3 = P1 / "baseline3_runtime"
CSGD = (
    P1 / "CRCICLR_CROSSBACKBONE_CSGD_WORK" / "experiments"
    / "persist_eeg_crossbackbone_csgd_v1" / "outputs"
    / "crossbackbone_csgd_v1" / "SESSION_SUBJECT_RESULTS.csv"
)
PSWA_WORK = (
    P1 / "CRCICLR_CROSSBACKBONE_PEEH_WORK" / "experiments"
    / "persist_eeg_crossbackbone_pswa_v1" / "outputs"
    / "crossbackbone_pswa_v1" / "CROSSBACKBONE_PSWA_SEED0.csv"
)
MANIFEST = (
    P1 / "CRCICLR_BACKBONE_GEN_WORK" / "experiments"
    / "persist_eeg_final_heldout_confirmation_v1" / "protocol"
    / "FINAL_HOLDOUT_MANIFEST.json"
)
TRUE_OUTER_WBCIC = {
    "sub-4", "sub-8", "sub-10", "sub-15", "sub-20",
    "sub-39", "sub-40", "sub-43", "sub-46", "sub-51",
}

TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
MODELS = ("EEGNet", "CBraMod", "TeCh", "ModernTCN", "Medformer", "LiteBN")
OLD = {"EEGNet": "eegnet", "CBraMod": "cbramod", "TeCh": "tech", "LiteBN": "litebn"}
NEW = {
    "ModernTCN": (
        "moderntcn", "codex/persist-eeg-moderntcn-4task-3seed-final-v1",
        "persist_eeg_moderntcn_4task_3seed_final_v1",
    ),
    "Medformer": (
        "medformer", "codex/persist-eeg-medformer-4task-3seed-final-v1",
        "persist_eeg_medformer_4task_3seed_final_v1",
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_root(model: str, task: str, fold: int, seed: int) -> Path:
    if model in OLD:
        return SEVEN / task.lower() / OLD[model] / f"fold{fold}_seed{seed}"
    slug = NEW[model][0]
    return BASELINE3 / slug / "cells" / task.lower() / f"fold{fold}_seed{seed}"


def source(model: str) -> tuple[str, str]:
    if model in NEW:
        return NEW[model][1:]
    return ("codex/persist-eeg-seven-backbone-fourtask-3seed-v1", "persist_eeg_seven_backbone_fourtask_3seed_v1")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    open_subjects = {str(x) for x in manifest["OpenBMI"]["subject_ids"]}
    internal_wbcic_subjects = {str(x) for x in manifest["WBCIC"]["subject_ids"]}
    wbcic_subjects = TRUE_OUTER_WBCIC
    if len(open_subjects) != 14 or len(internal_wbcic_subjects) != 10 or len(wbcic_subjects) != 10:
        raise RuntimeError("final heldout cohort size mismatch")
    if internal_wbcic_subjects & wbcic_subjects:
        raise RuntimeError("unexpected overlap between WBCIC internal and true outer subjects")

    sessions = read_csv(CSGD)
    session_index: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in sessions:
        if row.get("Model") in MODELS and row.get("Task") in TASKS:
            session_index[row["Model"], row["Task"]].append(row)

    pswa = read_csv(PSWA_WORK)
    pswa_index = {(row.get("Model"), row.get("Task")): row for row in pswa}
    audit_rows: list[dict[str, object]] = []
    checkpoint_detail: list[dict[str, object]] = []
    for model in MODELS:
        branch, experiment = source(model)
        for task in TASKS:
            present = []
            missing = []
            parameters = set()
            for fold in range(5):
                for seed in range(3):
                    root = checkpoint_root(model, task, fold, seed)
                    checkpoint = root / "selected.pt"
                    record_path = root / "record.json"
                    reason = ""
                    record = None
                    if not checkpoint.is_file() or not record_path.is_file():
                        reason = "missing selected.pt or record.json"
                    else:
                        try:
                            record = json.loads(record_path.read_text(encoding="utf-8"))
                            identity = (record.get("model"), record.get("task"), int(record.get("fold", -1)), int(record.get("seed", -1)))
                            if identity != (model, task, fold, seed):
                                reason = f"record identity mismatch: {identity}"
                            if not record.get("checkpoint_sha256") or not record.get("normalizer", {}).get("mean_std_sha256"):
                                reason = "missing frozen checkpoint or normalizer hash"
                            elif sha256(checkpoint) != record["checkpoint_sha256"]:
                                reason = "frozen checkpoint SHA-256 mismatch"
                        except Exception as error:
                            reason = f"record parse failed: {error}"
                    if reason:
                        missing.append(f"fold{fold}_seed{seed}: {reason}")
                    else:
                        present.append((fold, seed))
                        parameters.add(int(record.get("trainable_parameters", record.get("parameters", 0))))
                    checkpoint_detail.append({
                        "model": model, "task": task, "fold": fold, "seed": seed,
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_hash": "" if record is None else record.get("checkpoint_sha256", ""),
                        "normalizer_hash": "" if record is None else record.get("normalizer", {}).get("mean_std_sha256", ""),
                        "status": "PRESENT_HASH_VERIFIED" if not reason else "MISSING_OR_INVALID",
                        "reason": reason,
                    })

            current = session_index[model, task]
            cohort = wbcic_subjects if task == "WBCIC_MI" else open_subjects
            needed_sessions = {"S0", "S1", "S2"} if task == "WBCIC_MI" else {"S1", "S2"}
            actual = {
                (int(row["fold"]), int(row["seed"]), str(row["subject"]), row["session"])
                for row in current
            }
            expected = {
                (fold, seed, subject, session)
                for fold in range(5) for seed in range(3)
                for subject in cohort for session in needed_sessions
            }
            all_sessions = len(present) == 15 and actual == expected and len(current) == len(expected)
            future_name = "S2"  # OpenBMI paper S2; WBCIC file S2 is paper S3.
            future = [row for row in current if row["session"] == future_name]
            future_expected = {
                (fold, seed, subject)
                for fold in range(5) for seed in range(3) for subject in cohort
            }
            future_keys = {(int(row["fold"]), int(row["seed"]), str(row["subject"])) for row in future}
            future_complete = len(present) == 15 and future_keys == future_expected and len(future) == len(future_expected)
            # The recent baseline3 final heldout CSVs may be complete before CSGD all-session inference.
            recent_future = False
            recent_subjects: set[str] = set()
            if model in NEW:
                exp_name = NEW[model][2]
                path = P1 / f"CRCICLR_{NEW[model][0].upper()}_FINAL" / "experiments" / exp_name / "outputs" / "HELDOUT_SUBJECT_RESULTS.csv"
                new_rows = [row for row in read_csv(path) if row.get("task") == task]
                recent_subjects = {str(row["subject_id"]) for row in new_rows}
                keys = {(int(row["fold"]), int(row["seed"]), str(row["subject_id"])) for row in new_rows}
                recent_future = len(present) == 15 and keys == future_expected and len(new_rows) == len(future_expected)
            future_available = future_complete or recent_future
            pswa_row = pswa_index.get((model, task), {})
            pswa_state = pswa_row.get("Status", "NOT_FOUND")
            if len(present) != 15:
                status = "INCOMPLETE_FROZEN_CHECKPOINT_MATRIX"
            elif not future_available:
                status = "COMPLETE_CHECKPOINTS_BUT_FINAL_COHORT_METRIC_MISSING"
            else:
                status = "COMPLETE_PRIMARY_FUTURE"
            notes = list(missing)
            if task == "WBCIC_MI" and model in NEW and recent_subjects == internal_wbcic_subjects:
                notes.append("existing HELDOUT_SUBJECT_RESULTS uses V8 internal subjects, not WBCIC true outer; expected prompt BA/F1 refer to internal cohort")
            if not all_sessions:
                notes.append("full 15-checkpoint true-outer session matrix missing")
            audit_rows.append({
                "model": model, "task": task,
                "n_folds": len({fold for fold, _ in present}),
                "n_seeds": len({seed for _, seed in present}),
                "n_frozen_checkpoints": len(present),
                "future_BA_available": future_available,
                "future_macro_F1_available": future_available,
                "all_session_metrics_available": all_sessions,
                "WS_BA_available": all_sessions,
                "CSGD_available": all_sessions,
                "PSWA_available": "WORKTREE_UNCOMMITTED_" + pswa_state if pswa_row else "NO_COMMITTED_ESTIMATE",
                "parameters_available": len(parameters) == 1 and next(iter(parameters), 0) > 0,
                "MACs_available": False,
                "status": status,
                "source_branch": branch,
                "source_experiment": experiment,
                "notes": "; ".join(notes) if notes else "checkpoint and session matrices present",
            })

    write_csv(OUT / "BASELINE_COMPLETION_AUDIT.csv", audit_rows)
    write_csv(OUT / "FROZEN_CHECKPOINT_MANIFEST_AUDIT.csv", checkpoint_detail)
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    (PROTOCOL / "SESSION_MAPPING_AUDIT.json").write_text(json.dumps({
        "OpenBMI": {"file_S1": "paper_S1", "file_S2": "paper_S2"},
        "WBCIC": {"file_S0": "paper_S1", "file_S1": "paper_S2", "file_S2": "paper_S3"},
        "heldout_subject_counts": {"OpenBMI": 14, "WBCIC_true_outer": 10},
        "OpenBMI_heldout_subject_ids": sorted(open_subjects, key=int),
        "WBCIC_true_outer_subject_ids": sorted(wbcic_subjects),
        "WBCIC_V8_internal_subject_ids": sorted(internal_wbcic_subjects),
        "cohort_overlap": sorted(wbcic_subjects & internal_wbcic_subjects),
        "V8_internal_manifest_path": str(MANIFEST),
        "true_outer_source": "crossbackbone_csgd_v1 code TRUE_WBCIC_SUBJECTS and seven_backbone_true_outer_v1 outputs",
        "verified_from": "existing CSGD runner session definitions and heldout subject metadata; no EEG arrays or labels read",
    }, indent=2) + "\n", encoding="utf-8")
    for row in audit_rows:
        print(row["model"], row["task"], f"{row['n_frozen_checkpoints']}/15", row["status"], "sessions", row["all_session_metrics_available"])


if __name__ == "__main__":
    main()
