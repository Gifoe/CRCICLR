"""Resume exact frozen all-session inference on final biological holdouts.

The model/data path is the already-published cross-backbone CSGD runner. This
wrapper changes only the output namespace and expands evaluation to all 15
prescribed checkpoints where they exist. It never trains or selects a model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
P1 = Path(r"D:\nips-temp\TotalP\P1")
SOURCE = (P1 / "CRCICLR_CROSSBACKBONE_CSGD_WORK" / "experiments"
          / "persist_eeg_crossbackbone_csgd_v1" / "code" / "run_crossbackbone_csgd.py")
RUNTIME = P1 / "baseline_metrics_closure_v1_runtime"
OLD_RUNTIME = P1 / "crossbackbone_csgd_runtime"
LOCK = EXP / "protocol" / "FROZEN_INFERENCE_LOCK.json"
LOCK_SHA = LOCK.with_suffix(".sha256")
MODELS = ("EEGNet", "CBraMod", "TeCh", "ModernTCN", "Medformer", "LiteBN")
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
TRUE_WBCIC = ("sub-4", "sub-8", "sub-10", "sub-15", "sub-20", "sub-39",
              "sub-40", "sub-43", "sub-46", "sub-51")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    os.environ["CSGD_RUNTIME"] = str(RUNTIME)
    os.environ.setdefault("SEVEN_RUNTIME", str(P1 / "seven_backbone_fourtask_3seed_runtime"))
    spec = importlib.util.spec_from_file_location("closure_frozen_csgd", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    if Path(value.RUNTIME) != RUNTIME:
        raise RuntimeError("CSGD runtime redirection failed")
    return value


def prelock(csgd) -> None:
    if LOCK.exists():
        raise RuntimeError("frozen inference lock already exists; verify it instead")
    manifest = json.loads(csgd.HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    open_subjects = tuple(map(str, manifest["OpenBMI"]["subject_ids"]))
    if len(open_subjects) != 14 or len(set(open_subjects)) != 14:
        raise RuntimeError("OpenBMI cohort must be 14 unique subjects")
    actual_wbcic = tuple(csgd.natural_subjects(
        path.name for path in csgd.TRUE_WBCIC_CACHE.iterdir()
        if path.is_dir() and path.name.startswith("sub-")))
    if actual_wbcic != tuple(csgd.natural_subjects(TRUE_WBCIC)):
        raise RuntimeError(f"true-outer cache membership mismatch: {actual_wbcic}")
    for subject in TRUE_WBCIC:
        for session in (0, 1, 2):
            for suffix in ("epochs.npy", "labels.npy", "metadata.json"):
                if not (csgd.TRUE_WBCIC_CACHE / subject / f"ses-{session}_{suffix}").is_file():
                    raise RuntimeError(f"missing true-outer cache metadata/file: {subject} {session} {suffix}")
    rows = [csgd.audit_row(model, task, fold, seed)
            for model in MODELS for task in TASKS for fold in range(5) for seed in range(3)]
    invalid = [row for row in rows if str(row["status"]).startswith("INVALID")]
    if invalid:
        raise RuntimeError(f"invalid checkpoint: {invalid[0]}")
    lock = {
        "schema": "PERSIST_EEG_FROZEN_TRUE_OUTER_INFERENCE_LOCK_V1",
        "source_runner_path": str(SOURCE),
        "source_runner_sha256": sha(SOURCE),
        "source_runner_commit": "4020f0bb0a24d5ef70bb9ba0e2116b7b0f257ebb",
        "amendment_sha256": sha(EXP / "protocol" / "TRUE_OUTER_AMENDMENT_20260916.md"),
        "cohorts": {"OpenBMI": {"subjects": list(open_subjects)},
                    "WBCIC": {"subjects": list(TRUE_WBCIC)}},
        "session_file_to_paper": {"OpenBMI": {"S1": "S1", "S2": "S2"},
                                  "WBCIC": {"S0": "S1", "S1": "S2", "S2": "S3"}},
        "training_performed": False,
        "checkpoint_reselection": False,
        "evaluation_checkpoints": [row for row in rows if row["status"] == "PRESENT_VERIFIED"],
        "missing_checkpoints": [row for row in rows if row["status"] != "PRESENT_VERIFIED"],
    }
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOCK_SHA.write_text(sha(LOCK) + "\n", encoding="utf-8")
    print(f"FROZEN_LOCKED present={len(lock['evaluation_checkpoints'])} "
          f"missing={len(lock['missing_checkpoints'])} sha256={sha(LOCK)}", flush=True)


def read_lock(csgd) -> dict:
    if not LOCK.is_file() or not LOCK_SHA.is_file():
        raise RuntimeError("committed inference lock required")
    if sha(LOCK) != LOCK_SHA.read_text(encoding="utf-8").strip():
        raise RuntimeError("frozen inference lock hash mismatch")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if sha(SOURCE) != lock["source_runner_sha256"]:
        raise RuntimeError("source frozen inference code has changed")
    if sha(EXP / "protocol" / "TRUE_OUTER_AMENDMENT_20260916.md") != lock["amendment_sha256"]:
        raise RuntimeError("cohort amendment changed after lock")
    if tuple(lock["cohorts"]["WBCIC"]["subjects"]) != TRUE_WBCIC:
        raise RuntimeError("WBCIC cohort changed")
    return lock


def old_session(csgd, model: str, task: str, fold: int, seed: int, session: int) -> Path:
    return OLD_RUNTIME / "session_cells" / model.lower() / task.lower() / f"fold{fold}_seed{seed}_session{session}.json"


def complete_subjects(path: Path, expected: set[str]) -> bool:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        rows = result["subject_rows"]
        return len(rows) == len(expected) and {str(row["subject"]) for row in rows} == expected
    except Exception:
        return False


def reuse(csgd, row: dict, lock: dict) -> int:
    model, task, fold, seed = row["Model"], row["Task"], int(row["fold"]), int(row["seed"])
    expected = set(lock["cohorts"]["WBCIC" if task == "WBCIC_MI" else "OpenBMI"]["subjects"])
    sessions = (0, 1, 2) if task == "WBCIC_MI" else (1, 2)
    copied = 0
    for session in sessions:
        old = old_session(csgd, model, task, fold, seed, session)
        new = csgd.session_path(model, task, fold, seed, session)
        if new.exists():
            if not csgd.valid_cached_session(new, row, session) or not complete_subjects(new, expected):
                raise RuntimeError(f"invalid closure session cache: {new}")
            continue
        if csgd.valid_cached_session(old, row, session) and complete_subjects(old, expected):
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, new)
            copied += 1
    return copied


def run(csgd, model: str, task: str | None) -> None:
    lock = read_lock(csgd)
    csgd.torch.set_num_threads(int(os.environ.get("CLOSURE_CPU_THREADS", "4")))
    device = csgd.torch.device("cuda" if csgd.torch.cuda.is_available() else "cpu")
    grouped = defaultdict(list)
    for row in lock["evaluation_checkpoints"]:
        if row["Model"] != model or (task and row["Task"] != task):
            continue
        grouped[row["Task"], int(row["fold"])].append(row)
    counts = {current_task: sum(len(rows) for (name, _), rows in grouped.items() if name == current_task)
              for current_task, _ in grouped}
    for current_task, count in counts.items():
        if count != 15:
            print(f"INCOMPLETE_TASK_SKIP model={model} task={current_task} checkpoints={count}/15", flush=True)
    for (current_task, fold), rows in sorted(grouped.items(), key=lambda pair: (TASKS.index(pair[0][0]), pair[0][1])):
        if counts[current_task] != 15:
            continue
        if len(rows) != 3:
            print(f"INCOMPLETE_SKIP model={model} task={current_task} fold={fold} n={len(rows)}", flush=True)
            continue
        copied = sum(reuse(csgd, row, lock) for row in rows)
        print(f"GROUP_START model={model} task={current_task} fold={fold} reused_sessions={copied}", flush=True)
        csgd.run_group(model, current_task, fold, sorted(rows, key=lambda row: int(row["seed"])), lock, device)
    print(f"MODEL_DONE model={model} task={task or 'ALL'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prelock", "run"), required=True)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--task", choices=TASKS)
    args = parser.parse_args()
    csgd = source()
    if args.stage == "prelock":
        prelock(csgd)
    else:
        if not args.model:
            parser.error("--model is required for --stage run")
        run(csgd, args.model, args.task)


if __name__ == "__main__":
    main()
