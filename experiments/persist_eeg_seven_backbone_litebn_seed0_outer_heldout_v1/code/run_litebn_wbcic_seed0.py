"""Detached continuation of the five missing WBCIC LiteBN seed-0 cells.

Training is the unmodified frozen seven-backbone runner.  This supervisor
only supplies Windows-safe logs, no overlapping duplicate cells, and status.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("D:/nips-temp/TotalP/P1")
RUNTIME = ROOT / "seven_backbone_fourtask_3seed_runtime"
REPO = ROOT / "CRCICLR_LITEBN_OUTER_HELDOUT_WORK"
CODE = REPO / "experiments/persist_eeg_seven_backbone_fourtask_3seed_v1/code"
LOG = RUNTIME / "logs/litebn_seed0_wbcic_completion"
STATUS = RUNTIME / "litebn_seed0_outer_heldout_status.json"
PYTHON = Path("E:/Anaconda/envs/persist_stable_251/python.exe")


def status(**value: object) -> None:
    value.update(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), supervisor_pid=os.getpid())
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".part"); tmp.write_text(json.dumps(value, indent=2), encoding="utf-8"); os.replace(tmp, STATUS)


def complete(cell: Path) -> bool:
    return (cell / "record.json").is_file() and (cell / "selected.pt").is_file()


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"SEVEN_RUNTIME": str(RUNTIME), "SEVEN_REPO": str(REPO), "MODERN_REPO": str(REPO),
                "TASK_GENERALITY_REPO": str(REPO), "OFFICIAL_BACKBONE_ROOT": str(RUNTIME / "official"),
                "FULL_OPENBMI_CACHE": str(ROOT / "persist_eeg_stage0_repo_full/outputs/persist_eeg_stage0/cache/openbmi"),
                "FULL_WBCIC_CACHE": str(ROOT / "CRCICLR_SOURCE_ONLY_DIAGNOSTIC/experiments/persist_eeg_wbcic_independent_replication_v1/runtime/cache/wbcic_epochs"),
                "PYTHONFAULTHANDLER": "1", "PYTHONUNBUFFERED": "1", "CUDA_MODULE_LOADING": "LAZY"})
    for fold in range(5):
        cell = RUNTIME / "search_cells/wbcic_mi/litebn" / f"fold{fold}_seed0"
        if complete(cell):
            continue
        lock = cell / "RUNNING.lock"
        if lock.exists():
            raise RuntimeError(f"refusing to replace existing cell lock: {lock}")
        log = LOG / f"wbcic_mi_litebn_fold{fold}_seed0.log"
        with log.open("ab") as stream:
            child = subprocess.Popen([str(PYTHON), "-u", "-X", "faulthandler", str(CODE / "run_search.py"),
                                      "--task", "WBCIC_MI", "--model", "LiteBN", "--fold", str(fold), "--seed", "0"],
                                     cwd=CODE, env=env, stdout=stream, stderr=subprocess.STDOUT)
            status(state="RUNNING", task="WBCIC_MI", model="LiteBN", fold=fold, seed=0, child_pid=child.pid, log=str(log))
            code = child.wait()
        if code != 0 or not complete(cell):
            status(state="FAILED", task="WBCIC_MI", model="LiteBN", fold=fold, seed=0, exit_code=code, log=str(log))
            raise SystemExit(code or 1)
    status(state="COMPLETE", task="WBCIC_MI", model="LiteBN", cells=5)


if __name__ == "__main__":
    main()
