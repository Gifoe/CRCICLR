"""Static, non-overlapping full SEARCH scheduler for the seven-backbone grid."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


RUNTIME = Path(os.environ["SEVEN_RUNTIME"]).resolve()
CODE = Path(__file__).resolve().parent
MODELS = ("EEGNet", "LiteBN", "TCFormer", "TeCh", "ST-EEGFormer-small", "LaBraM-base", "CBraMod")
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")


def write(value: dict) -> None:
    path = RUNTIME / "search_queue_status.json"; path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.part"); temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def complete(task: str, model: str, fold: int, seed: int) -> bool:
    return (RUNTIME / "search_cells" / task.lower() / model.lower().replace("-", "_") / f"fold{fold}_seed{seed}" / "record.json").is_file()


def main() -> int:
    cells = [(task, model, fold, seed) for model in MODELS for task in TASKS for fold in range(5) for seed in (0, 1, 2)]
    begun = time.time(); done = sum(complete(*cell) for cell in cells)
    for ordinal, cell in enumerate(cells, 1):
        task, model, fold, seed = cell
        if complete(*cell):
            write({"state": "RUNNING", "completed": done, "total": len(cells), "current": None, "elapsed_seconds": time.time() - begun}); continue
        write({"state": "RUNNING", "completed": done, "total": len(cells), "current": {"task": task, "model": model, "fold": fold, "seed": seed, "ordinal": ordinal}, "elapsed_seconds": time.time() - begun})
        command = [sys.executable, str(CODE / "run_search.py"), "--task", task, "--model", model, "--fold", str(fold), "--seed", str(seed)]
        result = subprocess.run(command, cwd=str(CODE), env={**os.environ, "PYTHONUNBUFFERED": "1"})
        if result.returncode != 0:
            write({"state": "FAILED", "completed": done, "total": len(cells), "failed": {"task": task, "model": model, "fold": fold, "seed": seed, "exit_code": result.returncode}, "elapsed_seconds": time.time() - begun})
            return result.returncode
        done += 1
    write({"state": "COMPLETE", "completed": done, "total": len(cells), "elapsed_seconds": time.time() - begun})
    print("FULL_SEARCH_QUEUE_COMPLETE", flush=True); return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException:
        # Task Scheduler otherwise drops child-process tracebacks.  This is a
        # scheduler-observability change only; it cannot affect any cell.
        crash = RUNTIME / "search_queue_crash.txt"
        crash.write_text(traceback.format_exc(), encoding="utf-8")
        raise
