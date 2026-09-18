"""Run the frozen PU/U explanatory analysis in a fixed size-based order.

This is a scheduler only.  It never changes a checkpoint, split, selector,
rank, or evaluation rule.  Each cell is still executed by
``run_pu_u_interpretation.py``.  The order was frozen from the trainable
parameter counts recorded with the selected baseline checkpoints:

    FBCNet (21,026) < EEGNet (34,162) < TeCh (793,026)
    < EEGConformer (853,506) < Medformer (3,890,178)
    < CBraMod (4,884,002) < ModernTCN (17,391,170).

The original Experiment 1 cell for a model must be complete before its
explanatory cells are launched.  This makes a restart safe: completed cells
are reused, while a missing source cell remains a visible waiting condition.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_pu_u_interpretation import TASKS, path_for
from run_pu_u_interpretation_queue import RUNTIME, complete_original_model


PARAMETER_ORDER = (
    "FBCNet",
    "EEGNet",
    "TeCh",
    "EEGConformer",
    "Medformer",
    "CBraMod",
    "ModernTCN",
)
PARAMETER_COUNTS = {
    "FBCNet": 21_026,
    "EEGNet": 34_162,
    "TeCh": 793_026,
    "EEGConformer": 853_506,
    "Medformer": 3_890_178,
    "CBraMod": 4_884_002,
    "ModernTCN": 17_391_170,
}
CODE = Path(__file__).resolve().parent
STATUS_PATH = RUNTIME / "PU_U_ORDERED_ANALYSIS_STATUS.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def complete_derived_model(model: str) -> bool:
    for task in TASKS:
        for fold in range(5):
            for seed in range(3):
                path = path_for(model, task, fold, seed)
                if not path.is_file():
                    return False
                try:
                    if json.loads(path.read_text(encoding="utf-8")).get("identity") != [model, task, fold, seed]:
                        return False
                except (OSError, json.JSONDecodeError):
                    return False
    return True


def derived_count(model: str) -> int:
    return sum(path_for(model, task, fold, seed).is_file()
               for task in TASKS for fold in range(5) for seed in range(3))


def write_status(**status: object) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_utc": utc_now(), "parameter_order": list(PARAMETER_ORDER),
               "parameter_counts": PARAMETER_COUNTS, **status}
    temp = STATUS_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(STATUS_PATH)


def wait_for_source(model: str, seconds: float) -> None:
    while not complete_original_model(model):
        write_status(state="waiting_for_original_cells", model=model,
                     source_complete=False, derived_cells=derived_count(model))
        print(f"WAITING_SOURCE {model}", flush=True)
        time.sleep(seconds)


def run_model(model: str, workers: int) -> None:
    if complete_derived_model(model):
        write_status(state="derived_cells_cached", model=model, source_complete=True,
                     derived_cells=60)
        print(f"DERIVED_CACHED {model}", flush=True)
        return
    log_path = RUNTIME / f"ordered_interpretation_{model.lower()}.log"
    env = os.environ.copy()
    env["INCREMENTAL_DIRECT_ANALYSIS"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nORDERED_MODEL_START {model} {utc_now()} workers={workers}\n")
        log.flush()
        result = subprocess.run(
            [sys.executable, "-u", str(CODE / "run_pu_u_interpretation_queue.py"),
             "--models", model, "--workers", str(workers)],
            cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode or not complete_derived_model(model):
        write_status(state="failed", model=model, source_complete=True,
                     derived_cells=derived_count(model), exit_code=result.returncode,
                     log_path=str(log_path))
        raise RuntimeError(f"ordered analysis failed for {model}: exit={result.returncode}")
    write_status(state="model_complete", model=model, source_complete=True,
                 derived_cells=60, log_path=str(log_path))
    print(f"MODEL_COMPLETE {model}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--external-running", nargs="*", default=())
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 3:
        raise ValueError("workers must be in [1, 3]")
    external = set(args.external_running)
    for model in PARAMETER_ORDER:
        wait_for_source(model, args.poll_seconds)
        if model in external and not complete_derived_model(model):
            write_status(state="waiting_for_external_queue", model=model,
                         source_complete=True, derived_cells=derived_count(model))
            while not complete_derived_model(model):
                time.sleep(args.poll_seconds)
            print(f"EXTERNAL_MODEL_COMPLETE {model}", flush=True)
        else:
            run_model(model, args.workers if model != "ModernTCN" else 1)
    write_status(state="all_models_complete", model=None, source_complete=True,
                 derived_cells=60 * len(PARAMETER_ORDER))
    print("ORDERED_ANALYSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
