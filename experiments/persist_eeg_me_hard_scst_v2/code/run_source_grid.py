"""Persistent bounded parallel runner for the 60 source-search units."""
from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import source_search
import v2_common as c


PYTHON = Path(sys.executable)
SCRIPT = Path(__file__).with_name("source_search.py")
STATUS = c.RUNTIME / "source_grid_status.json"
LOGS = c.RUNTIME / "source_grid_logs"
LOCK = threading.Lock()


def expected_complete(dataset: str, fold: int, seed: int, scope: str) -> bool:
    root = c.RUNTIME / "source_units" / scope
    files = [root / "erm" / dataset / f"fold-{fold}" / f"seed-{seed}" / "metrics.json"]
    files.extend(root / f"q{q:.2f}_l{lam:.2f}" / dataset / f"fold-{fold}" / f"seed-{seed}" / "metrics.json" for q in (.25, .50) for lam in (.25, .50, 1.0))
    return all(path.is_file() for path in files)


def write_status(state: dict) -> None:
    with LOCK:
        c.write_json(STATUS, state)


def run_pair(unit: tuple[str, int, int], state: dict) -> tuple[tuple[str, int, int], int]:
    dataset, fold, seed = unit
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{dataset}_f{fold}_s{seed}_pair.log"
    commands = [
        [str(PYTHON), str(SCRIPT), "--dataset", dataset, "--fold", str(fold), "--seed", str(seed), "--scope", scope]
        for scope in ("A", "B") if not expected_complete(dataset, fold, seed, scope)
    ]
    return_code = 0
    with log.open("a", encoding="utf-8") as stream:
        for command in commands:
            stream.write(f"\n[runner-start] {time.time()} {' '.join(command)}\n")
            stream.flush()
            return_code = subprocess.run(command, cwd=c.REPO, stdout=stream, stderr=subprocess.STDOUT, text=True).returncode
            stream.write(f"[runner-exit] {time.time()} code={return_code}\n")
            stream.flush()
            if return_code:
                break
    with LOCK:
        complete_scopes = [scope for scope in ("A", "B") if expected_complete(dataset, fold, seed, scope)]
        key = f"{dataset}/f{fold}/s{seed}"
        state["units"][key] = {"return_code": return_code, "complete_scopes": complete_scopes, "log": str(log)}
        state["completed"] = state["initially_complete"] + sum(len(value["complete_scopes"]) for value in state["units"].values())
        state["failed"] = sum(value["return_code"] != 0 for value in state["units"].values())
        c.write_json(STATUS, state)
    return unit, return_code


def main() -> None:
    units = [(dataset, fold, seed, scope) for dataset in c.DATASETS for fold in c.FOLDS for seed in c.SEEDS for scope in ("A", "B")]
    initially_complete = sum(expected_complete(*unit) for unit in units)
    pairs = [(dataset, fold, seed) for dataset in c.DATASETS for fold in c.FOLDS for seed in c.SEEDS]
    pending = [unit for unit in pairs if not all(expected_complete(*unit, scope) for scope in ("A", "B"))]
    state = {"schema": "ME_HARD_SCST_SOURCE_GRID_STATUS_V1", "total": len(units), "initially_complete": initially_complete, "completed": initially_complete, "failed": 0, "workers": 3, "units": {}}
    write_status(state)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_pair, unit, state) for unit in pending]
        for future in concurrent.futures.as_completed(futures):
            unit, code = future.result()
            print(f"[grid] {unit} exit={code}", flush=True)
    failures = [unit for unit in units if not expected_complete(*unit)]
    if failures:
        state["terminal"] = "ENGINEERING_FAILURE"
        state["incomplete_units"] = [list(value) for value in failures]
        write_status(state)
        raise RuntimeError(f"source grid incomplete: {failures}")
    source_search.aggregate()
    state["completed"] = len(units)
    state["failed"] = 0
    state["terminal"] = "SOURCE_GRID_COMPLETE"
    write_status(state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
