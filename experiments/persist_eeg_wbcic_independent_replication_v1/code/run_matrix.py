"""Resume and schedule the 15 frozen EEGNet fold×seed units on one GPU."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import common


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
LOGS = EXP / "runtime" / "matrix_logs"
STATUS = EXP / "runtime" / "MATRIX_SCHEDULER_STATUS.json"


def complete(fold: int, seed: int) -> bool:
    path = common.unit_dir("eegnet", fold, seed) / "UNIT_COMPLETE.json"
    return path.is_file() and common.read_json(path).get("pass") is True


def write_status(payload: dict) -> None:
    common.write_json(STATUS, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=2)
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 3:
        raise ValueError("one RTX 5090 supports at most three full-cache workers safely")
    common.protocol()
    LOGS.mkdir(parents=True, exist_ok=True)
    queue = [(fold, seed) for fold in range(5) for seed in range(3) if not complete(fold, seed)]
    attempts = {unit: 0 for unit in queue}
    running: dict[tuple[int, int], tuple[subprocess.Popen, object, object]] = {}
    failures = []
    while queue or running:
        while queue and len(running) < args.max_workers:
            fold, seed = queue.pop(0)
            attempts[(fold, seed)] += 1
            stdout = (LOGS / f"fold-{fold}_seed-{seed}.stdout.log").open("a", encoding="utf-8")
            stderr = (LOGS / f"fold-{fold}_seed-{seed}.stderr.log").open("a", encoding="utf-8")
            command = [
                sys.executable,
                str(HERE / "run_unit.py"),
                "--backbone", "eegnet",
                "--fold", str(fold),
                "--seed", str(seed),
            ]
            process = subprocess.Popen(command, cwd=common.REPO, stdout=stdout, stderr=stderr)
            running[(fold, seed)] = (process, stdout, stderr)
            print(f"[scheduler start] fold={fold} seed={seed} pid={process.pid} attempt={attempts[(fold, seed)]}", flush=True)
        time.sleep(5)
        for unit, (process, stdout, stderr) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            stdout.close()
            stderr.close()
            del running[unit]
            fold, seed = unit
            if code == 0 and complete(fold, seed):
                print(f"[scheduler complete] fold={fold} seed={seed}", flush=True)
            elif attempts[unit] < args.max_attempts:
                print(f"[scheduler retry] fold={fold} seed={seed} exit={code}", flush=True)
                queue.append(unit)
            else:
                failures.append({"fold": fold, "seed": seed, "exit_code": code, "attempts": attempts[unit]})
                print(f"[scheduler failed] fold={fold} seed={seed} exit={code}", flush=True)
        completed = sum(complete(fold, seed) for fold in range(5) for seed in range(3))
        write_status(
            {
                "schema": "WBCIC_PHASE3_MATRIX_SCHEDULER_V1",
                "completed_units": completed,
                "total_units": 15,
                "running": [{"fold": fold, "seed": seed, "pid": process.pid} for (fold, seed), (process, _, _) in running.items()],
                "queued": [{"fold": fold, "seed": seed} for fold, seed in queue],
                "failures": failures,
                "updated_at_unix": time.time(),
            }
        )
    result = {
        "pass": not failures and all(complete(fold, seed) for fold in range(5) for seed in range(3)),
        "completed_units": sum(complete(fold, seed) for fold in range(5) for seed in range(3)),
        "failures": failures,
    }
    write_status({"schema": "WBCIC_PHASE3_MATRIX_SCHEDULER_V1", **result, "completed_at_unix": time.time()})
    print(json.dumps(result, indent=2), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
