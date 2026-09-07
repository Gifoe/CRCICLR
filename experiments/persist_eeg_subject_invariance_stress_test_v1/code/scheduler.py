"""Resumable server scheduler: EEGNet first, then EEGConformer, then analysis."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+", choices=common.BACKBONES, default=list(common.BACKBONES))
    args = parser.parse_args()
    common.ensure_dirs()
    subprocess.run([sys.executable, str(common.HERE / "preflight.py")], check=True, cwd=common.REPO)
    units = [(backbone, fold, seed) for backbone in args.backbones for fold in range(5) for seed in range(3)]
    started = time.time()
    for index, (backbone, fold, seed) in enumerate(units, start=1):
        common.write_json(
            common.RUNTIME / "SCHEDULER_STATUS.json",
            {
                "status": "RUNNING",
                "unit_index": index,
                "unit_total": len(units),
                "backbone": backbone,
                "fold": fold,
                "seed": seed,
                "elapsed_seconds": time.time() - started,
            },
        )
        print(f"[scheduler {index}/{len(units)}] {backbone} fold={fold} seed={seed}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(common.HERE / "run_stress.py"),
                "--backbone",
                backbone,
                "--fold",
                str(fold),
                "--seed",
                str(seed),
            ],
            check=True,
            cwd=common.REPO,
        )
    subprocess.run([sys.executable, str(common.HERE / "aggregate.py")], check=True, cwd=common.REPO)
    subprocess.run([sys.executable, str(common.HERE / "validate.py")], check=True, cwd=common.REPO)
    common.write_json(
        common.RUNTIME / "SCHEDULER_STATUS.json",
        {"status": "COMPLETE", "unit_total": len(units), "elapsed_seconds": time.time() - started},
    )
    print("STRESS_TEST_SCHEDULER_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
