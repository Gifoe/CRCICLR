"""Resumable two-phase scheduler; source freeze is a hard global barrier."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import common


def call(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=common.HERE, check=True)


def run_grid(phase: str) -> None:
    total = 30
    done = 0
    for backbone in common.BACKBONES:
        for fold in range(5):
            for seed in range(3):
                call("run_unit.py", "--phase", phase, "--backbone", backbone, "--fold", str(fold), "--seed", str(seed))
                done += 1
                common.write_json(
                    common.RUNTIME / "SCHEDULER_STATUS.json",
                    {"phase": phase, "completed": done, "total": total, "backbone": backbone, "fold": fold, "seed": seed, "updated_at_unix": time.time()},
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("source", "outcome", "all"), default="all")
    args = parser.parse_args()
    if args.phase in ("source", "all"):
        run_grid("source")
        call("freeze_source.py")
    if args.phase in ("outcome", "all"):
        if not (common.RUNTIME / "GLOBAL_SOURCE_FREEZE.json").exists():
            raise RuntimeError("outcome scheduler blocked before global source freeze")
        run_grid("outcome")
    if args.phase == "all":
        call("aggregate.py")
        call("validate.py")


if __name__ == "__main__":
    main()
