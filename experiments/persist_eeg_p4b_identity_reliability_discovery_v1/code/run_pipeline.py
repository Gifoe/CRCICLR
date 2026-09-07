from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CODE = Path(__file__).resolve().parent
EXP = CODE.parent
LOGS = EXP / "runtime" / "logs"


def write(name: str, value: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / name).write_text(value.rstrip() + "\n", encoding="utf-8")


def run(stage: str, script: str) -> None:
    write("pipeline.stage", stage)
    stem = script[:-3]
    with (LOGS / f"{stem}_stdout.log").open("w", encoding="utf-8") as stdout, (
        LOGS / f"{stem}_stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        result = subprocess.run([sys.executable, str(CODE / script)], cwd=EXP.parents[1], stdout=stdout, stderr=stderr)
    write(f"{stem}.exitcode", str(result.returncode))
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    try:
        run("DISCOVERY_UTILITY_AND_ANALYSIS", "run_discovery.py")
        run("FINAL_VALIDATE", "validate_p4b.py")
    except SystemExit as exc:
        write("pipeline.exitcode", str(exc.code))
        write("pipeline.stage", "FAILED")
        raise
    write("pipeline.exitcode", "0")
    write("pipeline.stage", "COMPLETE")


if __name__ == "__main__":
    main()
