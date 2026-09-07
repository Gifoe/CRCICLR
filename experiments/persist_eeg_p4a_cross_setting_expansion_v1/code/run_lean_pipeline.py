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
    with (LOGS / f"{script[:-3]}_stdout.log").open("w", encoding="utf-8") as out, (
        LOGS / f"{script[:-3]}_stderr.log"
    ).open("w", encoding="utf-8") as err:
        result = subprocess.run([sys.executable, str(CODE / script)], cwd=EXP.parents[1], stdout=out, stderr=err)
    write(f"{script[:-3]}.exitcode", str(result.returncode))
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    try:
        run("LEAN_AGGREGATE", "aggregate_lean.py")
        run("LEAN_VALIDATE", "validate_p4a_lean.py")
    except SystemExit as exc:
        write("pipeline.exitcode", str(exc.code))
        write("pipeline.stage", "LEAN_FAILED")
        raise
    write("pipeline.exitcode", "0")
    write("pipeline.stage", "LEAN_COMPLETE")


if __name__ == "__main__":
    main()
