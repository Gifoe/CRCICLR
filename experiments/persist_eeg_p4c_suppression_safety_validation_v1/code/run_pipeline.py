from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RUNTIME = EXP / "runtime"


def run(name: str) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    (RUNTIME / "pipeline.stage").write_text(name + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(HERE / name)], cwd=EXP, check=True)


def main() -> None:
    run("prepare_safety.py")
    run("run_safety_outcomes.py")
    run("analyze_safety.py")
    run("validate_p4c_safety.py")
    (RUNTIME / "pipeline.stage").write_text("COMPLETE\n", encoding="utf-8")


if __name__ == "__main__":
    main()
