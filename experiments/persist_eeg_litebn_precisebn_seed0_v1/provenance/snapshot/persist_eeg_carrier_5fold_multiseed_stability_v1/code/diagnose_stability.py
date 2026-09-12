"""Read-only stability diagnostics for the completed 5-fold x 3-seed grid."""
from pathlib import Path
import json


def main() -> int:
    path = Path(__file__).resolve().parents[1] / "outputs" / "DECISION.json"
    if not path.is_file():
        raise SystemExit("DECISION.json is not present; run train_grid.py")
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
