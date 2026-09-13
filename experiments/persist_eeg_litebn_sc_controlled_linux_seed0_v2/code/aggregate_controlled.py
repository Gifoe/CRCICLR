"""Print the controlled C1 WBCIC decision artifact."""
import json
from pathlib import Path


def main() -> int:
    path = Path(__file__).resolve().parents[1] / "outputs" / "CONTINUATION_DECISION.json"
    if not path.is_file():
        raise SystemExit("CONTINUATION_DECISION.json is not available")
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
