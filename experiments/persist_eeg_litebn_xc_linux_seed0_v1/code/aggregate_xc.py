"""Print the frozen XC WBCIC gate after run_xc_task completes."""
import json
from pathlib import Path


def main() -> int:
    path = Path(__file__).resolve().parents[1] / "outputs" / "WBCIC_XC_GATE.json"
    if not path.is_file():
        raise SystemExit("WBCIC XC outer/heldout gate is not complete")
    result = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(result, indent=2, sort_keys=True))
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
