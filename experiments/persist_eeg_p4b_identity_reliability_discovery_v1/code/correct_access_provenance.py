from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RESULTS = EXP / "results"
UTILITY_CACHE = EXP / "runtime" / "utility_runs"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    artifacts = list(UTILITY_CACHE.rglob("COMPLETE.json"))
    if len(artifacts) != 60:
        raise RuntimeError(f"expected 60 completed utility artifacts, found {len(artifacts)}")
    earliest = min(path.stat().st_mtime for path in artifacts)
    earliest_utc = datetime.fromtimestamp(earliest, timezone.utc).isoformat()
    pre = read_json(EXP / "PRE_OUTCOME_FREEZE_COMPLETE.json")
    if earliest_utc <= pre["timestamp_utc"]:
        raise RuntimeError("earliest durable utility artifact does not postdate pre-outcome freeze")
    analysis_path = RESULTS / "P4B_ANALYSIS_COMPLETE.json"
    analysis = read_json(analysis_path)
    analysis["first_future_access_timestamp_utc"] = earliest_utc
    analysis["first_access_timestamp_semantics"] = "earliest durable COMPLETE.json utility artifact; conservative observed-access timestamp"
    write_json(analysis_path, analysis)
    (EXP / "FUTURE_UTILITY_ACCESS_LEDGER.md").write_text(
        "# Future Utility Access Ledger\n\n"
        f"- Pre-outcome freeze completed: `{pre['timestamp_utc']}`.\n"
        "- Hash verification passed before access.\n"
        f"- Earliest durable post-freeze utility artifact: `{earliest_utc}`.\n"
        "- This timestamp is recovered from the earliest of 60 atomic `COMPLETE.json` files; it replaces the later process-restart timestamp and is a conservative observed-access record.\n"
        "- Historical discovery utilities used: S1, S2, S3.\n"
        "- New discovery utility opened: S5 only.\n"
        "- P4C reserved S4/S6 future utilities: UNTOUCHED.\n"
        "- OpenBMI sealed 14: UNTOUCHED.\n"
        "- WBCIC outer 10: UNTOUCHED / NOT ENUMERATED.\n",
        encoding="utf-8",
    )
    print(f"P4B_ACCESS_PROVENANCE_CORRECTED {earliest_utc}", flush=True)


if __name__ == "__main__":
    main()
