#!/usr/bin/env python3
"""Emit the bounded GitHub Actions transfer matrix without third-party imports."""

import json
import os


def main() -> None:
    requested = os.environ.get("REQUESTED_DATASET", "all").lower()
    requested_chunk = os.environ.get("REQUESTED_CHUNK", "").strip()
    chunks = {
        "openbmi-mi": ["01-09", "10-18", "19-27", "28-36", "37-45", "46-54"],
        "openbmi-erp": ["01-09", "10-18", "19-27", "28-36", "37-45", "46-54"],
        "openbmi-ssvep": ["01-09", "10-18", "19-27", "28-36", "37-45", "46-54"],
        "eegmmidb": ["001-020", "021-040", "041-060", "061-080", "081-100", "101-109"],
    }
    selected = list(chunks) if requested == "all" else [requested]
    if any(name not in chunks for name in selected):
        raise SystemExit(f"unknown dataset: {requested}")
    include = []
    for name in selected:
        values = [requested_chunk] if requested_chunk else chunks[name]
        for value in values:
            if value not in chunks[name]:
                raise SystemExit(f"invalid chunk {value} for {name}")
            include.append({"dataset": name, "chunk": value})
    print("matrix=" + json.dumps({"include": include}, separators=(",", ":")))
    print("strict=" + ("true" if requested == "all" and not requested_chunk else "false"))


if __name__ == "__main__":
    main()
