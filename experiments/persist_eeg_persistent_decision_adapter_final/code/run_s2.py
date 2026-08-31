"""Fail-closed WBCIC S2 one-shot runner.

This module deliberately checks the source gate and all locks before it even
constructs a path below the future-session resource.  On the current source
result it exits with the preregistered source-negative terminal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pda_core as c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    gate_path = c.RESULTS / "SOURCE_GATE.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    if gate.get("source_gate_pass") is not True:
        print("PERSIST_PDA_SOURCE_NOT_SUPPORTED")
        return
    c.assert_future_resource_locked(c.EXP / "protocol" / "DATA_ACCESS_LOCK.json")
    c.assert_future_resource_locked(c.EXP / "protocol" / "PERSIST_PDA_SOURCE_LOCK.json")
    c.assert_future_resource_locked(c.EXP / "protocol" / "PERSIST_PDA_FINAL_METHOD_LOCK.json")
    # The implementation is intentionally not generalized to silently open a
    # future archive.  A future run must provide the explicitly authorized
    # path and record it in the lock before this point.
    raise RuntimeError("S2 path is intentionally sealed in this source-only build")


if __name__ == "__main__":
    main()
