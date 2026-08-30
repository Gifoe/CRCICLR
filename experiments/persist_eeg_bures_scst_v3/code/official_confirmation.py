"""Locked model-specific confirmation stage.

The source V3 implementation is representation-cache based.  This entry point
therefore refuses to substitute the ATCNet-CleanRoom cache for
ATCNet-Official/EEGNeX.  When the source gate fails (the expected stopping
condition), it emits explicit not-run schemas and never touches WBCIC S3.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

import common as c


def _placeholder(path, columns, status):
    frame = pd.DataFrame(columns=columns + ["status"]); frame.loc[0, "status"] = status; c.write_csv(path, frame)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--model", choices=("ATCNet-Official", "EEGNeX", "all"), default="all"); args = parser.parse_args(); c.ensure_dirs()
    gate = c.read_json(c.RESULTS / "SOURCE_GATE.json") if (c.RESULTS / "SOURCE_GATE.json").is_file() else {"source_gate_pass": False, "terminal_if_stop": "BURES_SCST_SOURCE_GATE_FAILED"}
    if not gate.get("source_gate_pass", False):
        status = str(gate.get("terminal_if_stop", "BURES_SCST_SOURCE_GATE_FAILED")) + ":NOT_RUN"
        _placeholder(c.RESULTS / "ATCNET_OFFICIAL_PER_SUBJECT.csv", ["model", "method", "fold", "seed", "subject_id", "BA", "macro_F1"], status)
        _placeholder(c.RESULTS / "ATCNET_OFFICIAL_PER_FOLD.csv", ["model", "method", "fold", "seed", "BA", "macro_F1"], status)
        _placeholder(c.RESULTS / "EEGNEX_PER_SUBJECT.csv", ["model", "method", "fold", "seed", "subject_id", "BA", "macro_F1"], status)
        _placeholder(c.RESULTS / "EEGNEX_PER_FOLD.csv", ["model", "method", "fold", "seed", "BA", "macro_F1"], status)
        c.write_json(c.RESULTS / "CONFIRMATION_STATUS.json", {"status": status, "s3_opened": False, "outer_or_sealed_opened": False})
        print(status); return
    # A positive source gate authorises a separate one-shot implementation,
    # but it does not permit silently using another architecture's cache.
    raise RuntimeError("MODEL_SPECIFIC_CACHE_NOT_AVAILABLE_FOR_LOCKED_CONFIRMATION")


if __name__ == "__main__": main()
