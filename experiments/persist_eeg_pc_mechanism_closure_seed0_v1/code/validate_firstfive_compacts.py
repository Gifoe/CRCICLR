#!/usr/bin/env python3
"""Strict local integrity and scope audit for the first five Phase-2 previews.

This validates packaging/provenance and frozen-scope invariants. It is not a
scientific endorsement of a five-fold or cross-task conclusion.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "first_five"
EXPECTED = {
    "FINAL_P_BACKTRACE.csv": 624,
    "MECHANISM_REGIMES.csv": 68,
    "PC_DIRECTIONAL_INTERACTION.csv": 3400,
    "PC_DIRECTIONAL_RANDOM_CONTROL.csv": 100,
    "PC_LONG_RANGE_MEDIATION.csv": 204,
    "PC_MEDIATION_TRANSITIONS.csv": 272,
    "SUBJECT_SESSION_MECHANISM_DRIFT.csv": 34,
    "SUBJECT_SESSION_MECHANISM_SIGNATURE.csv": 68,
    "TRAINING_PC_MECHANISM.csv": 312,
    "TRAINING_TRAJECTORY_AUDIT.csv": 6,
}
FROZEN = {
    "analysis_lock_sha256": "ac12d33f7d30b2c846ec4c4fb07f8513bf25e372943a767b134883bf095a7c88",
    "projector_amendment_sha256": "aec1e604d38f1901bd9eca7e072e30a3d496a28061a726ebbe63133259b0e474",
    "upstream_gate_sha256": "7cdc2c0fe5b5e32dcb98b917a344d0eaaf0cfca75eec3a504d56650f2223df5a",
    "upstream_implementation_sha256": "4aab984969f4ba43128ed738a38ac138d94ee1c9b1252662af91b0c44e3f8eee",
    "upstream_protocol_sha256": "3ddb5006bf7159b00a0c80fed31856849ccdaa69ecf5060b0161c8673327e15c",
}
SOURCE_KEYS = {
    "source_backtrace_sha256": "FINAL_P_BACKTRACE_V1.json",
    "source_directional_manifest_sha256": "DIRECTIONAL_V4.json",
    "source_mediation_manifest_sha256": "MEDIATION_V2.json",
    "source_replay_audit_sha256": "replay_v2_REPLAY_AUDIT.json",
    "source_subject_v2_sha256": "SUBJECT_SESSION_AUDIT_V2.json",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-audit-root", type=Path, default=None,
                        help="Optional separately transferred source manifests to re-hash")
    args = parser.parse_args()
    errors: list[str] = []
    cells = []
    for fold in range(5):
        folder = OUT / f"eegnet_openbmi_mi_fold{fold}_seed0"
        p = json.loads((folder / "PROVENANCE.json").read_text(encoding="utf-8"))
        check(p.get("fold") == fold and p.get("seed") == 0, f"fold {fold}: identity mismatch", errors)
        check(p.get("model") == "EEGNet" and p.get("task") == "OpenBMI_MI", f"fold {fold}: scope mismatch", errors)
        check(p.get("status") in {"CELL_COMPACT_PREVIEW_COMPLETE_NOT_20_CELL_RESULT", "FIRST_CELL_COMPACT_PREVIEW_COMPLETE_NOT_20_CELL_RESULT"}, f"fold {fold}: status mismatch", errors)
        check(p.get("final_heldout_accessed") is False, f"fold {fold}: heldout flag not false", errors)
        for key, value in FROZEN.items():
            check(p.get(key) == value, f"fold {fold}: frozen {key} mismatch", errors)
        check(p.get("trajectory_provenance") == "RETRAINED_REPLICA_TRAJECTORY", f"fold {fold}: replay provenance mismatch", errors)
        check(p.get("outer_development_accessed") is True, f"fold {fold}: outer evaluation not recorded", errors)
        report = folder / "MECHANISM_CLOSURE_REPORT.md"
        check(sha(report) == p.get("report_sha256"), f"fold {fold}: report hash mismatch", errors)
        report_text = report.read_text(encoding="utf-8").lower()
        for phrase in ("not the 20-cell", "not the historical training path", "not a jacobian", "not prospective prediction or biological causation"):
            check(phrase in report_text, f"fold {fold}: report lacks limitation phrase {phrase!r}", errors)
        for name, n in EXPECTED.items():
            path = folder / name
            check(path.is_file(), f"fold {fold}: missing {name}", errors)
            if not path.is_file():
                continue
            check(sha(path) == p.get("output_sha256", {}).get(name), f"fold {fold}: {name} hash mismatch", errors)
            headers, rows = read_csv(path)
            check(len(rows) == n and p.get("output_rows", {}).get(name) == n, f"fold {fold}: {name} row count mismatch", errors)
            for row_i, row in enumerate(rows, start=2):
                for key, value in row.items():
                    if value is None or value == "":
                        continue
                    try:
                        number = float(value)
                    except (ValueError, TypeError):
                        continue
                    check(math.isfinite(number), f"fold {fold}: non-finite {name}:{row_i}:{key}", errors)
            # All output rows must retain this exact upstream identity.
            for field, value in (("model", "EEGNet"), ("task", "OpenBMI_MI"), ("fold", str(fold)), ("seed", "0")):
                if field in headers:
                    check(all(r.get(field) == value for r in rows), f"fold {fold}: {name} has mixed {field}", errors)
            if name in ("PC_MEDIATION_TRANSITIONS.csv", "PC_LONG_RANGE_MEDIATION.csv", "SUBJECT_SESSION_MECHANISM_SIGNATURE.csv", "SUBJECT_SESSION_MECHANISM_DRIFT.csv", "TRAINING_PC_MECHANISM.csv", "MECHANISM_REGIMES.csv") and "split" in headers:
                splits = {r.get("split") for r in rows}
                check(splits <= {"TRAIN", "OUTER_DEVELOPMENT"}, f"fold {fold}: unexpected split in {name}", errors)
            if name == "PC_DIRECTIONAL_RANDOM_CONTROL.csv":
                check(len(rows) == 100, f"fold {fold}: expected 100 controls", errors)
                stages = {}
                for row in rows:
                    stages.setdefault(row.get("stage"), set()).add(int(row.get("draw_id", row.get("draw", -1))))
                check(len(stages) == 5 and all(draws == set(range(20)) for draws in stages.values()), f"fold {fold}: controls are not 20 contiguous draws per stage", errors)
            if name == "TRAINING_TRAJECTORY_AUDIT.csv":
                check({r.get("trajectory_provenance") for r in rows} == {"RETRAINED_REPLICA_TRAJECTORY"}, f"fold {fold}: trajectory mislabelled", errors)
                check(all(r.get("outer_used_for_selection", "").lower() == "false" for r in rows), f"fold {fold}: outer selection flag not false", errors)
        # Match each compact provenance to its separately transferred source audit.
        srcdir = args.source_audit_root / f"fold{fold}" if args.source_audit_root else None
        source_hashes = {}
        for key, filename in SOURCE_KEYS.items():
            check(bool(p.get(key)), f"fold {fold}: missing provenance key {key}", errors)
            if srcdir:
                path = srcdir / filename
                check(path.is_file(), f"fold {fold}: missing source audit {filename}", errors)
                if path.is_file():
                    source_hashes[key] = sha(path)
                    check(source_hashes[key] == p.get(key), f"fold {fold}: source audit hash mismatch for {filename}", errors)
        replay_path = srcdir / SOURCE_KEYS["source_replay_audit_sha256"] if srcdir else None
        if replay_path and replay_path.is_file():
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            check(replay.get("status") == "REPLAY_COMPLETE", f"fold {fold}: replay not complete", errors)
            check(replay.get("trajectory_provenance") == "RETRAINED_REPLICA_TRAJECTORY", f"fold {fold}: replay source mislabeled", errors)
        cells.append({"fold": fold, "provenance_sha256": sha(folder / "PROVENANCE.json"), "report_sha256": sha(report), "source_audit_sha256_rechecked": source_hashes, "output_files": len(EXPECTED), "output_rows_each": EXPECTED})

    result = {
        "status": "PASS" if not errors else "FAIL",
        "scope": "EEGNet/OpenBMI_MI/seed0/folds0-4 compact previews only",
        "cell_count": len(cells),
        "not_a_20_cell_conclusion": True,
        "final_heldout_accessed": False,
        "frozen": FROZEN,
        "cells": cells,
        "errors": errors,
    }
    out = OUT / "VALIDATION_SUMMARY.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("FAIL: " + " | ".join(errors))
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    lines = [f"{sha(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    sums_path = OUT / "SHA256SUMS.txt"
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "cells": len(cells), "validation_summary_sha256": sha(out)}, indent=2))


if __name__ == "__main__":
    main()
