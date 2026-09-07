from __future__ import annotations

import time

import common as c


ALLOWED_CHANGED = {
    "ITERATION_LEDGER.md",
    "REPAIR_LOG.md",
    "code/analyze_stage0_repair2.py",
    "code/validate_stage0_repair2.py",
}


def main() -> None:
    original_path = c.EXP / "protocol" / "PRE_STAGE0_REPAIR2_FREEZE.json"
    original = c.read_json(original_path)
    if original.get("pass") is not True or original.get("frozen_before_repair2_outcomes") is not True:
        raise RuntimeError("missing valid original Repair-2 freeze")
    for relative, expected in original.get("file_sha256", {}).items():
        path = c.EXP / relative
        if relative in ALLOWED_CHANGED:
            continue
        if not path.is_file() or c.sha256(path) != expected:
            raise RuntimeError(f"non-E1 frozen input changed: {relative}")

    units = sorted((c.RUNTIME / "stage0_repair2_units").glob("*/fold-*/UNIT_COMPLETE.json"))
    if len(units) != 20 or any(c.read_json(path).get("pass") is not True for path in units):
        raise RuntimeError("E1 refreeze requires 20 intact Repair-2 raw units")
    result_names = [
        "STAGE0_REPAIR2_LAYER_SUMMARY.csv",
        "STAGE0_REPAIR2_ALPHA_DISTRIBUTION.csv",
        "STAGE0_REPAIR2_SUBJECT_FIDELITY.csv",
        "STAGE0_REPAIR2_CLASS_FIDELITY.csv",
        "STAGE0_REPAIR2_MANIFOLD_VALIDITY.csv",
        "STAGE0_REPAIR2_STATISTICS.json",
        "STAGE0_REPAIR2_FINAL_RESULT.json",
    ]
    missing = [name for name in result_names if not (c.RESULTS / name).is_file()]
    if missing:
        raise RuntimeError(f"E1 cannot preserve missing pre-fix results: {missing}")
    changed_paths = {relative: c.EXP / relative for relative in ALLOWED_CHANGED}
    if any(not path.is_file() for path in changed_paths.values()):
        raise RuntimeError("E1 changed file missing")
    c.write_json(
        c.EXP / "protocol" / "STAGE0_REPAIR2_E1_ENGINEERING_FREEZE.json",
        {
            "schema": "SCST_DR_STAGE0_REPAIR2_E1_ENGINEERING_FREEZE_V1",
            "pass": True,
            "scientific_change": False,
            "bug_scope": "figure label access only",
            "original_freeze_sha256": c.sha256(original_path),
            "allowed_changed_files": sorted(ALLOWED_CHANGED),
            "changed_file_sha256": {relative: c.sha256(path) for relative, path in changed_paths.items()},
            "pre_fix_result_sha256": {name: c.sha256(c.RESULTS / name) for name in result_names},
            "raw_unit_complete_sha256": {
                str(path.relative_to(c.RUNTIME)).replace("\\", "/"): c.sha256(path) for path in units
            },
            "created_unix": time.time(),
        },
    )
    print("SCST_DR_STAGE0_REPAIR2_E1_ENGINEERING_FREEZE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
