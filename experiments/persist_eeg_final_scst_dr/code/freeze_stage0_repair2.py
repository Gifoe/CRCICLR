from __future__ import annotations

import time

import common as c


def main() -> None:
    repair1 = c.read_json(c.RESULTS / "STAGE0_REPAIR1_VALIDATION.json")
    if repair1.get("pass") is not True or repair1.get("stage0_terminal") != "TRANSPORT_OFF_MANIFOLD":
        raise RuntimeError("Repair 2 requires the validated Repair-1 off-manifold terminal")
    if any((c.RUNTIME / "stage0_repair2_units").glob("*/fold-*/UNIT_COMPLETE.json")):
        raise RuntimeError("Repair-2 unit exists before the protocol freeze")
    files = [
        c.EXP / "protocol" / "STAGE0_REPAIR2_PROTOCOL_LOCK.json",
        c.EXP / "STAGE0_REPAIR2_PROTOCOL.md",
        c.EXP / "REPAIR_LOG.md",
        c.EXP / "ITERATION_LEDGER.md",
        c.EXP / "protocol" / "DATA_ACCESS_AUDIT.json",
        c.EXP / "protocol" / "SEALED_RESOURCE_AUDIT.json",
        c.RESULTS / "STAGE0_REPAIR1_FINAL_RESULT.json",
        c.RESULTS / "STAGE0_REPAIR1_VALIDATION.json",
        c.RESULTS / "STAGE0_REPAIR1_LAYER_SUMMARY.csv",
        c.HERE / "common.py",
        c.HERE / "run_stage0.py",
        c.HERE / "analyze_stage0.py",
        c.HERE / "run_stage0_repair2.py",
        c.HERE / "analyze_stage0_repair2.py",
        c.HERE / "validate_stage0_repair2.py",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"Repair-2 freeze missing files: {missing}")
    lock = c.read_json(c.EXP / "protocol" / "STAGE0_REPAIR2_PROTOCOL_LOCK.json")
    if lock.get("parent_commit") != "f5cb605954e624629d8a3849d33a732f29ef54ee":
        raise RuntimeError("Repair-2 scientific parent differs from the validated closure tip")
    c.write_json(
        c.EXP / "protocol" / "PRE_STAGE0_REPAIR2_FREEZE.json",
        {
            "schema": "SCST_DR_PRE_STAGE0_REPAIR2_FREEZE_V1",
            "pass": True,
            "git_commit_at_freeze": c.git_head(),
            "frozen_before_repair2_outcomes": True,
            "future_or_outer_performance_accessed": False,
            "file_sha256": {
                str(path.relative_to(c.EXP)).replace("\\", "/"): c.sha256(path)
                for path in files
            },
            "created_unix": time.time(),
        },
    )
    print("SCST_DR_PRE_STAGE0_REPAIR2_FREEZE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
