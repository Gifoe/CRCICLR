from __future__ import annotations

import time

import common as c


def main() -> None:
    v0_validation = c.read_json(c.RESULTS / "STAGE0_VALIDATION.json")
    if v0_validation.get("pass") is not True or v0_validation.get("stage0_terminal") != "TRANSPORT_NOT_SUBJECT_FAITHFUL":
        raise RuntimeError("repair 1 requires the validated V0 subject-fidelity failure")
    if any((c.RUNTIME / "stage0_repair1_units").glob("*/fold-*/*/UNIT_COMPLETE.json")):
        raise RuntimeError("repair-1 metric exists before repair freeze")
    files = [
        c.EXP / "protocol" / "STAGE0_REPAIR1_PROTOCOL_LOCK.json",
        c.RESULTS / "STAGE0_LAYER_SUMMARY.csv",
        c.RESULTS / "STAGE0_FINAL_RESULT.json",
        c.RESULTS / "STAGE0_VALIDATION.json",
        c.HERE / "common.py",
        c.HERE / "run_stage0_repair1.py",
        c.HERE / "analyze_stage0_repair1.py",
        c.HERE / "validate_stage0_repair1.py",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"repair freeze missing files: {missing}")
    c.write_json(c.EXP / "protocol" / "PRE_STAGE0_REPAIR1_FREEZE.json", {
        "schema": "SCST_DR_PRE_STAGE0_REPAIR1_FREEZE_V1",
        "pass": True,
        "git_commit_at_freeze": c.git_head(),
        "frozen_before_repair1_metrics": True,
        "v0_terminal": v0_validation["stage0_terminal"],
        "future_or_outer_performance_accessed": False,
        "file_sha256": {str(path.relative_to(c.EXP)).replace("\\", "/"): c.sha256(path) for path in files},
        "created_unix": time.time(),
    })
    print("SCST_DR_PRE_STAGE0_REPAIR1_FREEZE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
