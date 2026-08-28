from __future__ import annotations

import time

import common as c


def main() -> None:
    c.ensure_dirs()
    protocol = c.protocol()
    data_audit = c.EXP / "protocol" / "DATA_ACCESS_AUDIT.json"
    sealed_audit = c.EXP / "protocol" / "SEALED_RESOURCE_AUDIT.json"
    if not data_audit.is_file() or not sealed_audit.is_file():
        raise RuntimeError("data/sealed audits must exist before Stage-0 freeze")
    for path in c.RESULTS.glob("*FIDELITY*.csv"):
        raise RuntimeError(f"transport outcome already exists before freeze: {path}")
    files = [
        c.PROTOCOL,
        data_audit,
        sealed_audit,
        c.HERE / "common.py",
        c.HERE / "run_stage0.py",
        c.HERE / "analyze_stage0.py",
        c.HERE / "validate_stage0.py",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot freeze missing files: {missing}")
    payload = {
        "schema": "SCST_DR_PRE_STAGE0_FREEZE_V1",
        "pass": True,
        "git_commit_at_freeze": c.git_head(),
        "parent_commit": protocol["parent_commit"],
        "frozen_before_first_transport_metric": True,
        "future_session_or_outer_outcome_accessed": False,
        "file_sha256": {str(path.relative_to(c.EXP)).replace("\\", "/"): c.sha256(path) for path in files},
        "created_unix": time.time(),
    }
    c.write_json(c.EXP / "protocol" / "PRE_STAGE0_FREEZE.json", payload)
    print("SCST_DR_PRE_STAGE0_FREEZE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
