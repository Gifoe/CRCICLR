#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


ROOT = Path("/root/autodl-tmp/hsc_tta_eeg")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    args = parser.parse_args()
    path = ROOT / "state" / "full_gpu_experiment_state.json"
    payload = json.loads(path.read_text()) if path.exists() else {}
    payload.update({"current_stage": args.stage,
        "git_sha": subprocess.check_output(["git", "-C", str(ROOT / "repo"), "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint_status": "verified", "last_update_time": datetime.now(timezone.utc).isoformat()})
    payload.setdefault("completed_subjects", {}); payload.setdefault("failed_subjects", [])
    payload.setdefault("completed_seeds", {}); payload.setdefault("config_hashes", {})
    payload.setdefault("freeze_status", "not_frozen"); payload.setdefault("final_test_access_status", "locked")
    if args.stage == "GPU-11-frozen": payload["freeze_status"] = "methods_frozen"
    if args.stage == "GPU-13-decisions-frozen": payload.update({"freeze_status": "decisions_frozen", "final_test_access_status": "unlocked_for_offline_outcome"})
    if args.stage == "GPU-16-complete": payload["final_test_access_status"] = "completed"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
