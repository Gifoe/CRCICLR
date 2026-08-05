from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


STATES = (
    "INITIALIZED", "INPUT_AUDIT_COMPLETE", "V51_PROTOCOL_FROZEN",
    "RAW_DIAGNOSTIC_COMPLETE", "CALIBRATION_DIAGNOSTIC_COMPLETE",
    "OUTLIER_ANALYSIS_COMPLETE", "V51_DECISION_COMPLETE",
    "DELIVERY_COMPLETE", "STOPPED",
)


def hash_manifest(root: Path) -> str:
    entries = []
    if root.exists():
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            entries.append((str(path.relative_to(root)), path.stat().st_size))
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def transition(path: Path, state: str, *, git_commit: str, input_hashes: dict[str, str],
               config_hash: str, cohort_hash: str, source_result_hash: str, output_root: Path) -> None:
    if state not in STATES:
        raise ValueError(state)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        previous = payload["state"]
        expected = STATES[min(STATES.index(previous) + 1, len(STATES) - 1)]
        if state != expected:
            raise RuntimeError(f"invalid state transition {previous} -> {state}; expected {expected}")
        history = payload["history"]
    else:
        if state != STATES[0]:
            raise RuntimeError("state machine must start at INITIALIZED")
        previous = None; history = []
    record = {
        "state": state, "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit, "input_hashes": input_hashes,
        "config_hash": config_hash, "cohort_hash": cohort_hash,
        "source_result_hash": source_result_hash, "previous_state": previous,
        "output_manifest_hash": hash_manifest(output_root),
        "formal_calibration_opened": False, "internal_final_opened": False, "cap_opened": False,
    }
    history.append(record); payload = {**record, "history": history}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

