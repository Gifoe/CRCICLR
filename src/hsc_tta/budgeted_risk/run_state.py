from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any


ORDER = [
    "INITIALIZED", "AUDIT_COMPLETE", "COHORTS_VERIFIED",
    "SOURCE_MODELS_VERIFIED", "CACHE_COMPLETE", "STAGE0_PROTOCOL_FROZEN",
    "STAGE0_FULL_CONTEXT_COMPLETE", "STAGE0_BUDGET_COMPLETE",
    "STAGE0_ACQUISITION_COMPLETE", "STAGE0_DECISION_COMPLETE",
    "FULL_METHOD_FROZEN", "FORMAL_CALIBRATION_COMPLETE",
    "INTERNAL_FINAL_COMPLETE", "CAP_COMPLETE", "DELIVERY_COMPLETE",
]
TERMINAL = "STOPPED_NO_GO"
REQUIRED_HASHES = (
    "git_commit", "config_hash", "cohort_hash", "episode_hash",
    "source_model_hash", "feature_schema_hash", "output_manifest_hash",
)


class RunState:
    """Append-only experiment state with a single explicit no-go escape."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def read(self) -> dict[str, Any] | None:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else None

    def advance(self, state: str, **metadata: Any) -> dict[str, Any]:
        previous = self.read()
        prior = previous["state"] if previous else None
        if state == TERMINAL:
            if prior not in {"SOURCE_MODELS_VERIFIED", "STAGE0_FULL_CONTEXT_COMPLETE",
                             "STAGE0_BUDGET_COMPLETE", "STAGE0_DECISION_COMPLETE",
                             "DELIVERY_COMPLETE"}:
                raise RuntimeError(f"illegal no-go transition {prior} -> {state}")
        else:
            expected = ORDER[0] if prior is None else (
                ORDER[ORDER.index(prior) + 1] if prior in ORDER and ORDER.index(prior) + 1 < len(ORDER) else None
            )
            if state != expected:
                raise RuntimeError(f"illegal state transition {prior} -> {state}; expected {expected}")
        missing = [key for key in REQUIRED_HASHES if key not in metadata]
        if missing:
            raise ValueError(f"missing state metadata: {missing}")
        entry = {
            "state": state,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "previous_state": prior,
            **metadata,
        }
        history = list(previous.get("history", [])) if previous else []
        if previous:
            history.append({key: value for key, value in previous.items() if key != "history"})
        entry["history"] = history
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.part")
        temporary.write_text(json.dumps(entry, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)
        return entry

