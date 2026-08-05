from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _canonical_hash(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode()).hexdigest()


@dataclass
class ContextualAccessController:
    subject_id: str
    dataset: str
    seed: int
    role: str
    phase: str = "CONTEXT_PHASE"
    _decision: dict[str, Any] | None = None

    def read_context(self, context: Any) -> Any:
        if self.phase != "CONTEXT_PHASE":
            raise RuntimeError("context is readable only during CONTEXT_PHASE")
        return context

    def freeze_decision(self, payload: dict[str, Any], path: str | Path) -> dict[str, Any]:
        if self.phase != "CONTEXT_PHASE":
            raise RuntimeError("decision can be frozen exactly once")
        required = {
            "subject_id", "dataset", "seed", "role", "selected_branch", "alpha",
            "delta", "context_hash", "source_model_hash", "method_config_hash",
            "feature_hash", "certified_index",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"decision is missing {sorted(missing)}")
        if str(payload["subject_id"]) != self.subject_id or str(payload["dataset"]) != self.dataset:
            raise ValueError("decision identity mismatch")
        frozen = dict(payload)
        frozen["decision_hash"] = _canonical_hash(frozen)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_text(json.dumps(frozen, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
        self._decision = frozen
        self.phase = "FROZEN_DECISION"
        return frozen

    def open_future(self, future: Any, path: str | Path) -> Any:
        if self.phase != "FROZEN_DECISION" or self._decision is None:
            raise RuntimeError("Future cannot be opened before a frozen decision")
        disk = json.loads(Path(path).read_text(encoding="utf-8"))
        given = disk.pop("decision_hash", None)
        if given != _canonical_hash(disk):
            raise RuntimeError("frozen decision hash validation failed")
        if given != self._decision["decision_hash"]:
            raise RuntimeError("frozen decision changed after freezing")
        self.phase = "FUTURE_EVALUATION"
        return future
