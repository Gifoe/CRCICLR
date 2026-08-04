from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class AccessPhase(str, Enum):
    ADAPT_PHASE = "ADAPT_PHASE"
    PROBE_PHASE = "PROBE_PHASE"
    FROZEN_DECISION = "FROZEN_DECISION"
    FUTURE_EVALUATION = "FUTURE_EVALUATION"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


@dataclass
class EpisodeAccessController:
    subject_id: str
    config_hash: str
    phase: AccessPhase = AccessPhase.ADAPT_PHASE
    action_state_hashes: dict[str, str] = field(default_factory=dict)
    decision_path: Path | None = None
    decision_hash: str | None = None

    def access_adapt(self, inputs: Any) -> Any:
        if self.phase is not AccessPhase.ADAPT_PHASE:
            raise PermissionError(f"adapt input unavailable during {self.phase.value}")
        return inputs

    def begin_probe(self, action_state_hashes: dict[str, str]) -> None:
        if self.phase is not AccessPhase.ADAPT_PHASE:
            raise RuntimeError("probe transition requires ADAPT_PHASE")
        if not action_state_hashes or any(not value for value in action_state_hashes.values()):
            raise ValueError("all action states must be frozen and hashed before Probe")
        self.action_state_hashes = dict(sorted(action_state_hashes.items()))
        self.phase = AccessPhase.PROBE_PHASE

    def access_probe(self, inputs: Any, *, labels: Any | None = None) -> Any:
        if self.phase is not AccessPhase.PROBE_PHASE:
            raise PermissionError(f"probe input unavailable during {self.phase.value}")
        if labels is not None:
            raise PermissionError("Probe labels are never accessible")
        return inputs

    def freeze_decision(self, decision: dict[str, object], path: str | Path) -> str:
        if self.phase is not AccessPhase.PROBE_PHASE:
            raise RuntimeError("decision freeze requires PROBE_PHASE")
        required = {"subject_id", "selected_action", "action_state_hash", "config_hash", "lambda_index"}
        missing = required - decision.keys()
        if missing:
            raise ValueError(f"decision missing fields: {sorted(missing)}")
        if decision["subject_id"] != self.subject_id or decision["config_hash"] != self.config_hash:
            raise ValueError("decision subject/config mismatch")
        action = str(decision["selected_action"])
        if self.action_state_hashes.get(action) != decision["action_state_hash"]:
            raise ValueError("selected action state hash mismatch")
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        self.decision_path = target; self.decision_hash = file_sha256(target)
        freeze = target.with_suffix(target.suffix + ".freeze.json")
        payload = {"subject_id": self.subject_id, "decision_sha256": self.decision_hash,
                   "config_hash": self.config_hash, "future_opened": False}
        part = freeze.with_suffix(freeze.suffix + ".part")
        part.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(part, freeze)
        self.phase = AccessPhase.FROZEN_DECISION
        return self.decision_hash

    def access_future(self, inputs: Any, labels: Any) -> tuple[Any, Any]:
        if self.phase is not AccessPhase.FROZEN_DECISION or self.decision_path is None or self.decision_hash is None:
            raise PermissionError("Future is inaccessible before a frozen decision")
        if file_sha256(self.decision_path) != self.decision_hash:
            raise RuntimeError("decision hash changed after freeze")
        freeze = self.decision_path.with_suffix(self.decision_path.suffix + ".freeze.json")
        payload = json.loads(freeze.read_text(encoding="utf-8"))
        if payload.get("decision_sha256") != self.decision_hash or payload.get("config_hash") != self.config_hash:
            raise RuntimeError("freeze manifest hash/config mismatch")
        payload["future_opened"] = True
        part = freeze.with_suffix(freeze.suffix + ".part")
        part.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(part, freeze)
        self.phase = AccessPhase.FUTURE_EVALUATION
        return inputs, labels
