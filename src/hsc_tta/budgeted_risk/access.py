from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass
class BudgetedAccessController:
    dataset: str
    subject_id: str
    seed: int
    role: str
    phase: str = "UNLABELED_CONTEXT"
    _query_hash: str | None = None
    _decision_hash: str | None = None

    def begin_queries(self) -> None:
        if self.phase != "UNLABELED_CONTEXT": raise RuntimeError("invalid query transition")
        self.phase = "QUERY_PHASE"

    def freeze_queries(self, query_hash: str) -> None:
        if self.phase != "QUERY_PHASE": raise RuntimeError("queries are not open")
        self._query_hash = str(query_hash); self.phase = "QUERY_FROZEN"

    def freeze_decision(self, payload: dict[str, Any], path: str | Path) -> dict[str, Any]:
        if self.phase != "QUERY_FROZEN" or not self._query_hash:
            raise RuntimeError("query transcript must be frozen first")
        required={"dataset","subject_id","seed","role","budget","strategy","alpha","delta","query_hash","source_model_hash","episode_hash","certified_index"}
        missing=required.difference(payload)
        if missing: raise ValueError(f"missing decision fields: {sorted(missing)}")
        if payload["query_hash"] != self._query_hash: raise ValueError("query hash mismatch")
        decision=dict(payload);decision["decision_hash"]=_hash(decision)
        target=Path(path);target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_suffix(target.suffix+".part")
        temporary.write_text(json.dumps(decision,indent=2,sort_keys=True),encoding="utf-8");temporary.replace(target)
        self._decision_hash=decision["decision_hash"];self.phase="RISK_DECISION_FROZEN"
        return decision

    def open_future(self, future: Any, decision_path: str | Path) -> Any:
        if self.phase != "RISK_DECISION_FROZEN" or not self._decision_hash:
            raise RuntimeError("Future is closed before risk decision freeze")
        payload=json.loads(Path(decision_path).read_text(encoding="utf-8"));given=payload.pop("decision_hash",None)
        if given != _hash(payload) or given != self._decision_hash: raise RuntimeError("decision hash validation failed")
        self.phase="FUTURE_EVALUATION";return future
