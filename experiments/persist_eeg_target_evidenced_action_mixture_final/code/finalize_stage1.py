"""Write the immutable TEA-EEG source-stage lock.

This finalizer only records hashes and the already-computed source decision. It
does not open a dataset, train a model, or change the scientific recipe.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
CODE = EXP / "code"
PROTOCOL = EXP / "protocol"
RESULTS = EXP / "results"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=EXP.parents[1], text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return "UNAVAILABLE"


def _load(name: str, default: Any = None) -> Any:
    path = RESULTS / name
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    stats = _load("STATISTICS.json", {}) or {}
    split = _load("../protocol/AUTONOMOUS_RESEARCH_SPLIT.json", {}) or {}
    source_lock = {
        "status": "SOURCE_GATE_FAILED_NO_METHOD_FREEZE" if not stats.get("source_gate", {}).get("pass", False) else "SOURCE_GATE_PASS_PENDING_S2",
        "terminal": stats.get("terminal", "TEA_IMPLEMENTATION_INVALID"),
        "source_gate": stats.get("source_gate", {}),
        "selected_recipe": stats.get("selected_recipe"),
        "scientific_change_after_s2": False,
        "wbcic_s2_opened": False,
        "OUTER_TEST_USED": False,
    }
    (PROTOCOL / "TEA_SOURCE_LOCK.json").write_text(json.dumps(source_lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    final_lock = {
        "status": "NOT_AUTHORIZED_SOURCE_GATE_FAILED" if not stats.get("source_gate", {}).get("pass", False) else "SOURCE_METHOD_FROZEN_PENDING_S2",
        "method_frozen": bool(stats.get("source_gate", {}).get("pass", False)),
        "git_sha": _git(["rev-parse", "HEAD"]),
        "tree_sha": _git(["rev-parse", "HEAD^{tree}"]),
        "code_hashes": {p.name: _sha256(p) for p in sorted(CODE.glob("*.py"))},
        "action_definitions": {"primary": ["KEEP", "AMPLIFY", "GEOMETRY"], "diagnostic": ["KEEP", "AMPLIFY", "GEOMETRY", "ERASE"]},
        "target_blocks": {"count": 5, "rule": "deterministic temporal factorization; held block excluded from context"},
        "input_feature_policy": {"labels": False, "effects": False, "subject_id": False, "fold_id": False, "context_encoder": "DeepSets phi/rho, max width 64, max 2 layers"},
        "regret": "CE_KEEP - CE_action; conservative gain mu - kappa*sigma",
        "loss_coefficients": {"lambda_R": 1.0, "lambda_rank": 0.5, "lambda_safe": 1.0, "lambda_action": 0.05},
        "source_results": stats,
        "split_assignment_hash": split.get("assignment_hash"),
        "bootstrap_unit": "biological subject",
        "wbcic_s2_opened": False,
        "OUTER_TEST_USED": False,
    }
    (PROTOCOL / "TEA_FINAL_METHOD_LOCK.json").write_text(json.dumps(final_lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outer_lock = {
        "status": "NOT_AUTHORIZED_CROSS_BACKBONE_NOT_TESTED",
        "outer_test_used": False,
        "sealed_resources_opened": False,
        "OUTER_TEST_USED": False,
    }
    (PROTOCOL / "TEA_OUTER_CONFIRMATION_LOCK.json").write_text(json.dumps(outer_lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_terminal": stats.get("terminal"), "method_frozen": final_lock["method_frozen"], "git_sha": final_lock["git_sha"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
