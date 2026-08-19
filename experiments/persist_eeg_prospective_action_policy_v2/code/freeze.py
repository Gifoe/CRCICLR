from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import (
    EXPLORATION,
    EXPERIMENT_ROOT,
    FREEZE,
    PROTOCOL,
    SEED,
    canonical_hash,
    ensure_directories,
    sha256_file,
    write_json,
)


def _code_hashes() -> dict[str, str]:
    code = EXPERIMENT_ROOT / "code"
    return {path.name: sha256_file(path) for path in sorted(code.glob("*.py"))}


def freeze_candidates() -> dict[str, Any]:
    ensure_directories()
    decision_path = EXPLORATION / "EXPLORATION_DECISION.json"
    split_path = PROTOCOL / "AUTONOMOUS_RESEARCH_SPLIT.json"
    if not decision_path.exists() or not split_path.exists():
        raise FileNotFoundError("Exploration and split artifacts are required before freezing")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if decision["status"] != "STRONG_CANDIDATE_FOUND":
        raise RuntimeError("No exploration candidate is authorized for a holdout opening")
    expected = ["I003_CROSS_RUN_FULL", "I003_CROSS_RUN_PROTECTED_SAFE"]
    if decision["selected_candidates_for_freeze"] != expected:
        raise RuntimeError("Unexpected candidate selection")
    policy_specs = [
        {
            "policy_id": "I003_CROSS_RUN_FULL",
            "feature_rule": "majority class from all available frozen runs except the target run for the same manifest sample",
            "intervention_rule": "intervene iff leave-target-run majority disagrees with target KEEP prediction",
            "action_priority": ["AMPLIFY", "GEOMETRY", "ERASE"],
            "default_action": "KEEP",
            "threshold": 0.5,
            "calibration": "none; deterministic odd-vote majority",
            "checkpoint": None,
            "protected_constraint": "ERASE remains a disclosed protected-space risk and is separately ablated",
        },
        {
            "policy_id": "I003_CROSS_RUN_PROTECTED_SAFE",
            "feature_rule": "majority class from all available frozen runs except the target run for the same manifest sample",
            "intervention_rule": "intervene iff leave-target-run majority disagrees with target KEEP prediction",
            "action_priority": ["AMPLIFY", "GEOMETRY"],
            "default_action": "KEEP",
            "threshold": 0.5,
            "calibration": "none; deterministic odd-vote majority",
            "checkpoint": None,
            "protected_constraint": "full ERASE is forbidden",
        },
    ]
    payload: dict[str, Any] = {
        "status": "POLICY_FROZEN_BEFORE_DEVELOPMENT_HOLDOUT",
        "experiment": "persist_eeg_prospective_action_policy_v2",
        "seed": SEED,
        "split_assignment_hash": split["assignment_hash"],
        "exploration_decision_sha256": sha256_file(decision_path),
        "exploration_results_sha256": sha256_file(EXPLORATION / "EXPLORATION_POLICY_RESULTS.csv"),
        "code_sha256": _code_hashes(),
        "candidate_policies": policy_specs,
        "training": "none for frozen consensus policies",
        "model_capacity": "deterministic finite rule",
        "action_semantics": "select an existing historical intervention whose binary prediction equals the other-run majority",
        "inference_requirement": "at least two historical frozen run predictions for the same manifest sample; target run is excluded from its own vote",
        "holdout_openings_allowed": 1,
        "holdout_openings_completed": 0,
        "post_holdout_retuning_allowed": False,
        "outer_test_authorized": False,
        "DEVELOPMENT_HOLDOUT_OPENED": False,
        "OUTER_TEST_USED": False,
    }
    payload["policy_lock_hash"] = canonical_hash(payload)
    write_json(FREEZE / "FROZEN_POLICY_SPEC.json", payload)
    authorization = {
        "status": "DEVELOPMENT_HOLDOUT_OPEN_AUTHORIZED_ONCE",
        "policy_lock_hash": payload["policy_lock_hash"],
        "required_split_assignment_hash": split["assignment_hash"],
        "candidate_count": len(policy_specs),
        "remaining_openings": 1,
        "outer_test_authorized": False,
        "OUTER_TEST_USED": False,
    }
    write_json(FREEZE / "HOLDOUT_OPEN_AUTHORIZATION.json", authorization)
    md = f"""# Frozen policy lock

`OUTER_TEST_USED = false`

- Lock hash: `{payload['policy_lock_hash']}`
- Split hash: `{split['assignment_hash']}`
- Frozen candidates: `{', '.join(expected)}`
- Learned checkpoints: none (both policies are deterministic)
- Allowed development-holdout openings: one
- Post-holdout tuning: forbidden
- WBCIC outer evaluation: unauthorized

The full policy maximizes recovered exploration headroom but may invoke ERASE.
The protected-safe policy forbids ERASE and is frozen as the safety comparator.
"""
    (FREEZE / "FROZEN_POLICY_LOCK.md").write_text(md, encoding="utf-8")
    return payload


def verify_lock() -> dict[str, Any]:
    path = FREEZE / "FROZEN_POLICY_SPEC.json"
    if not path.exists():
        raise FileNotFoundError("No frozen policy lock")
    payload = json.loads(path.read_text(encoding="utf-8"))
    supplied = payload.pop("policy_lock_hash")
    actual = canonical_hash(payload)
    payload["policy_lock_hash"] = supplied
    if supplied != actual:
        raise RuntimeError("Frozen policy lock hash mismatch")
    current = _code_hashes()
    if current != payload["code_sha256"]:
        raise RuntimeError("Code changed after policy freeze")
    return payload
