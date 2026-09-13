"""Exact LiteBN-X minus only its sample-dependent temporal-scale gate."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

import litebn_x as authoritative


class LiteBNXNG(authoritative.LiteBNEnhanced):
    """LiteBN-X with scale_mlp/lambda_scale removed and the gate bypassed."""

    def __init__(self, channels: int, classes: int):
        # Construct exact X first so module creation and ordering stay authoritative.
        super().__init__(channels, classes, "LiteBN_X")
        del self.scale_mlp
        del self.lambda_scale
        self.scale_gate = False
        self.architecture = "LiteBN_XNG"


def _rng_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["python"] != right["python"]:
        return False
    la, ra = left["numpy"], right["numpy"]
    if la[0] != ra[0] or not np.array_equal(la[1], ra[1]) or la[2:] != ra[2:]:
        return False
    if not torch.equal(left["torch"], right["torch"]):
        return False
    lc, rc = left.get("cuda", []), right.get("cuda", [])
    return len(lc) == len(rc) and all(torch.equal(a, b) for a, b in zip(lc, rc))


def initialize_xng(task: str, fold: int, seed: int = 0) -> tuple[nn.Module, dict[str, Any]]:
    """Build authoritative X, copy every shared tensor, then restore post-X RNG."""
    spec = authoritative.TASKS[task]
    authoritative.set_seed(seed)
    reference = authoritative.LiteBNEnhanced(int(spec["channels"]), int(spec["classes"]), "LiteBN_X")
    post_reference_rng = authoritative.rng_state()

    candidate = LiteBNXNG(int(spec["channels"]), int(spec["classes"]))
    reference_state = reference.state_dict()
    candidate_keys = tuple(candidate.state_dict().keys())
    shared = {key: reference_state[key] for key in candidate_keys}
    candidate.load_state_dict(shared, strict=True)

    candidate_state = candidate.state_dict()
    exact = sum(int(torch.equal(candidate_state[key], reference_state[key])) for key in candidate_keys)
    maximum = max((float((candidate_state[key] - reference_state[key]).abs().max()) for key in candidate_keys), default=0.0)
    removed = sorted(set(reference_state) - set(candidate_state))
    expected_removed = sorted([
        "lambda_scale",
        "scale_mlp.0.bias",
        "scale_mlp.0.weight",
        "scale_mlp.2.bias",
        "scale_mlp.2.weight",
    ])
    active_scale = bool(
        candidate.scale_gate
        or any("scale_mlp" in name or "lambda_scale" in name for name, _ in candidate.named_parameters())
        or any("scale_mlp" in name for name, _ in candidate.named_modules())
    )

    authoritative.restore_rng(post_reference_rng)
    preserved = _rng_equal(post_reference_rng, authoritative.rng_state())
    status = "PASS" if exact == len(candidate_keys) and maximum == 0.0 and not active_scale and preserved and removed == expected_removed else "FAIL"
    audit = {
        "task": task,
        "fold": int(fold),
        "shared_tensor_count": len(candidate_keys),
        "shared_tensor_exact_matches": exact,
        "max_abs_shared_tensor_diff": maximum,
        "xng_has_active_scale_gate": active_scale,
        "rng_preserved": preserved,
        "removed_state_keys": ";".join(removed),
        "status": status,
    }
    if status != "PASS":
        raise RuntimeError(f"XNG_INITIALIZATION_PROTOCOL_FAIL: {audit}")
    return candidate, audit


def parameter_audit(task: str) -> dict[str, Any]:
    spec = authoritative.TASKS[task]
    authoritative.set_seed(0)
    baseline = authoritative.LiteBN_BASELINE(int(spec["channels"]), int(spec["classes"]))
    authoritative.set_seed(0)
    original = authoritative.LiteBNEnhanced(int(spec["channels"]), int(spec["classes"]), "LiteBN_X")
    authoritative.set_seed(0)
    candidate = LiteBNXNG(int(spec["channels"]), int(spec["classes"]))
    base_count = authoritative.parameter_count(baseline)
    x_count = authoritative.parameter_count(original)
    xng_count = authoritative.parameter_count(candidate)
    scale_count = sum(parameter.numel() for name, parameter in original.named_parameters() if name == "lambda_scale" or name.startswith("scale_mlp."))
    unexpected = sorted(set(candidate.state_dict()) - set(original.state_dict()))
    passed = x_count - xng_count == scale_count and not unexpected
    row = {
        "task": task,
        "litebn_parameter_count": base_count,
        "original_x_parameter_count": x_count,
        "xng_parameter_count": xng_count,
        "original_x_minus_xng": x_count - xng_count,
        "scale_gate_parameter_count": scale_count,
        "unexpected_xng_state_keys": ";".join(unexpected),
        "status": "PASS" if passed else "FAIL",
    }
    if not passed:
        raise RuntimeError(f"XNG_ARCHITECTURE_AUDIT_FAIL: {row}")
    return row
