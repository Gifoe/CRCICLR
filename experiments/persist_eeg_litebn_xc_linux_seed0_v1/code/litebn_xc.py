"""LiteBN-XC: authoritative XNG plus only the authoritative XS channel gate."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import litebn_x as authoritative
import litebn_xng as xng_source


class LiteBNXC(xng_source.LiteBNXNG):
    """Scale-gate-free XNG with the original shared per-electrode channel gate."""

    def __init__(self, channels: int, classes: int):
        super().__init__(channels, classes)
        self.channel_gate = True
        # Exact module definition and ordering from authoritative LiteBN-XS.
        self.channel_mlp = nn.Sequential(nn.Linear(2, 8), nn.GELU(), nn.Linear(8, 1))
        self.lambda_channel = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.architecture = "LiteBN_XC"


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


def initialize_xc(task: str, fold: int, seed: int = 0) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    """Reconstruct XNG init, copy its full state, add exact same-seed XS gate, restore RNG."""
    spec = authoritative.TASKS[task]
    xng, xng_audit = xng_source.initialize_xng(task, fold, seed)
    post_xng_rng = authoritative.rng_state()

    authoritative.set_seed(seed)
    xs = authoritative.LiteBNEnhanced(int(spec["channels"]), int(spec["classes"]), "LiteBN_XS")
    candidate = LiteBNXC(int(spec["channels"]), int(spec["classes"]))
    candidate_state = candidate.state_dict()
    xng_state = xng.state_dict()
    shared_keys = tuple(xng_state.keys())
    candidate_state.update({key: value for key, value in xng_state.items()})
    gate_keys = ("lambda_channel", "channel_mlp.0.weight", "channel_mlp.0.bias", "channel_mlp.2.weight", "channel_mlp.2.bias")
    candidate_state.update({key: xs.state_dict()[key] for key in gate_keys})
    candidate.load_state_dict(candidate_state, strict=True)

    after = candidate.state_dict()
    exact = sum(int(torch.equal(after[key], xng_state[key])) for key in shared_keys)
    maximum = max((float((after[key] - xng_state[key]).abs().max()) for key in shared_keys), default=0.0)
    gate_exact = all(torch.equal(after[key], xs.state_dict()[key]) for key in gate_keys)
    lambda_zero = bool(torch.equal(candidate.lambda_channel.detach(), torch.zeros_like(candidate.lambda_channel.detach())))
    scale_present = bool(
        candidate.scale_gate
        or any(name == "lambda_scale" or name.startswith("scale_mlp.") for name, _ in candidate.named_parameters())
        or any(name.startswith("scale_mlp") for name, _ in candidate.named_modules())
    )
    authoritative.restore_rng(post_xng_rng)
    preserved = _rng_equal(post_xng_rng, authoritative.rng_state())
    status = "PASS" if all((xng_audit["status"] == "PASS", exact == len(shared_keys), maximum == 0.0,
                            gate_exact, lambda_zero, not scale_present, preserved)) else "FAIL"
    audit = {
        "task": task, "fold": int(fold), "shared_tensor_count": len(shared_keys),
        "shared_tensor_exact_matches": exact, "max_abs_shared_tensor_diff": maximum,
        "channel_gate_tensor_count": len(gate_keys), "channel_gate_matches_authoritative_xs": gate_exact,
        "lambda_channel": float(candidate.lambda_channel.detach()), "lambda_channel_exactly_zero": lambda_zero,
        "scale_gate_present": scale_present, "rng_preserved": preserved, "status": status,
    }
    if status != "PASS":
        raise RuntimeError(f"XC_INITIALIZATION_AUDIT_FAIL: {audit}")
    return xng, candidate, audit


def function_audit(xng: nn.Module, xc: nn.Module, value: torch.Tensor) -> dict[str, Any]:
    """Verify function preservation in FP32/AMP and eval/train modes on real input."""
    device = value.device
    max_embedding = 0.0
    max_logits = 0.0
    mismatches = 0
    mode_values: dict[str, float | int] = {}
    with torch.no_grad():
        gate_diff = float((xc.apply_channel_gate(value) - value).abs().max())
    for training in (False, True):
        for amp in (False, True):
            left, right = copy.deepcopy(xng).to(device), copy.deepcopy(xc).to(device)
            left.train(training); right.train(training)
            state = authoritative.rng_state()
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                authoritative.restore_rng(state)
                l_logits, l_embedding = left(value)
                authoritative.restore_rng(state)
                r_logits, r_embedding = right(value)
            embedding_diff = float((l_embedding.float() - r_embedding.float()).abs().max())
            logit_diff = float((l_logits.float() - r_logits.float()).abs().max())
            mismatch = int((l_logits.argmax(dim=1) != r_logits.argmax(dim=1)).sum().item())
            label = ("train" if training else "eval") + ("_amp_fp16" if amp else "_fp32")
            mode_values[f"{label}_max_abs_embedding_difference"] = embedding_diff
            mode_values[f"{label}_max_abs_logit_difference"] = logit_diff
            mode_values[f"{label}_prediction_mismatch"] = mismatch
            max_embedding, max_logits, mismatches = max(max_embedding, embedding_diff), max(max_logits, logit_diff), mismatches + mismatch
            del left, right
    status = "PASS" if gate_diff == 0.0 and max_embedding <= 1e-6 and max_logits <= 1e-6 and mismatches == 0 else "FAIL"
    result = {
        "max_abs_input_gate_difference": gate_diff,
        "max_abs_embedding_difference": max_embedding,
        "max_abs_logit_difference": max_logits,
        "prediction_mismatch": mismatches,
        **mode_values,
        "status": status,
    }
    if status != "PASS":
        raise RuntimeError(f"XC_INITIALIZATION_AUDIT_FAIL: {result}")
    return result


def parameter_audit(task: str) -> dict[str, Any]:
    spec = authoritative.TASKS[task]
    authoritative.set_seed(0)
    xng = xng_source.LiteBNXNG(int(spec["channels"]), int(spec["classes"]))
    authoritative.set_seed(0)
    xc = LiteBNXC(int(spec["channels"]), int(spec["classes"]))
    xng_count, xc_count = authoritative.parameter_count(xng), authoritative.parameter_count(xc)
    added = xc_count - xng_count
    added_names = sorted(set(dict(xc.named_parameters())) - set(dict(xng.named_parameters())))
    expected_names = sorted(["lambda_channel", "channel_mlp.0.weight", "channel_mlp.0.bias", "channel_mlp.2.weight", "channel_mlp.2.bias"])
    passed = xng_count == 216705 and xc_count == 216739 and added == 34 and added_names == expected_names
    row = {
        "task": task, "xng_parameter_count": xng_count, "xc_parameter_count": xc_count,
        "added_parameters": added, "expected_added_parameters": 34,
        "added_parameter_names": ";".join(added_names), "status": "PASS" if passed else "FAIL",
    }
    if not passed:
        raise RuntimeError(f"XC_PARAMETER_AUDIT_FAIL: {row}")
    return row
