"""Controlled two-stream LiteBN stochastic-consistency primitives."""
from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import litebn_x as authoritative


BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def bn_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, BN_TYPES) and module.track_running_stats:
            for field in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(module, field)
                if value is not None:
                    result[f"{module_name}.{field}"] = value.detach().clone()
    return result


def torch_rng_state() -> dict[str, Any]:
    result: dict[str, Any] = {"cpu": torch.get_rng_state().detach().cpu().clone()}
    if torch.cuda.is_available():
        result["cuda"] = [value.detach().cpu().clone() for value in torch.cuda.get_rng_state_all()]
    return result


def restore_torch_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["cpu"].detach().cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.detach().cpu() for value in state["cuda"]])


def initialize_secondary_rng(seed: int) -> dict[str, Any]:
    """Create the independent persistent B stream without perturbing the A stream."""
    main = torch_rng_state()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    secondary = torch_rng_state()
    restore_torch_rng(main)
    return secondary


def controlled_dual_forward(
    model: nn.Module,
    value: torch.Tensor,
    secondary_rng: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Forward A on live buffers and B on cloned buffers and an independent RNG stream."""
    parameters = dict(model.named_parameters())
    buffers_b = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
    logits_a, embedding_a = model(value)
    main_after_a = torch_rng_state()
    restore_torch_rng(secondary_rng)
    try:
        logits_b, embedding_b = torch.func.functional_call(
            model, (parameters, buffers_b), (value,), strict=True
        )
        secondary_after_b = torch_rng_state()
    finally:
        restore_torch_rng(main_after_a)
    return logits_a, embedding_a, logits_b, embedding_b, secondary_after_b


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Locked symmetric KL, explicitly evaluated in FP32."""
    with torch.autocast(device_type=logits_a.device.type, enabled=False):
        logp_a = F.log_softmax(logits_a.float(), dim=-1)
        logp_b = F.log_softmax(logits_b.float(), dim=-1)
        p_a, p_b = logp_a.exp(), logp_b.exp()
        per_sample = 0.5 * (
            (p_a * (logp_a - logp_b)).sum(dim=-1)
            + (p_b * (logp_b - logp_a)).sum(dim=-1)
        )
        return per_sample.mean(), per_sample


def initialize_c1(task: str, fold: int, seed: int = 0) -> tuple[nn.Module, dict[str, Any]]:
    """Construct C1 from the identical historical LiteBN initialization."""
    spec = authoritative.TASKS[task]
    authoritative.set_seed(seed)
    reference = authoritative.LiteBN_BASELINE(int(spec["channels"]), int(spec["classes"]))
    post_reference_rng = authoritative.rng_state()
    authoritative.set_seed(seed)
    candidate = authoritative.LiteBN_BASELINE(int(spec["channels"]), int(spec["classes"]))
    ref_params, c1_params = dict(reference.named_parameters()), dict(candidate.named_parameters())
    ref_buffers, c1_buffers = dict(reference.named_buffers()), dict(candidate.named_buffers())
    max_parameter = max(
        (float((c1_params[key].detach() - ref_params[key].detach()).abs().max()) for key in ref_params),
        default=0.0,
    )
    max_buffer = max(
        (float((c1_buffers[key].float() - ref_buffers[key].float()).abs().max()) for key in ref_buffers),
        default=0.0,
    )
    exact_parameter = sum(int(torch.equal(c1_params[key], ref_params[key])) for key in ref_params)
    exact_buffer = sum(int(torch.equal(c1_buffers[key], ref_buffers[key])) for key in ref_buffers)
    authoritative.restore_rng(post_reference_rng)
    passed = (
        tuple(ref_params) == tuple(c1_params)
        and tuple(ref_buffers) == tuple(c1_buffers)
        and exact_parameter == len(ref_params)
        and exact_buffer == len(ref_buffers)
        and max_parameter == 0.0
        and max_buffer == 0.0
    )
    row = {
        "task": task,
        "fold": int(fold),
        "method": "C1_LiteBN_SC",
        "parameter_tensor_count": len(ref_params),
        "exact_parameter_matches": exact_parameter,
        "buffer_tensor_count": len(ref_buffers),
        "exact_buffer_matches": exact_buffer,
        "max_abs_parameter_diff": max_parameter,
        "max_abs_buffer_diff": max_buffer,
        "status": "PASS" if passed else "FAIL",
    }
    if not passed:
        raise RuntimeError(f"SC_PROTOCOL_FAIL initialization: {row}")
    return candidate, row


def inference_identity_audit(task: str) -> dict[str, Any]:
    spec = authoritative.TASKS[task]
    authoritative.set_seed(0)
    b0 = authoritative.LiteBN_BASELINE(int(spec["channels"]), int(spec["classes"]))
    authoritative.set_seed(0)
    c1 = authoritative.LiteBN_BASELINE(int(spec["channels"]), int(spec["classes"]))
    b0.eval(); c1.eval()
    value = torch.zeros(2, int(spec["channels"]), int(spec["samples"]))
    with torch.no_grad():
        b0_logits, b0_embedding = b0(value)
        c1_logits, c1_embedding = c1(value)
    b0_count, c1_count = authoritative.parameter_count(b0), authoritative.parameter_count(c1)
    state_shapes_equal = tuple((k, tuple(v.shape)) for k, v in b0.state_dict().items()) == tuple(
        (k, tuple(v.shape)) for k, v in c1.state_dict().items()
    )
    passed = all((
        type(b0) is type(c1), b0_count == c1_count, state_shapes_equal,
        torch.equal(b0_logits, c1_logits), torch.equal(b0_embedding, c1_embedding),
    ))
    row = {
        "task": task, "model_class": type(c1).__qualname__, "b0_parameter_count": b0_count,
        "c1_parameter_count": c1_count, "inference_parameter_increase": c1_count - b0_count,
        "state_shapes_identical": state_shapes_equal, "eval_logits_exact": torch.equal(b0_logits, c1_logits),
        "eval_embeddings_exact": torch.equal(b0_embedding, c1_embedding), "inference_forward_passes": 1,
        "extra_inference_branches": 0, "status": "PASS" if passed else "FAIL",
    }
    if not passed:
        raise RuntimeError(f"SC_PROTOCOL_FAIL inference identity: {row}")
    return row


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)
