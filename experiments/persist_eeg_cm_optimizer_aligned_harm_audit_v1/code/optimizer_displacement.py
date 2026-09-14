"""Exact one-step AdamW displacement and eval-mode gradient helpers."""
from __future__ import annotations

import copy
from typing import Iterable

import torch
import torch.nn.functional as F


LR = 3e-4
WEIGHT_DECAY = 5e-4
CLIP_NORM = 5.0
EPS = 1e-12


def named_space_parameters(model: torch.nn.Module, space: str) -> list[tuple[str, torch.nn.Parameter]]:
    selected = []
    for name, parameter in model.named_parameters():
        if name.startswith("base.") or not parameter.requires_grad:
            continue
        in_c = name == "lambda_channel" or name.startswith("channel_mlp.")
        in_m = name.startswith("mixers.")
        if (space == "C" and in_c) or (space == "M" and in_m) or (space == "CM" and (in_c or in_m)):
            selected.append((name, parameter))
    if not selected:
        raise RuntimeError(f"empty parameter space: {space}")
    return selected


def flatten(grads: Iterable[torch.Tensor | None], params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([
        (grad if grad is not None else torch.zeros_like(param)).detach().reshape(-1).float()
        for grad, param in zip(grads, params)
    ])


def eval_gradient(model: torch.nn.Module, params: list[torch.nn.Parameter], value: torch.Tensor,
                  labels: torch.Tensor) -> tuple[torch.Tensor, float, list[torch.Tensor | None]]:
    """Primary gradients: eval-mode and dropout-free, with no .grad mutation."""
    model.eval()
    logits, _ = model(value)
    loss = F.cross_entropy(logits, labels)
    raw = list(torch.autograd.grad(loss, params, allow_unused=True, create_graph=False))
    return flatten(raw, params), float(loss.detach().cpu()), raw


def deterministic_loss(model: torch.nn.Module, value: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    with torch.inference_mode():
        return float(F.cross_entropy(model(value)[0], labels).detach().cpu())


def one_step_displacement(identity_model: torch.nn.Module, space: str, value: torch.Tensor,
                          labels: torch.Tensor) -> dict[str, object]:
    """Run a real AdamW step on a disposable in-memory clone.

    This intentionally lets AdamW observe None gradients exactly as produced by
    autograd.  It does not reconstruct an update from a flattened gradient.
    """
    proposal = copy.deepcopy(identity_model)
    proposal.eval()
    named = named_space_parameters(proposal, space)
    params = [parameter for _, parameter in named]
    before = [parameter.detach().clone() for parameter in params]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer.zero_grad(set_to_none=True)
    logits, _ = proposal(value)
    loss = F.cross_entropy(logits, labels)
    raw = list(torch.autograd.grad(loss, params, allow_unused=True, create_graph=False))
    # Assigning only non-None gradients preserves AdamW's None-gradient skip semantics.
    for parameter, gradient in zip(params, raw):
        parameter.grad = None if gradient is None else gradient.detach().clone()
    raw_vector = flatten(raw, params)
    unclipped = float(torch.nn.utils.clip_grad_norm_(params, CLIP_NORM).detach().cpu())
    optimizer.step()
    delta = torch.cat([(parameter.detach() - start).reshape(-1).float()
                       for parameter, start in zip(params, before)])
    return {
        "proposal": proposal,
        "params": params,
        "g_a": raw_vector.detach(),
        "delta": delta.detach(),
        "loss_a": float(loss.detach().cpu()),
        "preclip_global_grad_norm": unclipped,
        "delta_norm": float(torch.linalg.vector_norm(delta).detach().cpu()),
        "g_a_norm": float(torch.linalg.vector_norm(raw_vector).detach().cpu()),
        "g_a_delta_dot": float(torch.dot(raw_vector, delta).detach().cpu()),
        "g_a_delta_cosine": float(torch.dot(raw_vector, delta).detach().cpu() /
                                    (torch.linalg.vector_norm(raw_vector) * torch.linalg.vector_norm(delta) + EPS)),
    }


def active_gradient_counts(raw: list[torch.Tensor | None], named: list[tuple[str, torch.nn.Parameter]]) -> dict[str, object]:
    scalar_total = int(sum(parameter.numel() for _, parameter in named))
    tensor_total = len(named)
    tensor_active = sum(int(gradient is not None and bool((gradient.detach().abs() > EPS).any())) for gradient in raw)
    scalar_active = int(sum(int((gradient.detach().abs() > EPS).sum().cpu()) for gradient in raw if gradient is not None))
    return {
        "parameter_tensors_total": tensor_total,
        "parameter_tensors_nonzero_gradient": tensor_active,
        "scalar_parameters_total": scalar_total,
        "scalar_elements_abs_grad_gt_1e12": scalar_active,
        "fraction_scalar_active": scalar_active / scalar_total if scalar_total else 0.0,
        "parameter_names": ";".join(name for name, _ in named),
    }
