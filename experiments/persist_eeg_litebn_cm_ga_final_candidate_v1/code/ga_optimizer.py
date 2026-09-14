"""Exact Gradient-Admissible optimizer rule used only if Stage 1 is authorized."""
from __future__ import annotations

from typing import Iterable

import torch

EPS = 1e-12


def flatten(values: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.reshape(-1).float() for value in values])


def split_like(vector: torch.Tensor, parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    output, offset = [], 0
    for parameter in parameters:
        count = parameter.numel()
        output.append(vector[offset:offset + count].reshape_as(parameter))
        offset += count
    if offset != vector.numel():
        raise RuntimeError("gradient/parameter cardinality mismatch")
    return output


def gradient_admissible_step(
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    gradient_a: torch.Tensor,
    gradient_b: torch.Tensor,
    clip_norm: float = 5.0,
) -> dict[str, float | bool]:
    """Apply one accepted step or a bitwise no-op for a rejected step."""
    norm_a = torch.linalg.vector_norm(gradient_a)
    norm_b = torch.linalg.vector_norm(gradient_b)
    rho = torch.dot(gradient_a, gradient_b) / (norm_a * norm_b + EPS)
    rho_value = float(rho.detach().cpu())
    if rho_value < 0.0:
        # Deliberately do not call zero_grad or optimizer.step: even a zero
        # gradient AdamW step would apply decoupled weight decay.
        return {
            "accepted": False,
            "rho": rho_value,
            "g_A_norm": float(norm_a.detach().cpu()),
            "g_B_norm": float(norm_b.detach().cpu()),
            "g_update_norm_preclip": 0.0,
            "g_update_norm_postclip": 0.0,
        }
    update = 0.5 * (gradient_a + gradient_b)
    preclip = torch.linalg.vector_norm(update)
    update = update * min(1.0, clip_norm / max(float(preclip.detach().cpu()), EPS))
    postclip = torch.linalg.vector_norm(update)
    optimizer.zero_grad(set_to_none=True)
    for parameter, chunk in zip(parameters, split_like(update, parameters)):
        parameter.grad = chunk.detach().clone()
    optimizer.step()
    return {
        "accepted": True,
        "rho": rho_value,
        "g_A_norm": float(norm_a.detach().cpu()),
        "g_B_norm": float(norm_b.detach().cpu()),
        "g_update_norm_preclip": float(preclip.detach().cpu()),
        "g_update_norm_postclip": float(postclip.detach().cpu()),
    }


def self_test() -> dict[str, bool]:
    rejected_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    rejected_optimizer = torch.optim.AdamW([rejected_parameter], lr=3e-4, weight_decay=5e-4)
    before = rejected_parameter.detach().clone()
    rejected = gradient_admissible_step(
        rejected_optimizer, [rejected_parameter],
        torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0]),
    )
    accepted_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    accepted_optimizer = torch.optim.AdamW([accepted_parameter], lr=3e-4, weight_decay=5e-4)
    accepted = gradient_admissible_step(
        accepted_optimizer, [accepted_parameter],
        torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]),
    )
    return {
        "rejected_no_optimizer_step": not rejected["accepted"],
        "rejected_bitwise_unchanged": torch.equal(before, rejected_parameter.detach()),
        "accepted_optimizer_step": bool(accepted["accepted"]),
        "accepted_parameter_changed": not torch.equal(torch.tensor([1.0, -2.0]), accepted_parameter.detach()),
    }

