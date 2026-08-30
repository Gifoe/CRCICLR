"""Auditable training-scope and EMA components for ME-HardSCST V2."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def configure_scope(model_name: str, net: nn.Module, scope: str) -> list[nn.Parameter]:
    """Apply exactly Scope A (head) or Scope B (last feature block + head)."""
    if scope not in ("A", "B"):
        raise ValueError(scope)
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    modules: list[nn.Module] = []
    if model_name == "ATCNet-CleanRoom":
        modules.append(net.head)
        if scope == "B":
            modules.extend([net.tcn, net.norm])
    elif model_name == "ATCNet-Official":
        modules.append(net.final_layer)
        if scope == "B":
            modules.extend([net.temporal_conv_nets, net.attention_blocks])
    elif model_name == "EEGNeX":
        modules.append(net.final_layer)
        if scope == "B":
            modules.append(net.block_5)
    else:
        raise KeyError(model_name)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    parameters = [parameter for parameter in net.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("scope selected no trainable parameters")
    return parameters


class EMATeacher:
    def __init__(self, student: nn.Module, decay: float = 0.99):
        if decay != 0.99:
            raise ValueError("EMA decay is frozen at 0.99")
        self.decay = float(decay)
        self.model = copy.deepcopy(student).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        teacher_state = self.model.state_dict()
        student_state = student.state_dict()
        for name, teacher in teacher_state.items():
            source = student_state[name].detach()
            if torch.is_floating_point(teacher):
                teacher.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                teacher.copy_(source)


@dataclass
class BankRefreshTracker:
    last_epoch: int = -1
    refreshes: int = 0

    def refresh(self, epoch: int) -> None:
        if int(epoch) == self.last_epoch:
            raise RuntimeError(f"BANK_REFRESH_DUPLICATE_EPOCH:{epoch}")
        if int(epoch) != self.last_epoch + 1:
            raise RuntimeError(f"BANK_REFRESH_NONSEQUENTIAL:{self.last_epoch}->{epoch}")
        self.last_epoch = int(epoch)
        self.refreshes += 1


def primary_total_loss(clean_logits: torch.Tensor, labels: torch.Tensor, cf_loss: torch.Tensor, lambda_h: float) -> torch.Tensor:
    if lambda_h not in (0.25, 0.50, 1.00):
        raise ValueError(lambda_h)
    # Intentionally only clean cross entropy plus counterfactual margin loss.
    # No symmetric KL, logit matching, GRL, or identity objective is present.
    return F.cross_entropy(clean_logits.float(), labels) + float(lambda_h) * cf_loss

