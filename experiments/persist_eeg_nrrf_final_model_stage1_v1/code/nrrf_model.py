"""NRRF-v1 model and exact preregistered subject-level losses.

The anchor is deliberately kept outside the trainable path: it always runs in
evaluation mode under ``torch.no_grad``.  This file contains no gate, learned
fusion coefficient, or subject identifier feature.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


ALPHA = 0.5
LAMBDA_TAIL = 0.5
LAMBDA_REGRET = 1.0
LAMBDA_EXPR = 0.25
EMA_BETA = 0.99


class NRRF(nn.Module):
    """Frozen EEGNet anchor plus a trainable exact CompactLite-BN branch."""

    def __init__(self, anchor: nn.Module, expressive: nn.Module) -> None:
        super().__init__()
        self.anchor = anchor
        self.expressive = expressive
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)
        self.anchor.eval()

    def train(self, mode: bool = True) -> "NRRF":
        # Calling ``super().train`` would place anchor BatchNorm in train mode.
        self.training = mode
        self.expressive.train(mode)
        self.anchor.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.anchor.eval()
        with torch.no_grad():
            logits_anchor, _ = self.anchor(x)
        logits_expr, _ = self.expressive(x)
        logits_fused = (1.0 - ALPHA) * logits_anchor + ALPHA * logits_expr
        return logits_anchor, logits_expr, logits_fused

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return self.expressive.parameters()


def subject_means(values: torch.Tensor, subject_slot: torch.Tensor, n_subjects: int) -> torch.Tensor:
    """Means where every slot occurs at least once, preserving autograd."""
    sums = torch.zeros(n_subjects, device=values.device, dtype=values.dtype)
    counts = torch.zeros(n_subjects, device=values.device, dtype=values.dtype)
    sums.scatter_add_(0, subject_slot, values)
    counts.scatter_add_(0, subject_slot, torch.ones_like(values))
    if bool((counts == 0).any()):
        raise RuntimeError("subject-balanced batch omitted a chosen subject")
    return sums / counts


def loss_terms(
    logits_anchor: torch.Tensor,
    logits_expr: torch.Tensor,
    logits_fused: torch.Tensor,
    labels: torch.Tensor,
    subject_slot: torch.Tensor,
    n_subjects: int,
    method: str,
) -> dict[str, torch.Tensor]:
    """Return the locked NRRF or JOINT-CE objective and all diagnostics."""
    anchor_ce = F.cross_entropy(logits_anchor.float(), labels, reduction="none").detach()
    expr_ce = F.cross_entropy(logits_expr.float(), labels, reduction="none")
    fused_ce = F.cross_entropy(logits_fused.float(), labels, reduction="none")
    loss_anchor = subject_means(anchor_ce, subject_slot, n_subjects).detach()
    loss_expr = subject_means(expr_ce, subject_slot, n_subjects)
    loss_fused = subject_means(fused_ce, subject_slot, n_subjects)
    mean = loss_fused.mean()
    q = int(math.ceil(0.25 * n_subjects))
    tail = loss_fused.topk(q, largest=True, sorted=True).values.mean()
    regret_per_subject = torch.relu(loss_fused - loss_anchor)
    regret = regret_per_subject.mean()
    expr = loss_expr.mean()
    if method == "JOINT-CE":
        total = mean + LAMBDA_EXPR * expr
    elif method == "NRRF-v1":
        total = mean + LAMBDA_TAIL * tail + LAMBDA_REGRET * regret + LAMBDA_EXPR * expr
    else:
        raise ValueError(f"unknown NRRF method: {method}")
    return {
        "total": total,
        "L_mean": mean,
        "L_tail": tail,
        "L_regret": regret,
        "L_expr": expr,
        "positive_regret_fraction": (regret_per_subject > 0).float().mean(),
        "mean_positive_regret": regret_per_subject[regret_per_subject > 0].mean()
        if bool((regret_per_subject > 0).any()) else regret_per_subject.new_zeros(()),
        "worst_subject_loss": loss_fused.max(),
        "mean_subject_loss": loss_fused.mean(),
        "per_subject_anchor": loss_anchor,
        "per_subject_fused": loss_fused,
    }


@torch.no_grad()
def update_ema(ema_parameters: dict[str, torch.Tensor], expressive: nn.Module) -> None:
    """EMA applies to parameters only; BatchNorm buffers remain final-epoch raw."""
    for name, parameter in expressive.named_parameters():
        ema_parameters[name].mul_(EMA_BETA).add_(parameter.detach(), alpha=1.0 - EMA_BETA)


def make_ema(expressive: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in expressive.named_parameters()}


@torch.no_grad()
def load_ema(expressive: nn.Module, ema_parameters: dict[str, torch.Tensor]) -> None:
    for name, parameter in expressive.named_parameters():
        parameter.copy_(ema_parameters[name])
