"""The preregistered bounded CFRF-v1 and matched GLOBAL-ALPHA control."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


ALPHA_CENTER = 0.5
ALPHA_RADIUS = 0.25
LAMBDA_DELTA = 0.10
EMA_BETA = 0.99


def fusion_input(h_a: torch.Tensor, h_e: torch.Tensor, z_a: torch.Tensor, z_e: torch.Tensor) -> torch.Tensor:
    h_a = F.normalize(h_a, p=2, dim=1, eps=1e-6)
    h_e = F.normalize(h_e, p=2, dim=1, eps=1e-6)
    p_a = torch.softmax(z_a, dim=1)[:, 1:2]
    p_e = torch.softmax(z_e, dim=1)[:, 1:2]
    return torch.cat((h_a, h_e, p_a, p_e, (p_a - p_e).abs()), dim=1)


class CFRF(nn.Module):
    def __init__(self, input_dim: int = 131) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_dim, 32)
        self.drop = nn.Dropout(0.10)
        self.out = nn.Linear(32, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h_a: torch.Tensor, h_e: torch.Tensor, z_a: torch.Tensor, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = fusion_input(h_a, h_e, z_a, z_e)
        delta = ALPHA_RADIUS * torch.tanh(self.out(self.drop(F.gelu(self.hidden(u)))))
        alpha = ALPHA_CENTER + delta
        logits = (1.0 - alpha) * z_a + alpha * z_e
        return logits, alpha.squeeze(1), delta.squeeze(1)


class GlobalAlpha(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.zeros(()))

    def forward(self, z_a: torch.Tensor, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = ALPHA_RADIUS * torch.tanh(self.q)
        alpha = ALPHA_CENTER + delta
        return (1.0 - alpha) * z_a + alpha * z_e, alpha.expand(z_a.shape[0]), delta.expand(z_a.shape[0])


def ema_start(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}


@torch.no_grad()
def ema_update(ema: dict[str, torch.Tensor], module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        ema[name].mul_(EMA_BETA).add_(parameter.detach(), alpha=1.0 - EMA_BETA)


@torch.no_grad()
def ema_load(ema: dict[str, torch.Tensor], module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        parameter.copy_(ema[name])
