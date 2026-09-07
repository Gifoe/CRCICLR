"""Minimal zero-initialized additive logit sidecar."""
from __future__ import annotations

from torch import nn


class ResidualSidecar(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(64, 32)
        self.activation = nn.GELU()
        self.out = nn.Linear(32, 2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feature):
        return self.out(self.activation(self.hidden(feature)))


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
