"""Frozen-base single-trial subject/session center estimator."""
from __future__ import annotations

from torch import nn


class OffsetEstimator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(64, 32)
        self.activation = nn.GELU()
        self.out = nn.Linear(32, 64)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h):
        return self.out(self.activation(self.hidden(h)))
