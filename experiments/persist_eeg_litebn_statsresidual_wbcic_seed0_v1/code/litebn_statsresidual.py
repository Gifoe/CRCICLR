"""Frozen LiteBN plus an exact-zero global-statistics residual logit head."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn


class LiteBNStatsResidual(nn.Module):
    """Preserve the complete LiteBN prediction path and add one 128 -> 2 head."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        self.residual = nn.Linear(128, 2, bias=True)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self._tap: torch.Tensor | None = None

        def capture_pool_input(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            if len(args) != 1:
                raise RuntimeError("STATSRES_PROTOCOL_FAIL unexpected pool input signature")
            self._tap = args[0]

        self._tap_handle = self.base.pool.register_forward_pre_hook(capture_pool_input)
        self.train(False)

    def train(self, mode: bool = True) -> "LiteBNStatsResidual":
        super().train(mode)
        self.base.eval()
        self.residual.train(mode)
        return self

    def base_forward_and_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.base.eval()
        self._tap = None
        with torch.no_grad():
            base_logits, _ = self.base(x)
        if self._tap is None:
            raise RuntimeError("STATSRES_PROTOCOL_FAIL LiteBN pre-pool tap did not fire")
        u = self._tap.detach()
        if u.ndim != 4 or u.shape[1] != 64 or u.shape[2] != 1:
            raise RuntimeError(f"STATSRES_PROTOCOL_FAIL tap shape {tuple(u.shape)}")
        mu = u.mean(dim=-1).squeeze(-1)
        sigma = u.std(dim=-1, unbiased=False).squeeze(-1)
        stats = torch.cat((mu, sigma), dim=1)
        if tuple(stats.shape) != (x.shape[0], 128):
            raise RuntimeError(f"STATSRES_PROTOCOL_FAIL stats shape {tuple(stats.shape)}")
        return base_logits.detach(), stats.detach()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_logits, stats = self.base_forward_and_stats(x)
        residual_logits = self.residual(stats)
        return base_logits + residual_logits, base_logits, stats

    def close(self) -> None:
        self._tap_handle.remove()


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def tensor_schema(state: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    return [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in state.items()
    ]


def split_parameter_buffer_schema(model: nn.Module) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parameters = OrderedDict(model.named_parameters())
    buffers = OrderedDict(model.named_buffers())
    return tensor_schema(parameters), tensor_schema(buffers)


def exact_state_audit(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict[str, Any]:
    names_equal = tuple(before) == tuple(after)
    equal = names_equal and all(torch.equal(before[name], after[name]) for name in before)
    maximum = 0.0
    if names_equal:
        for name in before:
            a, b = before[name], after[name]
            if a.numel() and (a.is_floating_point() or a.is_complex()):
                maximum = max(maximum, float((a - b).abs().max()))
            elif not torch.equal(a, b):
                maximum = float("inf")
    return {
        "state_names_equal": names_equal,
        "exact_tensor_equality": equal,
        "max_abs_difference": maximum,
        "tensor_count": len(before),
        "status": "PASS" if equal and maximum == 0.0 else "FAIL",
    }
