"""Leave-one-out, label-free offset target used only on training episodes."""
from __future__ import annotations

import torch


def leave_one_out_centers(h: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    """Return detached LOO center for every row; every group must contain >=2 rows."""
    target = torch.empty_like(h)
    for gid in torch.unique(group):
        mask = group == gid
        values = h[mask]
        if values.shape[0] < 2:
            raise ValueError("leave-one-out target requires at least two trials per group")
        target[mask] = (values.sum(dim=0, keepdim=True) - values) / (values.shape[0] - 1)
    return target.detach()


def offset_mse(offset_hat: torch.Tensor, h: torch.Tensor, group: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension MSE; class labels are deliberately absent from this API."""
    target = leave_one_out_centers(h, group)
    return torch.mean((offset_hat - target) ** 2), target
