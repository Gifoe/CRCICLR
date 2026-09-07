"""Label-free leave-one-out target for offset-estimator Stage A."""
from __future__ import annotations

import torch


def leave_one_out_centers(h: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    target = torch.empty_like(h)
    for identifier in torch.unique(group):
        mask = group == identifier
        values = h[mask]
        if values.shape[0] < 2:
            raise ValueError("LOO target requires two or more trials")
        target[mask] = (values.sum(0, keepdim=True) - values) / (values.shape[0] - 1)
    return target.detach()


def offset_mse(prediction: torch.Tensor, h: torch.Tensor, group: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """No label argument by design; only subject/session episode grouping enters."""
    target = leave_one_out_centers(h, group)
    return torch.mean((prediction - target) ** 2), target
