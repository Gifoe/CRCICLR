"""Subject-direction prospective relational loss (PRD)."""
from __future__ import annotations

from itertools import combinations
import torch
import torch.nn.functional as F


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def direction_from_centroids(mu0: torch.Tensor, mu1: torch.Tensor) -> torch.Tensor:
    """The binary K=2 instance used by the protocol."""
    return _unit(mu1 - mu0)


def _directions(z: torch.Tensor, y: torch.Tensor, subject: torch.Tensor, classes: list[int], pair: tuple[int, int]) -> torch.Tensor:
    values = []
    for sid in torch.unique(subject, sorted=True):
        mask_s = subject == sid
        mus = []
        for cls in pair:
            part = z[mask_s & (y == cls)]
            if part.shape[0] == 0:
                raise RuntimeError(f"PRD episode has no class {cls} for subject {int(sid)}")
            mus.append(part.mean(dim=0))
        values.append(_unit(mus[1] - mus[0]))
    if not values:
        raise RuntimeError("PRD episode has no subjects")
    return torch.stack(values, dim=0)


def prd_loss(z_support: torch.Tensor, y_support: torch.Tensor, subject_support: torch.Tensor,
             z_query: torch.Tensor, y_query: torch.Tensor, subject_query: torch.Tensor,
             num_classes: int = 2) -> torch.Tensor:
    """Generic pairwise K-class PRD; K=2 is exactly the explicit binary formula."""
    if num_classes < 2:
        raise ValueError("PRD requires at least two classes")
    losses = []
    for pair in combinations(range(num_classes), 2):
        ds = _directions(z_support, y_support, subject_support, list(range(num_classes)), pair)
        dq = _directions(z_query, y_query, subject_query, list(range(num_classes)), pair)
        population = _unit(ds.mean(dim=0, keepdim=True))
        losses.append((1.0 - F.cosine_similarity(dq, population.expand_as(dq), dim=1)).mean())
    return torch.stack(losses).mean()


def binary_prd_explicit(z_support: torch.Tensor, y_support: torch.Tensor, subject_support: torch.Tensor,
                        z_query: torch.Tensor, y_query: torch.Tensor, subject_query: torch.Tensor) -> torch.Tensor:
    ds = _directions(z_support, y_support, subject_support, [0, 1], (0, 1))
    dq = _directions(z_query, y_query, subject_query, [0, 1], (0, 1))
    population = _unit(ds.mean(dim=0, keepdim=True))
    return (1.0 - F.cosine_similarity(dq, population.expand_as(dq), dim=1)).mean()
