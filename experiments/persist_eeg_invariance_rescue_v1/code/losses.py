from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _domain_groups(
    features: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    offset: int,
    maximum_domains: int = 8,
    maximum_samples: int = 16,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    available = [int(value) for value in torch.unique(domains).tolist() if int(value) >= 0]
    available.sort()
    if not available:
        return []
    shift = int(offset) % len(available)
    available = (available[shift:] + available[:shift])[:maximum_domains]
    groups = []
    for domain in available:
        positions = torch.flatnonzero(domains == domain)
        if len(positions) < 2:
            continue
        positions = positions[:maximum_samples]
        groups.append((features[positions], labels[positions]))
    return groups


def rbf_mmd(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    total = torch.cat([source, target], dim=0).float()
    distance = torch.cdist(total, total).square()
    nonzero = distance.detach()[distance.detach() > 0]
    bandwidth = nonzero.median() if len(nonzero) else torch.tensor(1.0, device=total.device)
    bandwidth = bandwidth.clamp_min(1e-6)
    kernels = sum(torch.exp(-distance / (bandwidth * scale)) for scale in (0.25, 0.5, 1.0, 2.0, 4.0))
    n = len(source)
    return kernels[:n, :n].mean() + kernels[n:, n:].mean() - 2 * kernels[:n, n:].mean()


def marginal_mmd_loss(
    features: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor, offset: int
) -> torch.Tensor:
    groups = _domain_groups(features, labels, domains, offset)
    if len(groups) < 2:
        return features.sum() * 0.0
    values = [rbf_mmd(groups[index][0], groups[index + 1][0]) for index in range(0, len(groups) - 1, 2)]
    return torch.stack(values).mean() if values else features.sum() * 0.0


def conditional_alignment_loss(
    features: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor, offset: int
) -> torch.Tensor:
    groups = _domain_groups(features, labels, domains, offset)
    values: list[torch.Tensor] = []
    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            for label in (0, 1):
                a = groups[left][0][groups[left][1] == label]
                b = groups[right][0][groups[right][1] == label]
                if len(a) and len(b):
                    values.append(torch.mean(torch.square(a.mean(0) - b.mean(0))))
    return torch.stack(values).mean() if values else features.sum() * 0.0


def coral(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if len(source) < 2 or len(target) < 2:
        return source.sum() * 0.0
    source_centered = source - source.mean(0, keepdim=True)
    target_centered = target - target.mean(0, keepdim=True)
    source_cov = source_centered.T @ source_centered / (len(source) - 1)
    target_cov = target_centered.T @ target_centered / (len(target) - 1)
    return torch.square(source.mean(0) - target.mean(0)).mean() + torch.square(source_cov - target_cov).mean()


def coral_loss(features: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor, offset: int) -> torch.Tensor:
    groups = _domain_groups(features, labels, domains, offset)
    values = [coral(groups[left][0], groups[right][0]) for left in range(len(groups)) for right in range(left + 1, len(groups))]
    return torch.stack(values).mean() if values else features.sum() * 0.0


def _same_label_permutation(labels: torch.Tensor, shift: int) -> torch.Tensor:
    result = torch.arange(len(labels), device=labels.device)
    for label in torch.unique(labels):
        positions = torch.flatnonzero(labels == label)
        if len(positions) > 1:
            result[positions] = positions.roll(int(shift) % len(positions))
    return result


def supervised_contrastive_loss(
    projection: torch.Tensor,
    labels: torch.Tensor,
    offset: int,
    temperature: float = 0.07,
) -> torch.Tensor:
    if len(projection) < 4:
        return projection.sum() * 0.0
    first = _same_label_permutation(labels, 1 + offset % 7)
    second = _same_label_permutation(labels, 2 + offset % 11)
    mix = 0.90 + 0.01 * (offset % 10)
    view1 = F.normalize(mix * projection + (1.0 - mix) * projection[first], dim=1)
    view2 = F.normalize(mix * projection + (1.0 - mix) * projection[second], dim=1)
    features = torch.cat([view1, view2], dim=0)
    repeated_labels = torch.cat([labels, labels], dim=0)
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positive = repeated_labels[:, None].eq(repeated_labels[None, :]) & ~self_mask
    denominator = torch.logsumexp(logits.masked_fill(self_mask, float("-inf")), dim=1)
    log_probability = logits - denominator[:, None]
    count = positive.sum(dim=1).clamp_min(1)
    loss = -(log_probability.masked_fill(~positive, 0.0).sum(dim=1) / count)
    return loss.mean()

