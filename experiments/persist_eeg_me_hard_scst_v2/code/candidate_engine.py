"""Admissibility-gated candidate generation and upper-tail margin loss."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from mixed_effects import MixedEffectsBank


EPS = 1e-8


@dataclass
class CandidateSet:
    deltas: np.ndarray
    alphas: np.ndarray
    targets: np.ndarray
    structured: np.ndarray
    valid: np.ndarray
    support_pass: np.ndarray
    semantic_pass: np.ndarray
    teacher_pass: np.ndarray
    norm_pass: np.ndarray
    whitened_norm: np.ndarray


def margins(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    truth = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    return truth - masked.max(1).values


class AdmissibilityEngine:
    def __init__(self, features: np.ndarray, labels: np.ndarray, subjects: np.ndarray, row_ids: np.ndarray, bank: MixedEffectsBank):
        self.features = np.asarray(features, np.float64)
        self.labels = np.asarray(labels, np.int64)
        self.subjects = np.asarray(subjects).astype(str)
        self.row_ids = np.asarray(row_ids, np.int64)
        self.bank = bank
        if not np.array_equal(self.row_ids, bank.row_ids):
            raise ValueError("engine/bank row mismatch")
        # Local support uses clean training representations only.  Radius is
        # the 95th percentile of each point's 3NN mean distance within class.
        self.support_radius: dict[int, float] = {}
        for label in sorted(np.unique(self.labels).tolist()):
            values = self.features[self.labels == label]
            distance = np.linalg.norm(values[:, None] - values[None, :], axis=2)
            np.fill_diagonal(distance, np.inf)
            knn = np.partition(distance, kth=min(2, len(values) - 1), axis=1)[:, : min(3, len(values) - 1)]
            self.support_radius[int(label)] = float(np.quantile(knn.mean(1), 0.95))

    def _neighbors(self, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distance = np.linalg.norm(self.features - candidate[None], axis=1)
        order = np.argsort(distance, kind="stable")[:3]
        return order, distance[order]

    def generate(
        self,
        position: int,
        teacher_logits_fn,
        *,
        k_targets: int,
        alphas: tuple[float, ...],
        seed: int,
        factorized: bool,
        include_random: bool,
    ) -> CandidateSet:
        position = int(position)
        source = self.subjects[position]
        target_pool = np.asarray([s for s in self.bank.subjects if s != source])
        rng = np.random.default_rng(int(seed))
        if len(target_pool) > k_targets:
            target_pool = target_pool[rng.choice(len(target_pool), size=k_targets, replace=False)]
        rows: list[tuple[np.ndarray, float, str, bool]] = []
        for target in target_pool:
            structured = self.bank.direction(int(self.row_ids[position]), str(target), factorized=factorized)
            directions = [(structured, True)]
            if include_random:
                directions.append((self.bank.hard_random(structured, rng), False))
            for direction, is_structured in directions:
                for alpha in alphas:
                    rows.append((direction, float(alpha), str(target), is_structured))
        if not rows:
            shape = (0, self.features.shape[1])
            empty = np.empty(0)
            return CandidateSet(np.empty(shape, np.float32), empty, empty.astype(str), empty.astype(bool), empty.astype(bool), empty.astype(bool), empty.astype(bool), empty.astype(bool), empty)
        directions = np.stack([row[0] for row in rows]).astype(np.float32)
        alpha_values = np.asarray([row[1] for row in rows], np.float32)
        candidates = self.features[position][None] + alpha_values[:, None] * directions
        target_values = np.asarray([row[2] for row in rows])
        structured_values = np.asarray([row[3] for row in rows], bool)
        support_pass = np.zeros(len(rows), bool)
        semantic_pass = np.zeros(len(rows), bool)
        for idx, candidate in enumerate(candidates):
            near, dist = self._neighbors(candidate)
            support_pass[idx] = float(dist.mean()) <= self.support_radius[int(self.labels[position])]
            semantic_pass[idx] = int(np.sum(self.labels[near] == self.labels[position])) >= 2
        teacher_logits = np.asarray(teacher_logits_fn(candidates), np.float64)
        y = int(self.labels[position])
        other = np.max(np.delete(teacher_logits, y, axis=1), axis=1)
        teacher_pass = (teacher_logits.argmax(1) == y) & ((teacher_logits[:, y] - other) > 0)
        whitened = np.asarray([self.bank.whitened_norm(alpha * delta) for alpha, delta in zip(alpha_values, directions)])
        norm_pass = whitened <= self.bank.norm_radius + 1e-12
        valid = support_pass & semantic_pass & teacher_pass & norm_pass
        return CandidateSet(directions, alpha_values, target_values, structured_values, valid, support_pass, semantic_pass, teacher_pass, norm_pass, whitened)


def match_structured_random(valid: np.ndarray, structured: np.ndarray, *, seed: int) -> np.ndarray:
    """Deterministically subsample larger side to matched valid counts."""
    valid = np.asarray(valid, bool)
    structured = np.asarray(structured, bool)
    left = np.flatnonzero(valid & structured)
    right = np.flatnonzero(valid & ~structured)
    count = min(len(left), len(right))
    keep = np.zeros(len(valid), bool)
    if not count:
        return keep
    rng = np.random.default_rng(int(seed))
    keep[rng.choice(left, count, replace=False)] = True
    keep[rng.choice(right, count, replace=False)] = True
    return keep


def upper_tail_loss(
    clean_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_owner: torch.Tensor,
    valid: torch.Tensor,
    *,
    q: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss with detached ranking/top-tail membership and clean-correct gate."""
    if q not in (0.25, 0.50):
        raise ValueError(q)
    clean_margin = margins(clean_logits, labels)
    owner = candidate_owner.long()
    candidate_margin = margins(candidate_logits, labels[owner])
    hardness = (clean_margin.detach()[owner] - candidate_margin.detach())
    clean_correct = clean_logits.detach().argmax(1).eq(labels)
    selected: list[torch.Tensor] = []
    for anchor in range(len(labels)):
        positions = torch.nonzero(valid.bool() & owner.eq(anchor), as_tuple=False).flatten()
        if not clean_correct[anchor] or not len(positions):
            continue
        count = int(np.ceil(q * len(positions)))
        top = torch.topk(hardness[positions], k=count, largest=True, sorted=False).indices.detach()
        selected.append(positions[top])
    if not selected:
        zero = candidate_logits.sum() * 0.0
        return zero, {"selected": 0.0, "hardness": 0.0}
    picked = torch.cat(selected).detach()
    loss = F.softplus(-candidate_margin[picked]).mean()
    return loss, {"selected": float(len(picked)), "hardness": float(hardness[picked].mean().detach().cpu())}


def uniform_margin_loss(
    clean_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_owner: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    owner = candidate_owner.long()
    candidate_margin = margins(candidate_logits, labels[owner])
    clean_correct = clean_logits.detach().argmax(1).eq(labels)
    picked = valid.bool() & clean_correct[owner]
    if not torch.any(picked):
        return candidate_logits.sum() * 0.0, {"selected": 0.0, "hardness": 0.0}
    hardness = margins(clean_logits, labels).detach()[owner] - candidate_margin.detach()
    return F.softplus(-candidate_margin[picked]).mean(), {"selected": float(picked.sum()), "hardness": float(hardness[picked].mean().cpu())}

