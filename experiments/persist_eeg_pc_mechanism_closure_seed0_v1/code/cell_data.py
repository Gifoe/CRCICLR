"""Memory-bounded, split-isolated stage materialization for one frozen cell."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sampling_core import PairPlan, TrialView, select_pairs_from_norms


@dataclass(frozen=True)
class StageSelection:
    activations: np.ndarray
    logits: np.ndarray
    row_indices: np.ndarray
    recipient_local: np.ndarray
    donor_local: np.ndarray
    recipient_global: np.ndarray
    donor_global: np.ndarray
    donor_label: np.ndarray
    coverage: tuple[dict[str, object], ...]


def materialize_stage(runner: Any, view: TrialView, *, model: str, task: str,
                      fold: int, stage: str, batch_size: int = 16,
                      precomputed_norms: np.ndarray | None = None) -> StageSelection:
    """Search all eligible donors by norm, retain only selected activations.

    The first pass never stores full high-dimensional stage activations. The
    second pass visits just the deterministic recipient/donor union. This is
    semantically identical to selecting pairs from a full activation array.
    """
    if view.split not in {"TRAIN", "OUTER_DEVELOPMENT"} or batch_size < 1:
        raise ValueError("invalid legal stage view or batch size")
    if runner.m.training or runner.head.training:
        raise RuntimeError("frozen model and head must be in eval mode")
    if precomputed_norms is None:
        norms = stage_norms(runner, view, (stage,), batch_size=batch_size)[stage]
    else:
        norms = np.asarray(precomputed_norms, np.float64)
        if norms.shape != (len(view.x),) or not np.isfinite(norms).all():
            raise ValueError("invalid precomputed stage norms")
    plan: PairPlan = select_pairs_from_norms(norms, view.labels, view.subjects, view.sessions,
                                             model=model, task=task, fold=fold, stage=stage, split=view.split)
    if not len(plan.recipient):
        raise RuntimeError(f"no legal cross-class pairs at {stage} in {view.split}")
    selected = np.unique(np.concatenate([plan.recipient, plan.donor]))
    activations: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for start in range(0, len(selected), batch_size):
        indices = selected[start:start + batch_size]
        with torch.inference_mode():
            native, _, z = runner.all(runner.tensor(view.x[indices]))
        activations.append(native[stage].reshape(len(indices), -1).detach().float().cpu().numpy())
        logits.append(z.detach().float().cpu().numpy())
    a_selected = np.concatenate(activations).astype(np.float32)
    z_selected = np.concatenate(logits).astype(np.float32)
    if not np.isfinite(a_selected).all() or not np.isfinite(z_selected).all():
        raise RuntimeError("nonfinite selected native activation/logit")
    lookup = np.full(len(view.x), -1, dtype=np.int64)
    lookup[selected] = np.arange(len(selected), dtype=np.int64)
    recipient_local, donor_local = lookup[plan.recipient], lookup[plan.donor]
    if np.any(recipient_local < 0) or np.any(donor_local < 0):
        raise RuntimeError("selected pair materialization mismatch")
    return StageSelection(a_selected, z_selected, selected, recipient_local, donor_local,
                          plan.recipient, plan.donor, plan.donor_label, plan.coverage)


def stage_norms(runner: Any, view: TrialView, stages: tuple[str, ...], *,
                batch_size: int = 16) -> dict[str, np.ndarray]:
    """One native forward pass per batch gives norms for every audited stage."""
    if not len(stages) or batch_size < 1 or view.split not in {"TRAIN", "OUTER_DEVELOPMENT"}:
        raise ValueError("invalid stage-norm request")
    if runner.m.training or runner.head.training:
        raise RuntimeError("frozen model and head must be in eval mode")
    result = {stage: np.empty(len(view.x), np.float64) for stage in stages}
    for start in range(0, len(view.x), batch_size):
        stop = min(start + batch_size, len(view.x))
        with torch.inference_mode():
            native, _, _ = runner.all(runner.tensor(view.x[start:stop]))
        for stage in stages:
            if stage not in native:
                raise KeyError(f"stage absent from frozen runner: {stage}")
            a = native[stage].reshape(stop - start, -1).detach().float().cpu().numpy()
            result[stage][start:stop] = np.linalg.norm(a.astype(np.float64), axis=1)
    if any(not np.isfinite(values).all() for values in result.values()):
        raise RuntimeError("nonfinite native stage norm")
    return result
