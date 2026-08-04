from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .episodes import validate_three_way


@dataclass(frozen=True)
class PseudoEpisode:
    subject_id: str
    pseudo_id: str
    adapt_indices: np.ndarray
    probe_indices: np.ndarray
    future_indices: np.ndarray


def rolling_pseudo_episodes(subject_id: str, available: np.ndarray, *, n_adapt: int, n_probe: int,
                            n_future: int, stride: int) -> list[PseudoEpisode]:
    indices = np.asarray(available, dtype=int); width = n_adapt + n_probe + n_future
    if min(n_adapt, n_probe, n_future, stride) <= 0:
        raise ValueError("positive pseudo-episode sizes and stride required")
    episodes = []
    for start in range(0, len(indices) - width + 1, stride):
        a = indices[start:start+n_adapt]; p = indices[start+n_adapt:start+n_adapt+n_probe]
        v = indices[start+n_adapt+n_probe:start+width]; validate_three_way(a, p, v)
        episodes.append(PseudoEpisode(subject_id, f"{subject_id}:pseudo:{start}", a, p, v))
    return episodes


def assert_grouped_assignment(episodes: list[PseudoEpisode], fold_by_pseudo_id: dict[str, int]) -> None:
    by_subject: dict[str, set[int]] = {}
    for episode in episodes:
        if episode.pseudo_id not in fold_by_pseudo_id:
            raise ValueError("missing pseudo-episode fold")
        by_subject.setdefault(episode.subject_id, set()).add(fold_by_pseudo_id[episode.pseudo_id])
    leaked = [subject for subject, folds in by_subject.items() if len(folds) != 1]
    if leaked:
        raise RuntimeError(f"pseudo-episodes split across folds: {leaked}")
