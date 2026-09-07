"""Low-rank query/expert interaction features for regularized linear scoring."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD


def interaction(query: np.ndarray, expert_token: np.ndarray, rank: int, seed: int) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    token = np.asarray(expert_token, dtype=np.float32)
    q = TruncatedSVD(n_components=min(int(rank), query.shape[1] - 1), random_state=int(seed)).fit_transform(query)
    k = TruncatedSVD(n_components=min(int(rank), token.shape[1] - 1), random_state=int(seed) + 1).fit_transform(token)
    width = min(q.shape[1], k.shape[1])
    return np.column_stack([q, k, q[:, :width] * k[:, :width]]).astype(np.float32)
