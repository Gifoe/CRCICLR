from __future__ import annotations

import numpy as np
from scipy.special import softmax
from .base import SubjectAction


class EntropyAdapterMock(SubjectAction):
    """Deterministic CPU schema mock. It is not a real EEG TTA adapter."""

    name = "entropy_adapter"

    def __init__(self, temperature: float = 0.95):
        self.temperature = temperature
        self.adapted_ = False

    def adapt(self, context_embeddings: np.ndarray, context_logits: np.ndarray) -> "EntropyAdapterMock":
        if np.asarray(context_embeddings).shape[0] != np.asarray(context_logits).shape[0]:
            raise ValueError("context embeddings/logits must share sample dimension")
        self.adapted_ = True
        return self

    def predict_proba(self, embeddings: np.ndarray, logits: np.ndarray | None = None) -> np.ndarray:
        if not self.adapted_ or logits is None:
            raise RuntimeError("adapt must be called and logits supplied")
        return softmax(np.asarray(logits, dtype=float) / self.temperature, axis=1)

