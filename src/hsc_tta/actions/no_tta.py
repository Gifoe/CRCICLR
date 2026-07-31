from __future__ import annotations

import numpy as np
from scipy.special import softmax
from .base import SubjectAction


class NoTTA(SubjectAction):
    name = "no_tta"

    def adapt(self, context_embeddings: np.ndarray, context_logits: np.ndarray) -> "NoTTA":
        return self

    def predict_proba(self, embeddings: np.ndarray, logits: np.ndarray | None = None) -> np.ndarray:
        if logits is None:
            raise ValueError("no_tta requires base logits")
        return softmax(np.asarray(logits, dtype=float), axis=1)

