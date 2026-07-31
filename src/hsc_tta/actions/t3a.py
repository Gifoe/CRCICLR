from __future__ import annotations

import numpy as np
from scipy.special import softmax
from .base import SubjectAction


class T3A(SubjectAction):
    name = "t3a"

    def __init__(self, confidence: float = 0.7):
        self.confidence = confidence
        self.prototypes_: np.ndarray | None = None

    def adapt(self, context_embeddings: np.ndarray, context_logits: np.ndarray) -> "T3A":
        z = np.asarray(context_embeddings, dtype=float)
        p = softmax(np.asarray(context_logits, dtype=float), axis=1)
        if z.ndim != 2 or p.shape[0] != z.shape[0]:
            raise ValueError("context embeddings/logits must share sample dimension")
        labels, conf = p.argmax(1), p.max(1)
        keep = conf >= self.confidence
        k = p.shape[1]
        global_mean = z.mean(0)
        prototypes = []
        for cls in range(k):
            mask = keep & (labels == cls)
            prototypes.append(z[mask].mean(0) if np.any(mask) else global_mean)
        proto = np.vstack(prototypes)
        norms = np.linalg.norm(proto, axis=1, keepdims=True)
        self.prototypes_ = proto / np.maximum(norms, 1e-12)
        return self

    def predict_proba(self, embeddings: np.ndarray, logits: np.ndarray | None = None) -> np.ndarray:
        if self.prototypes_ is None:
            raise RuntimeError("adapt must be called per subject before prediction")
        z = np.asarray(embeddings, dtype=float)
        z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
        return softmax(z @ self.prototypes_.T, axis=1)

