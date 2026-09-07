from __future__ import annotations

import numpy as np
from scipy.special import softmax


class T3A:
    """Author-code-equivalent episodic T3A in the task head's hidden space."""

    name = "t3a"

    def __init__(self, classifier_weight: np.ndarray | float, filter_k: int = 20, confidence: float | None = None):
        weights = np.asarray(classifier_weight, dtype=np.float64)
        self._legacy_confidence = None
        if weights.ndim == 0:
            self._legacy_confidence = float(weights)
            self.source_supports = np.empty((0, 0), dtype=np.float64)
            self.filter_k = int(filter_k)
            self.confidence = self._legacy_confidence
            return
        if weights.ndim != 2:
            raise ValueError("classifier_weight must be [classes, hidden]")
        self.source_supports = weights.copy()
        self.filter_k = int(filter_k)
        self.confidence = confidence
        source_logits = weights @ weights.T
        self.source_labels = source_logits.argmax(1)
        self.source_entropy = self._entropy(source_logits)
        self.reset()

    @staticmethod
    def _entropy(logits: np.ndarray) -> np.ndarray:
        p = softmax(np.asarray(logits, dtype=np.float64), axis=1)
        return -(p * np.log(np.maximum(p, 1e-12))).sum(1)

    def reset(self) -> None:
        self.supports = self.source_supports.copy()
        self.support_labels = self.source_labels.copy()
        self.entropies = self.source_entropy.copy()
        self.prototypes = self._compute_prototypes()

    def _compute_prototypes(self) -> np.ndarray:
        selected: list[int] = []
        for cls in range(self.source_supports.shape[0]):
            indices = np.flatnonzero(self.support_labels == cls)
            indices = indices[np.argsort(self.entropies[indices], kind="stable")]
            selected.extend(indices[: self.filter_k].tolist() if self.filter_k >= 0 else indices.tolist())
        if not selected:
            raise RuntimeError("T3A support filtering removed all supports")
        supports = self.supports[selected]
        labels = self.support_labels[selected]
        normalized = supports / np.maximum(np.linalg.norm(supports, axis=1, keepdims=True), 1e-12)
        one_hot = np.eye(self.source_supports.shape[0], dtype=np.float64)[labels]
        weights = normalized.T @ one_hot
        return weights / np.maximum(np.linalg.norm(weights, axis=0, keepdims=True), 1e-12)

    def adapt(self, context_hidden: np.ndarray, context_logits: np.ndarray) -> "T3A":
        z = np.asarray(context_hidden, dtype=np.float64)
        logits = np.asarray(context_logits, dtype=np.float64)
        if z.ndim != 2 or logits.shape[0] != z.shape[0]:
            raise ValueError("context hidden/logits must share sample dimension")
        if self._legacy_confidence is not None:
            if z.shape[1] != logits.shape[1]:
                raise ValueError("legacy T3A fixture requires hidden dimension equal to class count")
            self.source_supports = np.eye(logits.shape[1], dtype=np.float64)
            self.source_labels = np.arange(logits.shape[1])
            self.source_entropy = self._entropy(self.source_supports @ self.source_supports.T)
            self.reset()
        probabilities = softmax(logits, axis=1)
        labels = probabilities.argmax(1)
        keep = np.ones(z.shape[0], dtype=bool)
        if self.confidence is not None:
            keep &= probabilities.max(1) >= float(self.confidence)
        self.supports = np.concatenate((self.source_supports, z[keep]), axis=0)
        self.support_labels = np.concatenate((self.source_labels, labels[keep]), axis=0)
        self.entropies = np.concatenate((self.source_entropy, self._entropy(logits[keep])), axis=0)
        self.prototypes = self._compute_prototypes()
        return self

    def predict_proba(self, hidden: np.ndarray) -> np.ndarray:
        z = np.asarray(hidden, dtype=np.float64)
        return softmax(z @ self.prototypes, axis=1)
