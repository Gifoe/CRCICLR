from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class SubjectAction(ABC):
    name: str

    @abstractmethod
    def adapt(self, context_embeddings: np.ndarray, context_logits: np.ndarray) -> "SubjectAction": ...

    @abstractmethod
    def predict_proba(self, embeddings: np.ndarray, logits: np.ndarray | None = None) -> np.ndarray: ...

