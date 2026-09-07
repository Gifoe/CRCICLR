"""Metric helpers."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
def score(y: np.ndarray,p: np.ndarray)->dict[str,float]: return {"BA":float(balanced_accuracy_score(y,p)),"macro_F1":float(f1_score(y,p,average="macro",zero_division=0)),"accuracy":float(accuracy_score(y,p)),"recall_0":float(recall_score(y,p,labels=[0],average=None,zero_division=0)[0]),"recall_1":float(recall_score(y,p,labels=[1],average=None,zero_division=0)[0])}

# Compatibility names required while importing the frozen v1 cache/split helper.
classification_metrics = score

def geometry(source_z: np.ndarray, source_y: np.ndarray, future_z: np.ndarray, future_y: np.ndarray, future_logits: np.ndarray) -> dict[str, float]:
    ds = source_z[source_y == 1].mean(0) - source_z[source_y == 0].mean(0)
    dq = future_z[future_y == 1].mean(0) - future_z[future_y == 0].mean(0)
    sep = float(np.linalg.norm(dq))
    means = [future_z[future_y == c].mean(0) for c in (0, 1)]
    per_trial_sq = [np.sum((future_z[future_y == c] - means[c]) ** 2, axis=1) for c in (0, 1)]
    scatter = float(np.mean(np.concatenate(per_trial_sq)))
    y = future_y.astype(int)
    margin = float(np.mean(future_logits[np.arange(len(y)), y] - future_logits[np.arange(len(y)), 1 - y]))
    return {
        "source_future_direction_cosine": float(ds @ dq / max(np.linalg.norm(ds) * np.linalg.norm(dq), 1e-12)),
        "class_separation": sep,
        "within_class_scatter": scatter,
        "fisher": sep * sep / max(scatter, 1e-12),
        "classifier_logit_margin": margin,
    }
