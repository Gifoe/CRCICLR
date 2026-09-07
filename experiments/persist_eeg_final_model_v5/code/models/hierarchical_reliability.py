from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ReliabilityConfig:
    family: str
    pool_name: str
    pool: tuple[int, ...]
    history_sessions: tuple[int, ...]
    shrinkage: float = 0.0
    temperature: float = 1.0
    c_value: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "pool_name": self.pool_name,
            "pool": list(self.pool),
            "history_sessions": list(self.history_sessions),
            "shrinkage": self.shrinkage,
            "temperature": self.temperature,
            "C": self.c_value,
        }


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def _history_mask(frame, subjects: Sequence[str], sessions: Sequence[int]) -> np.ndarray:
    return frame.subject_id.astype(str).isin(set(map(str, subjects))).to_numpy() & frame.session_id.astype(int).isin(set(map(int, sessions))).to_numpy()


def global_reliability(frame, train_subjects: Sequence[str], config: ReliabilityConfig) -> np.ndarray:
    mask = _history_mask(frame, train_subjects, config.history_sessions)
    labels = frame.loc[mask, "label"].to_numpy(int)
    logits = frame.loc[mask, [f"margin_{name}" for name in frame.attrs["expert_names"]]].to_numpy(float)[:, config.pool]
    if config.family == "soft_nll":
        probability = _sigmoid(logits)
        nll = -np.mean(labels[:, None] * np.log(np.clip(probability, 1e-7, 1.0)) + (1 - labels[:, None]) * np.log(np.clip(1 - probability, 1e-7, 1.0)), axis=0)
        return nll
    return np.mean((logits >= 0).astype(int) == labels[:, None], axis=0)


def _local_probability(
    frame,
    subject: str,
    target_logits: np.ndarray,
    config: ReliabilityConfig,
    global_prior: np.ndarray,
) -> np.ndarray:
    pool = np.asarray(config.pool, dtype=int)
    history = frame.subject_id.astype(str).eq(str(subject)).to_numpy() & frame.session_id.astype(int).isin(config.history_sessions).to_numpy()
    labels = frame.loc[history, "label"].to_numpy(int)
    all_logits = frame.loc[history, [f"margin_{name}" for name in frame.attrs["expert_names"]]].to_numpy(float)
    logits = all_logits[:, pool]
    target = np.asarray(target_logits, dtype=float)[:, pool]
    target_probability = _sigmoid(target)
    n = max(len(labels), 1)
    if config.family in {"soft_ba", "hard_best"}:
        local = np.mean((logits >= 0).astype(int) == labels[:, None], axis=0)
        posterior = (n * local + config.shrinkage * global_prior) / (n + config.shrinkage)
        if config.family == "hard_best":
            return target_probability[:, int(np.argmax(posterior))]
        score = config.temperature * (posterior - np.max(posterior))
        weight = np.exp(score)
        weight /= weight.sum()
        return target_probability @ weight
    if config.family == "soft_nll":
        history_probability = _sigmoid(logits)
        local = -np.mean(labels[:, None] * np.log(np.clip(history_probability, 1e-7, 1.0)) + (1 - labels[:, None]) * np.log(np.clip(1 - history_probability, 1e-7, 1.0)), axis=0)
        posterior = (n * local + config.shrinkage * global_prior) / (n + config.shrinkage)
        score = config.temperature * (posterior - np.max(posterior))
        weight = np.exp(score)
        weight /= weight.sum()
        return target_probability @ weight
    if config.family == "local_logistic":
        model = LogisticRegression(
            C=float(config.c_value),
            penalty="l2",
            solver="lbfgs",
            max_iter=500,
            class_weight="balanced",
            random_state=0,
        )
        model.fit(logits, labels)
        return model.predict_proba(target)[:, 1]
    raise ValueError(config.family)


def predict_subjects(
    frame,
    target_frame,
    subjects: Sequence[str],
    config: ReliabilityConfig,
    train_subjects: Sequence[str],
) -> np.ndarray:
    expert_names = list(frame.attrs["expert_names"])
    columns = [f"margin_{name}" for name in expert_names]
    prior = global_reliability(frame, train_subjects, config)
    result = np.full(len(target_frame), np.nan, dtype=float)
    for subject in map(str, subjects):
        mask = target_frame.subject_id.astype(str).eq(subject).to_numpy()
        if not mask.any():
            continue
        target_logits = target_frame.loc[mask, columns].to_numpy(float)
        result[mask] = _local_probability(frame, subject, target_logits, config, prior)
    if np.isnan(result).any():
        raise RuntimeError("Incomplete cross-session reliability predictions")
    return result


def starting_configurations(families: set[str] | None = None) -> list[ReliabilityConfig]:
    families = families or {"soft_ba", "soft_nll", "hard_best", "local_logistic"}
    pools = {
        "stable_deep": (0, 2),
        "competent3": (0, 1, 2),
        "no_conformer4": (0, 1, 2, 4),
        "all5": (0, 1, 2, 3, 4),
    }
    histories = {"S1S2": (0, 1), "S2": (1,)}
    result: list[ReliabilityConfig] = []
    for pool_name, pool in pools.items():
        for _, history in histories.items():
            if "soft_ba" in families:
                for shrinkage in (20.0, 100.0):
                    for temperature in (2.0, 8.0):
                        result.append(ReliabilityConfig("soft_ba", pool_name, pool, history, shrinkage, temperature))
            if "soft_nll" in families:
                for shrinkage in (20.0, 100.0):
                    for temperature in (1.0, 4.0):
                        result.append(ReliabilityConfig("soft_nll", pool_name, pool, history, shrinkage, temperature))
            if "hard_best" in families:
                result.append(ReliabilityConfig("hard_best", pool_name, pool, history, 100.0, 1.0))
            if "local_logistic" in families:
                for c_value in (0.01, 0.1, 1.0):
                    result.append(ReliabilityConfig("local_logistic", pool_name, pool, history, c_value=c_value))
    return result
