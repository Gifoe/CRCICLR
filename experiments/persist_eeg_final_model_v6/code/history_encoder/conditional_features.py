from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from protocol.datasets import FoldDataset


EPS = 1e-6


@dataclass
class FeatureContext:
    scaler: StandardScaler
    protected_importance: np.ndarray
    random_importance: np.ndarray
    population_weight: np.ndarray
    population_bias: float


def _contrast(z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return z[labels == 1].mean(axis=0) - z[labels == 0].mean(axis=0)


def fit_context(data: FoldDataset, seed: int) -> FeatureContext:
    sessions = list(data.history_sessions) + [data.future_session]
    fit_mask = data.mask(list(data.model_fit_subjects), sessions)
    scaler = StandardScaler().fit(data.embeddings[fit_mask])
    z_all = scaler.transform(data.embeddings)
    stability_rows = []
    for subject in data.model_fit_subjects:
        history_mask = data.mask([subject], list(data.history_sessions))
        future_mask = data.mask([subject], [data.future_session])
        d_history = _contrast(z_all[history_mask], data.metadata.loc[history_mask, "label"].to_numpy(int))
        d_future = _contrast(z_all[future_mask], data.metadata.loc[future_mask, "label"].to_numpy(int))
        d_history /= max(np.linalg.norm(d_history), EPS)
        d_future /= max(np.linalg.norm(d_future), EPS)
        stability_rows.append(d_history * d_future)
    stability = np.stack(stability_rows)
    sign_consistency = np.mean(stability > 0, axis=0)
    magnitude = np.median(np.abs(stability), axis=0)
    score = np.maximum(sign_consistency - 0.5, 0.0) * np.sqrt(magnitude + EPS)
    order = np.argsort(np.argsort(score, kind="stable"), kind="stable")
    omega = order.astype(float) / max(len(order) - 1, 1)
    omega = np.clip((omega - 0.35) / 0.65, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    random_omega = omega[rng.permutation(len(omega))]
    # Closed-form balanced population direction supplies a stable scalar without
    # giving the target future labels to the history encoder.
    labels = data.metadata.loc[fit_mask, "label"].to_numpy(int)
    z_fit = z_all[fit_mask]
    population_weight = _contrast(z_fit, labels)
    midpoint = 0.5 * (z_fit[labels == 0].mean(axis=0) + z_fit[labels == 1].mean(axis=0))
    population_bias = -float(midpoint @ population_weight)
    return FeatureContext(scaler, omega, random_omega, population_weight, population_bias)


def subject_features(
    data: FoldDataset,
    context: FeatureContext,
    subject: str,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history_mask = data.mask([subject], list(data.history_sessions))
    future_mask = data.mask([subject], [data.future_session])
    z_history = context.scaler.transform(data.embeddings[history_mask])
    z_future = context.scaler.transform(data.embeddings[future_mask])
    y_history = data.metadata.loc[history_mask, "label"].to_numpy(int)
    y_future = data.metadata.loc[future_mask, "label"].to_numpy(int)
    means = np.stack([z_history[y_history == label].mean(axis=0) for label in (0, 1)])
    variances = np.stack([z_history[y_history == label].var(axis=0) for label in (0, 1)])
    scale = np.sqrt(0.5 * variances.sum(axis=0) + 0.1)
    midpoint = means.mean(axis=0)
    normalized = (z_future - midpoint) / scale
    direction = (means[1] - means[0]) / scale
    direction /= max(np.linalg.norm(direction), EPS)
    if variant == "generic_bilinear":
        adapted = normalized
    elif variant == "persist_protected":
        omega = context.protected_importance
        adapted = omega[None, :] * z_future + (1.0 - omega[None, :]) * normalized
    elif variant == "random_protected":
        omega = context.random_importance
        adapted = omega[None, :] * z_future + (1.0 - omega[None, :]) * normalized
    elif variant == "affine_only":
        adapted = normalized
        direction = np.zeros_like(direction)
    else:
        raise ValueError(variant)
    interaction = adapted * direction[None, :]
    distance = np.column_stack(
        [
            np.mean(((z_future - means[0]) / scale) ** 2, axis=1),
            np.mean(((z_future - means[1]) / scale) ** 2, axis=1),
        ]
    )
    population_margin = z_future @ context.population_weight + context.population_bias
    session_contrasts = []
    for session in data.history_sessions:
        mask = data.mask([subject], [session])
        z = context.scaler.transform(data.embeddings[mask])
        y = data.metadata.loc[mask, "label"].to_numpy(int)
        value = _contrast(z, y)
        value /= max(np.linalg.norm(value), EPS)
        session_contrasts.append(value)
    if len(session_contrasts) >= 2:
        stability = float(np.mean([a @ b for index, a in enumerate(session_contrasts) for b in session_contrasts[index + 1 :]]))
    else:
        # K=1 stability uses two deterministic, label-stratified halves of the
        # legal history rather than inventing or observing a future session.
        parts = []
        for parity in (0, 1):
            selected = np.arange(len(y_history)) % 2 == parity
            value = _contrast(z_history[selected], y_history[selected])
            value /= max(np.linalg.norm(value), EPS)
            parts.append(value)
        stability = float(parts[0] @ parts[1])
    scalars = np.column_stack(
        [
            distance,
            distance[:, 0] - distance[:, 1],
            population_margin,
            np.full(len(z_future), np.linalg.norm((means[1] - means[0]) / scale)),
            np.full(len(z_future), stability),
        ]
    )
    features = np.column_stack([z_future, adapted, interaction, scalars]).astype(np.float32)
    return features, y_future, data.metadata.loc[future_mask, "trial_uid"].astype(str).to_numpy()
