from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler


EPS = 1e-7


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def probability_logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=float), EPS, 1.0 - EPS)
    return np.log(value) - np.log1p(-value)


def ce(labels: np.ndarray, logits: np.ndarray) -> float:
    return float(log_loss(labels, sigmoid(logits), labels=[0, 1]))


def ba(labels: np.ndarray, logits: np.ndarray) -> float:
    return float(balanced_accuracy_score(labels, np.asarray(logits) >= 0.0))


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    family: str
    strength: float = 0.5
    c_value: float = 0.1
    group: int = -1

    def payload(self) -> dict:
        return asdict(self)


@dataclass
class PopulationState:
    scaler: StandardScaler
    pca: PCA
    classifier: LogisticRegression
    dimension: int

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(np.asarray(features, dtype=np.float64)))

    def population_logit(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.classifier.decision_function(self.transform(features)), dtype=float)


def component_bank(dimension: int, include_latest: bool) -> list[ComponentSpec]:
    specs = [
        ComponentSpec("CALIBRATE_C01_A50", "calibration", strength=0.50, c_value=0.1),
        ComponentSpec("CALIBRATE_C1_A50", "calibration", strength=0.50, c_value=1.0),
        ComponentSpec("RIDGE_C001_A50", "ridge_head", strength=0.50, c_value=0.01),
        ComponentSpec("RIDGE_C01_A50", "ridge_head", strength=0.50, c_value=0.1),
        ComponentSpec("RIDGE_C1_A50", "ridge_head", strength=0.50, c_value=1.0),
        ComponentSpec("SHRINKAGE_LDA_A50", "shrinkage_lda", strength=0.50),
        ComponentSpec("PROTOTYPE_TRANSPORT_A50", "prototype_transport", strength=0.50),
        ComponentSpec("META_SGD_FULL_S025", "meta_sgd", strength=0.25),
        ComponentSpec("META_SGD_FULL_S050", "meta_sgd", strength=0.50),
    ]
    for group, _ in enumerate(np.array_split(np.arange(dimension), 8)):
        specs.append(ComponentSpec(f"META_SGD_PC_GROUP_{group:02d}", "projected_gradient", strength=0.50, group=group))
    if include_latest:
        specs.append(ComponentSpec("LATEST_SESSION_RIDGE_A50", "session_transport", strength=0.50, c_value=0.1))
    return specs


def fit_population_state(features: np.ndarray, labels: np.ndarray) -> PopulationState:
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    dimension = int(scaled.shape[1])
    pca = PCA(n_components=dimension, whiten=False, svd_solver="full").fit(scaled)
    z = pca.transform(scaled)
    classifier = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2_000,
        random_state=20260820,
    ).fit(z, labels)
    return PopulationState(scaler=scaler, pca=pca, classifier=classifier, dimension=dimension)


def base_logit(kind: str, raw_logit: np.ndarray, population_logit: np.ndarray) -> np.ndarray:
    if kind == "raw":
        return np.asarray(raw_logit, dtype=float)
    if kind == "population":
        return np.asarray(population_logit, dtype=float)
    if kind.startswith("blend"):
        weight = float(kind.replace("blend", "")) / 100.0
        return (1.0 - weight) * np.asarray(raw_logit, dtype=float) + weight * np.asarray(population_logit, dtype=float)
    raise ValueError(kind)


def _safe_binary_logistic(x: np.ndarray, y: np.ndarray, c_value: float) -> LogisticRegression | None:
    if len(np.unique(y)) < 2:
        return None
    try:
        return LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            solver="liblinear",
            max_iter=2_000,
            random_state=20260820,
        ).fit(x, y)
    except Exception:
        return None


def _blend(base: np.ndarray, candidate: np.ndarray, strength: float) -> np.ndarray:
    return (1.0 - float(strength)) * np.asarray(base, dtype=float) + float(strength) * np.asarray(candidate, dtype=float)


def action_logits(
    spec: ComponentSpec,
    fit_z: np.ndarray,
    fit_y: np.ndarray,
    fit_base: np.ndarray,
    target_z: np.ndarray,
    target_base: np.ndarray,
    fit_sessions: np.ndarray | None = None,
) -> np.ndarray:
    fit_z = np.asarray(fit_z, dtype=float)
    fit_y = np.asarray(fit_y, dtype=int)
    fit_base = np.asarray(fit_base, dtype=float)
    target_z = np.asarray(target_z, dtype=float)
    target_base = np.asarray(target_base, dtype=float)
    family = spec.family
    if family == "calibration":
        model = _safe_binary_logistic(fit_base[:, None], fit_y, spec.c_value)
        if model is None:
            return target_base.copy()
        return _blend(target_base, model.decision_function(target_base[:, None]), spec.strength)
    if family in {"ridge_head", "session_transport"}:
        selected = np.ones(len(fit_y), dtype=bool)
        if family == "session_transport" and fit_sessions is not None:
            selected = np.asarray(fit_sessions) == np.max(fit_sessions)
        model = _safe_binary_logistic(fit_z[selected], fit_y[selected], spec.c_value)
        if model is None:
            return target_base.copy()
        return _blend(target_base, model.decision_function(target_z), spec.strength)
    if family == "shrinkage_lda":
        if len(np.unique(fit_y)) < 2:
            return target_base.copy()
        try:
            model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(fit_z, fit_y)
            candidate = probability_logit(model.predict_proba(target_z)[:, 1])
            return _blend(target_base, candidate, spec.strength)
        except Exception:
            return target_base.copy()
    if family == "prototype_transport":
        if len(np.unique(fit_y)) < 2:
            return target_base.copy()
        mean0 = fit_z[fit_y == 0].mean(axis=0)
        mean1 = fit_z[fit_y == 1].mean(axis=0)
        direction = mean1 - mean0
        norm = float(np.linalg.norm(direction))
        if norm < EPS:
            return target_base.copy()
        midpoint = 0.5 * (mean0 + mean1)
        fit_score = ((fit_z - midpoint) @ direction / norm)[:, None]
        target_score = ((target_z - midpoint) @ direction / norm)[:, None]
        calibrator = _safe_binary_logistic(fit_score, fit_y, 0.1)
        if calibrator is None:
            return target_base.copy()
        return _blend(target_base, calibrator.decision_function(target_score), spec.strength)
    if family in {"meta_sgd", "projected_gradient"}:
        residual = sigmoid(fit_base) - fit_y
        gradient = np.mean(residual[:, None] * fit_z, axis=0)
        if family == "projected_gradient":
            groups = np.array_split(np.arange(fit_z.shape[1]), 8)
            selected = groups[int(spec.group)]
        else:
            selected = np.arange(fit_z.shape[1])
        fit_delta = -(fit_z[:, selected] @ gradient[selected])
        target_delta = -(target_z[:, selected] @ gradient[selected])
        rms = float(np.sqrt(np.mean(np.square(fit_delta))))
        if rms < EPS:
            return target_base.copy()
        return target_base + float(spec.strength) * target_delta / rms
    raise ValueError(spec)


def deterministic_history_halves(labels: np.ndarray, sessions: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    if sessions is not None and len(np.unique(sessions)) >= 2:
        values = np.sort(np.unique(sessions))
        return np.asarray(sessions) == values[0], np.asarray(sessions) == values[-1]
    first = np.zeros(len(labels), dtype=bool)
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        first[indices[::2]] = True
    return first, ~first


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator < EPS else float(np.dot(a, b) / denominator)


def history_context(z: np.ndarray, labels: np.ndarray, base: np.ndarray, sessions: np.ndarray | None) -> dict[str, float]:
    probability = sigmoid(base)
    entropy = -(probability * np.log(np.clip(probability, EPS, 1.0)) + (1.0 - probability) * np.log(np.clip(1.0 - probability, EPS, 1.0)))
    mean0 = z[labels == 0].mean(axis=0)
    mean1 = z[labels == 1].mean(axis=0)
    gradient = np.mean((probability - labels)[:, None] * z, axis=0)
    session_drift = 0.0
    if sessions is not None and len(np.unique(sessions)) >= 2:
        values = np.sort(np.unique(sessions))
        session_drift = float(np.linalg.norm(z[np.asarray(sessions) == values[0]].mean(axis=0) - z[np.asarray(sessions) == values[-1]].mean(axis=0)))
    return {
        "history_base_ce": ce(labels, base),
        "history_base_ba": ba(labels, base),
        "history_margin_mean": float(np.mean(np.abs(base))),
        "history_margin_std": float(np.std(np.abs(base))),
        "history_entropy_mean": float(np.mean(entropy)),
        "history_prototype_separation": float(np.linalg.norm(mean1 - mean0)),
        "history_gradient_norm": float(np.linalg.norm(gradient)),
        "history_session_drift": session_drift,
        "history_samples": float(len(labels)),
    }

def component_descriptors(
    spec: ComponentSpec,
    z: np.ndarray,
    labels: np.ndarray,
    base: np.ndarray,
    sessions: np.ndarray | None,
    candidate_history: np.ndarray,
) -> dict[str, float]:
    delta = np.asarray(candidate_history) - np.asarray(base)
    first, second = deterministic_history_halves(labels, sessions)
    prediction_first = action_logits(spec, z[first], labels[first], base[first], z, base, None if sessions is None else np.asarray(sessions)[first])
    prediction_second = action_logits(spec, z[second], labels[second], base[second], z, base, None if sessions is None else np.asarray(sessions)[second])
    delta_first = prediction_first - base
    delta_second = prediction_second - base
    cross_gain = 0.5 * (
        ce(labels[second], base[second]) - ce(labels[second], prediction_first[second])
        + ce(labels[first], base[first]) - ce(labels[first], prediction_second[first])
    )
    loss_direction = labels - sigmoid(base)
    return {
        "update_rms": float(np.sqrt(np.mean(np.square(delta)))),
        "update_mean_abs": float(np.mean(np.abs(delta))),
        "history_component_ce_gain": float(ce(labels, base) - ce(labels, candidate_history)),
        "history_component_ba_gain": float(ba(labels, candidate_history) - ba(labels, base)),
        "P_persistence": cosine(delta_first, delta_second),
        "D_decision_dependence": float(np.mean((candidate_history >= 0.0) != (base >= 0.0))),
        "G_task_overlap": cosine(delta, loss_direction),
        "R_history_transfer": float(cross_gain),
        "split_update_disagreement": float(np.sqrt(np.mean(np.square(delta_first - delta_second)))),
    }
