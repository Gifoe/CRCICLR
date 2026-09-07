from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA

from hsc_tta.prediction_sets import prediction_sets


def fit_pca(embeddings: np.ndarray, path: str | Path, seed: int) -> PCA:
    model = PCA(n_components=32, svd_solver="randomized", random_state=seed)
    model.fit(np.asarray(embeddings, dtype=np.float32))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


def probability_entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    return -(p * np.log(np.maximum(p, 1e-12))).sum(1)


def context_features(
    embeddings: np.ndarray,
    probabilities: np.ndarray,
    base_probabilities: np.ndarray,
    pca: PCA,
    quality_flags: np.ndarray,
    action: str,
    diagnostics: dict[str, object],
) -> dict[str, float | int | str | bool]:
    z = np.asarray(embeddings, np.float32)
    p = np.asarray(probabilities, float)
    base = np.asarray(base_probabilities, float)
    reduced = pca.transform(z)
    entropy = probability_entropy(p)
    maximum = p.max(1)
    prediction = p.argmax(1)
    base_entropy = probability_entropy(base)
    result: dict[str, float | int | str | bool] = {
        "action": action,
        "n_context": int(len(z)),
        "entropy_q10": float(np.quantile(entropy, .1)),
        "entropy_q50": float(np.quantile(entropy, .5)),
        "entropy_q90": float(np.quantile(entropy, .9)),
        "max_probability_q10": float(np.quantile(maximum, .1)),
        "max_probability_q50": float(np.quantile(maximum, .5)),
        "max_probability_q90": float(np.quantile(maximum, .9)),
        "prediction_instability": float(np.mean(prediction[1:] != prediction[:-1])) if len(prediction) > 1 else 0.0,
        "channel_mask_summary": 1.0,
        "missingness": 0.0,
        "action_entropy_change": float(entropy.mean() - base_entropy.mean()),
        "prediction_kl": float(np.mean(np.sum(p * (np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(base, 1e-12))), axis=1))),
        "prototype_shift": float(diagnostics.get("prototype_shift", 0.0)),
        "adapter_update_norm": float(diagnostics.get("adapter_update_norm", 0.0)),
        "pseudo_label_balance": float(diagnostics.get("pseudo_label_balance", 0.0)),
        "collapse_status": int(bool(diagnostics.get("collapse_flag", False))),
        "action_no_tta": int(action == "no_tta"),
        "action_t3a": int(action == "t3a"),
        "action_entropy_adapter": int(action == "entropy_adapter"),
    }
    for index in range(32):
        result[f"pca_mean_{index}"] = float(reduced[:, index].mean())
        result[f"pca_std_{index}"] = float(reduced[:, index].std())
    for cls, proportion in enumerate(np.bincount(prediction, minlength=p.shape[1]) / len(prediction)):
        result[f"predicted_class_proportion_{cls}"] = float(proportion)
    quality = np.asarray(quality_flags, float)
    for index, name in enumerate(("quality_nonfinite_rate", "quality_flat_rate", "quality_peak_abs")):
        result[name] = float(np.nanmean(quality[:, index]))
    return result


def context_set_utility(probabilities: np.ndarray, selected_lambda: float) -> tuple[float, float]:
    sets = prediction_sets(np.asarray(probabilities, float), np.asarray([selected_lambda], float))[:, 0, :]
    sizes = sets.sum(1)
    return float(sizes.mean()), float(np.mean(sizes == 1))


def feature_columns(row: dict[str, object] | list[str]) -> list[str]:
    columns = list(row) if isinstance(row, dict) else list(row)
    excluded = {"dataset", "seed", "subject_id", "episode_id", "split_role", "action", "alpha",
                "adaptation_status", "adaptation_runtime", "parameter_reset_hash"}
    return sorted(name for name in columns if name not in excluded and not name.startswith("future_"))
