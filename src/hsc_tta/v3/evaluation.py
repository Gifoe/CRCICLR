from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import f1_score

from hsc_tta.actions import T3A
from hsc_tta.prediction_sets import evaluate_prediction_sets
from hsc_tta.v2.development_surfaces import _labels, _outputs, _tokens, load_source_model

from .actions import AdapterConfig, CorrectedResidualAdapter, FrozenT3AAction, T3AConfig, source_probabilities


LAMBDAS = np.r_[np.linspace(.50, .99, 20), 1.0]
ALPHAS = (.10, .20)
ACTIONS = ("no_tta", "official_t3a", "robust_residual_adapter")


def deterministic_seed(dataset: str, seed: int, subject_id: str, action: str) -> int:
    payload = f"{dataset}|{seed}|{subject_id}|{action}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def safe_point(probabilities: np.ndarray, labels: np.ndarray, alpha: float,
               lambdas: np.ndarray = LAMBDAS) -> dict[str, float]:
    curve = evaluate_prediction_sets(probabilities, labels, lambdas)
    valid = [row for row in curve if row["future_risk"] <= alpha]
    selected = min(valid, key=lambda row: (row["average_set_size"], row["lambda_index"]))
    return {**selected, "alpha": float(alpha)}


class SubjectEvaluator:
    """Evaluate frozen V3 actions while keeping the caller responsible for split access."""

    def __init__(self, root: str | Path, dataset: str, seed: int, device: str = "cuda"):
        self.root = Path(root); self.dataset = dataset; self.seed = int(seed); self.device = device
        self.model, self.source_payload = load_source_model(self.root, dataset, seed, device)

    def load_episode(self, row) -> dict[str, object]:
        subject = str(getattr(row, "subject_id", row.name))
        return {
            "subject_id": subject,
            "adapt": _tokens(self.root, self.dataset, subject, np.asarray(row.adapt_indices, int)),
            "probe": _tokens(self.root, self.dataset, subject, np.asarray(row.probe_indices, int)),
            "future": _tokens(self.root, self.dataset, subject, np.asarray(row.future_indices, int)),
            "labels": _labels(self.root, self.dataset, subject, np.asarray(row.future_indices, int)),
        }

    def source(self, episode: dict[str, object]) -> dict[str, np.ndarray]:
        if "source_probabilities" in episode:
            return episode["source_probabilities"]
        return {part: source_probabilities(self.model, episode[part], self.device)
                for part in ("adapt", "probe", "future")}

    def prepare_episode(self, row) -> dict[str, object]:
        episode = self.load_episode(row); probabilities = {}; hidden = {}; logits = {}
        for part in ("adapt", "probe", "future"):
            current_logits, current_hidden = _outputs(self.model, episode[part], self.device)
            logits[part] = current_logits; hidden[part] = current_hidden
            probabilities[part] = softmax(current_logits.astype(np.float64), axis=1)
        episode["source_probabilities"] = probabilities; episode["source_hidden"] = hidden; episode["source_logits"] = logits
        return episode

    def action(self, episode: dict[str, object], action: str, config: dict[str, object]) -> dict[str, object]:
        subject = str(episode["subject_id"])
        torch.manual_seed(deterministic_seed(self.dataset, self.seed, subject, action))
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(deterministic_seed(self.dataset, self.seed, subject, action))
        if action == "official_t3a":
            if "source_hidden" in episode:
                current = T3AConfig(**config); initial = self.model.classifier.weight.detach().cpu().numpy().copy()
                instance = T3A(initial, current.filter_k, current.confidence_threshold)
                try:
                    instance.adapt(episode["source_hidden"]["adapt"], episode["source_logits"]["adapt"])
                    prototypes = (1 - current.prototype_interpolation) * initial + current.prototype_interpolation * instance.prototypes
                    prototypes = prototypes / np.maximum(np.linalg.norm(prototypes, axis=0, keepdims=True), 1e-12)
                    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()); digest.update(prototypes.tobytes())
                    shift = float(np.linalg.norm(prototypes - initial) / np.sqrt(initial.size))
                    return {"available": True, "status": "frozen", "state_hash": digest.hexdigest(),
                            "probe": softmax(episode["source_hidden"]["probe"] @ prototypes, axis=1),
                            "future": softmax(episode["source_hidden"]["future"] @ prototypes, axis=1),
                            "diagnostics": {"status": "frozen", "normalized_update_magnitude": shift,
                                            "prototype_shift": shift, "support_count": len(instance.supports)}, "config": config}
                except Exception as error:
                    source = self.source(episode)
                    return {"available": False, "status": f"unavailable:{type(error).__name__}:{error}",
                            "state_hash": "fallback-source", "probe": source["probe"], "future": source["future"],
                            "diagnostics": {"status": f"unavailable:{type(error).__name__}:{error}"}, "config": config}
            instance = FrozenT3AAction(self.model, T3AConfig(**config), self.device)
        elif action == "robust_residual_adapter":
            instance = CorrectedResidualAdapter(self.model, AdapterConfig(**config), self.device)
        else:
            raise ValueError(f"unsupported action: {action}")
        instance.adapt_on_adapt(episode["adapt"])
        status = instance.failure_status(); diagnostics = instance.diagnostics()
        if status != "adapted":
            return {"available": False, "status": status, "state_hash": "fallback-source",
                    "probe": source_probabilities(self.model, episode["probe"], self.device),
                    "future": source_probabilities(self.model, episode["future"], self.device),
                    "diagnostics": diagnostics, "config": config}
        state_hash = instance.freeze_state()
        return {"available": True, "status": "frozen", "state_hash": state_hash,
                "probe": instance.predict_probe(episode["probe"]),
                "future": instance.predict_future(episode["future"]),
                "diagnostics": instance.diagnostics(), "config": config}


def subject_action_rows(dataset: str, seed: int, subject_id: str, action: str,
                        source_future: np.ndarray, action_future: np.ndarray, labels: np.ndarray,
                        *, available: bool, status: str, config_id: str,
                        lambdas: np.ndarray = LAMBDAS, alphas: tuple[float, ...] = ALPHAS) -> list[dict[str, object]]:
    source_error = float(np.mean(source_future.argmax(1) != labels))
    action_error = float(np.mean(action_future.argmax(1) != labels))
    macro_f1 = float(f1_score(labels, action_future.argmax(1), labels=np.arange(action_future.shape[1]),
                              average="macro", zero_division=0))
    rows = []
    for alpha in alphas:
        source_point = safe_point(source_future, labels, alpha, lambdas)
        action_point = safe_point(action_future, labels, alpha, lambdas)
        gain = float(source_point["average_set_size"] - action_point["average_set_size"])
        rows.append({"dataset": dataset, "seed": int(seed), "subject_id": subject_id, "action": action,
                     "config_id": config_id, "alpha": float(alpha), "action_available": bool(available),
                     "status": status, "source_argmax_error": source_error, "argmax_error": action_error,
                     "classification_degradation": action_error - source_error, "macro_f1": macro_f1,
                     "source_safe_size": source_point["average_set_size"],
                     "safe_size": action_point["average_set_size"], "safe_risk": action_point["future_risk"],
                     "singleton_rate": action_point["singleton_rate"], "lambda_index": int(action_point["lambda_index"]),
                     "oracle_gain": gain, "relative_gain": gain / max(float(source_point["average_set_size"]), 1e-12)})
    return rows
