from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import cohen_kappa_score
import torch

from hsc_tta.actions import EntropyAdapter, ResidualAdapter, T3A
from hsc_tta.gpu.features import context_features, feature_columns, fit_pca, probability_entropy
from hsc_tta.gpu.training import load_task_head
from hsc_tta.prediction_sets import evaluate_prediction_sets


ACTIONS = ("no_tta", "t3a", "entropy_adapter")
ALPHAS = (0.10, 0.20)
LAMBDAS = np.r_[np.linspace(0.50, 0.99, 20), 1.0]


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def embedding_path(root: Path, dataset: str, subject: str) -> Path:
    return root / "outputs" / "full_experiment" / "embeddings" / dataset / f"{subject.split(':', 1)[1]}.h5"


def read_segment(root: Path, dataset: str, subject: str, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = np.asarray(indices, dtype=int)
    if index.ndim != 1 or np.any(np.diff(index) <= 0):
        raise ValueError("segment indices must be strictly increasing")
    with h5py.File(embedding_path(root, dataset, subject), "r") as handle:
        return (handle["embedding"][index].astype(np.float32), handle["label"][index].astype(np.int64),
                handle["quality_flags"][index].astype(np.float32))


def head_outputs(head: torch.nn.Module, embeddings: np.ndarray, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    logits, hidden = [], []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(embeddings), 2048):
            x = torch.as_tensor(embeddings[start : start + 2048], dtype=torch.float32, device=device)
            current_logits, current_hidden = head(x, return_hidden=True)
            logits.append(current_logits.cpu().numpy())
            hidden.append(current_hidden.cpu().numpy())
    return np.concatenate(logits), np.concatenate(hidden)


def adapt_context(head: torch.nn.Module, embeddings: np.ndarray, t3a_config: dict[str, object],
                  entropy_config: dict[str, object], device: str = "cuda") -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]], dict[str, object]]:
    logits, hidden = head_outputs(head, embeddings, device)
    base = softmax(logits.astype(np.float64), axis=1)
    probabilities: dict[str, np.ndarray] = {"no_tta": base}
    diagnostics: dict[str, dict[str, object]] = {
        "no_tta": {"adaptation_runtime": 0.0, "adaptation_status": "ok", "collapse_flag": False}}
    states: dict[str, object] = {"no_tta": None}
    started = time.perf_counter()
    t3a = T3A(head.classifier.weight.detach().cpu().numpy(), filter_k=int(t3a_config["filter_k"]),
              confidence=t3a_config.get("confidence"))
    initial = t3a.prototypes.copy()
    t3a.adapt(hidden, logits)
    probabilities["t3a"] = t3a.predict_proba(hidden)
    pseudo = base.argmax(1)
    proportions = np.bincount(pseudo, minlength=base.shape[1]) / len(pseudo)
    diagnostics["t3a"] = {
        "adaptation_runtime": time.perf_counter() - started, "adaptation_status": "ok", "collapse_flag": False,
        "prototype_shift": float(np.linalg.norm(t3a.prototypes - initial)),
        "pseudo_label_balance": float(1.0 - proportions.max()),
        "entropy_before": float(probability_entropy(base).mean()),
        "entropy_after": float(probability_entropy(probabilities["t3a"]).mean()),
    }
    states["t3a"] = t3a.prototypes.astype(np.float32)
    started = time.perf_counter()
    adapter = EntropyAdapter(head, steps=int(entropy_config["steps"]),
        learning_rate=float(entropy_config["learning_rate"]), beta=float(entropy_config["beta"]),
        gamma=float(entropy_config.get("gamma", 1e-4)), bottleneck=int(entropy_config.get("bottleneck", 64)),
        device=device)
    adapter.adapt(embeddings)
    probabilities["entropy_adapter"] = adapter.predict_proba(embeddings)
    diagnostics["entropy_adapter"] = {**adapter.diagnostics,
        "adaptation_runtime": time.perf_counter() - started,
        "adaptation_status": "failed" if adapter.diagnostics["collapse_flag"] else "ok",
        "pseudo_label_balance": 0.0, "prototype_shift": 0.0}
    states["entropy_adapter"] = {name: value.detach().cpu() for name, value in adapter.adapter.state_dict().items()}
    return probabilities, diagnostics, states


def predict_with_states(head: torch.nn.Module, embeddings: np.ndarray, states: dict[str, object],
                        device: str = "cuda") -> dict[str, np.ndarray]:
    logits, hidden = head_outputs(head, embeddings, device)
    result = {"no_tta": softmax(logits.astype(np.float64), axis=1)}
    prototypes = np.asarray(states["t3a"], dtype=np.float64)
    result["t3a"] = softmax(np.asarray(hidden, float) @ prototypes, axis=1)
    adapter = ResidualAdapter(200, 64).to(device)
    adapter.load_state_dict(states["entropy_adapter"])
    adapter.eval()
    with torch.inference_mode():
        output = []
        for start in range(0, len(embeddings), 2048):
            x = torch.as_tensor(embeddings[start : start + 2048], dtype=torch.float32, device=device)
            output.append(torch.softmax(head(adapter(x)), dim=1).cpu().numpy())
    result["entropy_adapter"] = np.concatenate(output)
    return result


def load_episode_frame(root: Path, dataset: str, seed: int) -> pd.DataFrame:
    frame = pd.read_parquet(root / "data" / "episodes_main120" / dataset / f"seed_{seed}.parquet")
    if frame["exclusion_reason"].notna().any():
        raise RuntimeError(f"formal {dataset} episode contains exclusions")
    return frame


def head_path(root: Path, dataset: str, seed: int) -> Path:
    inherited = "hmc" if dataset == "cap" else dataset
    return root / "outputs" / "full_experiment" / "task_heads" / inherited / f"seed_{seed}" / "task_head_best.pt"


def ensure_pca(root: Path, dataset: str, seed: int):
    source_dataset = "hmc" if dataset == "cap" else dataset
    path = root / "outputs" / "full_experiment" / "pca" / source_dataset / f"seed_{seed}" / "pca_32.joblib"
    if path.exists():
        return joblib.load(path)
    split = json.loads((root / "data" / "splits_internal" / source_dataset / f"seed_{seed}.json").read_text())
    values = []
    for subject in split["task_head_fit"]:
        with h5py.File(embedding_path(root, source_dataset, subject), "r") as handle:
            values.append(handle["embedding"][...].astype(np.float32))
    model = fit_pca(np.concatenate(values), path, seed)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".hash.json").write_text(json.dumps({"sha256": digest, "fit_subjects": split["task_head_fit"]}, indent=2), encoding="utf-8")
    return model


def _future_error(head: torch.nn.Module, context: np.ndarray, future: np.ndarray, future_y: np.ndarray,
                  action: str, config: dict[str, object], device: str) -> tuple[float, bool]:
    default_t3a = {"confidence": None, "filter_k": -1}
    default_entropy = {"steps": 5, "learning_rate": 1e-4, "beta": .1, "gamma": 1e-4, "bottleneck": 64}
    context_probs, diagnostics, states = adapt_context(head, context,
        config if action == "t3a" else default_t3a,
        config if action == "entropy_adapter" else default_entropy, device)
    future_probs = predict_with_states(head, future, states, device)[action]
    return float(np.mean(future_probs.argmax(1) != future_y)), bool(diagnostics[action].get("collapse_flag", False))


def select_action_hyperparameters(root: Path, dataset: str, seed: int, device: str = "cuda",
                                  resume: bool = True) -> dict[str, dict[str, object]]:
    if dataset == "cap":
        return select_action_hyperparameters(root, "hmc", seed, device, resume)
    output = root / "outputs" / "full_experiment" / "action_hyperparameters" / dataset / f"seed_{seed}"
    selected_file = output / "selected.json"
    if resume and selected_file.exists():
        return json.loads(selected_file.read_text())
    head, _ = load_task_head(head_path(root, dataset, seed), device)
    episodes = load_episode_frame(root, dataset, seed)
    meta = episodes[episodes["split_role"] == "meta_risk_train"]
    fold_map = json.loads((root / "data" / "splits_internal" / dataset / f"seed_{seed}.json").read_text())["meta_risk_folds"]
    subject_data = []
    for row in meta.itertuples(index=False):
        context, _, _ = read_segment(root, dataset, row.subject_id, row.context_indices)
        future, labels, _ = read_segment(root, dataset, row.subject_id, row.future_indices)
        base_logits, _ = head_outputs(head, future, device)
        base_error = float(np.mean(base_logits.argmax(1) != labels))
        subject_data.append((row.subject_id, context, future, labels, base_error, int(fold_map[row.subject_id])))
    candidates = {
        "t3a": [{"confidence": confidence, "filter_k": budget}
                for confidence in (None, .7, .8) for budget in (-1, 10, 20)],
        "entropy_adapter": [{"steps": steps, "learning_rate": lr, "beta": beta, "gamma": 1e-4, "bottleneck": 64}
                            for steps in (5, 10) for lr in (1e-4, 5e-4) for beta in (.1, .5)],
    }
    selected: dict[str, dict[str, object]] = {}
    all_rows = []
    for action, grid in candidates.items():
        scores = []
        for candidate_index, config in enumerate(grid):
            rows = []
            for subject, context, future, labels, base_error, fold in subject_data:
                error, collapse = _future_error(head, context, future, labels, action, config, device)
                rows.append({"subject_id": subject, "fold": fold, "error": error,
                             "negative_adaptation": int(error > base_error), "collapse": int(collapse)})
            frame = pd.DataFrame(rows)
            fold_error = frame.groupby("fold")["error"].mean().mean()
            score = {"action": action, "candidate": candidate_index, **config,
                     "grouped_validation_mean_argmax_error": float(fold_error),
                     "negative_adaptation_rate": float(frame["negative_adaptation"].mean()),
                     "class_collapse_rate": float(frame["collapse"].mean())}
            scores.append(score)
            all_rows.append(score)
        if action == "t3a":
            best = min(scores, key=lambda x: (x["grouped_validation_mean_argmax_error"], x["negative_adaptation_rate"],
                not (x["confidence"] is None and x["filter_k"] == -1), 10**9 if x["filter_k"] == -1 else x["filter_k"]))
        else:
            best = min(scores, key=lambda x: (x["grouped_validation_mean_argmax_error"], x["negative_adaptation_rate"],
                x["class_collapse_rate"], x["steps"], x["learning_rate"]))
        selected[action] = {key: value for key, value in best.items() if key not in {
            "action", "candidate", "grouped_validation_mean_argmax_error", "negative_adaptation_rate", "class_collapse_rate"}}
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output / "grid_results.csv", index=False)
    selected_file.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return selected


def evaluate_history_seed(root: Path, dataset: str, seed: int, device: str = "cuda", resume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = root / "outputs" / "full_experiment"
    fpath = base / "subject_context_features" / dataset / f"seed_{seed}.parquet"
    dpath = base / "action_context_diagnostics" / dataset / f"seed_{seed}.parquet"
    opath = base / "historical_action_outcomes" / dataset / f"seed_{seed}.parquet"
    if resume and all(path.exists() for path in (fpath, dpath, opath)):
        return pd.read_parquet(fpath), pd.read_parquet(dpath), pd.read_parquet(opath)
    pca = ensure_pca(root, dataset, seed)
    config = select_action_hyperparameters(root, "hmc" if dataset == "cap" else dataset, seed, device, resume)
    head, _ = load_task_head(head_path(root, dataset, seed), device)
    episodes = load_episode_frame(root, dataset, seed)
    roles = {"target_site_calibration"} if dataset == "cap" else {"meta_risk_train", "conformal_calibration"}
    episodes = episodes[episodes["split_role"].isin(roles)]
    feature_rows, diagnostic_rows, outcome_rows = [], [], []
    for row in episodes.itertuples(index=False):
        context, _, quality = read_segment(root, dataset, row.subject_id, row.context_indices)
        future, future_y, _ = read_segment(root, dataset, row.subject_id, row.future_indices)
        context_probs, diagnostics, states = adapt_context(head, context, config["t3a"], config["entropy_adapter"], device)
        future_probs = predict_with_states(head, future, states, device)
        base_prob = context_probs["no_tta"]
        for action in ACTIONS:
            identity = {"dataset": dataset, "seed": seed, "subject_id": row.subject_id,
                        "episode_id": row.episode_id, "split_role": row.split_role}
            features = context_features(context, context_probs[action], base_prob, pca, quality, action, diagnostics[action])
            feature_rows.append({**identity, **features})
            diagnostic_rows.append({**identity, "action": action,
                **{k: json.dumps(v) if isinstance(v, list) else v for k, v in diagnostics[action].items()}})
            kappa = cohen_kappa_score(future_y, future_probs[action].argmax(1))
            for alpha in ALPHAS:
                for curve in evaluate_prediction_sets(future_probs[action], future_y, LAMBDAS):
                    outcome_rows.append({**identity, "action": action, "alpha": alpha,
                                         "cohen_kappa": float(kappa), **curve})
    features, diagnostics_frame, outcomes = map(pd.DataFrame, (feature_rows, diagnostic_rows, outcome_rows))
    atomic_parquet(features, fpath); atomic_parquet(diagnostics_frame, dpath); atomic_parquet(outcomes, opath)
    return features, diagnostics_frame, outcomes
