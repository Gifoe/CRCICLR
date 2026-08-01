from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from hsc_tta.certification import (apply_critical_index_certificate, critical_index_table,
                                   fit_actionwise_simultaneous_quantile)
from hsc_tta.freeze import file_sha256
from hsc_tta.gpu.experiment import (ACTIONS, ALPHAS, LAMBDAS, adapt_context, atomic_parquet,
                                    ensure_pca, evaluate_history_seed, head_path, load_episode_frame,
                                    predict_with_states, read_segment, select_action_hyperparameters)
from hsc_tta.gpu.features import context_features, context_set_utility, feature_columns
from hsc_tta.gpu.training import load_task_head
from hsc_tta.prediction_sets import evaluate_prediction_sets
from hsc_tta.risk_prediction import CriticalIndexPredictor
from hsc_tta.selection import select_safe_action


L = 20


def _model_dir(root: Path, dataset: str, seed: int, alpha: float) -> Path:
    return root / "outputs" / "full_experiment" / "critical_index_models" / dataset / f"seed_{seed}" / f"alpha_{alpha:.2f}"


def _calibration_dir(root: Path, dataset: str, seed: int, alpha: float) -> Path:
    return root / "outputs" / "full_experiment" / "calibration" / dataset / f"seed_{seed}" / f"alpha_{alpha:.2f}"


def _critical_frame(outcomes: pd.DataFrame) -> pd.DataFrame:
    columns = {"dataset", "seed", "episode_id", "subject_id", "action", "alpha", "lambda",
               "lambda_index", "future_risk"}
    return critical_index_table(outcomes[list(columns)])


def train_predictors_and_calibrate(root: Path, dataset: str, seed: int, device: str = "cuda",
                                   resume: bool = True) -> tuple[pd.DataFrame, dict[float, dict[str, object]]]:
    features, _, outcomes = evaluate_history_seed(root, dataset, seed, device, resume)
    critical = _critical_frame(outcomes)
    joined = features.merge(critical, on=["dataset", "seed", "episode_id", "subject_id", "action"],
                            validate="one_to_many")
    predictions = []
    quantiles: dict[float, dict[str, object]] = {}
    for alpha in ALPHAS:
        current = joined[np.isclose(joined["alpha"], alpha)].copy()
        if dataset == "cap":
            predictor = CriticalIndexPredictor.load(_model_dir(root, "hmc", seed, alpha) / "model.joblib")
        else:
            columns = feature_columns(features.columns.tolist())
            meta = current[current["split_role"] == "meta_risk_train"].copy()
            fold_map = json.loads((root / "data" / "splits_internal" / dataset / f"seed_{seed}.json").read_text())["meta_risk_folds"]
            meta["fixed_fold"] = meta["subject_id"].map(fold_map).astype(int)
            predictor = CriticalIndexPredictor(columns, alpha=alpha, n_nontrivial_lambdas=L, random_state=seed)
            predictor.fit(meta, fold_column="fixed_fold")
            directory = _model_dir(root, dataset, seed, alpha)
            directory.mkdir(parents=True, exist_ok=True)
            predictor.save(directory / "model.joblib")
            (directory / "feature_columns.json").write_text(json.dumps(columns, indent=2), encoding="utf-8")
            (directory / "cv_results.json").write_text(json.dumps(predictor.cv_results, indent=2), encoding="utf-8")
            (directory / "model_hash.json").write_text(json.dumps({"model_id": predictor.model_id,
                "file_sha256": file_sha256(directory / "model.joblib")}, indent=2), encoding="utf-8")
        current["predicted_critical_index"] = predictor.predict(current)
        current["predictor_source_dataset"] = "hmc" if dataset == "cap" else dataset
        predictions.append(current[["dataset", "seed", "episode_id", "subject_id", "split_role", "action",
                                    "alpha", "critical_index", "predicted_critical_index", "predictor_source_dataset"]])
        calibration_role = "target_site_calibration" if dataset == "cap" else "conformal_calibration"
        calibration = current[current["split_role"] == calibration_role].copy()
        quantile = fit_actionwise_simultaneous_quantile(calibration, delta=.10, n_nontrivial_lambdas=L)
        directory = _calibration_dir(root, dataset, seed, alpha)
        directory.mkdir(parents=True, exist_ok=True)
        residuals = calibration.copy()
        residuals["residual"] = residuals["critical_index"] - residuals["predicted_critical_index"]
        subject_residuals = residuals.groupby(["dataset", "seed", "episode_id", "subject_id", "alpha"], as_index=False)["residual"].max()
        atomic_parquet(subject_residuals, directory / "calibration_residuals.parquet")
        q_payload = {key: getattr(quantile, key) for key in quantile.__dataclass_fields__}
        (directory / "calibration_quantile.json").write_text(json.dumps(q_payload, indent=2), encoding="utf-8")
        (directory / "calibration_subjects.json").write_text(json.dumps(sorted(calibration["subject_id"].unique()), indent=2), encoding="utf-8")
        (directory / "quantile_provenance.json").write_text(json.dumps({"score": "max_actions(J-Jhat)",
            "delta": .10, "order": quantile.order_k, "provenance": quantile.provenance}, indent=2), encoding="utf-8")
        (directory / "calibration_model_hashes.json").write_text(json.dumps({
            "predictor": file_sha256(_model_dir(root, "hmc" if dataset == "cap" else dataset, seed, alpha) / "model.joblib"),
            "task_head": file_sha256(head_path(root, dataset, seed))}, indent=2), encoding="utf-8")
        quantiles[alpha] = q_payload
    result = pd.concat(predictions, ignore_index=True)
    path = root / "outputs" / "full_experiment" / "critical_index_predictions" / dataset / f"seed_{seed}.parquet"
    atomic_parquet(result, path)
    return result, quantiles


def _collect_hashes(paths: list[Path]) -> dict[str, dict[str, object]]:
    return {str(path): {"sha256": file_sha256(path), "size": path.stat().st_size} for path in sorted(paths)}


def freeze_methods(root: Path) -> Path:
    output = root / "outputs" / "full_experiment" / "freezes"
    output.mkdir(parents=True, exist_ok=True)
    repository = root / "repo"
    git_sha = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    method = """method: HSC-TTA critical-index certificate
actions: [no_tta, t3a, entropy_adapter]
alphas: [0.10, 0.20]
delta: 0.10
lambda_grid: 20 values from 0.50 to 0.99 plus 1.0 sentinel
selector: U-only lexicographic
legacy_empirical_bernstein_formal_path: false
"""
    (output / "METHOD_FREEZE.yaml").write_text(method, encoding="utf-8")
    split_files = list((root / "data" / "splits").glob("*/*")) + list((root / "data" / "splits_internal").glob("*/*"))
    episode_files = list((root / "data" / "episodes_main120").glob("*/*"))
    (output / "SPLIT_FREEZE.json").write_text(json.dumps({"splits": _collect_hashes(split_files),
        "episodes_main120": _collect_hashes(episode_files)}, indent=2), encoding="utf-8")
    channel = json.loads((repository / "CHANNEL_PROTOCOL.json").read_text())
    (output / "CHANNEL_PROTOCOL.json").write_text(json.dumps(channel, indent=2), encoding="utf-8")
    action_files = list((root / "outputs" / "full_experiment" / "action_hyperparameters").glob("*/*/selected.json"))
    (output / "ACTION_HYPERPARAMETERS.yaml").write_text("files:\n" + "".join(
        f"  - path: {path}\n    sha256: {file_sha256(path)}\n" for path in sorted(action_files)), encoding="utf-8")
    backbone = root / "outputs" / "full_experiment" / "environment" / "backbone_revision.json"
    (output / "BACKBONE_REVISION.json").write_text(backbone.read_text(), encoding="utf-8")
    heads = list((root / "outputs" / "full_experiment" / "task_heads").glob("*/*/task_head_best.pt"))
    (output / "TASK_HEAD_CONFIG.yaml").write_text("checkpoints:\n" + "".join(
        f"  - path: {path}\n    sha256: {file_sha256(path)}\n" for path in sorted(heads)), encoding="utf-8")
    cert = {"alphas": ALPHAS, "delta": .10, "lambda_grid": LAMBDAS.tolist(),
            "predictor": "alpha-specific HistGradientBoostingRegressor", "calibration_score": "max over 3 actions only"}
    (output / "CRITICAL_INDEX_CERTIFICATE_CONFIG.yaml").write_text(json.dumps(cert, indent=2), encoding="utf-8")
    predictor_files = list((root / "outputs" / "full_experiment" / "critical_index_models").glob("*/*/*/model.joblib"))
    q_files = list((root / "outputs" / "full_experiment" / "calibration").glob("*/*/*/calibration_quantile.json"))
    pca_files = list((root / "outputs" / "full_experiment" / "pca").glob("*/*/pca_32.joblib"))
    frozen_files = list(output.glob("METHOD_FREEZE.yaml")) + list(output.glob("SPLIT_FREEZE.json")) + [
        output / "CHANNEL_PROTOCOL.json", output / "ACTION_HYPERPARAMETERS.yaml", output / "BACKBONE_REVISION.json",
        output / "TASK_HEAD_CONFIG.yaml", output / "CRITICAL_INDEX_CERTIFICATE_CONFIG.yaml"]
    manifest = {"freeze_version": "hsc-gpu-full-v1", "git_commit": git_sha,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "files": _collect_hashes(frozen_files),
        "task_heads": _collect_hashes(heads), "pca": _collect_hashes(pca_files),
        "critical_index_predictors": _collect_hashes(predictor_files), "calibration_quantiles": _collect_hashes(q_files),
        "checkpoint_sha256": "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178",
        "cbramod_revision": "0ff6be918985689e7df679bc731ffb70e6c6224f", "alphas": ALPHAS,
        "delta": .10, "lambda_grid": LAMBDAS.tolist(), "pre_outcome_decisions": None}
    path = output / "EXPERIMENT_FREEZE_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def verify_method_freeze(path: Path, require_decisions: bool = False) -> dict[str, object]:
    payload = json.loads(path.read_text())
    for group in ("files", "task_heads", "pca", "critical_index_predictors", "calibration_quantiles"):
        for name, entry in payload[group].items():
            if file_sha256(name) != entry["sha256"]:
                raise RuntimeError(f"freeze hash mismatch: {name}")
    if require_decisions:
        decisions = payload.get("pre_outcome_decisions")
        if not decisions:
            raise RuntimeError("pre-outcome decisions are not frozen")
        for name, entry in decisions.items():
            if file_sha256(name) != entry["sha256"]:
                raise RuntimeError(f"pre-outcome decision hash mismatch: {name}")
    return payload


def _quantile(root: Path, dataset: str, seed: int, alpha: float) -> dict[str, object]:
    return json.loads((_calibration_dir(root, dataset, seed, alpha) / "calibration_quantile.json").read_text())


def make_final_decisions(root: Path, dataset: str, seed: int, device: str = "cuda") -> tuple[pd.DataFrame, pd.DataFrame]:
    verify_method_freeze(root / "outputs" / "full_experiment" / "freezes" / "EXPERIMENT_FREEZE_MANIFEST.json")
    pca = ensure_pca(root, dataset, seed)
    config = select_action_hyperparameters(root, "hmc" if dataset == "cap" else dataset, seed, device, True)
    head, _ = load_task_head(head_path(root, dataset, seed), device)
    episodes = load_episode_frame(root, dataset, seed)
    final_role = "external_final_test" if dataset == "cap" else "final_test"
    episodes = episodes[episodes["split_role"] == final_role]
    candidate_rows, decision_rows, feature_rows = [], [], []
    state_dir = root / "outputs" / "full_experiment" / "action_states_final" / dataset / f"seed_{seed}"
    state_dir.mkdir(parents=True, exist_ok=True)
    for row in episodes.itertuples(index=False):
        # Context-only API: no future indices or labels are passed to this read.
        context, _, quality = read_segment(root, dataset, row.subject_id, row.context_indices)
        probabilities, diagnostics, states = adapt_context(head, context, config["t3a"], config["entropy_adapter"], device)
        state_file = state_dir / f"{row.subject_id.split(':',1)[1]}.pt"
        torch.save({"dataset": dataset, "seed": seed, "subject_id": row.subject_id,
                    "episode_id": row.episode_id, "states": states}, state_file)
        base_probability = probabilities["no_tta"]
        by_action = {}
        for action in ACTIONS:
            feature = {"dataset": dataset, "seed": seed, "subject_id": row.subject_id,
                       "episode_id": row.episode_id, "split_role": final_role,
                       **context_features(context, probabilities[action], base_probability, pca, quality,
                                          action, diagnostics[action])}
            feature_rows.append(feature)
            by_action[action] = feature
        for alpha in ALPHAS:
            source = "hmc" if dataset == "cap" else dataset
            predictor = CriticalIndexPredictor.load(_model_dir(root, source, seed, alpha) / "model.joblib")
            q = _quantile(root, dataset, seed, alpha)
            current = pd.DataFrame([by_action[action] for action in ACTIONS])
            predicted = predictor.predict(current)
            certified = np.clip(np.ceil(predicted + float(q["q_alpha"])), 0, L).astype(int)
            subject_candidates = []
            for index, action in enumerate(ACTIONS):
                status = str(diagnostics[action].get("adaptation_status", "ok"))
                critical = L if status != "ok" else int(certified[index])
                selected_lambda = float(LAMBDAS[critical])
                average_size, singleton = context_set_utility(probabilities[action], selected_lambda)
                candidate = {"dataset": dataset, "seed": seed, "episode_id": row.episode_id,
                    "subject_id": row.subject_id, "alpha": alpha, "action": action,
                    "predicted_critical_index": float(predicted[index]), "q_alpha": float(q["q_alpha"]),
                    "certified_critical_index": critical, "selected_lambda": selected_lambda,
                    "nontrivial_candidate": bool(critical < L and average_size < probabilities[action].shape[1] and selected_lambda < 1),
                    "context_average_set_size": average_size, "context_singleton_rate": singleton,
                    "adaptation_cost": ACTIONS.index(action), "adaptation_status": status,
                    "adaptation_runtime": float(diagnostics[action].get("adaptation_runtime", 0.0)),
                    "n_classes": probabilities[action].shape[1], "n_nontrivial_lambdas": L}
                candidate_rows.append(candidate); subject_candidates.append(candidate)
            selection = select_safe_action(pd.DataFrame(subject_candidates))
            selected_row = selection["selected_row"]
            decision_rows.append({"dataset": dataset, "seed": seed, "episode_id": row.episode_id,
                "subject_id": row.subject_id, "alpha": alpha, "selected_action": selection["selected_action"],
                "selected_lambda": selection["selected_lambda"], "certified_critical_index": selection["certified_critical_index"],
                "status": selection["status"], "certified": selection["certified"],
                "nontrivial_certified": selection["nontrivial_certified"],
                "selection_reason": selection["selection_reason"], "state_file": str(state_file),
                "q_alpha": float(selected_row["q_alpha"]),
                "context_average_set_size": float(selected_row["context_average_set_size"]),
                "context_singleton_rate": float(selected_row["context_singleton_rate"]),
                "adaptation_status": str(selected_row["adaptation_status"]),
                "adaptation_runtime": float(selected_row["adaptation_runtime"])})
    candidates, decisions = pd.DataFrame(candidate_rows), pd.DataFrame(decision_rows)
    atomic_parquet(candidates, root / "outputs" / "full_experiment" / "certified_action_candidates" / dataset / f"seed_{seed}.parquet")
    atomic_parquet(decisions, root / "outputs" / "full_experiment" / "pre_outcome_decisions" / dataset / f"seed_{seed}.parquet")
    atomic_parquet(pd.DataFrame(feature_rows), root / "outputs" / "full_experiment" / "subject_context_features" / dataset / f"seed_{seed}_final.parquet")
    return candidates, decisions


def freeze_decisions(root: Path) -> None:
    path = root / "outputs" / "full_experiment" / "freezes" / "EXPERIMENT_FREEZE_MANIFEST.json"
    payload = verify_method_freeze(path)
    files = list((root / "outputs" / "full_experiment" / "pre_outcome_decisions").glob("*/*.parquet"))
    files += list((root / "outputs" / "full_experiment" / "certified_action_candidates").glob("*/*.parquet"))
    files += list((root / "outputs" / "full_experiment" / "action_states_final").glob("*/*/*.pt"))
    payload["pre_outcome_decisions"] = _collect_hashes(files)
    payload["decision_freeze_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    verify_method_freeze(path, require_decisions=True)


def evaluate_final_outcomes(root: Path, dataset: str, seed: int, device: str = "cuda") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    verify_method_freeze(root / "outputs" / "full_experiment" / "freezes" / "EXPERIMENT_FREEZE_MANIFEST.json", True)
    head, _ = load_task_head(head_path(root, dataset, seed), device)
    episodes = load_episode_frame(root, dataset, seed).set_index("subject_id")
    candidates = pd.read_parquet(root / "outputs" / "full_experiment" / "certified_action_candidates" / dataset / f"seed_{seed}.parquet")
    decisions = pd.read_parquet(root / "outputs" / "full_experiment" / "pre_outcome_decisions" / dataset / f"seed_{seed}.parquet")
    counterfactual = []
    for subject, group in candidates.groupby("subject_id", sort=True):
        row = episodes.loc[subject]
        future, labels, _ = read_segment(root, dataset, subject, row.future_indices)
        state_file = Path(decisions[decisions["subject_id"] == subject]["state_file"].iloc[0])
        payload = torch.load(state_file, map_location="cpu", weights_only=False)
        probabilities = predict_with_states(head, future, payload["states"], device)
        for candidate in group.itertuples(index=False):
            curve = evaluate_prediction_sets(probabilities[candidate.action], labels,
                                             np.asarray([candidate.selected_lambda], float))[0]
            counterfactual.append({"dataset": dataset, "seed": seed, "subject_id": subject,
                "episode_id": candidate.episode_id, "alpha": candidate.alpha, "action": candidate.action,
                "lambda": candidate.selected_lambda, "critical_index": candidate.certified_critical_index,
                "nontrivial_candidate": candidate.nontrivial_candidate,
                "true_future_risk": curve["future_risk"], "future_average_set_size": curve["average_set_size"],
                "future_singleton_rate": curve["singleton_rate"], "argmax_error": curve["argmax_error"],
                "macro_f1": curve["macro_f1"], "balanced_accuracy": curve["balanced_accuracy"],
                "cohen_kappa": float(__import__("sklearn.metrics").metrics.cohen_kappa_score(labels, probabilities[candidate.action].argmax(1))),
                "adaptation_status": candidate.adaptation_status})
    counter = pd.DataFrame(counterfactual)
    selected = decisions.merge(counter, left_on=["dataset", "seed", "subject_id", "episode_id", "alpha", "selected_action"],
        right_on=["dataset", "seed", "subject_id", "episode_id", "alpha", "action"], validate="one_to_one",
        suffixes=("_decision", "_outcome"))
    selected["adaptation_status"] = selected["adaptation_status_decision"]
    selected = selected.drop(columns=["adaptation_status_decision", "adaptation_status_outcome"])
    joined = selected.copy()
    atomic_parquet(counter, root / "outputs" / "full_experiment" / "final_counterfactual_action_outcomes" / dataset / f"seed_{seed}.parquet")
    atomic_parquet(selected, root / "outputs" / "full_experiment" / "final_test_outcomes" / dataset / f"seed_{seed}.parquet")
    atomic_parquet(joined, root / "outputs" / "full_experiment" / "joined_decisions" / dataset / f"seed_{seed}.parquet")
    return counter, selected, joined
