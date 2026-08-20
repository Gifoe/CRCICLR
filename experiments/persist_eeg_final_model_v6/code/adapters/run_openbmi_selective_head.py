"""History-only selective subject head on the MI-specific OpenBMI backbone."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from backbones import train_openbmi_mi as mi
from common import ABLATIONS, CACHE, DIAGNOSTICS, LEADERBOARD, PROTOCOL, RESEARCH_LOG, V6_SEED, logit, sigmoid, stable_seed, stage0_root, write_csv, write_json
from evaluation.metrics import summarize
from protocol.datasets import load_openbmi_fold


CONFIGS: tuple[dict[str, float], ...] = tuple(
    {"C": c_value, "alpha": alpha}
    for c_value in (0.001, 0.01, 0.1, 1.0)
    for alpha in (0.25, 0.5, 0.75, 1.0)
)


def _extract(model, x: np.ndarray, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    features, logits = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            xb = mi._normalize(torch.as_tensor(x[start : start + 128], dtype=torch.float32, device=device), mean, std)
            feature = model.forward_features(xb)
            value = model.head(feature)
            features.append(feature.cpu().numpy())
            logits.append((value[:, 1] - value[:, 0]).cpu().numpy())
    return np.concatenate(features), np.concatenate(logits)


def _head(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("head", LogisticRegression(C=c_value, class_weight="balanced", solver="liblinear", max_iter=2_000, random_state=seed)),
        ]
    )


def _select_history(features: np.ndarray, raw_logit: np.ndarray, labels: np.ndarray, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float]]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(splitter.split(features, labels))
    anchor_fold_ba, anchor_fold_nll = [], []
    for _, validation in folds:
        probability = sigmoid(raw_logit[validation])
        anchor_fold_ba.append(balanced_accuracy_score(labels[validation], probability >= 0.5))
        anchor_fold_nll.append(log_loss(labels[validation], np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1]))
    anchor = {
        "configuration": {"C": 0.0, "alpha": 0.0},
        "history_cv_BA": float(np.mean(anchor_fold_ba)),
        "history_cv_NLL": float(np.mean(anchor_fold_nll)),
        "history_cv_worst_delta_vs_anchor": 0.0,
        "candidate_order": -1,
    }
    rows: list[dict[str, Any]] = [anchor]
    for order, configuration in enumerate(CONFIGS):
        fold_ba, fold_nll, fold_delta = [], [], []
        for fold_index, (fit, validation) in enumerate(folds):
            model = _head(float(configuration["C"]), seed + order * 17 + fold_index)
            model.fit(features[fit], labels[fit])
            head_probability = model.predict_proba(features[validation])[:, 1]
            probability = sigmoid((1.0 - configuration["alpha"]) * raw_logit[validation] + configuration["alpha"] * logit(head_probability))
            value = balanced_accuracy_score(labels[validation], probability >= 0.5)
            fold_ba.append(value)
            fold_nll.append(log_loss(labels[validation], np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1]))
            fold_delta.append(value - anchor_fold_ba[fold_index])
        rows.append(
            {
                "configuration": dict(configuration),
                "history_cv_BA": float(np.mean(fold_ba)),
                "history_cv_NLL": float(np.mean(fold_nll)),
                "history_cv_worst_delta_vs_anchor": float(np.min(fold_delta)),
                "candidate_order": order,
            }
        )
    selected = max(rows, key=lambda row: (row["history_cv_BA"], -row["history_cv_NLL"], -row["configuration"]["alpha"], -row["candidate_order"]))
    improvement = float(selected["history_cv_BA"] - anchor["history_cv_BA"])
    persist_accept = improvement >= 0.01 and float(selected["history_cv_worst_delta_vs_anchor"]) >= -0.05
    audit = {
        "anchor_history_cv_BA": float(anchor["history_cv_BA"]),
        "selected_history_cv_BA": float(selected["history_cv_BA"]),
        "selected_history_cv_improvement": improvement,
        "selected_history_cv_worst_delta": float(selected["history_cv_worst_delta_vs_anchor"]),
        "persist_accept": bool(persist_accept),
    }
    return dict(selected["configuration"]), rows, audit


def _future_probability(
    configuration: dict[str, float],
    history_features: np.ndarray,
    history_labels: np.ndarray,
    future_features: np.ndarray,
    future_raw_logit: np.ndarray,
    seed: int,
) -> np.ndarray:
    if float(configuration["alpha"]) == 0.0:
        return sigmoid(future_raw_logit)
    model = _head(float(configuration["C"]), seed)
    model.fit(history_features, history_labels)
    head_probability = model.predict_proba(future_features)[:, 1]
    return sigmoid((1.0 - configuration["alpha"]) * future_raw_logit + configuration["alpha"] * logit(head_probability))


def run() -> None:
    manifest = pd.read_parquet(stage0_root() / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet")
    manifest = manifest.loc[manifest.paradigm.eq("mi")].copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions, selections = [], []
    for fold in range(5):
        checkpoint = CACHE / f"OPENBMI_MI_SPECIFIC_BACKBONE_FOLD_{fold}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = mi.build(payload["configuration"]).to(device)
        model.load_state_dict(payload["model"], strict=True)
        mean_np, std_np = mi._normalizer(fold)
        mean = torch.as_tensor(mean_np, device=device)
        std = torch.as_tensor(std_np, device=device)
        data = load_openbmi_fold(fold)
        for subject in data.outcome_subjects:
            history_x, history_y, future_x, future_y, uid = mi._raw_subject(manifest, subject)
            history_features, history_raw_logit = _extract(model, history_x, mean, std, device)
            future_features, future_raw_logit = _extract(model, future_x, mean, std, device)
            selected, rows, audit = _select_history(
                history_features, history_raw_logit, history_y,
                stable_seed(V6_SEED, "OpenBMI-selective-head", fold, subject),
            )
            generic_probability = _future_probability(
                selected, history_features, history_y, future_features, future_raw_logit,
                stable_seed(V6_SEED, "OpenBMI-selective-head-fit", fold, subject),
            )
            persist_configuration = selected if audit["persist_accept"] else {"C": 0.0, "alpha": 0.0}
            persist_probability = _future_probability(
                persist_configuration, history_features, history_y, future_features, future_raw_logit,
                stable_seed(V6_SEED, "OpenBMI-selective-head-persist", fold, subject),
            )
            for method_id, probability, history_used in (
                ("MI_BACKBONE_FROZEN", sigmoid(future_raw_logit), False),
                ("MI_GENERIC_HISTORY_HEAD", generic_probability, float(selected["alpha"]) > 0),
                ("PERSIST_MI_SELECTIVE_HISTORY_HEAD", persist_probability, float(persist_configuration["alpha"]) > 0),
            ):
                predictions.append(
                    pd.DataFrame(
                        {
                            "benchmark": data.benchmark,
                            "method_id": method_id,
                            "trial_uid": uid,
                            "subject_id": subject,
                            "outer_fold": fold,
                            "label": future_y,
                            "probability": probability,
                            "prediction": (probability >= 0.5).astype(int),
                            "target_history_labels_used": history_used,
                            "target_future_labels_used_for_fit": False,
                            "exploratory": True,
                            "OUTER_TEST_USED": False,
                        }
                    )
                )
            for row in rows:
                selections.append(
                    {
                        "outer_fold": fold,
                        "subject_id": subject,
                        "configuration": json.dumps(row["configuration"], sort_keys=True),
                        "history_cv_BA": row["history_cv_BA"],
                        "history_cv_NLL": row["history_cv_NLL"],
                        "history_cv_worst_delta_vs_anchor": row["history_cv_worst_delta_vs_anchor"],
                        "selected_generic": row["configuration"] == selected,
                        "selected_persist": row["configuration"] == persist_configuration,
                        **audit,
                        "target_future_labels_used_for_fit": False,
                        "OUTER_TEST_USED": False,
                    }
                )
        print(f"[OpenBMI selective head] fold={fold} complete", flush=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    baseline = pd.read_csv(DIAGNOSTICS / "OPENBMI_BASELINE_PREDICTIONS.csv")
    reference = baseline.loc[baseline.method_id.eq("B_HISTORY_FUSION_LDA")].copy()
    rows, subject_parts, fold_parts = [], [], []
    for method in prediction_frame.method_id.unique():
        part = prediction_frame.loc[prediction_frame.method_id.eq(method)].copy()
        row, subjects, folds = summarize(part, reference=reference)
        rows.append(row); subject_parts.append(subjects); fold_parts.append(folds)
    table = pd.DataFrame(rows).sort_values("mean_subject_BA", ascending=False)
    write_csv(LEADERBOARD / "OPENBMI_MI_SELECTIVE_HEAD.csv", table)
    write_csv(DIAGNOSTICS / "OPENBMI_MI_SELECTIVE_HEAD_SELECTIONS.csv", pd.DataFrame(selections))
    write_csv(DIAGNOSTICS / "OPENBMI_MI_SELECTIVE_HEAD_PREDICTIONS.csv", prediction_frame)
    write_csv(DIAGNOSTICS / "OPENBMI_MI_SELECTIVE_HEAD_SUBJECT_RESULTS.csv", pd.concat(subject_parts, ignore_index=True))
    write_csv(DIAGNOSTICS / "OPENBMI_MI_SELECTIVE_HEAD_FOLD_RESULTS.csv", pd.concat(fold_parts, ignore_index=True))
    write_csv(ABLATIONS / "OPENBMI_MI_SELECTIVE_HEAD_ABLATION.csv", table)
    write_json(
        PROTOCOL / "OPENBMI_MI_SELECTIVE_HEAD_AUDIT.json",
        {
            "history": "outcome subject S1 labels and MI-backbone representations only",
            "selection": "five-fold stratified S1-only cross-validation independently per subject",
            "generic_rule": "best history-CV ridge head and anchor blend",
            "PERSIST_gate": "accept only if mean history-CV BA improves >=1 pp and worst fold delta >=-5 pp",
            "future_S2_labels_used_for_selection_or_fit": False,
            "OUTER_TEST_USED": False,
        },
    )
    (RESEARCH_LOG / "ITERATION_005_OPENBMI.md").write_text(
        "# Iteration 005 — MI representation selective history head\n\n"
        "A low-capacity subject head is selected and safety-gated entirely within S1, then applied once to S2.\n\n"
        "```text\n" + table.to_string(index=False) + "\n```\n",
        encoding="utf-8",
    )
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
