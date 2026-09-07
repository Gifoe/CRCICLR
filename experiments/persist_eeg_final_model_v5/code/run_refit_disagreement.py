"""Loss-aligned disagreement fitting and fixed-hyperparameter outer refitting."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from aggregation import anchored_postprocess
from common import DIAGNOSTICS, LEADERBOARD, RESEARCH_LOG, logit, sigmoid, stable_seed, ensure_directories, write_csv, write_json
from datasets import load_wbcic
from evaluation import summarize
from models import output_competence
from run_reliability_stack import _aligned_local_probabilities, _build_features
from training import OOFResult, _sample_weight


def _lean_features(data, local_probability):
    raw_logits = np.asarray(data.expert_logits, dtype=float)
    raw_probability = sigmoid(raw_logits)
    local_logits = logit(local_probability)
    return np.column_stack(
        [
            raw_logits,
            local_logits,
            local_logits - raw_logits,
            logit(data.current_probability),
            raw_probability.mean(axis=1),
            raw_probability.std(axis=1),
            local_probability.mean(axis=1),
            local_probability.std(axis=1),
            (raw_logits >= 0).mean(axis=1),
            (local_logits >= 0).mean(axis=1),
        ]
    ).astype(np.float32)


def _fixed_outer(
    data,
    method_id: str,
    x: np.ndarray,
    *,
    fit_scope: str,
    refit_nonoutcome: bool,
    c_value: float = 1.0,
) -> OOFResult:
    probability = np.full(len(data.labels), np.nan)
    prediction = np.full(len(data.labels), -1, dtype=int)
    assignment = np.full(len(data.labels), -1, dtype=int)
    records = []
    votes = (data.expert_logits >= 0).astype(int)
    nonunanimous = votes.min(axis=1) != votes.max(axis=1)
    for fold in data.folds:
        fold_id = int(fold["outer_fold"])
        test = np.isin(data.subjects, fold["test_subjects"])
        if refit_nonoutcome:
            fit = ~test
        else:
            fit = np.isin(data.subjects, fold["train_subjects"])
        if fit_scope == "disagreement":
            fit &= nonunanimous
        elif fit_scope != "all":
            raise ValueError(fit_scope)
        configuration = {"family": "logistic", "C": float(c_value), "pca_components": None}
        model = output_competence.build(
            configuration, stable_seed("V5_FIXED_OUTER_REFIT", method_id, fold_id)
        )
        output_competence.fit(model, x[fit], data.labels[fit], _sample_weight(data, fit))
        raw_probability = model.predict_proba(x[test])[:, 1]
        p_test, y_test = anchored_postprocess(
            data.current_probability[test],
            data.current_prediction[test],
            raw_probability,
            data.expert_logits[test],
            alpha=1.0,
            gate="not_unanimous",
            threshold=0.5,
        )
        probability[test] = p_test
        prediction[test] = y_test
        assignment[test] = fold_id
        records.append(
            {
                "method_id": method_id,
                "outer_fold": fold_id,
                "configuration": configuration,
                "fit_scope": fit_scope,
                "fit_subject_count": int(np.unique(data.subjects[fit]).size),
                "refit_on_all_nonoutcome_subjects": bool(refit_nonoutcome),
                "gate": "not_unanimous",
                "threshold": 0.5,
                "configuration_selected_from_this_outer_fold": False,
                "target_prior_sessions_used": True,
                "target_S3_labels_used_for_fit": False,
                "OUTER_TEST_USED": False,
            }
        )
    if np.isnan(probability).any() or np.any(prediction < 0):
        raise RuntimeError(f"Incomplete fixed outer result: {method_id}")
    return OOFResult(method_id, probability, prediction, assignment, pd.DataFrame(records))


def run() -> None:
    ensure_directories()
    data = load_wbcic()
    simple, _, _ = _build_features(data)
    local_probability = _aligned_local_probabilities(data)
    lean = _lean_features(data, local_probability)
    specs = [
        ("M12_LEAN_ALL_TRAIN3", lean, "all", False),
        ("M12_LEAN_ALL_REFIT4", lean, "all", True),
        ("M12_LEAN_DISAGREEMENT_TRAIN3", lean, "disagreement", False),
        ("M12_LEAN_DISAGREEMENT_REFIT4", lean, "disagreement", True),
        ("M12_SIMPLE_ALL_REFIT4", simple, "all", True),
        ("M12_SIMPLE_DISAGREEMENT_REFIT4", simple, "disagreement", True),
    ]
    rows, subjects, folds, selections, predictions = [], [], [], [], []
    for method_id, matrix, fit_scope, refit in specs:
        result = _fixed_outer(
            data,
            method_id,
            matrix,
            fit_scope=fit_scope,
            refit_nonoutcome=refit,
            c_value=1.0,
        )
        row, subject, fold = summarize(
            data, method_id, result.prediction, result.probability, result.outer_fold, baseline="current"
        )
        row.update(
            {
                "architecture_family": "loss_aligned_subject_local_stack",
                "fit_scope": fit_scope,
                "refit_nonoutcome": refit,
                "target_prior_sessions_used": True,
            }
        )
        rows.append(row); subjects.append(subject); folds.append(fold); selections.append(result.selections)
        predictions.append(
            pd.DataFrame(
                {
                    "dataset": data.dataset_id,
                    "trial_uid": data.trial_uid,
                    "subject_id": data.subjects,
                    "method_id": method_id,
                    "outer_fold": result.outer_fold,
                    "label": data.labels,
                    "B_STRONG_CURRENT_prediction": data.current_prediction,
                    "prediction": result.prediction,
                    "probability": result.probability,
                    "target_prior_sessions_used": True,
                    "target_S3_labels_used_for_fit": False,
                    "OUTER_TEST_USED": False,
                }
            )
        )
        print(json.dumps(row, indent=2), flush=True)
    leaderboard = pd.DataFrame(rows).sort_values(["Delta_BA", "NLL"], ascending=[False, True])
    write_csv(LEADERBOARD / "WBCIC_REFIT_DISAGREEMENT.csv", leaderboard)
    write_csv(DIAGNOSTICS / "WBCIC_REFIT_DISAGREEMENT_SUBJECT_RESULTS.csv", pd.concat(subjects, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_REFIT_DISAGREEMENT_FOLD_RESULTS.csv", pd.concat(folds, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_REFIT_DISAGREEMENT_SELECTIONS.csv", pd.concat(selections, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_REFIT_DISAGREEMENT_OOF_PREDICTIONS.csv", pd.concat(predictions, ignore_index=True))
    best = leaderboard.iloc[0].to_dict()
    write_json(
        RESEARCH_LOG / "ITERATION_007.json",
        {
            "previous_failure": "The fixed local-history stack reached +0.867 pp, just below target, while training loss still included deployment-ineligible unanimous trials.",
            "hypothesis": "Training only on the deployed disagreement regime improves loss alignment; after configuration freeze, using all non-outcome subjects reduces estimator variance.",
            "what_changed": "Compared lean versus expanded stacks, all-trial versus disagreement-only loss, and three-fold fit versus fixed-config four-fold non-outcome refit. Threshold and gate remained fixed.",
            "grouped_result": best,
            "development_reuse_note": "Exploratory iteration after inspecting prior OOF results; no outer test data were accessed.",
            "conclusion": "KEEP" if best["Delta_BA"] >= 0.01 else "MODIFY",
            "target_prior_sessions_used": True,
            "target_S3_labels_used_for_fit": False,
            "OUTER_TEST_USED": False,
        },
    )
    print(leaderboard.to_string(index=False), flush=True)


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
