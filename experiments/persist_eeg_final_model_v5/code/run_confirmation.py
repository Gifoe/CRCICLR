"""Determinism/multi-seed confirmation and OpenBMI non-degradation fallback."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from aggregation import anchored_postprocess
from common import DIAGNOSTICS, LEADERBOARD, PROTOCOL, RESEARCH_LOG, logit, stable_seed, ensure_directories, write_csv, write_json
from datasets import load_openbmi, load_wbcic
from evaluation import summarize
from models import output_competence
from nested_cv import fold_assignment
from run_reliability_stack import _build_features
from training import OOFResult, _sample_weight


SEEDS = (20260820, 20260821, 20260822, 20260823, 20260824)


def _prediction_hash(prediction: np.ndarray, probability: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(prediction, dtype=np.int8).tobytes())
    digest.update(np.asarray(probability, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _csp_probability(data) -> np.ndarray:
    frame = pd.read_csv(DIAGNOSTICS / "WBCIC_CSP_AUGMENTATION_OOF_PREDICTIONS.csv")
    part = frame.loc[frame.method_id.eq("M13_SUBJECT_CSP_CONTROL")].set_index("trial_uid")
    if len(part) != len(data.labels):
        raise RuntimeError("CSP confirmation source missing")
    return part.loc[data.trial_uid].probability.to_numpy(float)


def _confirm_seed(data, matrix, seed: int) -> OOFResult:
    method_id = f"M13_CSP_AUGMENTED_REFIT4_SEED_{seed}"
    probability = np.full(len(data.labels), np.nan)
    prediction = np.full(len(data.labels), -1, dtype=int)
    assignment = np.full(len(data.labels), -1, dtype=int)
    records = []
    configuration = {"family": "logistic", "C": 1.0, "pca_components": None}
    for fold in data.folds:
        fold_id = int(fold["outer_fold"])
        test = np.isin(data.subjects, fold["test_subjects"])
        fit = ~test
        model = output_competence.build(
            configuration, stable_seed("V5_CONFIRM", seed, fold_id)
        )
        output_competence.fit(model, matrix[fit], data.labels[fit], _sample_weight(data, fit))
        raw = model.predict_proba(matrix[test])[:, 1]
        p_test, y_test = anchored_postprocess(
            data.current_probability[test], data.current_prediction[test], raw, data.expert_logits[test],
            alpha=1.0, gate="not_unanimous", threshold=0.5,
        )
        probability[test] = p_test
        prediction[test] = y_test
        assignment[test] = fold_id
        records.append(
            {
                "seed": seed,
                "outer_fold": fold_id,
                "configuration": configuration,
                "fit_subject_count": int(np.unique(data.subjects[fit]).size),
                "target_prior_sessions_used": True,
                "target_S3_labels_used_for_fit": False,
                "OUTER_TEST_USED": False,
            }
        )
    return OOFResult(method_id, probability, prediction, assignment, pd.DataFrame(records))


def run() -> None:
    ensure_directories()
    data = load_wbcic()
    simple, _, _ = _build_features(data)
    csp_probability = _csp_probability(data)
    matrix = np.column_stack(
        [
            simple,
            logit(csp_probability),
            csp_probability,
            np.abs(csp_probability - 0.5),
            (csp_probability >= 0.5).astype(float),
        ]
    ).astype(np.float32)
    rows, subjects, folds, selections, predictions, hashes = [], [], [], [], [], []
    for seed in SEEDS:
        result = _confirm_seed(data, matrix, seed)
        row, subject, fold = summarize(
            data, result.method_id, result.prediction, result.probability, result.outer_fold, baseline="current"
        )
        value_hash = _prediction_hash(result.prediction, result.probability)
        row.update({"seed": seed, "prediction_sha256": value_hash, "target_prior_sessions_used": True})
        rows.append(row); subjects.append(subject); folds.append(fold); selections.append(result.selections)
        hashes.append(value_hash)
        predictions.append(
            pd.DataFrame(
                {
                    "dataset": data.dataset_id,
                    "trial_uid": data.trial_uid,
                    "subject_id": data.subjects,
                    "method_id": result.method_id,
                    "seed": seed,
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
    seed_table = pd.DataFrame(rows)
    write_csv(LEADERBOARD / "WBCIC_MULTI_SEED_CONFIRMATION.csv", seed_table)
    write_csv(DIAGNOSTICS / "WBCIC_MULTI_SEED_SUBJECT_RESULTS.csv", pd.concat(subjects, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_MULTI_SEED_FOLD_RESULTS.csv", pd.concat(folds, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_MULTI_SEED_SELECTIONS.csv", pd.concat(selections, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_MULTI_SEED_OOF_PREDICTIONS.csv", pd.concat(predictions, ignore_index=True))

    openbmi = load_openbmi()
    openbmi_folds = fold_assignment(openbmi)
    fallback_id = "M13_OPENBMI_SAFE_FALLBACK_A1_DYNAMIC_KEEP"
    fallback_row, fallback_subject, fallback_fold = summarize(
        openbmi,
        fallback_id,
        openbmi.current_prediction.copy(),
        openbmi.current_probability.copy(),
        openbmi_folds,
        baseline="current",
    )
    fallback_row.update(
        {
            "fallback_reason": "fewer_than_two_legal_prior_sessions",
            "target_prior_sessions_used": False,
            "prediction_identical_to_A1": True,
        }
    )
    write_csv(LEADERBOARD / "OPENBMI_V5.csv", pd.DataFrame([fallback_row]))
    write_csv(DIAGNOSTICS / "OPENBMI_V5_SUBJECT_RESULTS.csv", fallback_subject)
    write_csv(DIAGNOSTICS / "OPENBMI_V5_FOLD_RESULTS.csv", fallback_fold)
    write_csv(
        DIAGNOSTICS / "OPENBMI_V5_OOF_PREDICTIONS.csv",
        pd.DataFrame(
            {
                "dataset": openbmi.dataset_id,
                "trial_uid": openbmi.trial_uid,
                "subject_id": openbmi.subjects,
                "method_id": fallback_id,
                "outer_fold": openbmi_folds,
                "label": openbmi.labels,
                "B_STRONG_CURRENT_prediction": openbmi.current_prediction,
                "prediction": openbmi.current_prediction,
                "probability": openbmi.current_probability,
                "fallback_reason": "fewer_than_two_legal_prior_sessions",
                "OUTER_TEST_USED": False,
            }
        ),
    )
    cross = pd.concat(
        [
            seed_table.loc[seed_table.seed.eq(SEEDS[0])].assign(benchmark="WBCIC-development"),
            pd.DataFrame([fallback_row]).assign(benchmark="OpenBMI"),
        ],
        ignore_index=True,
    )
    write_csv(LEADERBOARD / "CROSS_BENCHMARK_V5.csv", cross)
    deterministic = len(set(hashes)) == 1
    write_json(
        PROTOCOL / "MULTI_SEED_CONFIRMATION.json",
        {
            "seeds": list(SEEDS),
            "mean_Delta_BA": float(seed_table.Delta_BA.mean()),
            "std_Delta_BA": float(seed_table.Delta_BA.std(ddof=0)),
            "min_CI95_L": float(seed_table.CI95_L.min()),
            "min_positive_fold_fraction": float(seed_table.positive_fold_fraction.min()),
            "prediction_hashes": hashes,
            "identical_across_solver_seeds": deterministic,
            "explanation": "CSP eigendecomposition, history selection, and liblinear optimization are deterministic for fixed inputs; seeds only audit solver initialization.",
            "OpenBMI_fallback_identical_to_A1": True,
            "OUTER_TEST_USED": False,
        },
    )
    write_json(
        RESEARCH_LOG / "ITERATION_009.json",
        {
            "previous_failure": "A target candidate requires seed/retraining stability and cross-benchmark non-degradation before freeze.",
            "hypothesis": "The fixed CSP-context stack is deterministic and a fail-closed fallback preserves OpenBMI when two prior sessions are unavailable.",
            "what_changed": "Repeated all fold fits under five solver seeds and executed an exact A1 fallback on OpenBMI.",
            "grouped_result": seed_table.iloc[0].to_dict(),
            "identical_across_solver_seeds": deterministic,
            "OpenBMI_Delta_BA": float(fallback_row["Delta_BA"]),
            "conclusion": "KEEP" if deterministic and seed_table.CI95_L.min() > 0 else "MODIFY",
            "OUTER_TEST_USED": False,
        },
    )
    print(seed_table.to_string(index=False), flush=True)
    print(json.dumps(fallback_row, indent=2), flush=True)


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
