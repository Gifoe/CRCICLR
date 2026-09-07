"""Run fast, strict history-matched representation baselines."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from baselines.methods import configurations, fit_population, population_probability, target_probability
from common import BASELINES, DIAGNOSTICS, LEADERBOARD, PROTOCOL, V6_SEED, ensure_directories, stable_seed, write_csv, write_json
from evaluation.metrics import summarize
from protocol.datasets import FoldDataset, load_fold


FAMILIES = (
    ("B_POPULATION_LINEAR", "population_linear"),
    ("B_SUBJECT_LAST_LAYER", "target_logistic"),
    ("B_SUBJECT_SHRINKAGE_LDA", "target_lda"),
    ("B_HISTORY_CALIBRATED", "history_calibrated"),
    ("B_SUPERVISED_COSINE_PROTO", "cosine_prototype"),
    ("B_SHRINKAGE_PROTO", "shrinkage_prototype"),
    ("B_HISTORY_FUSION_LOGISTIC", "fusion_logistic"),
    ("B_HISTORY_FUSION_LDA", "fusion_lda"),
)


def _arrays(data: FoldDataset, subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    history = data.mask([subject], list(data.history_sessions))
    future = data.mask([subject], [data.future_session])
    if history.sum() == 0 or future.sum() == 0:
        raise RuntimeError(f"Missing history/future rows for {subject}")
    return (
        data.embeddings[history],
        data.metadata.loc[history, "label"].to_numpy(int),
        data.embeddings[future],
        data.metadata.loc[future, "label"].to_numpy(int),
        data.metadata.loc[future].reset_index(drop=True),
    )


def _score(labels: list[np.ndarray], probabilities: list[np.ndarray]) -> tuple[float, float]:
    bas = []
    nlls = []
    for y, p in zip(labels, probabilities):
        bas.append(float(balanced_accuracy_score(y, np.asarray(p) >= 0.5)))
        nlls.append(float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])))
    return float(np.mean(bas)), float(np.mean(nlls))


def _prediction_frame(data: FoldDataset, method_id: str, subject: str, future_meta: pd.DataFrame, probability: np.ndarray, configuration: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "benchmark": data.benchmark,
            "method_id": method_id,
            "trial_uid": future_meta.trial_uid.astype(str),
            "subject_id": str(subject),
            "outer_fold": data.fold,
            "label": future_meta.label.to_numpy(int),
            "probability": np.asarray(probability, dtype=float),
            "prediction": (np.asarray(probability) >= 0.5).astype(int),
            "selected_configuration": json.dumps(configuration, sort_keys=True),
            "history_sessions": "+".join(map(str, data.history_sessions)),
            "future_session": data.future_session,
            "target_history_labels_used": method_id not in {"B_POPULATION_LINEAR", "B_EEGNET_FROZEN_HEAD"},
            "target_future_labels_used_for_fit": False,
            "exploratory": True,
            "OUTER_TEST_USED": False,
        }
    )


def run_fold(benchmark: str, fold: int) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    data = load_fold(benchmark, fold)
    fit_mask = data.mask(list(data.model_fit_subjects), list(data.history_sessions) + [data.future_session])
    x_fit = data.embeddings[fit_mask]
    y_fit = data.metadata.loc[fit_mask, "label"].to_numpy(int)
    predictions: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []

    contexts: dict[float, Any] = {}
    pop_records = []
    for config in configurations("population_linear"):
        c = float(config["C"])
        context = fit_population(x_fit, y_fit, c, stable_seed(V6_SEED, benchmark, fold, "population", c))
        contexts[c] = context
        labels, probabilities = [], []
        for subject in data.discovery_subjects:
            _, _, x_future, y_future, _ = _arrays(data, subject)
            labels.append(y_future)
            probabilities.append(population_probability(context, x_future))
        ba, nll = _score(labels, probabilities)
        pop_records.append({"configuration": config, "discovery_BA": ba, "discovery_NLL": nll})
    pop_selected = max(pop_records, key=lambda item: (item["discovery_BA"], -item["discovery_NLL"], -float(item["configuration"]["C"])))
    pop_configuration = dict(pop_selected["configuration"])
    context = contexts[float(pop_configuration["C"])]
    selections.append(
        {
            "benchmark": data.benchmark,
            "outer_fold": fold,
            "method_id": "B_POPULATION_LINEAR",
            **pop_selected,
            "candidate_count": len(pop_records),
            "OUTER_TEST_USED": False,
        }
    )
    for subject in data.outcome_subjects:
        _, _, x_future, _, future_meta = _arrays(data, subject)
        predictions.append(_prediction_frame(data, "B_POPULATION_LINEAR", subject, future_meta, population_probability(context, x_future), pop_configuration))

    if data.backbone_logits is not None:
        for subject in data.outcome_subjects:
            future = data.mask([subject], [data.future_session])
            logits = data.backbone_logits[future]
            margin = logits[:, 1] - logits[:, 0]
            probability = 1.0 / (1.0 + np.exp(-np.clip(margin, -50, 50)))
            predictions.append(
                _prediction_frame(data, "B_EEGNET_FROZEN_HEAD", subject, data.metadata.loc[future].reset_index(drop=True), probability, {"frozen": True})
            )
        selections.append(
            {
                "benchmark": data.benchmark,
                "outer_fold": fold,
                "method_id": "B_EEGNET_FROZEN_HEAD",
                "configuration": {"frozen": True},
                "discovery_BA": None,
                "discovery_NLL": None,
                "candidate_count": 1,
                "OUTER_TEST_USED": False,
            }
        )

    for method_id, family in FAMILIES[1:]:
        candidates = []
        for order, configuration in enumerate(configurations(family)):
            labels, probabilities = [], []
            for subject in data.discovery_subjects:
                x_history, y_history, x_future, y_future, _ = _arrays(data, subject)
                probability = target_probability(
                    family,
                    configuration,
                    context,
                    x_history,
                    y_history,
                    x_future,
                    stable_seed(V6_SEED, benchmark, fold, family, subject, order),
                )
                labels.append(y_future)
                probabilities.append(probability)
            ba, nll = _score(labels, probabilities)
            candidates.append({"configuration": configuration, "candidate_order": order, "discovery_BA": ba, "discovery_NLL": nll})
        selected = max(candidates, key=lambda item: (item["discovery_BA"], -item["discovery_NLL"], -item["candidate_order"]))
        configuration = dict(selected["configuration"])
        selections.append(
            {
                "benchmark": data.benchmark,
                "outer_fold": fold,
                "method_id": method_id,
                **selected,
                "candidate_count": len(candidates),
                "OUTER_TEST_USED": False,
            }
        )
        for subject in data.outcome_subjects:
            x_history, y_history, x_future, _, future_meta = _arrays(data, subject)
            probability = target_probability(
                family,
                configuration,
                context,
                x_history,
                y_history,
                x_future,
                stable_seed(V6_SEED, benchmark, fold, family, subject, "outcome"),
            )
            predictions.append(_prediction_frame(data, method_id, subject, future_meta, probability, configuration))
        print(
            f"[{benchmark}] fold={fold} {method_id} discovery_BA={selected['discovery_BA']:.4f} config={configuration}",
            flush=True,
        )
    return predictions, selections


def run(benchmark: str) -> None:
    ensure_directories()
    started = time.time()
    prediction_parts: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    for fold in range(5):
        parts, chosen = run_fold(benchmark, fold)
        prediction_parts.extend(parts)
        selections.extend(chosen)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    selection_frame = pd.DataFrame(selections)
    summaries = []
    subject_parts = []
    fold_parts = []
    methods = sorted(predictions.method_id.unique())
    for method in methods:
        frame = predictions.loc[predictions.method_id.eq(method)].copy()
        row, subjects, folds = summarize(frame)
        summaries.append(row)
        subject_parts.append(subjects)
        fold_parts.append(folds)
    table = pd.DataFrame(summaries).sort_values(["mean_subject_BA", "NLL"], ascending=[False, True]).reset_index(drop=True)
    history_methods = set(methods) - {"B_POPULATION_LINEAR", "B_EEGNET_FROZEN_HEAD"}
    matched = table.loc[table.method_id.isin(history_methods)].iloc[0].method_id
    reference = predictions.loc[predictions.method_id.eq(matched)].copy()
    final_rows, final_subjects, final_folds = [], [], []
    for method in methods:
        frame = predictions.loc[predictions.method_id.eq(method)].copy()
        row, subjects, folds = summarize(frame, reference=reference)
        final_rows.append(row)
        final_subjects.append(subjects)
        final_folds.append(folds)
    final_table = pd.DataFrame(final_rows).sort_values(["mean_subject_BA", "NLL"], ascending=[False, True]).reset_index(drop=True)
    prefix = "OPENBMI" if benchmark.lower().startswith("open") else "WBCIC"
    write_csv(BASELINES / f"{prefix}_MATCHED_BASELINES.csv", final_table)
    write_csv(DIAGNOSTICS / f"{prefix}_BASELINE_SELECTIONS.csv", selection_frame)
    write_csv(DIAGNOSTICS / f"{prefix}_BASELINE_SUBJECT_RESULTS.csv", pd.concat(final_subjects, ignore_index=True))
    write_csv(DIAGNOSTICS / f"{prefix}_BASELINE_FOLD_RESULTS.csv", pd.concat(final_folds, ignore_index=True))
    write_csv(DIAGNOSTICS / f"{prefix}_BASELINE_PREDICTIONS.csv", predictions)
    write_csv(LEADERBOARD / f"{prefix}_V6.csv", final_table)
    write_json(
        PROTOCOL / f"{prefix}_BASELINE_MATCHING_AUDIT.json",
        {
            "benchmark": str(predictions.benchmark.iloc[0]),
            "strongest_information_matched_baseline": matched,
            "same_target_history_budget_enforced": True,
            "history_sessions": sorted(predictions.history_sessions.unique().tolist()),
            "future_session": sorted(map(int, predictions.future_session.unique().tolist())),
            "model_selection_roles": "discovery subjects only",
            "outcome_future_labels_used_for_fit": False,
            "exploratory": True,
            "OUTER_TEST_USED": False,
            "runtime_seconds": time.time() - started,
        },
    )
    print(final_table.to_string(index=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("openbmi", "wbcic"), required=True)
    args = parser.parse_args()
    run(args.benchmark)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
