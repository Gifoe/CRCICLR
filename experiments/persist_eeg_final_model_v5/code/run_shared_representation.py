"""Evaluate fold-compatible frozen EEGNet representations without coordinate mixing."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import CACHE, DIAGNOSTICS, LEADERBOARD, RESEARCH_LOG, logit, ensure_directories, write_csv, write_json
from datasets import load_wbcic
from evaluation import summarize
from run_search import _aligned_context, _configs, _output_features
from training import OOFResult, run_nested_foldwise_direct


def _shared_embedding(data, fold_id: int) -> np.ndarray:
    metadata = pd.read_parquet(CACHE / f"WBCIC_SHARED_FOLD_{fold_id}_EEGNET_STABLE_METADATA.parquet")
    values = np.load(
        CACHE / f"WBCIC_SHARED_FOLD_{fold_id}_EEGNET_STABLE_EMBEDDINGS.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if len(metadata) != len(values) or set(metadata.fold_representation.astype(int)) != {fold_id}:
        raise RuntimeError(f"Malformed shared representation for fold {fold_id}")
    positions = pd.Series(np.arange(len(metadata)), index=metadata.trial_uid.astype(str))
    if positions.index.duplicated().any() or not set(map(str, data.trial_uid)).issubset(set(positions.index)):
        raise RuntimeError(f"Shared representation trial coverage mismatch for fold {fold_id}")
    indices = positions.loc[list(map(str, data.trial_uid))].to_numpy(int)
    aligned = np.asarray(values[indices], dtype=np.float32)
    if aligned.shape != (len(data.labels), 32) or not np.isfinite(aligned).all():
        raise RuntimeError(f"Invalid shared representation shape for fold {fold_id}: {aligned.shape}")
    return aligned


def run() -> None:
    ensure_directories()
    data = load_wbcic()
    output = _output_features(data)
    current = np.column_stack(
        [
            data.current_probability,
            logit(data.current_probability),
            np.abs(data.current_probability - 0.5),
            data.current_prediction,
        ]
    ).astype(np.float32)
    output_current = np.column_stack([output, current]).astype(np.float32)
    compact = _aligned_context(
        "WBCIC_S3_COMPACT_EEG_CONTEXT.npy",
        "WBCIC_S3_COMPACT_EEG_CONTEXT_METADATA.parquet",
        data.trial_uid,
    )
    shared = {fold: _shared_embedding(data, fold) for fold in range(5)}
    providers = {
        "SHARED": lambda fold: shared[fold],
        "OUTPUT_SHARED": lambda fold: np.column_stack([output_current, shared[fold]]),
        "OUTPUT_COMPACT_SHARED": lambda fold: np.column_stack([output_current, compact, shared[fold]]),
    }
    jobs = [
        ("M2_SHARED_EMBEDDING_DIRECT_CONTROL", "SHARED", "logistic", False),
        ("M2_OUTPUT_SHARED_LOGISTIC", "OUTPUT_SHARED", "logistic", True),
        ("M2_OUTPUT_SHARED_HISTGB", "OUTPUT_SHARED", "histgb", True),
        ("M2_OUTPUT_ALL_SHARED_EXTRATREES", "OUTPUT_COMPACT_SHARED", "extra_trees", True),
    ]
    rows, subjects, folds, selections, predictions = [], [], [], [], []
    for method_id, feature_name, family, anchored in jobs:
        dimension = providers[feature_name](0).shape[1]
        print(f"[shared representation] start {method_id} dimension={dimension}", flush=True)
        result: OOFResult = run_nested_foldwise_direct(
            data,
            method_id,
            providers[feature_name],
            _configs(family, dimension),
            anchored=anchored,
        )
        row, subject, fold = summarize(
            data,
            method_id,
            result.prediction,
            result.probability,
            result.outer_fold,
            baseline="current",
        )
        row.update(
            {
                "feature_set": feature_name,
                "model_family": family,
                "anchored": anchored,
                "fold_compatible_representation": True,
            }
        )
        rows.append(row)
        subjects.append(subject)
        folds.append(fold)
        selections.append(result.selections)
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
                    "target_prior_sessions_used": False,
                    "target_S3_labels_used_for_fit": False,
                    "OUTER_TEST_USED": False,
                }
            )
        )
        print(json.dumps(row, indent=2), flush=True)
    leaderboard = pd.DataFrame(rows).sort_values(["Delta_BA", "NLL"], ascending=[False, True])
    write_csv(LEADERBOARD / "WBCIC_SHARED_REPRESENTATION_SEARCH.csv", leaderboard)
    write_csv(DIAGNOSTICS / "WBCIC_SHARED_REPRESENTATION_SUBJECT_RESULTS.csv", pd.concat(subjects, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_SHARED_REPRESENTATION_FOLD_RESULTS.csv", pd.concat(folds, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_SHARED_REPRESENTATION_SELECTIONS.csv", pd.concat(selections, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_SHARED_REPRESENTATION_OOF_PREDICTIONS.csv", pd.concat(predictions, ignore_index=True))
    best = leaderboard.iloc[0].to_dict()
    write_json(
        RESEARCH_LOG / "ITERATION_003.json",
        {
            "previous_failure": "The first frozen-embedding control mixed five checkpoint coordinate systems.",
            "hypothesis": "A single compatible fold-specific EEGNet coordinate system provides transferable morphology for competence prediction.",
            "what_changed": "All model-fit, calibration, and held-out representations inside an outer fold now come from the same heldout-safe checkpoint.",
            "grouped_result": best,
            "conclusion": "KEEP" if best["Delta_BA"] >= 0.003 else "MODIFY",
            "OUTER_TEST_USED": False,
        },
    )
    print(leaderboard.to_string(index=False), flush=True)


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
