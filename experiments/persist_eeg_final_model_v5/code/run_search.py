from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import CACHE, DIAGNOSTICS, LEADERBOARD, RESEARCH_LOG, ensure_directories, sigmoid, write_csv, write_json
from datasets import load_wbcic
from evaluation import summarize
from training import OOFResult, run_nested_direct


def _output_features(data) -> np.ndarray:
    logits = data.expert_logits
    probabilities = sigmoid(logits)
    mean = probabilities.mean(axis=1)
    vote = (logits >= 0).mean(axis=1)
    sorted_probability = np.sort(probabilities, axis=1)
    pairwise = [probabilities[:, left] - probabilities[:, right] for left in range(logits.shape[1]) for right in range(left + 1, logits.shape[1])]
    return np.column_stack(
        [
            logits,
            probabilities,
            probabilities - mean[:, None],
            sorted_probability,
            mean,
            probabilities.std(axis=1),
            probabilities.max(axis=1) - probabilities.min(axis=1),
            vote,
            np.abs(mean - 0.5),
            *pairwise,
        ]
    ).astype(np.float32)


def _aligned_context(values_name: str, metadata_name: str, trial_uid: np.ndarray) -> np.ndarray:
    values = np.load(CACHE / values_name, mmap_mode="r", allow_pickle=False)
    metadata = pd.read_parquet(CACHE / metadata_name)
    positions = pd.Series(np.arange(len(metadata)), index=metadata.trial_uid.astype(str))
    indices = positions.loc[list(map(str, trial_uid))].to_numpy(int)
    return np.asarray(values[indices], dtype=np.float32)


def _wbcic_stable_embedding(trial_uid: np.ndarray) -> np.ndarray:
    all_sessions = pd.read_parquet(CACHE / "WBCIC_DEV_ALL_SESSION_EXPERTS.parquet")
    values = np.load(CACHE / "WBCIC_DEV_ALL_SESSION_EEGNet_STABLE_EMBEDDINGS.npy", mmap_mode="r", allow_pickle=False)
    positions = pd.Series(np.arange(len(all_sessions)), index=all_sessions.trial_uid.astype(str))
    indices = positions.loc[list(map(str, trial_uid))].to_numpy(int)
    return np.asarray(values[indices], dtype=np.float32)


def _configs(kind: str, dimension: int) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    if kind == "logistic":
        components = [None]
        for value in (16, 32, 64):
            if value < dimension:
                components.append(value)
        for component in components:
            for c_value in (0.001, 0.01, 0.1, 1.0):
                result.append({"family": "logistic", "C": c_value, "pca_components": component})
    elif kind == "histgb":
        for leaves in (7, 15):
            for l2 in (1.0, 10.0):
                result.append(
                    {
                        "family": "histgb",
                        "learning_rate": 0.05,
                        "max_leaf_nodes": leaves,
                        "max_iter": 200,
                        "l2": l2,
                        "min_samples_leaf": 40,
                    }
                )
    elif kind == "extra_trees":
        for leaf in (10, 30):
            result.append(
                {
                    "family": "extra_trees",
                    "n_estimators": 400,
                    "max_depth": None,
                    "min_samples_leaf": leaf,
                    "max_features": "sqrt",
                }
            )
    else:
        raise ValueError(kind)
    return result


def run() -> None:
    ensure_directories()
    print("[V5 search] load WBCIC", flush=True)
    data = load_wbcic()
    print("[V5 search] build output features", flush=True)
    output = _output_features(data)
    print(f"[V5 search] output shape={output.shape}", flush=True)
    compact = _aligned_context(
        "WBCIC_S3_COMPACT_EEG_CONTEXT.npy",
        "WBCIC_S3_COMPACT_EEG_CONTEXT_METADATA.parquet",
        data.trial_uid,
    )
    print(f"[V5 search] compact shape={compact.shape}", flush=True)
    embedding = _wbcic_stable_embedding(data.trial_uid)
    print(f"[V5 search] embedding shape={embedding.shape}", flush=True)
    matrices = {
        "OUTPUT": output,
        "COMPACT": compact,
        "OUTPUT_COMPACT": np.column_stack([output, compact]),
        "EMBEDDING": embedding,
        "OUTPUT_EMBEDDING": np.column_stack([output, embedding]),
        "OUTPUT_COMPACT_EMBEDDING": np.column_stack([output, compact, embedding]),
    }
    jobs = [
        ("M1_OUTPUT_LOGISTIC", "OUTPUT", "logistic", True),
        ("M1_OUTPUT_HISTGB", "OUTPUT", "histgb", True),
        ("M2_COMPACT_DIRECT_CONTROL", "COMPACT", "logistic", False),
        ("M2_OUTPUT_COMPACT_LOGISTIC", "OUTPUT_COMPACT", "logistic", True),
        ("M2_OUTPUT_COMPACT_HISTGB", "OUTPUT_COMPACT", "histgb", True),
        ("M2_EMBEDDING_DIRECT_CONTROL", "EMBEDDING", "logistic", False),
        ("M2_OUTPUT_EMBEDDING_LOGISTIC", "OUTPUT_EMBEDDING", "logistic", True),
        ("M2_OUTPUT_ALL_CONTEXT_LOGISTIC", "OUTPUT_COMPACT_EMBEDDING", "logistic", True),
        ("M2_OUTPUT_ALL_CONTEXT_HISTGB", "OUTPUT_COMPACT_EMBEDDING", "histgb", True),
        ("M2_OUTPUT_ALL_CONTEXT_EXTRATREES", "OUTPUT_COMPACT_EMBEDDING", "extra_trees", True),
    ]
    rows, subjects, folds, selections, predictions = [], [], [], [], []
    for method_id, matrix_name, family, anchored in jobs:
        matrix = matrices[matrix_name]
        print(f"[V5 search] start {method_id} shape={matrix.shape}", flush=True)
        result: OOFResult = run_nested_direct(
            data,
            method_id,
            matrix,
            _configs(family, matrix.shape[1]),
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
        row.update({"feature_set": matrix_name, "model_family": family, "anchored": anchored})
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
    write_csv(LEADERBOARD / "WBCIC_OUTPUT_CONTEXT_SEARCH.csv", leaderboard)
    write_csv(DIAGNOSTICS / "WBCIC_OUTPUT_CONTEXT_SUBJECT_RESULTS.csv", pd.concat(subjects, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_OUTPUT_CONTEXT_FOLD_RESULTS.csv", pd.concat(folds, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_OUTPUT_CONTEXT_SELECTIONS.csv", pd.concat(selections, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_OUTPUT_CONTEXT_OOF_PREDICTIONS.csv", pd.concat(predictions, ignore_index=True))
    best = leaderboard.iloc[0].to_dict()
    write_json(
        RESEARCH_LOG / "ITERATION_002.json",
        {
            "previous_failure": "Prior-session expert reliability did not transfer stably to S3.",
            "hypothesis": "Trial EEG morphology or a frozen OOF EEGNet representation adds local information missing from expert outputs.",
            "what_changed": "Added compact label-free spectral/temporal/covariance context, frozen 32-D EEGNet context, direct-classifier controls, nonlinear models, and conservative anchoring.",
            "grouped_result": best,
            "conclusion": "KEEP" if best["Delta_BA"] > 0 else "MODIFY",
            "OUTER_TEST_USED": False,
        },
    )
    print(leaderboard.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
