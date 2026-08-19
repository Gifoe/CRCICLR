from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss

from common import LEADERBOARD, OUTPUTS, ensure_directories, sigmoid, write_csv, write_json


EXPERTS = ("EEGNet_STABLE", "EEGNet_STD", "DeepConvNet", "EEGConformer", "TeCh")


def _mean_subject_ba(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    values = []
    for subject in sorted(frame.subject_id.astype(str).unique()):
        mask = frame.subject_id.astype(str).to_numpy() == subject
        values.append(balanced_accuracy_score(frame.label.to_numpy(dtype=int)[mask], prediction[mask]))
    return float(np.mean(values))


def run(expert_table: Path) -> pd.DataFrame:
    ensure_directories()
    frame = pd.read_parquet(expert_table).sort_values(
        ["outer_fold", "subject_id", "trial_index_within_subject_session"]
    ).reset_index(drop=True)
    if len(frame) != 8195 or frame.subject_id.nunique() != 41 or frame.OUTER_TEST_USED.astype(bool).any():
        raise RuntimeError("Illegal or incomplete WBCIC development expert table")
    logits = np.column_stack([frame[f"margin_{name}"].to_numpy(dtype=float) for name in EXPERTS])
    probabilities = sigmoid(logits)
    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, expert in enumerate(EXPERTS):
        candidates[f"STATIC_{expert}"] = (logits[:, index], probabilities[:, index])
    candidates.update(
        {
            "STATIC_EEGNET_PAIR_LOGIT_MEAN": (
                logits[:, [0, 1]].mean(axis=1),
                sigmoid(logits[:, [0, 1]].mean(axis=1)),
            ),
            "STATIC_STABLE_DEEP_LOGIT_MEAN": (
                logits[:, [0, 2]].mean(axis=1),
                sigmoid(logits[:, [0, 2]].mean(axis=1)),
            ),
            "STATIC_STABLE_STD_DEEP_LOGIT_MEAN": (
                logits[:, [0, 1, 2]].mean(axis=1),
                sigmoid(logits[:, [0, 1, 2]].mean(axis=1)),
            ),
            "STATIC_ALL_LOGIT_MEAN": (logits.mean(axis=1), sigmoid(logits.mean(axis=1))),
            "STATIC_ALL_PROBABILITY_MEAN": (
                np.log(np.clip(probabilities.mean(axis=1), 1e-7, 1 - 1e-7))
                - np.log1p(-np.clip(probabilities.mean(axis=1), 1e-7, 1 - 1e-7)),
                probabilities.mean(axis=1),
            ),
            "STATIC_ALL_LOGIT_MEDIAN": (
                np.median(logits, axis=1),
                sigmoid(np.median(logits, axis=1)),
            ),
            "STATIC_ALL_MAJORITY_VOTE": (
                np.where((logits >= 0).mean(axis=1) >= 0.5, 1.0, -1.0),
                (logits >= 0).mean(axis=1),
            ),
        }
    )
    labels = frame.label.to_numpy(dtype=int)
    rows = []
    for method_id, (margin, probability) in candidates.items():
        prediction = (margin >= 0).astype(int)
        rows.append(
            {
                "method_id": method_id,
                "mean_subject_BA": _mean_subject_ba(frame, prediction),
                "accuracy": float(np.mean(prediction == labels)),
                "NLL": float(log_loss(labels, np.column_stack([1 - probability, probability]), labels=[0, 1])),
                "OUTER_TEST_USED": False,
            }
        )
    result = pd.DataFrame(rows).sort_values(["mean_subject_BA", "NLL"], ascending=[False, True]).reset_index(drop=True)
    result["selected_B_STRONG"] = False
    result.loc[0, "selected_B_STRONG"] = True
    write_csv(LEADERBOARD / "WBCIC_DEV_STATIC_ENSEMBLE_AUDIT.csv", result)
    write_json(
        OUTPUTS / "protocol" / "WBCIC_B_STRONG_LOCK.json",
        {
            "status": "WBCIC_DEVELOPMENT_STATIC_REFERENCE_LOCKED",
            "method_id": str(result.iloc[0].method_id),
            "mean_subject_BA": float(result.iloc[0].mean_subject_BA),
            "candidate_family_frozen": list(candidates),
            "scope": "41 authorized development subjects, S3 only",
            "outer_subject_ids_loaded": False,
            "OUTER_TEST_USED": False,
        },
    )
    print(result.to_string(index=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expert-table",
        type=Path,
        default=OUTPUTS / "cache" / "WBCIC_DEV_KEEP_EXPERTS.parquet",
    )
    args = parser.parse_args()
    run(args.expert_table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
