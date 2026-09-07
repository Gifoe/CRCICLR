from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss

from common import DIAGNOSTICS, ensure_directories, sigmoid, write_csv
from datasets import V5Dataset, load_openbmi, load_wbcic
from evaluation import ece


def audit_dataset(data: V5Dataset) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reliability = []
    predictions = (data.expert_logits >= 0).astype(int)
    correctness = predictions == data.labels[:, None]
    errors = (~correctness).astype(float)
    for index, expert in enumerate(data.expert_names):
        valid = data.expert_mask[:, index]
        labels = data.labels[valid]
        prediction = predictions[valid, index]
        probability = sigmoid(data.expert_logits[valid, index])
        subject_bas = []
        session_bas = []
        for subject in np.unique(data.subjects[valid]):
            mask = valid & (data.subjects == subject)
            subject_bas.append(balanced_accuracy_score(data.labels[mask], predictions[mask, index]))
        for session in np.unique(data.sessions[valid]):
            mask = valid & (data.sessions == session)
            session_bas.append(balanced_accuracy_score(data.labels[mask], predictions[mask, index]))
        minority = valid & (predictions[:, index] != data.static_prediction)
        rescue = valid & (data.current_prediction != data.labels) & (predictions[:, index] == data.labels)
        harm = valid & (data.current_prediction == data.labels) & (predictions[:, index] != data.labels)
        reliability.append(
            {
                "dataset": data.dataset_id,
                "expert": expert,
                "available_trials": int(valid.sum()),
                "standalone_BA": float(balanced_accuracy_score(labels, prediction)),
                "sensitivity": float(np.mean(prediction[labels == 1] == 1)),
                "specificity": float(np.mean(prediction[labels == 0] == 0)),
                "NLL": float(log_loss(labels, np.column_stack([1 - probability, probability]), labels=[0, 1])),
                "Brier": float(np.mean((probability - labels) ** 2)),
                "ECE": ece(labels, probability),
                "subject_BA_std": float(np.std(subject_bas)),
                "session_BA_std": float(np.std(session_bas)),
                "unique_rescue_count": int(rescue.sum()),
                "unique_harm_count": int(harm.sum()),
                "disagreement_accuracy": float(np.mean(predictions[minority, index] == data.labels[minority])) if minority.any() else 0.0,
                "minority_expert_rescue_accuracy": float(np.mean(predictions[minority, index] == data.labels[minority])) if minority.any() else 0.0,
                "OUTER_TEST_USED": False,
            }
        )
    correlation = np.full((len(data.expert_names), len(data.expert_names)), np.nan)
    complementarity = []
    for left in range(len(data.expert_names)):
        for right in range(len(data.expert_names)):
            valid = data.expert_mask[:, left] & data.expert_mask[:, right]
            if valid.sum() > 1 and np.std(errors[valid, left]) > 0 and np.std(errors[valid, right]) > 0:
                correlation[left, right] = np.corrcoef(errors[valid, left], errors[valid, right])[0, 1]
            if left < right:
                left_only = valid & correctness[:, left] & ~correctness[:, right]
                right_only = valid & correctness[:, right] & ~correctness[:, left]
                complementarity.append(
                    {
                        "dataset": data.dataset_id,
                        "expert_a": data.expert_names[left],
                        "expert_b": data.expert_names[right],
                        "shared_trials": int(valid.sum()),
                        "a_only_correct": int(left_only.sum()),
                        "b_only_correct": int(right_only.sum()),
                        "symmetric_unique_correct": int(left_only.sum() + right_only.sum()),
                        "error_correlation": correlation[left, right],
                        "OUTER_TEST_USED": False,
                    }
                )
    corr_frame = pd.DataFrame(correlation, index=data.expert_names, columns=data.expert_names)
    corr_frame.insert(0, "expert", corr_frame.index)
    corr_frame.insert(0, "dataset", data.dataset_id)
    corr_frame["OUTER_TEST_USED"] = False
    return pd.DataFrame(reliability), pd.DataFrame(complementarity), corr_frame.reset_index(drop=True)


def run() -> None:
    ensure_directories()
    reliability, complementarity, correlations = [], [], []
    for data in (load_openbmi(), load_wbcic()):
        rel, comp, corr = audit_dataset(data)
        reliability.append(rel)
        complementarity.append(comp)
        correlations.append(corr)
    write_csv(DIAGNOSTICS / "EXPERT_RELIABILITY.csv", pd.concat(reliability, ignore_index=True))
    write_csv(DIAGNOSTICS / "EXPERT_COMPLEMENTARITY.csv", pd.concat(complementarity, ignore_index=True))
    write_csv(DIAGNOSTICS / "EXPERT_ERROR_CORRELATION.csv", pd.concat(correlations, ignore_index=True))
    print(pd.concat(reliability, ignore_index=True).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
