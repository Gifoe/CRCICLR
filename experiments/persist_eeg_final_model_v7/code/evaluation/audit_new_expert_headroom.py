from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import DIAGNOSTICS, PROTOCOL, v6_outputs, write_csv, write_json


SPECS = {
    "openbmi": {
        "anchor": (v6_outputs() / "diagnostics" / "OPENBMI_MI_SPECIFIC_BACKBONE_PREDICTIONS.csv", "MI_SPECIFIC_BACKBONE_ADAPTED"),
        "identity_head": (DIAGNOSTICS / "OPENBMI_CAPACITY_MATCHED_EEGNET_PREDICTIONS.csv", "CAPACITY_MATCHED_EEGNET_FIXED_HEAD"),
        "conformer_head": (DIAGNOSTICS / "OPENBMI_CONFORMER_NORM_PREDICTIONS.csv", "CONFORMER_NORM_FIXED_HEAD"),
        "fbc_head": (DIAGNOSTICS / "OPENBMI_FBCVARIANCE_NORM_PREDICTIONS.csv", "FBCVARIANCE_NORM_FIXED_HEAD"),
        "initial_persist": (DIAGNOSTICS / "V7_PREDICTIONS.csv", "ANCHOR_PLUS_PERSIST_META_RESIDUAL"),
    },
    "wbcic": {
        "anchor": (v6_outputs() / "diagnostics" / "WBCIC_FUTURE_SESSION_POPULATION_PREDICTIONS.csv", "V5_FIXED_LOGIT_BLEND__A_FUTURE_SESSION_TARGET_ADAPTED"),
        "identity_head": (DIAGNOSTICS / "WBCIC_CAPACITY_MATCHED_EEGNET_PREDICTIONS.csv", "CAPACITY_MATCHED_EEGNET_FIXED_HEAD"),
        "conformer_head": (DIAGNOSTICS / "WBCIC_CONFORMER_NORM_PREDICTIONS.csv", "CONFORMER_NORM_FIXED_HEAD"),
        "initial_persist": (DIAGNOSTICS / "V7_PREDICTIONS.csv", "ANCHOR_PLUS_PERSIST_META_RESIDUAL"),
    },
}


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(probability) - np.log1p(-probability)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40, 40)))


def _load(benchmark: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frames = {}
    for name, (path, method) in SPECS[benchmark].items():
        frame = pd.read_csv(path)
        frame = frame.loc[frame.method_id.astype(str).eq(method)].copy()
        prefix = "OpenBMI_" if benchmark == "openbmi" else "WBCIC_"
        frame = frame.loc[frame.trial_uid.astype(str).str.startswith(prefix)].copy()
        if frame.trial_uid.duplicated().any() or frame.OUTER_TEST_USED.astype(bool).any():
            raise RuntimeError(f"Malformed {benchmark} {name}")
        frames[name] = frame.set_index("trial_uid")
    anchor = frames["anchor"]
    uids = anchor.index
    for name, frame in frames.items():
        if set(frame.index) != set(uids):
            raise RuntimeError(f"Trial mismatch {benchmark} {name}")
        frame = frame.loc[uids]
        if not np.array_equal(frame.label.to_numpy(int), anchor.label.to_numpy(int)):
            raise RuntimeError(f"Label mismatch {benchmark} {name}")
        frames[name] = frame
    return anchor.reset_index(), {name: _logit(frame.probability.to_numpy(float)) for name, frame in frames.items()}


def _mean_subject_ba(metadata: pd.DataFrame, logits: np.ndarray) -> float:
    values = []
    for subject, group in metadata.assign(row=np.arange(len(metadata))).groupby("subject_id"):
        index = group.row.to_numpy(int)
        values.append(balanced_accuracy_score(group.label, logits[index] >= 0.0))
    return float(np.mean(values))


def run_one(benchmark: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metadata, logits = _load(benchmark)
    labels = metadata.label.to_numpy(int)
    rows = []
    names = list(logits)
    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(names, size):
            value = np.mean([logits[name] for name in subset], axis=0)
            rows.append({
                "benchmark": benchmark,
                "subset": "+".join(subset),
                "experts": size,
                "mean_subject_BA": _mean_subject_ba(metadata, value),
                "NLL": float(log_loss(labels, _sigmoid(value), labels=[0, 1])),
                "outcome_labels_used_to_rank_subset": True,
                "used_to_tune_router": False,
                "OUTER_TEST_USED": False,
            })
    combinations = pd.DataFrame(rows).sort_values(["mean_subject_BA", "NLL"], ascending=[False, True])
    subject_rows = []
    oracle_values = []
    for subject, group in metadata.assign(row=np.arange(len(metadata))).groupby("subject_id"):
        index = group.row.to_numpy(int)
        candidates = []
        for name, value in logits.items():
            score = float(balanced_accuracy_score(group.label, value[index] >= 0.0))
            candidates.append((score, name, value[index]))
        selected = max(candidates, key=lambda item: (item[0], item[1]))
        subject_rows.append({
            "benchmark": benchmark,
            "subject_id": subject,
            "oracle_expert": selected[1],
            "oracle_BA": selected[0],
            "anchor_BA": next(value[0] for value in candidates if value[1] == "anchor"),
            "used_to_tune_router": False,
            "OUTER_TEST_USED": False,
        })
        oracle_values.append(selected[0])
    subjects = pd.DataFrame(subject_rows)
    matrix = np.column_stack([value >= 0.0 for value in logits.values()])
    trial_oracle_accuracy = float(np.mean(np.any(matrix == labels[:, None], axis=1)))
    summary = {
        "benchmark": benchmark,
        "anchor_BA": _mean_subject_ba(metadata, logits["anchor"]),
        "best_global_equal_weight_subset": combinations.iloc[0].to_dict(),
        "subject_oracle_BA": float(np.mean(oracle_values)),
        "subject_oracle_headroom_pp": float(100 * (np.mean(oracle_values) - _mean_subject_ba(metadata, logits["anchor"]))),
        "trial_oracle_accuracy": trial_oracle_accuracy,
        "outcome_labels_used_for_headroom_only": True,
        "used_to_tune_router": False,
        "OUTER_TEST_USED": False,
    }
    return combinations, subjects, summary


def run() -> None:
    combination_parts = []
    subject_parts = []
    summary = {}
    for benchmark in ("openbmi", "wbcic"):
        combinations, subjects, payload = run_one(benchmark)
        combination_parts.append(combinations)
        subject_parts.append(subjects)
        summary[benchmark] = payload
    write_csv(DIAGNOSTICS / "NEW_BACKBONE_EQUAL_WEIGHT_HEADROOM.csv", pd.concat(combination_parts, ignore_index=True))
    write_csv(DIAGNOSTICS / "NEW_BACKBONE_SUBJECT_ORACLE.csv", pd.concat(subject_parts, ignore_index=True))
    write_json(DIAGNOSTICS / "NEW_BACKBONE_HEADROOM.json", summary)
    write_json(PROTOCOL / "NEW_BACKBONE_ORACLE_LEGALITY.json", {
        "purpose": "headroom diagnosis only",
        "outcome_labels_used": True,
        "used_to_tune_router": False,
        "WBCIC_outer_split_opened": False,
        "OUTER_TEST_USED": False,
    })
    print(summary, flush=True)


if __name__ == "__main__":
    run()
