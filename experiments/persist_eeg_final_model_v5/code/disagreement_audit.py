from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from common import DIAGNOSTICS, ensure_directories, sigmoid, write_csv
from datasets import V5Dataset, load_openbmi, load_wbcic


def _ba(labels: np.ndarray, prediction: np.ndarray) -> float:
    return float(balanced_accuracy_score(labels, prediction))


def _stratum_rows(data: V5Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    logits = data.expert_logits.copy()
    mask = data.expert_mask
    probability = sigmoid(logits)
    votes = np.where(mask, logits >= 0, np.nan)
    n_experts = mask.sum(axis=1)
    class1_votes = np.nansum(votes, axis=1).astype(int)
    majority_count = np.maximum(class1_votes, n_experts - class1_votes)
    vote_split = np.array([f"{int(major)}-{int(total-major)}" for major, total in zip(majority_count, n_experts)])
    mean_probability = np.nanmean(probability, axis=1)
    entropy = -(np.clip(mean_probability, 1e-7, 1 - 1e-7) * np.log(np.clip(mean_probability, 1e-7, 1 - 1e-7)) + (1 - np.clip(mean_probability, 1e-7, 1 - 1e-7)) * np.log1p(-np.clip(mean_probability, 1e-7, 1 - 1e-7)))
    variance = np.nanvar(probability, axis=1)
    base = data.static_prediction
    expert_prediction = (logits >= 0).astype(int)
    expert_correct = expert_prediction == data.labels[:, None]
    oracle = np.any(expert_correct & mask, axis=1)
    best_expert = np.argmax(
        [
            np.mean(expert_correct[:, index][mask[:, index]])
            for index in range(logits.shape[1])
        ]
    )
    best_static_prediction = expert_prediction[:, best_expert]
    frame = pd.DataFrame(
        {
            "vote_split": vote_split,
            "unanimous": majority_count == n_experts,
            "base_margin_abs": np.abs(mean_probability - 0.5),
            "probability_variance": variance,
            "vote_entropy": entropy,
            "base_correct": base == data.labels,
            "best_static_correct": best_static_prediction == data.labels,
            "oracle_correct": oracle,
            "predicted_class": base,
            "session": data.sessions,
        }
    )
    frame["margin_bin"] = pd.qcut(frame.base_margin_abs, q=5, duplicates="drop").astype(str)
    frame["variance_bin"] = pd.qcut(frame.probability_variance, q=5, duplicates="drop").astype(str)
    rows = []
    for variable in ("vote_split", "unanimous", "margin_bin", "variance_bin", "predicted_class", "session"):
        for level, indices in frame.groupby(variable, observed=True).groups.items():
            idx = np.asarray(list(indices), dtype=int)
            labels = data.labels[idx]
            base_prediction = base[idx]
            best_prediction = best_static_prediction[idx]
            oracle_prediction = np.where(oracle[idx], labels, base_prediction)
            rows.append(
                {
                    "dataset": data.dataset_id,
                    "stratifier": variable,
                    "stratum": str(level),
                    "trials": len(idx),
                    "B_STRONG_CURRENT_BA": _ba(labels, data.current_prediction[idx]),
                    "static_ensemble_BA": _ba(labels, base_prediction),
                    "best_static_expert_BA": _ba(labels, best_prediction),
                    "oracle_BA": _ba(labels, oracle_prediction),
                    "available_oracle_gain": _ba(labels, oracle_prediction) - _ba(labels, data.current_prediction[idx]),
                    "majority_wrong_minority_correct": int(np.sum((base_prediction != labels) & oracle[idx])),
                    "OUTER_TEST_USED": False,
                }
            )
    headroom_rows = []
    for index, expert in enumerate(data.expert_names):
        valid = mask[:, index]
        prediction = expert_prediction[:, index]
        rescue = valid & (data.current_prediction != data.labels) & (prediction == data.labels)
        harm = valid & (data.current_prediction == data.labels) & (prediction != data.labels)
        minority = valid & (prediction != base)
        headroom_rows.append(
            {
                "dataset": data.dataset_id,
                "expert": expert,
                "available_trials": int(valid.sum()),
                "unique_rescue_vs_current": int(rescue.sum()),
                "harm_vs_current": int(harm.sum()),
                "minority_trials": int(minority.sum()),
                "minority_rescue_accuracy": float(np.mean(prediction[minority] == data.labels[minority])) if minority.any() else 0.0,
                "OUTER_TEST_USED": False,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(headroom_rows)


def run() -> None:
    ensure_directories()
    strata, maps = [], []
    for data in (load_openbmi(), load_wbcic()):
        part_strata, part_map = _stratum_rows(data)
        strata.append(part_strata)
        maps.append(part_map)
    strata_frame = pd.concat(strata, ignore_index=True)
    map_frame = pd.concat(maps, ignore_index=True)
    write_csv(DIAGNOSTICS / "DISAGREEMENT_STRATA.csv", strata_frame)
    write_csv(DIAGNOSTICS / "ORACLE_HEADROOM_MAP.csv", map_frame)
    lines = ["# Disagreement geometry", ""]
    for dataset, group in strata_frame.groupby("dataset"):
        vote = group[group.stratifier.eq("vote_split")].sort_values("stratum")
        richest = vote.sort_values("available_oracle_gain", ascending=False).iloc[0]
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"The largest oracle gap is in vote stratum `{richest.stratum}`: "
                f"{int(richest.trials)} trials and {100*richest.available_oracle_gain:.2f} pp available BA.",
                "",
                vote.to_markdown(index=False),
                "",
            ]
        )
    (DIAGNOSTICS / "DISAGREEMENT_GEOMETRY.md").write_text("\n".join(lines), encoding="utf-8")
    print(strata_frame[strata_frame.stratifier.eq("vote_split")].to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
