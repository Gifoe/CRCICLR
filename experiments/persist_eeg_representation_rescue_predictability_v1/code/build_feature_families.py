#!/usr/bin/env python3
"""Build fixed F0/F1/F2 feature matrices without target-derived predictors."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

F0 = [
    "b0_top1_probability", "b0_margin", "b0_entropy", "b0_logit_l2_norm",
    "candidate_top1_probability", "candidate_margin", "candidate_entropy", "candidate_logit_l2_norm",
    "top1_probability_difference", "margin_difference", "entropy_difference",
    "probability_L1_distance", "JS_divergence",
]
KEY = ["comparison", "task", "seed", "fold", "subject_id", "session", "trial_id"]


def parse(series: pd.Series) -> np.ndarray:
    return np.vstack(series.map(json.loads))


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    repo, runtime = args.repo.resolve(), args.runtime.resolve()
    source = repo / "experiments/persist_eeg_stable_rescue_predictability_audit_v1/outputs/PREDICTABILITY_FEATURES.csv"
    outputs = repo / "experiments/persist_eeg_representation_rescue_predictability_v1/outputs"
    base = pd.read_csv(source, dtype={"subject_id": str})
    reps = pd.read_csv(runtime / "FULL_REPRESENTATIONS.csv.gz", dtype={"subject_id": str})
    frame = base.merge(reps, on=KEY, validate="one_to_one")
    task_schema: dict[str, Any] = {}
    max_classes = 0
    for task, index in frame.groupby("task").groups.items():
        idx = np.asarray(list(index), dtype=int)
        b0_logits = parse(frame.loc[idx, "B0_logits"])
        candidate_logits = parse(frame.loc[idx, "candidate_logits"])
        b0_prob = parse(frame.loc[idx, "B0_probabilities"])
        candidate_prob = parse(frame.loc[idx, "candidate_probabilities"])
        classes = b0_logits.shape[1]
        if candidate_logits.shape[1] != classes or b0_prob.shape[1] != classes or candidate_prob.shape[1] != classes:
            raise RuntimeError(f"class coordinate mismatch for {task}")
        max_classes = max(max_classes, classes)
        b0_centered = b0_logits - b0_logits.mean(axis=1, keepdims=True)
        candidate_centered = candidate_logits - candidate_logits.mean(axis=1, keepdims=True)
        blocks = {
            "b0_centered_logit": b0_centered,
            "candidate_centered_logit": candidate_centered,
            "delta_centered_logit": candidate_centered - b0_centered,
            "b0_probability": b0_prob,
            "candidate_probability": candidate_prob,
            "delta_probability": candidate_prob - b0_prob,
        }
        f1 = list(F0)
        for prefix, matrix in blocks.items():
            for coordinate in range(classes):
                name = f"{prefix}_{coordinate}"
                frame.loc[idx, name] = matrix[:, coordinate]
                f1.append(name)
        if classes == 2:
            signed_b0 = b0_logits[:, 1] - b0_logits[:, 0]
            signed_candidate = candidate_logits[:, 1] - candidate_logits[:, 0]
            frame.loc[idx, "b0_signed_logit_margin"] = signed_b0
            frame.loc[idx, "candidate_signed_logit_margin"] = signed_candidate
            frame.loc[idx, "signed_margin_delta"] = signed_candidate - signed_b0
            f1 += ["b0_signed_logit_margin", "candidate_signed_logit_margin", "signed_margin_delta"]
        task_schema[str(task)] = {"class_count": classes, "class_coordinate_order": list(range(classes)), "F0": list(F0), "F1": f1}

    b0_embedding = parse(frame.B0_embedding)
    candidate_embedding = parse(frame.candidate_embedding)
    if b0_embedding.shape[1] != 64 or candidate_embedding.shape[1] != 128:
        raise RuntimeError(f"embedding dimensions are {b0_embedding.shape[1]}/{candidate_embedding.shape[1]}, expected 64/128")
    for coordinate in range(b0_embedding.shape[1]):
        frame[f"b0_embedding_{coordinate}"] = b0_embedding[:, coordinate]
    for coordinate in range(candidate_embedding.shape[1]):
        frame[f"candidate_embedding_{coordinate}"] = candidate_embedding[:, coordinate]
    for task in task_schema:
        task_schema[task]["F2"] = task_schema[task]["F1"] + [f"b0_embedding_{i}" for i in range(64)] + [f"candidate_embedding_{i}" for i in range(128)]
        task_schema[task]["b0_embedding_dim"] = 64
        task_schema[task]["candidate_embedding_dim"] = 128

    forbidden = ["true_label", "correct", "outcome", "subject_id", "fold", "seed", "task"]
    for task, spec in task_schema.items():
        for family in ("F0", "F1", "F2"):
            offenders = [column for column in spec[family] if any(token in column.lower() for token in forbidden)]
            if offenders:
                raise RuntimeError(f"forbidden predictors in {task}/{family}: {offenders}")
            if frame.loc[frame.task == task, spec[family]].isna().any().any():
                raise RuntimeError(f"missing feature values in {task}/{family}")

    keep = list(dict.fromkeys(list(base.columns) + [column for spec in task_schema.values() for family in ("F1", "F2") for column in spec[family]]))
    frame[keep].to_csv(runtime / "REPRESENTATION_FEATURES.csv.gz", index=False, compression="gzip")
    schema = {
        "feature_families": {"F0": "CONF_ONLY", "F1": "SIGNED_LOGITS", "F2": "SIGNED_PLUS_EMBEDDINGS"},
        "tasks": task_schema,
        "predictor_exclusions": forbidden,
        "rows": len(frame),
        "source_control": str(source),
        "ordered_class_coordinates_preserved": True,
        "embedding_reduction": "NONE",
    }
    atomic_json(outputs / "REPRESENTATION_FEATURE_SCHEMA.json", schema)
    print(f"FEATURE_FAMILIES_COMPLETE rows={len(frame)} max_classes={max_classes} b0_dim=64 candidate_dim=128", flush=True)


if __name__ == "__main__":
    main()
