#!/usr/bin/env python3
"""One-to-one pairing and descriptive B0-versus-candidate error audit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEY = ["task", "seed", "fold", "subject_id", "session", "trial_id"]
STATES = ("CC", "RESCUE", "HARM", "WW")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part"); frame.to_csv(temp, index=False); os.replace(temp, path)


def vector(value: str) -> np.ndarray:
    result = np.asarray(json.loads(value), dtype=np.float64)
    if result.ndim != 1 or len(result) < 2 or not np.isfinite(result).all():
        raise RuntimeError("PAIR_ALIGNMENT_FAIL: invalid output vector")
    return result


def entropy(probability: np.ndarray) -> float:
    return float(-(probability * np.log(np.clip(probability, 1e-12, 1.0))).sum())


def margin(probability: np.ndarray) -> float:
    ordered = np.sort(probability)
    return float(ordered[-1] - ordered[-2])


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    middle = 0.5 * (left + right)
    kl = lambda p: float((p * np.log(np.clip(p / np.clip(middle, 1e-12, None), 1e-12, None))).sum())
    return 0.5 * kl(left) + 0.5 * kl(right)


def state(b0: bool, candidate: bool) -> str:
    if b0 and candidate: return "CC"
    if not b0 and candidate: return "RESCUE"
    if b0 and not candidate: return "HARM"
    return "WW"


def balanced_rate(group: pd.DataFrame, target: str) -> float:
    return float(group.assign(hit=group.error_state.eq(target)).groupby("true_label").hit.mean().mean())


def balanced_accuracy_from_correct(group: pd.DataFrame, column: str) -> float:
    return float(group.groupby("true_label")[column].mean().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.repo.resolve() / "experiments/persist_eeg_b0_x_xs_complementarity_audit_v1"
    outputs = experiment / "outputs"; cells_dir = args.runtime.resolve() / "replay_cells"
    status = pd.read_csv(outputs / "REPLAY_CELL_STATUS.csv")
    frames = []
    for cell in status.itertuples(index=False):
        if cell.status != "PASS":
            continue
        path = cells_dir / f"{cell.comparison}__{cell.task}__seed{int(cell.seed)}__fold{int(cell.fold)}.csv.gz"
        if not path.is_file():
            raise RuntimeError(f"missing passed replay cell: {path}")
        long = pd.read_csv(path, dtype={"subject_id": str})
        candidate_name = "LiteBN_X" if cell.comparison == "X" else "LiteBN_XS"
        b0 = long[long.method == "LiteBN_BASELINE"].copy(); candidate = long[long.method == candidate_name].copy()
        if len(b0) != len(candidate) or b0.duplicated(KEY).any() or candidate.duplicated(KEY).any():
            raise RuntimeError(f"PAIR_ALIGNMENT_FAIL: cardinality/duplicate {path}")
        b0 = b0.set_index(KEY).sort_index(); candidate = candidate.set_index(KEY).sort_index()
        if not b0.index.equals(candidate.index) or not np.array_equal(b0.true_label.to_numpy(), candidate.true_label.to_numpy()):
            raise RuntimeError(f"PAIR_ALIGNMENT_FAIL: keys/labels {path}")
        if b0.normalizer_sha256.nunique() != 1 or candidate.normalizer_sha256.nunique() != 1 or b0.normalizer_sha256.iloc[0] != candidate.normalizer_sha256.iloc[0]:
            raise RuntimeError(f"PAIR_ALIGNMENT_FAIL: normalizer {path}")
        wide = b0.reset_index()[KEY + ["true_label"]]
        wide["B0_logits"] = b0.logits.to_numpy(); wide["B0_probabilities"] = b0.probabilities.to_numpy()
        wide["B0_prediction"] = b0.prediction.astype(int).to_numpy(); wide["B0_correct"] = wide.B0_prediction.eq(wide.true_label)
        wide["candidate_name"] = candidate_name; wide["candidate_logits"] = candidate.logits.to_numpy()
        wide["candidate_probabilities"] = candidate.probabilities.to_numpy(); wide["candidate_prediction"] = candidate.prediction.astype(int).to_numpy()
        wide["candidate_correct"] = wide.candidate_prediction.eq(wide.true_label); wide["B0_checkpoint_sha256"] = b0.checkpoint_sha256.to_numpy()
        wide["candidate_checkpoint_sha256"] = candidate.checkpoint_sha256.to_numpy(); wide["normalizer_sha256"] = b0.normalizer_sha256.to_numpy()
        observations: list[dict[str, Any]] = []
        for row in wide.itertuples(index=False):
            p0, pc = vector(row.B0_probabilities), vector(row.candidate_probabilities)
            if p0.shape != pc.shape or abs(p0.sum() - 1) > 1e-5 or abs(pc.sum() - 1) > 1e-5:
                raise RuntimeError(f"PAIR_ALIGNMENT_FAIL: probabilities {path}")
            observations.append({"B0_margin": margin(p0), "candidate_margin": margin(pc), "B0_entropy": entropy(p0),
                "candidate_entropy": entropy(pc), "probability_L1_distance": float(np.abs(p0 - pc).sum()), "JS_divergence": js_divergence(p0, pc),
                "error_state": state(bool(row.B0_correct), bool(row.candidate_correct))})
        frames.append(pd.concat([wide, pd.DataFrame(observations)], axis=1))
        print(f"PAIRED {cell.comparison} {cell.task} seed{int(cell.seed)} fold{int(cell.fold)} PASS", flush=True)
    if not frames:
        raise RuntimeError("no replay-passed cells available")
    paired = pd.concat(frames, ignore_index=True).sort_values(KEY + ["candidate_name"]).reset_index(drop=True)
    atomic_csv(outputs / "PAIRED_TRIAL_RESULTS.csv", paired)

    subject_rows = []
    subject_keys = ["task", "seed", "fold", "subject_id", "candidate_name"]
    for keys, group in paired.groupby(subject_keys, sort=True):
        counts = group.error_state.value_counts(); b0_ba = balanced_accuracy_from_correct(group, "B0_correct")
        candidate_ba = balanced_accuracy_from_correct(group, "candidate_correct")
        rescue, harm = balanced_rate(group, "RESCUE"), balanced_rate(group, "HARM")
        subject_rows.append(dict(zip(subject_keys, keys)) | {"B0_BA": b0_ba, "candidate_BA": candidate_ba, "delta_BA": candidate_ba - b0_ba,
            **{f"N_{name}": int(counts.get(name, 0)) for name in STATES}, "balanced_rescue_rate": rescue, "balanced_harm_rate": harm,
            "identity_abs_error": abs((candidate_ba - b0_ba) - (rescue - harm)), "status": "PASS"})
    subjects = pd.DataFrame(subject_rows)
    if subjects.identity_abs_error.max() > 1e-12:
        raise RuntimeError(f"balanced rescue/harm identity failed: {subjects.identity_abs_error.max()}")
    atomic_csv(outputs / "PAIRWISE_SUBJECT_SUMMARY.csv", subjects)

    task_rows = []
    for keys, group in paired.groupby(["task", "seed", "candidate_name"], sort=True):
        counts = group.error_state.value_counts(); n_total = len(group)
        subject_group = subjects[(subjects.task == keys[0]) & (subjects.seed == keys[1]) & (subjects.candidate_name == keys[2])]
        n_b0_wrong = int(counts.get("RESCUE", 0) + counts.get("WW", 0)); n_b0_correct = int(counts.get("HARM", 0) + counts.get("CC", 0))
        rescue = float(subject_group.balanced_rescue_rate.mean()); harm = float(subject_group.balanced_harm_rate.mean())
        task_rows.append({"task": keys[0], "seed": int(keys[1]), "candidate": keys[2], **{f"N_{name}": int(counts.get(name, 0)) for name in STATES}, "N_total": int(n_total),
            "rescue_fraction_all": float(counts.get("RESCUE", 0) / n_total), "harm_fraction_all": float(counts.get("HARM", 0) / n_total),
            "rescue_given_B0_wrong": float(counts.get("RESCUE", 0) / n_b0_wrong) if n_b0_wrong else np.nan,
            "harm_given_B0_correct": float(counts.get("HARM", 0) / n_b0_correct) if n_b0_correct else np.nan,
            "balanced_rescue_rate": rescue, "balanced_harm_rate": harm,
            "subject_equal_B0_BA": float(subject_group.B0_BA.mean()), "subject_equal_candidate_BA": float(subject_group.candidate_BA.mean()),
            "identity_abs_error": abs(float(subject_group.delta_BA.mean()) - (rescue - harm)),
            "eligible_folds": int(subject_group.fold.nunique()), "requested_folds": 5,
            "status": "PASS" if subject_group.fold.nunique() == 5 else "PROTOCOL_PARTIAL"})
    summary = pd.DataFrame(task_rows)
    if summary.identity_abs_error.max() > 1e-12:
        raise RuntimeError("task-level balanced rescue/harm identity failed")
    atomic_csv(outputs / "PAIRWISE_ERROR_SUMMARY.csv", summary)
    print(f"PAIRWISE_COMPLETE rows={len(paired)} subjects={len(subjects)}", flush=True)


if __name__ == "__main__":
    main()
