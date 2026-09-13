#!/usr/bin/env python3
"""Compute pairwise and seed-0 union label-informed oracle upper bounds."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


TRIAL_KEY = ["task", "seed", "fold", "subject_id", "session", "trial_id"]


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_name(path.name + ".part"); frame.to_csv(temp, index=False); os.replace(temp, path)


def ba(group: pd.DataFrame, correct: pd.Series) -> float:
    return float(pd.DataFrame({"label": group.true_label.to_numpy(), "correct": correct.astype(float).to_numpy()}).groupby("label").correct.mean().mean())


def category(headroom_pp: float) -> str:
    if headroom_pp >= 2.0: return "HIGH_COMPLEMENTARITY"
    if headroom_pp >= 1.0: return "MODERATE_COMPLEMENTARITY"
    if headroom_pp >= 0.3: return "LOW_COMPLEMENTARITY"
    return "NEGLIGIBLE_COMPLEMENTARITY"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, required=True); args = parser.parse_args()
    outputs = args.repo.resolve() / "experiments/persist_eeg_b0_x_xs_complementarity_audit_v1/outputs"
    paired = pd.read_csv(outputs / "PAIRED_TRIAL_RESULTS.csv", dtype={"subject_id": str})
    subject_rows = []
    group_columns = ["task", "seed", "fold", "subject_id", "candidate_name"]
    for keys, group in paired.groupby(group_columns, sort=True):
        b0 = ba(group, group.B0_correct); candidate = ba(group, group.candidate_correct)
        oracle = ba(group, group.B0_correct | group.candidate_correct); headroom = oracle - b0
        rescue = float(group.assign(rescue=group.error_state.eq("RESCUE")).groupby("true_label").rescue.mean().mean())
        if abs(headroom - rescue) > 1e-12:
            raise RuntimeError(f"oracle/rescue identity failed: {keys}")
        subject_rows.append(dict(zip(group_columns, keys)) | {"B0_BA": b0, "candidate_BA": candidate, "oracle_BA": oracle,
            "candidate_delta_pp": 100 * (candidate - b0), "oracle_headroom_pp": 100 * headroom,
            "realized_fraction_of_oracle": (candidate - b0) / headroom if headroom > 0 else np.nan,
            "oracle_rescue_identity_abs_error": abs(headroom - rescue), "status": "PASS"})
    subjects = pd.DataFrame(subject_rows); atomic_csv(outputs / "ORACLE_SUBJECT_RESULTS.csv", subjects)
    summary = subjects.groupby(["task", "seed", "candidate_name"], as_index=False).agg(
        B0_BA=("B0_BA", "mean"), candidate_BA=("candidate_BA", "mean"), oracle_BA=("oracle_BA", "mean"),
        candidate_delta_pp=("candidate_delta_pp", "mean"), SUBJECT_EQUAL_ORACLE_HEADROOM_PP=("oracle_headroom_pp", "mean"),
        oracle_rescue_identity_max_abs_error=("oracle_rescue_identity_abs_error", "max"), eligible_folds=("fold", "nunique"))
    summary["requested_folds"] = 5
    summary["headroom_category"] = summary.SUBJECT_EQUAL_ORACLE_HEADROOM_PP.map(category)
    summary["status"] = np.where(summary.eligible_folds.eq(5), "PASS", "PROTOCOL_PARTIAL")
    atomic_csv(outputs / "ORACLE_TASK_SUMMARY.csv", summary)

    x = paired[(paired.seed == 0) & paired.candidate_name.eq("LiteBN_X")].copy()
    xs = paired[(paired.seed == 0) & paired.candidate_name.eq("LiteBN_XS")].copy()
    union_rows = []
    for task in sorted(set(x.task).intersection(xs.task)):
        left = x[x.task == task].set_index(TRIAL_KEY).sort_index(); right = xs[xs.task == task].set_index(TRIAL_KEY).sort_index()
        common = left.index.intersection(right.index)
        if common.empty:
            raise RuntimeError(f"PAIR_ALIGNMENT_FAIL seed0 union has no common keys: {task}")
        left, right = left.loc[common], right.loc[common]
        if not np.array_equal(left.true_label.to_numpy(), right.true_label.to_numpy()):
            raise RuntimeError(f"PAIR_ALIGNMENT_FAIL seed0 union labels: {task}")
        if not np.array_equal(left.normalizer_sha256.to_numpy(), right.normalizer_sha256.to_numpy()):
            raise RuntimeError(f"PAIR_ALIGNMENT_FAIL seed0 union normalizers: {task}")
        joined = left.reset_index()[TRIAL_KEY + ["true_label", "B0_correct", "candidate_correct", "B0_checkpoint_sha256", "candidate_checkpoint_sha256", "normalizer_sha256"]].rename(columns={"candidate_correct": "X_correct", "candidate_checkpoint_sha256": "X_checkpoint_sha256"})
        joined["XS_correct"] = right.candidate_correct.to_numpy(); joined["XS_checkpoint_sha256"] = right.candidate_checkpoint_sha256.to_numpy()
        per_subject = []
        for (fold, subject), group in joined.groupby(["fold", "subject_id"], sort=True):
            b0 = ba(group, group.B0_correct)
            x_oracle = ba(group, group.B0_correct | group.X_correct)
            xs_oracle = ba(group, group.B0_correct | group.XS_correct)
            oracle = ba(group, group.B0_correct | group.X_correct | group.XS_correct); wrong = ~group.B0_correct
            per_subject.append({"fold": int(fold), "subject_id": str(subject), "B0_BA": b0, "union_oracle_BA": oracle,
                "X_oracle_headroom_pp": 100 * (x_oracle - b0), "XS_oracle_headroom_on_union_B0_pp": 100 * (xs_oracle - b0),
                "union_oracle_headroom_pp": 100 * (oracle - b0), "X_only_rescue": int((wrong & group.X_correct & ~group.XS_correct).sum()),
                "XS_only_rescue": int((wrong & ~group.X_correct & group.XS_correct).sum()),
                "X_and_XS_both_rescue": int((wrong & group.X_correct & group.XS_correct).sum())})
        subject_frame = pd.DataFrame(per_subject)
        x_headroom = float(subject_frame.X_oracle_headroom_pp.mean())
        xs_headroom = float(subject_frame.XS_oracle_headroom_on_union_B0_pp.mean())
        union_headroom = float(subject_frame.union_oracle_headroom_pp.mean())
        union_rows.append({"task": task, "seed": 0, "B0_BA": float(subject_frame.B0_BA.mean()),
            "union_oracle_BA": float(subject_frame.union_oracle_BA.mean()), "union_oracle_headroom_pp": union_headroom,
            "X_oracle_headroom_pp": x_headroom, "XS_oracle_headroom_on_union_B0_pp": xs_headroom,
            "union_gain_over_best_single_pp": union_headroom - max(x_headroom, xs_headroom),
            "X_only_rescue": int(subject_frame.X_only_rescue.sum()), "XS_only_rescue": int(subject_frame.XS_only_rescue.sum()),
            "X_and_XS_both_rescue": int(subject_frame.X_and_XS_both_rescue.sum()),
            "eligible_folds": int(subject_frame.fold.nunique()), "requested_folds": 5,
            "B0_provenance": "X-seed0 historical cell",
            "status": "PASS" if subject_frame.fold.nunique() == 5 else "PROTOCOL_PARTIAL"})
    union = pd.DataFrame(union_rows)
    if len(union) != 4:
        raise RuntimeError(f"seed0 union task cardinality {len(union)} != 4")
    atomic_csv(outputs / "SEED0_UNION_ORACLE_SUMMARY.csv", union)
    print(f"ORACLE_COMPLETE pairwise={len(summary)} union={len(union)}", flush=True)


if __name__ == "__main__":
    main()
