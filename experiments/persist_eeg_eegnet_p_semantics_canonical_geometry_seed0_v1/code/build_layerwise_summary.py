"""Create an equal-fold descriptive layer summary without tuning on OUTER data.

"Simplest material transform" is a post-hoc mechanism descriptor using the
locked BA and geometric-residual thresholds (the same residual threshold is
applied descriptively to either centroid or class relation), not a fitted
hyperparameter or a deployable selection. All family outcomes remain in the
full complexity-curve CSV.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("temporal_bn", "spatial_elu_pool1", "depth_point_elu_pool2", "embedding_64d")
FAMILIES = ("FULL", "P", "C", "RANDOM")
TRANSFORMS = ("translation", "diagonal_affine", "orthogonal_translation",
              "lowrank_residual_affine_trainCV_selected_rank")


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / f"{name}.csv", keep_default_na=False)


def fold_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        raise RuntimeError(f"empty summary slice: {column}")
    numeric = frame.copy()
    numeric[column] = pd.to_numeric(numeric[column], errors="raise")
    per_fold = numeric.groupby("fold", sort=True)[column].mean()
    if set(per_fold.index) != set(range(5)):
        raise RuntimeError(f"not five folds: {column}, {sorted(per_fold.index)}")
    return float(per_fold.mean())


def case(frame: pd.DataFrame, stage: str, family: str) -> pd.DataFrame:
    return frame[(frame.stage == stage) & (frame.family == family)]


def simplest_material(transform: pd.DataFrame, stage: str, family: str, lock: dict) -> tuple[str, float, int]:
    t = case(transform, stage, family)
    t = t[(t["mode"] == "UNSUPERVISED") & (t.eval_population == "OUTER_DEVELOPMENT")]
    grouped = t.groupby(["fold", "transform"], sort=True)[
        ["BA_recovery", "class_relation_residual", "centroid_residual"]].mean()
    selected_rank = t.groupby("fold", sort=True)["train_CV_lowrank_rank"].first()
    ba_limit = lock["analysis"]["descriptive_material_BA_recovery_threshold"]
    residual_limit = lock["analysis"]["descriptive_material_relation_residual_reduction_fraction"]
    if {f for f, _ in grouped.index} != set(range(5)):
        raise RuntimeError(f"missing transform fold {stage} {family}")
    for name in TRANSFORMS:
        passes = []
        recoveries = []
        for fold in range(5):
            identity = grouped.loc[(fold, "identity")]
            actual_name = (f"lowrank_residual_affine_r{int(selected_rank.loc[fold])}"
                           if name == "lowrank_residual_affine_trainCV_selected_rank" else name)
            candidate = grouped.loc[(fold, actual_name)]
            reduction = (identity.class_relation_residual - candidate.class_relation_residual) / max(
                identity.class_relation_residual, 1e-8)
            centroid_reduction = (identity.centroid_residual - candidate.centroid_residual) / max(
                identity.centroid_residual, 1e-8)
            passes.append(candidate.BA_recovery >= ba_limit and
                          max(reduction, centroid_reduction) >= residual_limit)
            recoveries.append(candidate.BA_recovery)
        if sum(passes) >= 4:
            return name, float(np.mean(recoveries)), int(sum(passes))
    return "NONE", 0.0, 0


def main() -> None:
    out = ROOT / "outputs" / "LAYERWISE_STABLE_TASK_SUMMARY.csv"
    if out.exists():
        raise RuntimeError(f"refusing to overwrite {out}")
    lock = json.loads((ROOT / "protocol" / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    base = read("BASELINE_PERSISTENCE")
    within = read("TASK_RELATION_WITHIN_SUBJECT")
    cross = read("TASK_RELATION_CROSS_SUBJECT")
    probes = read("P_SEMANTIC_PROBES")
    session = read("FROZEN_DECODER_SESSION_TRANSFER")
    subject = read("FROZEN_DECODER_CROSS_SUBJECT_TRANSFER")
    boundary = read("DECODER_BOUNDARY_STABILITY")
    transform = read("TRANSFORMATION_COMPLEXITY_CURVE")
    rows = []
    for stage in STAGES:
        for family in FAMILIES:
            b = case(base, stage, family)
            b = b[(b.population == "OUTER_DEVELOPMENT") & (b.metric == "cosine")]
            w = case(within, stage, family)
            w = w[(w.population == "OUTER_DEVELOPMENT") & (w.metric == "cosine")]
            c = case(cross, stage, family)
            c = c[(c.population == "OUTER_DEVELOPMENT") & (c.metric == "cosine") &
                  (c.target_session.astype(str) == c.reference_session.astype(str))]
            p = case(probes, stage, family)
            p = p[p.probe == "subject_ID"]
            s = case(session, stage, family)
            s = s[s.eval_population == "OUTER_DEVELOPMENT"]
            s12 = s[(s.fit_session.astype(str) == "1") & (s.eval_session.astype(str) == "2")]
            x = case(subject, stage, family)
            x = x[(x.eval_population == "OUTER_DEVELOPMENT") & (x.eval_session.astype(str) == "both")]
            angle = case(boundary, stage, family)
            complexity, recovery, supporting = simplest_material(transform, stage, family, lock)
            rows.append({
                "stage": stage, "family": family, "folds": 5,
                "subject_baseline_persistence_cosine_outer_equal_fold": fold_mean(b, "subject_equal_mean"),
                "within_subject_task_relation_cosine_outer_equal_fold": fold_mean(w, "subject_equal_mean"),
                "cross_subject_task_relation_cosine_outer_same_session_equal_fold": fold_mean(c, "subject_equal_mean"),
                "subject_ID_probe_BA_train_equal_fold": fold_mean(p, "balanced_accuracy"),
                "task_decoder_session_transfer_BA_outer_bidirectional_equal_fold": fold_mean(s, "BA"),
                "task_decoder_S1_to_S2_BA_outer_equal_fold": fold_mean(s12, "BA"),
                "task_decoder_cross_subject_BA_outer_both_sessions_equal_fold": fold_mean(x, "BA"),
                "decoder_boundary_rotation_degrees_equal_fold": fold_mean(angle, "normal_angle_degrees"),
                "descriptive_simplest_material_unsupervised_transform_complexity": complexity,
                "descriptive_simplest_transform_BA_recovery_outer_equal_fold": recovery,
                "descriptive_material_transform_supporting_folds": supporting,
                "protected_erasure_consequence_prior": "NOT_COMPARABLE_TO_FRESH_TRAIN_ONLY_GEOMETRY",
                "selection_scope": "posthoc descriptive mechanism criterion; no outer outcome used for fit or hyperparameter selection",
            })
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.9g", lineterminator="\n")
    print(out)


if __name__ == "__main__":
    main()
