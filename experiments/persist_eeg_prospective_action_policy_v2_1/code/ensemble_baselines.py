from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


BASELINE_IDS = (
    "B0_TARGET_KEEP",
    "B1_OTHER_RUN_HARD_MAJORITY",
    "B2_ALL_RUN_HARD_MAJORITY",
    "B3_OTHER_RUN_PROBABILITY_MEAN",
    "B4_ALL_RUN_PROBABILITY_MEAN",
    "B5_OTHER_RUN_LOGIT_MEAN",
    "B6_ALL_RUN_LOGIT_MEAN",
    "B7_CONFIDENCE_WEIGHTED_KEEP_ENSEMBLE",
)


@dataclass(frozen=True)
class PredictionSpec:
    method_id: str
    prediction: np.ndarray
    probability: np.ndarray | None
    score_kind: str
    frozen_model_count: np.ndarray
    keep_forward_passes: np.ndarray
    intervention_logits_required: bool
    representation_projection_required: bool
    target_run_specific: bool
    description: str

    def validate(self, rows: int) -> None:
        if self.prediction.shape != (rows,):
            raise RuntimeError(f"{self.method_id}: prediction shape mismatch")
        if not np.isin(self.prediction, (0, 1)).all():
            raise RuntimeError(f"{self.method_id}: predictions are not binary")
        if self.probability is not None:
            if self.probability.shape != (rows,):
                raise RuntimeError(f"{self.method_id}: probability shape mismatch")
            if not np.isfinite(self.probability).all() or np.any((self.probability < 0) | (self.probability > 1)):
                raise RuntimeError(f"{self.method_id}: invalid probabilities")


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def validate_manifest_groups(frame: pd.DataFrame) -> None:
    required = {
        "manifest_index",
        "subject_id",
        "session_id",
        "fold_id",
        "seed_id",
        "pred_noop",
        "p1_noop",
        "margin_noop",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing ensemble columns: {missing}")
    if frame.duplicated(["manifest_index", "fold_id", "seed_id"]).any():
        raise RuntimeError("A frozen run occurs more than once for a manifest trial")
    grouped = frame.groupby("manifest_index", sort=False)
    for column in ("subject_id", "session_id"):
        if int(grouped[column].nunique().max()) != 1:
            raise RuntimeError(f"manifest_index maps to multiple {column} values")
    if "outcome_label" in frame and int(grouped.outcome_label.nunique().max()) != 1:
        raise RuntimeError("manifest_index maps to multiple labels")
    if int(grouped.size().min()) < 2:
        raise RuntimeError("Leave-target-run consensus requires at least two frozen runs")


def _group_sum(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    temporary = pd.Series(np.asarray(values, dtype=float), index=frame.index)
    return temporary.groupby(frame.manifest_index, sort=False).transform("sum").to_numpy(dtype=float)


def _group_count(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby("manifest_index", sort=False).manifest_index.transform("size").to_numpy(dtype=int)


def confidence_weighted_probability(frame: pd.DataFrame, probabilities: np.ndarray) -> np.ndarray:
    """Frozen B7 rule; it has no label or outcome dependency."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(frame),):
        raise ValueError("probability shape mismatch")
    weights = np.maximum(np.abs(probabilities - 0.5), 1e-12)
    numerator = _group_sum(frame, weights * probabilities)
    denominator = _group_sum(frame, weights)
    return numerator / denominator


def build_ensemble_baselines(frame: pd.DataFrame) -> dict[str, PredictionSpec]:
    """Build the finite, predeclared B0--B7 family without reading labels."""
    validate_manifest_groups(frame)
    rows = len(frame)
    target_prediction = frame.pred_noop.to_numpy(dtype=int)
    target_probability = frame.p1_noop.to_numpy(dtype=float)
    target_margin = frame.margin_noop.to_numpy(dtype=float)
    count = _group_count(frame)
    other_count = count - 1

    vote_total = _group_sum(frame, target_prediction)
    probability_total = _group_sum(frame, target_probability)
    margin_total = _group_sum(frame, target_margin)
    other_vote = (vote_total - target_prediction) / other_count
    all_vote = vote_total / count
    other_probability = (probability_total - target_probability) / other_count
    all_probability = probability_total / count
    other_margin = (margin_total - target_margin) / other_count
    all_margin = margin_total / count
    weighted_probability = confidence_weighted_probability(frame, target_probability)

    one = np.ones(rows, dtype=int)
    all_models = count.astype(int)
    other_models = other_count.astype(int)
    specs = {
        "B0_TARGET_KEEP": PredictionSpec(
            "B0_TARGET_KEEP",
            target_prediction.copy(),
            target_probability.copy(),
            "target KEEP probability",
            one,
            one,
            False,
            False,
            True,
            "Original target frozen-run KEEP output.",
        ),
        "B1_OTHER_RUN_HARD_MAJORITY": PredictionSpec(
            "B1_OTHER_RUN_HARD_MAJORITY",
            (other_vote >= 0.5).astype(int),
            other_vote,
            "leave-target-run hard-vote fraction",
            other_models,
            other_models,
            False,
            False,
            True,
            "Target run excluded; ties map to class 1, although available V2 groups yield odd other-run counts.",
        ),
        "B2_ALL_RUN_HARD_MAJORITY": PredictionSpec(
            "B2_ALL_RUN_HARD_MAJORITY",
            (all_vote >= 0.5).astype(int),
            all_vote,
            "all-run hard-vote fraction",
            all_models,
            all_models,
            False,
            False,
            False,
            "All frozen KEEP classes; exact ties deterministically map to class 1.",
        ),
        "B3_OTHER_RUN_PROBABILITY_MEAN": PredictionSpec(
            "B3_OTHER_RUN_PROBABILITY_MEAN",
            (other_probability >= 0.5).astype(int),
            other_probability,
            "leave-target-run arithmetic mean KEEP probability",
            other_models,
            other_models,
            False,
            False,
            True,
            "Target run excluded; fixed 0.5 threshold.",
        ),
        "B4_ALL_RUN_PROBABILITY_MEAN": PredictionSpec(
            "B4_ALL_RUN_PROBABILITY_MEAN",
            (all_probability >= 0.5).astype(int),
            all_probability,
            "all-run arithmetic mean KEEP probability",
            all_models,
            all_models,
            False,
            False,
            False,
            "All frozen KEEP probabilities; fixed 0.5 threshold.",
        ),
        "B5_OTHER_RUN_LOGIT_MEAN": PredictionSpec(
            "B5_OTHER_RUN_LOGIT_MEAN",
            (other_margin >= 0.0).astype(int),
            sigmoid(other_margin),
            "sigmoid of leave-target-run mean KEEP margin",
            other_models,
            other_models,
            False,
            False,
            True,
            "Target run excluded; margins averaged and thresholded at zero.",
        ),
        "B6_ALL_RUN_LOGIT_MEAN": PredictionSpec(
            "B6_ALL_RUN_LOGIT_MEAN",
            (all_margin >= 0.0).astype(int),
            sigmoid(all_margin),
            "sigmoid of all-run mean KEEP margin",
            all_models,
            all_models,
            False,
            False,
            False,
            "All frozen KEEP margins averaged and thresholded at zero.",
        ),
        "B7_CONFIDENCE_WEIGHTED_KEEP_ENSEMBLE": PredictionSpec(
            "B7_CONFIDENCE_WEIGHTED_KEEP_ENSEMBLE",
            (weighted_probability >= 0.5).astype(int),
            weighted_probability,
            "all-run confidence-weighted KEEP probability",
            all_models,
            all_models,
            False,
            False,
            False,
            "Weights are max(abs(p-0.5), 1e-12); no fitted parameter and fixed 0.5 threshold.",
        ),
    }
    if tuple(specs) != BASELINE_IDS:
        raise RuntimeError("Predeclared baseline family changed")
    for spec in specs.values():
        spec.validate(rows)
    return specs

