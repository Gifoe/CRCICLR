from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ensemble_baselines import PredictionSpec


CONTROL_IDS = (
    "C1_GATED_DIRECT_CONSENSUS",
    "C2_ACTION_MASKED_DIRECT_CONSENSUS_FULL",
    "C3_ACTION_MASKED_DIRECT_CONSENSUS_PROTECTED_SAFE",
)


def _control_spec(
    method_id: str,
    prediction: np.ndarray,
    model_count: np.ndarray,
    description: str,
) -> PredictionSpec:
    return PredictionSpec(
        method_id=method_id,
        prediction=np.asarray(prediction, dtype=int),
        probability=None,
        score_kind="hard-label consensus control; no probability defined",
        frozen_model_count=np.asarray(model_count, dtype=int),
        keep_forward_passes=np.asarray(model_count, dtype=int),
        intervention_logits_required=False,
        representation_projection_required=False,
        target_run_specific=True,
        description=description,
    )


def build_consensus_controls(
    frame: pd.DataFrame,
    v2_policies: dict[str, dict[str, np.ndarray]],
) -> dict[str, PredictionSpec]:
    baseline = frame.pred_noop.to_numpy(dtype=int)
    majority = frame.other_run_base_majority.to_numpy(dtype=int)
    disagreement = majority != baseline
    model_count = frame.groupby("manifest_index").manifest_index.transform("size").to_numpy(dtype=int)

    c1 = baseline.copy()
    c1[disagreement] = majority[disagreement]

    full_action = v2_policies["I003_CROSS_RUN_FULL"]["selected"]
    safe_action = v2_policies["I003_CROSS_RUN_PROTECTED_SAFE"]["selected"]
    c2 = baseline.copy()
    c3 = baseline.copy()
    full_available = full_action != "noop"
    safe_available = safe_action != "noop"
    c2[full_available] = majority[full_available]
    c3[safe_available] = majority[safe_available]

    controls = {
        "C1_GATED_DIRECT_CONSENSUS": _control_spec(
            "C1_GATED_DIRECT_CONSENSUS",
            c1,
            model_count,
            "On target/other disagreement, directly emit the leave-target-run majority.",
        ),
        "C2_ACTION_MASKED_DIRECT_CONSENSUS_FULL": _control_spec(
            "C2_ACTION_MASKED_DIRECT_CONSENSUS_FULL",
            c2,
            model_count,
            "Use the exact frozen FULL action-availability mask, but emit the majority label without an intervention.",
        ),
        "C3_ACTION_MASKED_DIRECT_CONSENSUS_PROTECTED_SAFE": _control_spec(
            "C3_ACTION_MASKED_DIRECT_CONSENSUS_PROTECTED_SAFE",
            c3,
            model_count,
            "Use the exact frozen AMPLIFY+GEOMETRY availability mask, but emit the majority label directly.",
        ),
    }
    for spec in controls.values():
        spec.validate(len(frame))
    return controls


def compare_predictions(
    pool: str,
    left_id: str,
    left: np.ndarray,
    right_id: str,
    right: np.ndarray,
    intervened: np.ndarray,
) -> dict[str, Any]:
    left = np.asarray(left, dtype=int)
    right = np.asarray(right, dtype=int)
    intervened = np.asarray(intervened, dtype=bool)
    if not (left.shape == right.shape == intervened.shape):
        raise ValueError("Equivalence arrays have inconsistent shapes")
    disagreement = left != right
    intervened_count = int(intervened.sum())
    return {
        "pool": pool,
        "left_method": left_id,
        "right_method": right_id,
        "trials": int(len(left)),
        "intervened_trials": intervened_count,
        "exact_prediction_agreement": float(np.mean(~disagreement)),
        "number_of_disagreements": int(disagreement.sum()),
        "fraction_final_predicted_class_differs": float(np.mean(disagreement)),
        "fraction_intervened_predicted_class_differs": (
            float(np.mean(disagreement[intervened])) if intervened_count else np.nan
        ),
        "classification_metrics_prediction_equivalent": bool(not disagreement.any()),
        "OUTER_TEST_USED": False,
    }


def mandatory_equivalence_rows(
    pool: str,
    controls: dict[str, PredictionSpec],
    v2_policies: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    return [
        compare_predictions(
            pool,
            "C2_ACTION_MASKED_DIRECT_CONSENSUS_FULL",
            controls["C2_ACTION_MASKED_DIRECT_CONSENSUS_FULL"].prediction,
            "I003_CROSS_RUN_FULL",
            v2_policies["I003_CROSS_RUN_FULL"]["prediction"],
            v2_policies["I003_CROSS_RUN_FULL"]["selected"] != "noop",
        ),
        compare_predictions(
            pool,
            "C3_ACTION_MASKED_DIRECT_CONSENSUS_PROTECTED_SAFE",
            controls["C3_ACTION_MASKED_DIRECT_CONSENSUS_PROTECTED_SAFE"].prediction,
            "I003_CROSS_RUN_PROTECTED_SAFE",
            v2_policies["I003_CROSS_RUN_PROTECTED_SAFE"]["prediction"],
            v2_policies["I003_CROSS_RUN_PROTECTED_SAFE"]["selected"] != "noop",
        ),
    ]

