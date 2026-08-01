from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hsc_tta.certification import apply_certificate, fit_simultaneous_quantile
from hsc_tta.schemas.models import ActionSurfaceRow, ContextFeatureRow, SubjectDecisionRow
from hsc_tta.selection import select_safe_action
from hsc_tta.simulation.core import generate_subject_surface


def write_mock_gpu_interface(output_dir: str | Path, seed: int = 0, n_subjects: int = 120) -> dict[str, Path]:
    """Write validated synthetic rows using the frozen future-GPU schemas."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    surface = generate_subject_surface(n_subjects=n_subjects, seed=seed)
    calibration = surface[surface.split_role == "conformal_calibration"]
    test = surface[surface.split_role == "final_test"].copy()
    quantile = fit_simultaneous_quantile(calibration, delta=0.10)
    test["certified_upper_bound"] = apply_certificate(test.predicted_risk, quantile)
    test["dataset"] = "synthetic"
    test["seed"] = seed
    test["episode_id"] = test.subject_id.map(lambda sid: f"synthetic:{seed}:{sid}")
    test["within_subject_empirical_risk"] = np.clip(test.future_risk - 0.02, 0, 1)
    test["within_subject_margin"] = np.maximum(test.upper_risk - test.within_subject_empirical_risk, 0)
    test["within_subject_upper_risk"] = test.upper_risk
    test["macro_f1"] = np.clip(1 - test.argmax_error, 0, 1)
    test["balanced_accuracy"] = np.clip(1 - test.argmax_error, 0, 1)
    test["n_context"] = 180
    test["n_future"] = 360
    test["n_future_blocks"] = 12
    test["status"] = "evaluated"
    action_columns = [
        "dataset", "seed", "subject_id", "split_role", "episode_id", "action", "lambda",
        "predicted_risk", "within_subject_empirical_risk", "within_subject_margin",
        "within_subject_upper_risk", "certified_upper_bound", "future_risk", "argmax_error",
        "macro_f1", "balanced_accuracy", "average_set_size", "singleton_rate", "n_context",
        "n_future", "n_future_blocks", "status",
    ]
    action_surface = test[action_columns].reset_index(drop=True)

    context_rows = []
    for subject_id, group in test.groupby("subject_id", sort=True):
        proportions = rng.dirichlet(np.ones(5))
        row = {
            "dataset": "synthetic", "seed": seed, "subject_id": subject_id,
            "split_role": "final_test", "backbone": "mock", "episode_id": f"synthetic:{seed}:{subject_id}",
            "n_context": 180, "embedding_mean_0": float(rng.normal()), "embedding_std_0": float(rng.uniform(0.5, 1.5)),
            "entropy_q10": 0.2, "entropy_q50": 0.5, "entropy_q90": 0.9,
            "maxprob_q10": 0.3, "maxprob_q50": 0.6, "maxprob_q90": 0.95,
            "prediction_instability": float(rng.uniform(0, 0.2)), "channel_missing_rate": 0.0,
            "signal_quality_peak_abs": float(rng.uniform(0.5, 2.0)),
            "action_specific_entropy_delta": float(group.predicted_risk.mean() - group.future_risk.mean()),
        }
        row.update({f"predicted_class_proportion_{i}": float(value) for i, value in enumerate(proportions)})
        context_rows.append(row)
    context = pd.DataFrame(context_rows)

    decision_rows = []
    for subject_id, group in test.groupby("subject_id", sort=True):
        no_tta_error = float(group.loc[group.action == "no_tta", "argmax_error"].mean())
        choice = select_safe_action(group, alpha=0.20)
        if choice["status"] == "certified":
            selected = choice["selected_row"]
            selected_action = str(selected["action"])
            selected_lambda = float(selected["lambda"])
            predicted_risk = float(selected["predicted_risk"])
            upper = float(selected["certified_upper_bound"])
            true_risk = float(selected["future_risk"])
            average_set_size = float(selected["average_set_size"])
            singleton_rate = float(selected["singleton_rate"])
            selected_error = float(selected["argmax_error"])
            certified = True
            reason = str(choice["selection_reason"])
        else:
            selected_action = None
            selected_lambda = None
            predicted_risk = 1.0
            upper = 1.0
            true_risk = no_tta_error
            average_set_size = 5.0
            singleton_rate = 0.0
            selected_error = no_tta_error
            certified = False
            reason = "no_action_lambda_with_bound_at_or_below_alpha"
        decision_rows.append({
            "dataset": "synthetic", "seed": seed, "subject_id": subject_id, "alpha": 0.20,
            "selected_action": selected_action, "selected_lambda": selected_lambda,
            "predicted_risk": predicted_risk, "certified_upper_bound": upper,
            "true_future_risk": true_risk, "certified": certified,
            "nontrivial_certified": bool(certified and average_set_size < 5),
            "average_set_size": average_set_size, "singleton_rate": singleton_rate,
            "no_tta_error": no_tta_error, "selected_error": selected_error,
            "harmful_adaptation": bool(selected_action not in (None, "no_tta") and selected_error > no_tta_error),
            "status": choice["status"], "selection_reason": reason,
        })
    decisions = pd.DataFrame(decision_rows)

    for row in context.to_dict("records"):
        ContextFeatureRow.model_validate(row)
    for row in action_surface.to_dict("records"):
        ActionSurfaceRow.model_validate(row)
    for row in decisions.to_dict("records"):
        SubjectDecisionRow.model_validate(row)

    outputs = {
        "subject_context_features": output / "subject_context_features.parquet",
        "subject_action_surface": output / "subject_action_surface.parquet",
        "subject_decisions": output / "subject_decisions.parquet",
    }
    context.to_parquet(outputs["subject_context_features"], index=False)
    action_surface.to_parquet(outputs["subject_action_surface"], index=False)
    decisions.to_parquet(outputs["subject_decisions"], index=False)
    return outputs
