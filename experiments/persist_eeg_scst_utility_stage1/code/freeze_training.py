"""Create the immutable Stage-1 utility lock from source-only audit results."""
from __future__ import annotations

import json

import pandas as pd

import stage1_common as c


def main() -> None:
    if (c.RUNTIME / "utility_metrics").exists() or list(c.RESULTS.glob("SCST_PER_SUBJECT_*.csv")):
        raise RuntimeError("future utility artifacts already exist; lock cannot be back-filled")
    audit_path = c.RESULTS / "STAGE1_ADMISSIBILITY.csv"
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    frame = pd.read_csv(audit_path)
    eligible = []
    for model, group in frame.groupby("model"):
        datasets = set(group.dataset.astype(str))
        if datasets == set(c.DATASETS) and group.all_stage1_gates.fillna(False).astype(bool).all():
            eligible.append(str(model))
    if "ATCNet-CleanRoom" not in eligible:
        raise RuntimeError("central ATCNet-CleanRoom model did not revalidate; prospective utility forbidden")
    source_files = [audit_path, c.RESULTS / "MODEL_COMPETENCE.csv", c.RESULTS / "STAGE1_ADMISSIBILITY_PER_FOLD.csv"]
    lock = {
        "schema": "SCST_STAGE1_TRAINING_LOCK_V1",
        "git_sha_before_lock": c.git_head(),
        "eligible_models": sorted(eligible),
        "input_preprocessing": "per-trial per-channel temporal mean removal and standard-deviation scaling",
        "representations": "frozen anchor final pre-classifier feature; official ATCNet concatenates all five window features",
        "competence_thresholds": c.THRESHOLDS,
        "stage1_gates": {
            "residual_stability": "mean > 0 and subject-bootstrap CI95 lower > 0",
            "subject_fidelity": "mean > 0 and subject-bootstrap CI95 lower > 0",
            "matched_random_advantage": "mean > 0 and subject-bootstrap CI95 lower > 0",
            "class_accuracy_loss_max": 0.02,
            "true_class_log_probability_change_min": -0.05,
            "independent_probe_BA_min": 0.55,
            "off_manifold_excess_vs_random_max": 0.02,
            "independent_session_3NN_ratio_max": 1.30
        },
        "historical_strict_sensitivity": {"label": "HISTORICAL_STRICT_GATE", "independent_session_3NN_ratio_max": 1.25},
        "manifold_sensitivity_thresholds": list(c.MANIFOLD_THRESHOLDS),
        "SCST_operator": "class-conditional subject residual difference r_target,y - r_source,y",
        "alpha_rule": "largest alpha in {0,1/64,...,16/64} within source-session-0 95th-percentile 3NN support radius",
        "bank_construction": "Option A: fixed anchor geometry; combined model-fit and source-validation subjects; session-0 class-subject centroids",
        "trainable_scope": "encoder and classifier trainable; frozen transport vectors",
        "loss": "clean CE + 0.5 transported CE + 0.1 symmetric KL for Full-SCST",
        "hyperparameters": {"lambda_T": 0.5, "lambda_C": 0.1, "epochs": 15, "batch_size": 192, "optimizer": "AdamW", "learning_rate": 0.0001, "weight_decay": 0.001},
        "controls": ["ERM", "Mixup", "RandomTransport", "SCST-NoConsistency", "Full-SCST"],
        "seeds": list(c.SEEDS), "folds": list(c.FOLDS),
        "primary_metric": "future-session subject-balanced Balanced Accuracy",
        "secondary_metric": "subject macro-F1",
        "statistics": "10,000 subject bootstrap draws; seeds repeated within subject; fold sign count",
        "success_criteria": {
            "mean_Full_SCST_minus_ERM_gt_zero": True,
            "subject_bootstrap_CI95_lower_gt_zero": True,
            "positive_folds_min": 3,
            "Full_SCST_beats_RandomTransport": True,
            "class_fidelity_catastrophe_forbidden": True,
            "extra_iterations_explanation_forbidden": "fixed identical 15 epochs and optimizer steps"
        },
        "future_utility_accessed_before_lock": False,
        "outer_resources_forbidden": True,
        "source_artifact_hashes": {str(path.relative_to(c.EXP)): c.sha256(path) for path in source_files},
    }
    c.write_json(c.PROTOCOL / "SCST_STAGE1_TRAINING_LOCK.json", lock)
    print(json.dumps(c.clean(lock), indent=2), flush=True)


if __name__ == "__main__":
    main()
