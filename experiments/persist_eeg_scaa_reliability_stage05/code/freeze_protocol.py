from __future__ import annotations

import datetime as dt

import common as c


FROZEN_CODE = (
    "code/common.py",
    "code/extract_features.py",
    "code/analyze.py",
    "code/plot_publication.py",
    "code/validate.py",
)

STAGE0_PROVENANCE = (
    "protocol/DATA_ACCESS_LOCK.json",
    "protocol/ADAPTATION_RECIPE_SELECTION.json",
    "protocol/SCAA_STAGE0_PROTOCOL_LOCK.json",
    "results/PER_SUBJECT_SEED_UTILITY.csv",
    "results/PER_SUBJECT_UTILITY.csv",
    "results/VALIDATION.json",
    "SCAA_STAGE0_FINAL_REPORT.json",
)


def main() -> None:
    c.ensure_dirs()
    stage0_lock = c.read_json(c.STAGE0_PROTOCOL / "SCAA_STAGE0_PROTOCOL_LOCK.json")
    stage0_validation = c.read_json(c.STAGE0_RESULTS / "VALIDATION.json")
    stage0_report = c.read_json(c.STAGE0_ROOT / "SCAA_STAGE0_FINAL_REPORT.json")
    if not stage0_validation.get("pass"):
        raise RuntimeError("Stage-0 validator did not pass")
    if stage0_report.get("terminal") != "TARGET_HISTORY_UTILITY_TRANSFER_PARTIAL":
        raise RuntimeError("the frozen Stage-0 terminal changed")
    if stage0_report.get("authorization") != "SCAA_DEVELOPMENT_NOT_AUTHORIZED":
        raise RuntimeError("the frozen Stage-0 authorization changed")
    if not all((c.EXP / relative).is_file() for relative in FROZEN_CODE):
        raise RuntimeError("all frozen implementation files must exist before locking")

    subjects = list(map(str, stage0_lock["development_subjects"]))
    folds = stage0_lock["folds"]
    if len(subjects) != 41 or set(c.target_fold_map()) != set(subjects):
        raise RuntimeError("development subject/fold audit failed")

    data_lock = {
        "schema": "PERSIST_EEG_SCAA_RELIABILITY_STAGE05_DATA_ACCESS_LOCK_V1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": "WBCIC/NEMAR nm000348",
        "development_subject_count": 41,
        "development_subjects": subjects,
        "folds": folds,
        "session_roles": {
            "S1_ses_0": "adaptation and S1 chronological validation",
            "S2_ses_1": "historical certificate and all predictor features",
            "S3_ses_2": "outcome definition only after feature lock",
        },
        "feature_extraction_signal_sessions_permitted": [0, 1],
        "feature_extraction_signal_sessions_forbidden": [2],
        "outcome_merge_source": "committed Stage-0 PER_SUBJECT_UTILITY.csv only",
        "outer_10": {
            "status": "UNTOUCHED_UNENUMERATED_UNPREPROCESSED_UNEVALUATED",
            "identifiers_present": False,
        },
        "OpenBMI": "NOT_ACCESSED",
        "raw_cache_commit_policy": "never commit raw EEG, cache, or checkpoints",
    }
    c.write_json(c.PROTOCOL / "DATA_ACCESS_LOCK.json", data_lock)

    model_definitions = {
        "M0": {"status": "available", "features": [], "description": "training-fold prevalence"},
        "M1": {"status": "available", "features": ["backbone"], "description": "backbone only"},
        "M2": {"status": "available", "features": ["raw_delta2"], "description": "raw S2 utility"},
        "M3": {"status": "available", "features": ["certificate_snr"], "description": "S2 certificate precision"},
        "M4": {
            "status": "unavailable",
            "features": [],
            "description": "No legally available frozen target-subject Identity I. Existing identity_probe.csv is a model-fit-domain aggregate, not a target-level score; it is not remapped, retrained, or fabricated.",
        },
        "M5": {"status": "available", "features": ["representation_stability"], "description": "final-embedding temporal stability"},
        "M6": {"status": "available", "features": ["decision_stability"], "description": "anchor decision-margin temporal stability"},
        "M7": {"status": "available", "features": ["adaptation_effect_stability"], "description": "adaptation-effect temporal stability"},
        "M8": {
            "status": "available",
            "features": [
                "adaptation_effect_stability",
                "decision_stability",
                "certificate_snr",
                "representation_stability",
            ],
            "description": "fixed four-scalar combined mechanism model",
        },
    }

    feature_lock = {
        "schema": "PERSIST_EEG_SCAA_RELIABILITY_FEATURE_PROTOCOL_LOCK_V1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "created_from_commit": c.git_head(),
        "stage0_validated_tip": "46b8ecf2c39b0e32045cad9d78ca12327f0a3f0d",
        "stage0_terminal_preserved": "TARGET_HISTORY_UTILITY_TRANSFER_PARTIAL",
        "stage0_authorization_preserved": "SCAA_DEVELOPMENT_NOT_AUTHORIZED",
        "development_subjects": subjects,
        "folds": folds,
        "backbones": list(c.BACKBONES),
        "seeds": list(c.SEEDS),
        "stage0_provenance_hashes": {
            relative: c.sha256(c.STAGE0_ROOT / relative) for relative in STAGE0_PROVENANCE
        },
        "data_access_lock_sha256": c.sha256(c.PROTOCOL / "DATA_ACCESS_LOCK.json"),
        "frozen_code_hashes": {relative: c.sha256(c.EXP / relative) for relative in FROZEN_CODE},
        "session_roles": data_lock["session_roles"],
        "adapter": {
            "name": "classifier_head_only_supervised",
            "encoder_frozen": True,
            "normalization_frozen": True,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "maximum_epochs": 50,
            "minimum_epochs": 10,
            "patience": 8,
            "S1_split": "within-class chronological first 70% train, final 30% validation",
            "target_anchor": "target-specific outcome-fold ERM anchor that never saw the target",
        },
        "feature_priority": [
            "adaptation_effect_stability",
            "decision_stability",
            "certificate_precision",
            "representation_stability",
            "Identity_and_simple_controls",
        ],
        "feature_definitions": {
            "binary_correct_class_margin": "z[y] - z[1-y] (equivalently twice the correct-class centered logit)",
            "decision_stability": "For anchor correct-class margins, within each class compute abs(mean(S1val)-mean(S2))/sqrt((var(S1val)+var(S2))/2); score is negative mean across classes, so larger is more stable.",
            "representation_layer": "frozen anchor final embedding returned by forward_features; no alternate layer",
            "representation_stability": "Within each class compute Euclidean S1val-to-S2 final-embedding centroid distance divided by sqrt(mean of the two within-session mean squared radii); score is negative mean across classes.",
            "adaptation_effect": "adapted correct-class margin minus anchor correct-class margin for the same trial",
            "adaptation_effect_stability": "For the adaptation-effect scalar, within each class compute abs(mean(S1val)-mean(S2))/sqrt((var(S1val)+var(S2))/2); score is negative mean across classes.",
            "certificate_precision": "Within each seed, paired class-stratified S2 bootstrap (2000 resamples) of Delta2 BA; SE=bootstrap SD, SNR=Delta2/SE, LCB90=Delta2-1.2815515655*SE.",
            "controls": "raw Delta2, S1 head parameter-relative change, and S1-validation anchor mean maximum softmax confidence",
            "identity": model_definitions["M4"]["description"],
        },
        "trial_and_seed_aggregation": {
            "S1_validation": "frozen within-class chronological final 30 percent",
            "S2": "all S2 trials",
            "class_aggregation": "unweighted arithmetic mean of class-0 and class-1 scalars",
            "seed_aggregation": "compute every scalar per seed, then arithmetic mean across the three matched seeds within subject/backbone",
            "statistical_unit": "subject; paired backbone rows retained together in all folds and bootstrap resamples",
        },
        "outcomes": {
            "R_sign": "1[np.sign(Delta2)==np.sign(Delta3)]; exact zero is concordant only with exact zero",
            "R_safe": "1[Delta2>0 and Delta3>=0]",
            "H": "1[Delta2>0 and Delta3<0]",
            "safe_given_positive_certificate": "1[Delta3>=0] evaluated only where Delta2>0",
            "signed_persistence": "Delta2*Delta3",
        },
        "cross_validation": {
            "outer": "the existing five subject-disjoint outcome folds; both backbone rows of each held-out subject remain together",
            "preprocessing": "continuous predictors standardized using training fold mean and SD only; backbone is a binary indicator",
            "classifier": "sklearn LogisticRegression, L2, C=1, solver=liblinear, max_iter=2000, random_state=0",
            "constant_fallback": "training-fold prevalence if no features or a training fold contains one class",
            "models": model_definitions,
            "metrics": ["AUROC", "balanced_accuracy_at_0.5", "Brier"],
            "uncertainty": "10000 subject bootstrap resamples; both backbone rows resampled as one cluster",
        },
        "backbone_decomposition": {
            "models": ["backbone_only", "backbone_plus_mechanism", "backbone_plus_mechanism_plus_interaction"],
            "predictor_scaling": "mechanism score standardized over the 82 development rows for descriptive full-development decomposition",
            "coefficient_uncertainty": "subject-cluster bootstrap",
        },
        "policy": {
            "model": "M8 fixed four-scalar logistic model fit only among Delta2-positive training rows to predict safe_given_positive_certificate",
            "execution_eligibility": "At least one of M3/M5/M6/M7/M8 must have OOF AUROC>=0.55 for R_sign or safe_given_positive_certificate and improve over the better of M1/M2 by >=0.02 on the same outcome.",
            "base_requirement": "Delta2>0",
            "threshold_selection": "For each outer fold, create inner OOF training-subject predictions using the remaining frozen folds; among thresholds giving overall training-row coverage >=0.20, minimize training harmful-accepted rate, then prefer greater coverage/lower threshold. Never use held-out outcomes.",
            "held_out_application": "fit M8 on all eligible outer-training positive certificates and apply the training-selected threshold to held-out subjects",
            "comparators": ["Anchor", "Always Adapt", "Simple S2 Gate", "Reliability-Gated S2"],
        },
        "success_gates": {
            "A": "All final predictors use only S1/S2; mandatory.",
            "B": "A predeclared mechanism model has OOF AUROC>=0.60, subject-bootstrap lower 95% bound>0.50, and AUROC improvement>=0.03 over both M1 and M2 on R_sign or safe_given_positive_certificate.",
            "C": "For a Gate-B mechanism, adding it reduces absolute backbone coefficient by >=20% and improves logistic deviance by >=2 versus backbone-only; the interaction magnitude must not exceed twice the main mechanism magnitude.",
            "D": "Reliability gate reduces harmful accepted certificates by >=25% relative to the simple Delta2-positive gate.",
            "E": "Reliability-gated adaptation coverage >=0.20.",
            "F": "Reliability-gated mean S3 BA is no more than 0.002 below Anchor mean S3 BA.",
        },
        "terminal_rule": {
            "supported": "all Gates A-F pass",
            "partial": "Gate A passes and there is directionally positive predeclared mechanism evidence (OOF AUROC>=0.55 for R_sign or safe certificates) but all Gates A-F do not pass",
            "not_supported": "no predeclared mechanism reaches directional OOF AUROC>=0.55, or any prospective-purity gate fails",
        },
        "outer_10": data_lock["outer_10"],
        "OpenBMI": "NOT_ACCESSED",
        "feature_definition_changes_after_S3_association": False,
    }
    c.write_json(c.PROTOCOL / "RELIABILITY_FEATURE_PROTOCOL_LOCK.json", feature_lock)
    print("SCAA_RELIABILITY_STAGE05_PROTOCOL_FROZEN")


if __name__ == "__main__":
    main()

