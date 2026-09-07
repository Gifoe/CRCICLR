from __future__ import annotations

import subprocess

import common as c


CODE_FILES = (
    "common.py",
    "audit_data.py",
    "run_competence.py",
    "freeze_protocol.py",
    "run_utility.py",
    "analyze.py",
    "validate.py",
)


def main() -> None:
    data_lock_path = c.PROTOCOL / "DATA_ACCESS_LOCK.json"
    recipe_path = c.PROTOCOL / "ADAPTATION_RECIPE_SELECTION.json"
    data_lock = c.read_json(data_lock_path)
    recipe = c.read_json(recipe_path)
    if data_lock.get("pass") is not True:
        raise RuntimeError("DATA_ACCESS_LOCK did not pass")
    if data_lock.get("S2_or_S3_adaptation_utility_inspected") is not False:
        raise RuntimeError("outcome utility was already inspected")
    if recipe.get("competence_gate_pass") is not True:
        raise RuntimeError("head-only competence did not pass")
    if recipe.get("S2_or_S3_utility_accessed") is not False:
        raise RuntimeError("adapter recipe was not selected outcome-cleanly")
    if recipe.get("adapter") != "classifier_head_only_supervised":
        raise RuntimeError("unexpected adapter recipe")

    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", str(c.EXP.relative_to(c.REPO))],
        cwd=c.REPO,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"experiment tree must be committed before protocol freeze:\n{dirty}")

    code_hashes = {}
    for name in CODE_FILES:
        path = c.HERE / name
        if not path.is_file():
            raise FileNotFoundError(path)
        code_hashes[str(path.relative_to(c.EXP)).replace("\\", "/")] = c.sha256(path)

    folds = []
    for fold in c.FOLDS:
        role = c.roles(fold)
        folds.append({"fold": fold, **{key: list(value) for key, value in role.items()}})

    lock = {
        "schema": "PERSIST_EEG_SCAA_STAGE0_PROTOCOL_LOCK_V1",
        "created_at_code_commit": c.git_head(),
        "code_hashes": code_hashes,
        "data_access_lock_sha256": c.sha256(data_lock_path),
        "adaptation_recipe_selection_sha256": c.sha256(recipe_path),
        "dataset": "WBCIC / NEMAR nm000348",
        "development_subjects": data_lock["development_subjects"],
        "development_subject_count": 41,
        "sealed_outer": {
            "count": 10,
            "identifiers_present": False,
            "enumerated": False,
            "accessed": False,
            "preprocessed": False,
            "evaluated": False,
        },
        "sessions": {"S1_adaptation": 0, "S2_certificate": 1, "S3_future": 2},
        "folds": folds,
        "target_to_outcome_fold": data_lock["target_to_outcome_fold"],
        "backbones": list(c.BACKBONES),
        "seeds": list(c.SEEDS),
        "anchor_rule": "for each target, use its outcome-fold ERM checkpoint; target absent from model-fit",
        "anchor_checkpoint_hashes": [
            {
                "backbone": item["backbone"],
                "fold": item["fold"],
                "seed": item["seed"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "normalizer_sha256": item["normalizer_sha256"],
            }
            for item in data_lock["anchors"]
        ],
        "adapter": {
            "name": "classifier_head_only_supervised",
            "encoder_frozen": True,
            "normalization_state_frozen": True,
            "target_data_used_for_training": "S1_train only",
            "S1_split": "within-class chronological first 70% train / final 30% validation",
            "optimizer": "AdamW",
            "learning_rate": float(recipe["selected_lr"]),
            "weight_decay": float(recipe["weight_decay"]),
            "maximum_epochs": int(recipe["maximum_epochs"]),
            "minimum_epochs": int(recipe["minimum_epochs"]),
            "patience": int(recipe["patience"]),
            "checkpoint_rule": recipe["checkpoint_rule"],
            "global_recipe_across_subjects_backbones_seeds": True,
            "selection_completed_before_S2_or_S3": True,
            "last_block_repair_used": False,
        },
        "primary_metric": "balanced_accuracy",
        "secondary_metric": "macro_F1",
        "utility": {
            "Delta_S2": "BA(M1 frozen after S1, S2) - BA(M0, S2)",
            "Delta_S3": "BA(M1 frozen after S1, S3) - BA(M0, S3)",
            "reported_scale": "fraction with percentage-point rendering in reports",
        },
        "aggregation": {
            "seed": "mean within subject and backbone across three matched seeds",
            "pooled_backbone": "mean EEGNet and EEGConformer utilities within each subject",
            "statistical_unit": "subject",
            "subject_count": 41,
            "seeds_or_backbone_subject_pairs_are_independent_samples": False,
        },
        "primary_certificate": "Delta_S2 > 0",
        "zero_sign_rule": "numpy.sign; zero is concordant only with zero",
        "statistics": {
            "subject_bootstrap_resamples": 10000,
            "bootstrap_seed": c.stable_seed("SCAA-Stage0-primary-bootstrap-v1"),
            "correlations": ["Pearson", "Spearman"],
            "correlation_CI": "percentile 95% subject bootstrap",
            "sign_test": "two-sided exact binomial against 0.5 with exact 95% CI",
            "harm_bootstrap": "percentile 95% subject bootstrap",
            "policy_bootstrap": "percentile 95% subject bootstrap",
            "secondary_LCB": {
                "status": "descriptive_only_not_a_rescue_gate",
                "confidence": "90% one-sided",
                "paired_class_stratified_trial_bootstrap_resamples": 2000,
                "seed": c.stable_seed("SCAA-Stage0-secondary-LCB-v1"),
            },
        },
        "strong_support_gates": {
            "A_utility_transfer": "pooled Spearman > 0, 95% bootstrap CI lower > 0, and both backbone estimates > 0",
            "B_sign_persistence": "pooled concordance >= 0.65, exact two-sided p < 0.05, and exact CI lower > 0.5",
            "C_reduced_future_harm": "certified harm < always harm, bootstrap CI lower for always-minus-certified > 0, and relative harm reduction >= 0.30",
            "D_nontrivial_coverage": "pooled coverage >= 0.25",
            "E_policy_usefulness": "pooled gated mean BA >= max(anchor, always-adapt) - 0.005",
            "authorization": "all five gates pass",
        },
        "partial_support_rule": "not all strong gates pass, but pooled Spearman > 0, sign concordance > 0.5, certified harm < always harm, coverage >= 0.25, and no backbone Spearman is negative",
        "terminal_states": [
            "TARGET_HISTORY_UTILITY_TRANSFER_SUPPORTED",
            "TARGET_HISTORY_UTILITY_TRANSFER_PARTIAL",
            "TARGET_HISTORY_UTILITY_TRANSFER_NOT_SUPPORTED",
        ],
        "forbidden_post_outcome_changes": [
            "adapter", "trainable layers", "learning rate", "epochs", "S1 split",
            "checkpoint selection", "subjects", "folds", "backbones", "primary metric",
            "certificate threshold",
        ],
        "exact_commands": [
            r"D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe experiments\persist_eeg_scaa_stage0\code\run_utility.py",
            r"D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe experiments\persist_eeg_scaa_stage0\code\analyze.py",
            r"D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe experiments\persist_eeg_scaa_stage0\code\validate.py",
        ],
        "outcome_access_authorized_only_after_this_lock_is_committed": True,
    }
    c.write_json(c.PROTOCOL / "SCAA_STAGE0_PROTOCOL_LOCK.json", lock)
    print("SCAA_STAGE0_PROTOCOL_FROZEN_PRE_OUTCOME", flush=True)


if __name__ == "__main__":
    main()
