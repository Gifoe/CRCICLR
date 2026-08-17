"""Write the pre-outcome roster, task-search, and multiplicity locks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import (
    BACKBONES,
    BLOCKS,
    IMPLEMENTATION_ID,
    PRIMARY_SEED,
    PROTOCOL,
    RANDOM_DRAWS,
    REPLICATION_SEEDS,
    REPO_ROOT,
    git_commit,
    sha256_file,
    write_once,
)
from models import build_model, count_parameters


FROZEN_AT = "2026-08-17T19:45:34.8781405Z"
REFERENCE_COMMIT = "61e4157817bc9c04f50471fb9dd6b865d74e21e4"
PRE_OUTCOME_IMPLEMENTATION_COMMIT = "7baec3e17104703be65a1b6eb7e4ccd71ac3420e"

CONFIGS: dict[str, list[dict[str, Any]]] = {
    "FBCNet": [
        {
            "id": "FBC_S8_LR1E3",
            "model": {"spatial_filters": 8, "windows": 4, "dropout": 0.25},
            "batch_size": 128,
            "epochs": 30,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "label_smoothing": 0.0,
            "gradient_clip": 1.0,
        },
        {
            "id": "FBC_S8_STABLE",
            "model": {"spatial_filters": 8, "windows": 4, "dropout": 0.50},
            "batch_size": 128,
            "epochs": 40,
            "learning_rate": 3e-4,
            "weight_decay": 5e-4,
            "label_smoothing": 0.0,
            "gradient_clip": 1.0,
        },
        {
            "id": "FBC_S16_STABLE",
            "model": {"spatial_filters": 16, "windows": 4, "dropout": 0.35},
            "batch_size": 96,
            "epochs": 35,
            "learning_rate": 3e-4,
            "weight_decay": 5e-4,
            "label_smoothing": 0.0,
            "gradient_clip": 1.0,
        },
    ],
    "EEGConformer": [
        {
            "id": "CONF_D40_BASE",
            "model": {"d_model": 40, "depth": 3, "heads": 4, "dropout": 0.25, "temporal_kernel": 25},
            "batch_size": 64,
            "epochs": 35,
            "learning_rate": 3e-4,
            "weight_decay": 5e-4,
            "label_smoothing": 0.0,
            "gradient_clip": 1.0,
        },
        {
            "id": "CONF_D64_BASE",
            "model": {"d_model": 64, "depth": 3, "heads": 4, "dropout": 0.35, "temporal_kernel": 25},
            "batch_size": 64,
            "epochs": 35,
            "learning_rate": 3e-4,
            "weight_decay": 5e-4,
            "label_smoothing": 0.0,
            "gradient_clip": 1.0,
        },
        {
            "id": "CONF_D64_CONSERVATIVE",
            "model": {"d_model": 64, "depth": 3, "heads": 4, "dropout": 0.25, "temporal_kernel": 25},
            "batch_size": 64,
            "epochs": 40,
            "learning_rate": 1e-4,
            "weight_decay": 1e-3,
            "label_smoothing": 0.02,
            "gradient_clip": 1.0,
        },
    ],
    "DeepConvNet": [
        {
            "id": "DEEP_F25_LR1E3",
            "model": {"base_filters": 25, "dropout": 0.50},
            "batch_size": 64,
            "epochs": 30,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "label_smoothing": 0.0,
            "gradient_clip": 1.0,
        },
        {
            "id": "DEEP_F25_STABLE",
            "model": {"base_filters": 25, "dropout": 0.50},
            "batch_size": 64,
            "epochs": 40,
            "learning_rate": 3e-4,
            "weight_decay": 5e-4,
            "label_smoothing": 0.0,
            "gradient_clip": 1.0,
        },
        {
            "id": "DEEP_F32_STABLE",
            "model": {"base_filters": 32, "dropout": 0.40},
            "batch_size": 64,
            "epochs": 35,
            "learning_rate": 3e-4,
            "weight_decay": 5e-4,
            "label_smoothing": 0.02,
            "gradient_clip": 1.0,
        },
    ],
    "TeCh": [
        {
            "id": "TECH_D64_BASE",
            "model": {"d_model": 64, "channel_depth": 2, "temporal_depth": 2, "patch_len": 20, "dropout": 0.20, "train_jitter": 0.0},
            "batch_size": 64,
            "epochs": 35,
            "learning_rate": 3e-4,
            "weight_decay": 5e-4,
            "label_smoothing": 0.0,
            "gradient_clip": 1.0,
        },
        {
            "id": "TECH_D64_JITTER",
            "model": {"d_model": 64, "channel_depth": 2, "temporal_depth": 2, "patch_len": 20, "dropout": 0.30, "train_jitter": 0.01},
            "batch_size": 64,
            "epochs": 40,
            "learning_rate": 1e-4,
            "weight_decay": 1e-3,
            "label_smoothing": 0.02,
            "gradient_clip": 1.0,
        },
        {
            "id": "TECH_D128_BASE",
            "model": {"d_model": 128, "channel_depth": 1, "temporal_depth": 2, "patch_len": 20, "dropout": 0.30, "train_jitter": 0.0},
            "batch_size": 48,
            "epochs": 35,
            "learning_rate": 1e-4,
            "weight_decay": 5e-4,
            "label_smoothing": 0.0,
            "gradient_clip": 1.0,
        },
    ],
}


PROVENANCE = {
    "EEGNet": {
        "status": "frozen_read_only_reference",
        "source": "CRCICLR persist_eeg_wbcic_actionability_v2",
        "commit": REFERENCE_COMMIT,
        "license": "repository license",
    },
    "FBCNet": {
        "source": "https://github.com/ravikiran-mane/FBCNet",
        "upstream_commit_audited": "de1bbdd8a54cb1e466830e3d47070e0e56761a37",
        "license": "MIT",
        "implementation": "adapted reimplementation; deterministic internal 4-Hz FFT filter bank preserves frozen cache",
    },
    "EEGConformer": {
        "source": "https://github.com/eeyhsong/EEG-Conformer and public paper",
        "upstream_commit_audited": "9ae149ba62487ceae723277d13adac27837113d2",
        "license": "GPL-3.0 with separate commercial-use notice",
        "implementation": "independent compact paper-level reimplementation; no upstream source vendored",
    },
    "DeepConvNet": {
        "source": "Schirrmeister et al. 2017; Braindecode Deep4Net/Deep4Net successor",
        "upstream_commit_audited": "eb3ddee6b72b07ff7457f9273cff179400b925c5",
        "license": "BSD-3-Clause (Braindecode)",
        "implementation": "independent architecture-family reimplementation",
    },
    "TeCh": {
        "source": "https://github.com/Levi-Ackman/TeCh and ICLR 2026 paper",
        "upstream_commit_audited": "9a378cc546a5d97c871eff282148175b3c7cd75b",
        "license": "NO REPOSITORY-WIDE LICENSE FOUND",
        "implementation": "clean-room faithful reimplementation from public architecture; no upstream source vendored",
    },
}


def main() -> None:
    if set(CONFIGS) != set(BACKBONES) or any(len(values) > 6 for values in CONFIGS.values()):
        raise RuntimeError("Backbone roster or task-only budget violation")
    architecture_inventory = {}
    for backbone, configs in CONFIGS.items():
        inventory = []
        for config in configs:
            model = build_model(backbone, config)
            if int(model.representation_dim) < 32 or not hasattr(model, "head"):
                raise RuntimeError(f"Invalid representation contract: {backbone}/{config['id']}")
            inventory.append(
                {
                    "config": config["id"],
                    "parameter_count": count_parameters(model),
                    "native_representation_dim": int(model.representation_dim),
                    "classifier_head": "single nn.Linear(native_representation_dim, 2)",
                }
            )
        architecture_inventory[backbone] = inventory

    roster = {
        "status": "BACKBONE_ROSTER_FROZEN_PRE_OUTCOME",
        "implementation_id": IMPLEMENTATION_ID,
        "frozen_at_utc": FROZEN_AT,
        "git_sha_at_lock": REFERENCE_COMMIT,
        "pre_outcome_implementation_commit": PRE_OUTCOME_IMPLEMENTATION_COMMIT,
        "frozen_reference_commit": REFERENCE_COMMIT,
        "exact_roster": ["EEGNet", *BACKBONES],
        "new_prospective_backbones": list(BACKBONES),
        "no_sixth_backbone": True,
        "provenance": PROVENANCE,
        "architecture_inventory": architecture_inventory,
        "task_only_config_count": {name: len(values) for name, values in CONFIGS.items()},
        "primary_seed": PRIMARY_SEED,
        "conditional_replication_seeds": list(REPLICATION_SEEDS),
        "competence_gate": {
            "mean_subject_BA_min": 0.60,
            "bootstrap_LCB95_strictly_greater_than": 0.55,
            "fraction_subject_BA_gt_0p5_min": 0.70,
        },
        "representation_rule": "native penultimate vector immediately before frozen final linear head; dimension >=32",
        "replacement_policy": "no replacement after any new H1-H5; hard pre-outcome implementation block must be recorded",
    }
    write_once(PROTOCOL / "BACKBONE_ROSTER_LOCK.json", roster)

    for backbone, configs in CONFIGS.items():
        write_once(
            PROTOCOL / f"BACKBONE_{backbone.upper()}_TASK_SEARCH_LOCK.json",
            {
                "status": "TASK_ONLY_SEARCH_FROZEN_PRE_OUTCOME",
                "backbone": backbone,
                "frozen_at_utc": FROZEN_AT,
                "configs": configs,
                "finite_budget": len(configs),
                "maximum_budget": 6,
                "folds": 5,
                "train": "all non-outcome development subjects S1+S2",
                "validation": "development unseen-subject S3",
                "primary_selection_metric": "mean subject balanced accuracy",
                "tie_break": ["lower mean subject NLL", "lexicographic config id"],
                "secondary_reporting": ["median subject BA", "macro-F1", "worst-20%-subject BA", "NLL"],
                "PERSIST_metrics_forbidden_during_selection": True,
                "outer_test_state": "OUTER_TEST_LOCKED",
            },
        )

    protocol = {
        "status": "MULTIBACKBONE_PROTOCOL_FROZEN_PRE_OUTCOME",
        "frozen_at_utc": FROZEN_AT,
        "reference_commit": REFERENCE_COMMIT,
        "scope": "same 41 WBCIC development subjects; ten outer subjects sealed",
        "cache": "read-only persist_eeg_wbcic_actionability_v2 development epoch cache",
        "preprocessing": "immutable frozen 58x1000 Pz-subtracted 0.5-40Hz cache",
        "session_protocol": "S1+S2 -> S3",
        "cross_fitting": "five folds: outcome F_k; discovery F_(k+1); model-fit remaining three",
        "blocks": [{"name": name, "start": start, "end": end} for name, start, end in BLOCKS],
        "persistence_basis": "fold-specific symmetrized S1/S2 cross-session subject-centroid covariance",
        "random_controls": RANDOM_DRAWS,
        "bootstrap_draws": 10_000,
        "statistical_unit": "subject",
        "H1": "LCB95 persistence-specific advantage >0 and one-sided p<0.05",
        "H2": "UCB95 u_spec<0 and one-sided harmful p<0.05",
        "H3": "exact binary Haar local LCB>1 and p<0.05; finite LCB>1 and p<0.05",
        "H4": "LCB95 delta_BA_specific>0, mean>=0.005, one-sided p<0.05",
        "H5": "all LOFO means>0; all LOSO means>0; >=60% subjects nonnegative",
        "protected": "H1+H3 and LCB95(u_spec)>0 with one-sided protected p<0.05",
        "outer_materialization_forbidden": True,
    }
    write_once(PROTOCOL / "MULTIBACKBONE_PROTOCOL_LOCK.json", protocol)

    multiplicity = {
        "status": "MULTIBACKBONE_MULTIPLICITY_FROZEN_PRE_OUTCOME",
        "frozen_at_utc": FROZEN_AT,
        "maximum_family": 16,
        "family_definition": "all four pre-registered blocks for every competent new backbone; incompetent backbone/block slots remain in the 16-slot family with p_joint=1",
        "candidate_p_joint": "max(p_H1,p_H2_harmful,p_H3_local_exact,p_H3_finite,p_H4)",
        "global_procedure": "Holm FWER correction across exactly 16 p_joint values",
        "alpha": 0.05,
        "component_effect_and_CI_gates_also_required": True,
        "local_test": "exact per-fold Haar Beta(rank/2,(d-rank)/2) null, convolved across five folds with frozen 2^20 scrambled-Sobol integration; avoids 100-draw resolution failure",
        "within_backbone_component_holm": "reported as diagnostic continuity only; global 16-candidate Holm is confirmatory",
        "positive_target_requires": ["H1", "H2", "H3", "H4", "H5", "global_Holm"],
        "conditional_replication_seeds": list(REPLICATION_SEEDS),
        "AGDI_authorization_requires_seed_replication": True,
    }
    write_once(PROTOCOL / "MULTIBACKBONE_MULTIPLICITY_LOCK.json", multiplicity)
    print(json.dumps({"status": "MULTIBACKBONE_LOCKS_FROZEN", "protocol": str(PROTOCOL)}, indent=2))


if __name__ == "__main__":
    main()
