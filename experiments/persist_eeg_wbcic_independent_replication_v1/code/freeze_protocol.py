"""Freeze the Phase-3 WBCIC replication protocol before any outcome evaluation."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
SCOPE = EXP / "provenance" / "DEVELOPMENT_SCOPE_LOCK.json"
TARGET = EXP / "WBCIC_REPLICATION_PROTOCOL_FROZEN.json"
START_SHA = "3654486141c91333e0507e95be98f4bdc41c0254"
SUBJECT_HASH = "dae8e7ec00cbcf6dcc8c5b25829f2148fd0b5fdf162f75a0cddc18b096af7db4"


def sha_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    scope = json.loads(SCOPE.read_text(encoding="utf-8"))
    allowed_bids = list(map(str, scope["allowed_subjects"]))
    if len(allowed_bids) != 41 or sha_lines(allowed_bids) != SUBJECT_HASH:
        raise RuntimeError("authoritative 41-subject scope/hash mismatch")
    subjects = [value.removeprefix("sub-") for value in allowed_bids]
    frozen_folds = [[value.removeprefix("sub-") for value in scope["folds"][f"F{k}"]] for k in range(5)]
    if set().union(*map(set, frozen_folds)) != set(subjects) or sum(map(len, frozen_folds)) != 41:
        raise RuntimeError("frozen WBCIC folds are not a disjoint exhaustive partition")
    folds = []
    for fold in range(5):
        outcome = frozen_folds[fold]
        validation = frozen_folds[(fold + 1) % 5]
        excluded = set(outcome) | set(validation)
        model_fit = [subject for subject in subjects if subject not in excluded]
        folds.append(
            {
                "fold": fold,
                "outcome": outcome,
                "validation_discovery": validation,
                "model_fit": model_fit,
            }
        )
    actual_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if actual_head != START_SHA:
        raise RuntimeError(f"new branch is not rooted at the frozen Phase-2.5 SHA: {actual_head}")
    protocol = {
        "schema": "PERSIST_EEG_WBCIC_INDEPENDENT_REPLICATION_V1",
        "frozen_before_training": True,
        "frozen_before_outcome_evaluation": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_branch": "codex/persist-eeg-prospective-utility-gate-v1",
        "repository_start_sha": START_SHA,
        "execution_branch": "codex/persist-eeg-wbcic-independent-replication-v1",
        "historical_infrastructure": {
            "branch": "codex/wbcic-eegnet-actionability",
            "experiment": "experiments/persist_eeg_wbcic_actionability_v2",
            "reuse_scope": [
                "development subject whitelist and folds",
                "BIDS event interpretation",
                "Pz reference and 58-channel policy",
                "epoch preprocessing and cache infrastructure",
                "WBCIC EEGNet architecture",
            ],
            "historical_outcomes_reused": False,
        },
        "dataset": {
            "name": "NEMAR nm000348 / WBCIC Yang2025",
            "task": "two-class motor imagery",
            "labels": {"left_hand": 0, "right_hand": 1},
            "subject_pool": subjects,
            "subject_count": 41,
            "allowed_subjects_hash": SUBJECT_HASH,
            "folds": folds,
            "fold_role_rule": "outcome=F_k; validation/discovery=F_(k+1 mod 5); model-fit=remaining three folds",
        },
        "session_semantics": {
            "BIDS_ses-0": "S1",
            "BIDS_ses-1": "S2",
            "BIDS_ses-2": "S3",
            "model_fit": "model-fit subjects S1+S2 only",
            "selection": "validation/discovery subjects S3 only",
            "outcome": "outcome subjects S3 only after run-level source freeze",
        },
        "preprocessing": {
            "input_sampling_hz": 1000,
            "input_EEG_channels": 59,
            "reference": "subtract Pz from every other EEG channel, then drop Pz",
            "output_EEG_channels": 58,
            "bandpass_hz": [0.5, 40.0],
            "filter": "zero-phase fourth-order Butterworth SOS on event-centered window",
            "resampling": "scipy.signal.resample_poly 1000 Hz to 250 Hz",
            "epoch_seconds": [0.0, 4.0],
            "extra_event_offset_seconds": 0.0,
            "amplitude": "microvolts / 20, clipped to [-12.5, 12.5]",
            "cross_subject_normalization": False,
        },
        "backbones": {
            "eegnet": {
                "status": "mandatory primary and only Phase-3 backbone",
                "channels": 58,
                "embedding_dim": 32,
                "dropout": 0.25,
                "lr": 0.0003,
                "weight_decay": 0.0005,
                "batch_size": 64,
            }
        },
        "training": {
            "seeds": [0, 1, 2],
            "max_epochs": 40,
            "minimum_epochs": 10,
            "patience": 6,
            "optimizer": "AdamW",
            "task_loss": "cross entropy",
            "selection_metric": "validation/discovery S3 mean subject balanced accuracy",
            "selection_tie_break": "lower lambda, then earlier epoch",
            "matched_initialization": True,
            "matched_minibatch_order": True,
            "AMP": "bfloat16 autocast; float32 penalties and loss accumulation",
        },
        "methods": {
            "ERM": {"lambda_grid": [0.0], "objective": "task CE"},
            "DANN": {
                "lambda_grid": [0.01, 0.1, 1.0],
                "objective": "task CE plus lambda times source-subject CE through GRL",
                "subject_head": "Linear(32,128)-ReLU-Dropout(0.2)-Linear(128,K)",
            },
            "CORAL": {
                "lambda_grid": [0.01, 0.1, 1.0],
                "objective": "mean pairwise source-subject covariance discrepancy divided by 4*d^2",
            },
            "MMD": {
                "lambda_grid": [0.01, 0.1, 1.0],
                "objective": "mean pairwise source-subject multi-kernel RBF MMD",
                "bandwidth_rule": "source-only deterministic median pairwise-distance heuristic",
                "multipliers": [0.5, 1.0, 2.0],
            },
        },
        "identity": {
            "scope": "model-fit subjects only",
            "definition": "0.5*((log K-CE_S1_to_S2)+(log K-CE_S2_to_S1))",
            "probe": "feature-standardized multiclass ridge linear classifier",
            "ridge_alpha": 1.0,
        },
        "D_finite": {
            "scope": "model-fit S1/S2 only",
            "definition": "sqrt(mean(sum(center_classes(logits_erased-logits_intact)^2)))",
        },
        "secondary_direction_audit": {
            "method": "ERM only",
            "construction": "eigendecomposition of symmetrized S1/S2 subject-centroid cross-covariance",
            "candidate_count": 8,
            "blocks": {"P01_04": [1, 2, 3, 4], "P05_08": [5, 6, 7, 8]},
            "random_controls_per_block_run": 100,
            "random_control_rule": "deterministic orthonormal rank-4 subspace with per-trial displacement-norm matching",
            "intervention": "frozen task head; no retraining",
        },
        "competence_gate": {
            "mean_outcome_S3_BA_strictly_greater_than": 0.60,
            "fold_means_above_chance_at_least": 4,
            "systematic_data_or_preprocessing_failure": False,
        },
        "R1_gate": {
            "strong": "predeclared block BA harm CI lower>0 and persistent-minus-random CI lower>0",
            "partial": "positive task consequence but matched-random superiority uncertain",
            "not_supported": "no predeclared block has reliable task-supportive consequence",
        },
        "R2_gate": {
            "models": {
                "M0": ["persistence", "geometry_strength", "rank"],
                "MI": ["persistence", "geometry_strength", "rank", "identity_score"],
                "MD": ["persistence", "geometry_strength", "rank", "D_finite"],
                "MID": ["persistence", "geometry_strength", "rank", "identity_score", "D_finite"],
            },
            "ridge_alpha": 1.0,
            "evaluation": "leave-one-fold-by-seed run out",
            "strong": "RMSE_MD<RMSE_MI and run-cluster bootstrap CI lower for MI-MD >0",
            "partial": "RMSE_MD<RMSE_MI but CI crosses zero",
            "not_supported": "RMSE_MD>=RMSE_MI",
        },
        "R3_gate": {
            "meaningful_identity_reduction": "mean S_I >= max(0.05, 0.10*matched ERM absolute I) and paired CI lower>0",
            "strong_counterexample": "meaningful reduction and Delta_G CI upper<=0",
            "weak_counterexample": "meaningful reduction, mean Delta_G<=0, and CI crosses zero",
            "positive_alignment": "global slope CI lower>0",
            "misalignment_strong": "at least two families meaningful; at least one counterexample; no positive alignment",
            "misalignment_partial": "at least one family meaningful; no reliable BA gain; no positive alignment",
        },
        "bootstrap": {
            "draws": 10000,
            "seed": 20260826,
            "hierarchy": ["outer fold", "seed/run", "direction where applicable", "outcome subject where applicable"],
            "trial_level_inference": False,
        },
        "terminal_states": [
            "WBCIC_INDEPENDENT_REPLICATION_STRONG_SUPPORTED",
            "WBCIC_INDEPENDENT_REPLICATION_PARTIAL_SUPPORTED",
            "WBCIC_INDEPENDENT_REPLICATION_NOT_SUPPORTED",
            "WBCIC_INVARIANCE_MANIPULATION_INCONCLUSIVE",
            "WBCIC_REPRESENTATION_COMPETENCE_FAIL",
        ],
        "restricted_data_policy": {
            "authorized_WBCIC_development_subjects": 41,
            "sealed_WBCIC_outer_subject_count": 10,
            "sealed_WBCIC_outer_subject_ids_present": False,
            "sealed_WBCIC_outer_access_permitted": False,
            "OpenBMI_EEG_access_permitted": False,
            "outcome_S3_before_source_freeze_permitted": False,
        },
    }
    if TARGET.exists():
        existing = json.loads(TARGET.read_text(encoding="utf-8"))
        comparable_existing = {key: value for key, value in existing.items() if key != "frozen_at_utc"}
        comparable_new = {key: value for key, value in protocol.items() if key != "frozen_at_utc"}
        if comparable_existing != comparable_new:
            raise RuntimeError("existing frozen protocol differs from the requested Phase-3 protocol")
        print(f"PROTOCOL_ALREADY_FROZEN {TARGET}")
        return
    write_json(TARGET, protocol)
    print(f"WBCIC_REPLICATION_PROTOCOL_FROZEN {TARGET}")


if __name__ == "__main__":
    main()
