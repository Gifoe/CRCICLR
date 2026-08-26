"""Create the immutable Phase-2.5 protocol before any new training."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
TARGET = EXP / "UTILITY_GATE_PROTOCOL_FROZEN.json"
BASE = REPO / "experiments" / "persist_eeg_subject_invariance_stress_test_v1" / "STRESS_TEST_PROTOCOL_FROZEN.json"
BASE_SHA = "32f88afa76ace3a19ab9c2cfefc1c0b916fd3eb3"
EXPERIMENT = "persist_eeg_prospective_utility_gate_v1"


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def sort_subjects(values: list[str] | set[str]) -> list[str]:
    return sorted(map(str, values), key=lambda value: int(value))


def main() -> None:
    if TARGET.exists():
        value = json.loads(TARGET.read_text(encoding="utf-8"))
        if value.get("schema") != "PERSIST_EEG_PROSPECTIVE_UTILITY_GATE_V1":
            raise RuntimeError("existing protocol has the wrong schema")
        print(f"[protocol cached] {TARGET}")
        return
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() != BASE_SHA:
        raise RuntimeError("protocol must be frozen at the exact Phase-2 tip before the first Phase-2.5 commit")
    phase2 = json.loads(BASE.read_text(encoding="utf-8"))
    pool = set(map(str, phase2["dataset"]["subject_pool"]))
    folds = []
    for old in phase2["dataset"]["folds"]:
        fold = int(old["fold"])
        outcome = set(map(str, old["outcome"]))
        pseudo = set(map(str, old["inner_validation"]))
        inner_train = sort_subjects(pool - outcome - pseudo)
        rng = np.random.default_rng(stable_seed(EXPERIMENT, fold))
        fit_validation = sort_subjects([inner_train[i] for i in sorted(rng.choice(len(inner_train), size=4, replace=False))])
        fit_train = sort_subjects(set(inner_train) - set(fit_validation))
        folds.append(
            {
                "fold": fold,
                "fit_train": fit_train,
                "fit_validation": fit_validation,
                "pseudo_target": sort_subjects(pseudo),
                "outcome": sort_subjects(outcome),
                "fit_validation_seed": stable_seed(EXPERIMENT, fold),
                "split_rule": "stable SHA256 seed from experiment name and fold; RNG choice 4/24 without replacement",
            }
        )
    protocol = {
        "schema": "PERSIST_EEG_PROSPECTIVE_UTILITY_GATE_V1",
        "frozen_before_training": True,
        "frozen_before_outcome_evaluation": True,
        "repository_start_branch": "codex/persist-eeg-subject-invariance-stress-test-v1",
        "repository_start_sha": BASE_SHA,
        "experiment_branch": "codex/persist-eeg-prospective-utility-gate-v1",
        "scientific_question": "Can source-only pseudo-target suppression utility predict utility on unseen future subjects for the same frozen direction?",
        "dataset": {
            "name": "OpenBMI MI",
            "scope": "exact authorized V8_SEARCH 40-subject pool",
            "subject_pool": sort_subjects(pool),
            "subject_count": 40,
            "folds": folds,
            "sessions": {"fit_train": [1, 2], "fit_validation": [2], "pseudo_target": [2], "outcome": [2]},
            "restricted_data": ["OpenBMI internal 14-subject holdout", "all WBCIC data and membership"],
            "restricted_membership_enumerated": False,
        },
        "backbones": phase2["backbones"],
        "training": {
            **phase2["training"],
            "seeds": [0, 1, 2],
            "method": "ERM only",
            "fit_train_subjects": 20,
            "fit_validation_subjects": 4,
            "pseudo_target_used_for_training_or_selection": False,
        },
        "directions": {
            "candidate_rule": "exact Phase-2 source-only PCA directions ranked by symmetric cross-session subject-mean correlation",
            "construction_scope": "fit_train Sessions 1+2 only",
            "candidate_count": 8,
            "center": "fit_train trial embedding mean",
            "erasure": "h_erased = h - ((h - mu) dot v) v",
            "head": "frozen task head; no retraining",
        },
        "diagnostics": {
            "P": "Phase-2 symmetric cross-session subject-mean projection correlation",
            "I": "Phase-2 identity skill intact minus identity skill after erasure on fit_train",
            "D_finite": "sqrt(mean(sum(center_class(erased_logits-intact_logits)^2))) on fit_train Sessions 1+2",
            "C_train_BA_harm": "mean_subject(BA_intact-BA_erased) on fit_train Session 2",
            "C_train_CE_harm": "mean_subject(CE_erased-CE_intact) on fit_train Session 2",
            "C_validation": "same harm definitions on fit_validation Session 2",
        },
        "utility": {
            "sign": "positive means suppression helps",
            "U_pseudo_BA": "mean_subject(BA_erased-BA_intact), pseudo_target Session 2",
            "U_future_BA": "mean_subject(BA_erased-BA_intact), outcome Session 2",
            "U_F1": "mean_subject(F1_erased-F1_intact)",
            "U_CE": "mean_subject(CE_intact-CE_erased)",
            "historical_conversion": "U_suppress_BA = - historical_erasure_harm_BA",
        },
        "execution_guard": {
            "phase_1": "all 30 models/directions/diagnostics/U_pseudo persisted and hashed",
            "global_marker": "runtime/GLOBAL_SOURCE_FREEZE.json",
            "phase_2": "outer outcome labels become loader-authorized only after marker validation",
            "scope": "fold/run-specific role boundary; folds retain the frozen cross-validation overlap pattern",
        },
        "prediction_models": {
            "M0": ["persistence", "geometry_strength", "direction_rank"],
            "MI": ["persistence", "geometry_strength", "direction_rank", "identity_score"],
            "MD": ["persistence", "geometry_strength", "direction_rank", "D_finite"],
            "MC": ["persistence", "geometry_strength", "direction_rank", "C_train_BA_harm", "C_train_CE_harm"],
            "MIDC": ["persistence", "geometry_strength", "direction_rank", "identity_score", "D_finite", "C_train_BA_harm", "C_train_CE_harm"],
            "MU": ["persistence", "geometry_strength", "direction_rank", "U_pseudo_BA"],
            "MALLU": ["persistence", "geometry_strength", "direction_rank", "identity_score", "D_finite", "C_train_BA_harm", "C_train_CE_harm", "U_pseudo_BA"],
            "ridge_alpha": 1.0,
            "evaluation": "leave one complete backbone-fold-seed run out",
        },
        "statistics": {
            "bootstrap_draws": 10000,
            "bootstrap_seed": 25173,
            "hierarchy": ["outer_fold", "backbone", "seed/run", "direction"],
            "permutation_draws": 1000,
            "permutation_seed": 25174,
        },
        "policy": {
            "primary": "argmax U_pseudo_BA Top1",
            "comparators": ["Random expected", "Highest identity", "Lowest D", "Lowest source consequence", "Oracle Top1"],
            "random": "exact mean U_future_BA over 8 directions",
            "oracle": "upper bound unavailable prospectively",
        },
        "gates": {
            "G1": "positive mean and median within-run Spearman; strong if hierarchical 95% CI lower > 0",
            "G2": "RMSE(MIDC)-RMSE(MALLU) > 0; strong if run-cluster 95% CI lower > 0",
            "G3": "PseudoUtility-Top1 minus Random > 0; strong if hierarchical 95% CI lower > 0",
            "G4": "both backbones positive for G1 and G3; at least one lower CI > 0 per endpoint; neither reliably negative",
            "G5": "purity/protocol integrity passes",
            "H_threshold": 0.005,
            "H": "at least 15% helpful and 15% harmful direction-run cells",
            "low_variation": "at least 80% of runs have both pseudo and future within-run range < 0.005",
            "partial_support": "H and G5 and at least two of G1/G2/G3 have positive point estimates, with no reliable contradictory backbone",
            "precedence": ["purity hard fail", "low variation", "no heterogeneity/headroom", "strong", "partial", "not supported"],
        },
        "terminal_states": [
            "PROSPECTIVE_UTILITY_STRONG_TRANSPORT_SUPPORTED",
            "PROSPECTIVE_UTILITY_PARTIAL_TRANSPORT_SUPPORTED",
            "PROSPECTIVE_UTILITY_NOT_SUPPORTED",
            "NO_ACTIONABLE_SUPPRESSION_HEADROOM",
            "UTILITY_DISCOVERY_INCONCLUSIVE_LOW_VARIATION",
        ],
        "recommendations": {
            "PROSPECTIVE_UTILITY_STRONG_TRANSPORT_SUPPORTED": "AUTHORIZED",
            "PROSPECTIVE_UTILITY_PARTIAL_TRANSPORT_SUPPORTED": "DEFER_TO_WBCIC_REPLICATION",
            "PROSPECTIVE_UTILITY_NOT_SUPPORTED": "CLOSE_CONSTRUCTIVE_ROUTE",
            "NO_ACTIONABLE_SUPPRESSION_HEADROOM": "CLOSE_CONSTRUCTIVE_ROUTE",
            "UTILITY_DISCOVERY_INCONCLUSIVE_LOW_VARIATION": "INCONCLUSIVE",
        },
        "purity": {
            "OpenBMI_internal_holdout_accessed": False,
            "OpenBMI_holdout_membership_enumerated": False,
            "WBCIC_accessed": False,
            "outer_outcome_used_before_global_source_freeze": False,
            "pseudo_target_used_for_training_or_selection": False,
        },
    }
    TARGET.write_text(json.dumps(protocol, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[protocol frozen] {TARGET}")


if __name__ == "__main__":
    main()
