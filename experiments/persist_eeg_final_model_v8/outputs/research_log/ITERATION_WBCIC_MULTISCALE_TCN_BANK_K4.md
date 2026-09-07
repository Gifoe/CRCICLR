# WBCIC_MULTISCALE_TCN_BANK_K4

Structural hypothesis: a stronger multi-scale temporal-spatial TCN trained with subject-level future coverage can create competent complementary experts beyond frozen EEGNet/Conformer representations.

```json
{
  "benchmark": "WBCIC_S1S2_to_S3_authorized_development",
  "family_id": "WBCIC_S1S2_to_S3_authorized_development__MULTISCALE_TCN_BANK_K4",
  "subjects": 5,
  "baseline_method": "B_STRONG_MATCHED_V7",
  "baseline_BA": 0.7949999999999999,
  "strongest_single_candidate": "MULTISCALE_TCN_BANK_K4__E1_HISTORY_RESIDUAL50",
  "strongest_single_candidate_BA": 0.7989999999999999,
  "mean_expert_BA": 0.7857500000000001,
  "subject_oracle_BA": 0.8150000000000001,
  "oracle_headroom_pp": 2.000000000000004,
  "subjects_rescued_ge_2pp_fraction": 0.4,
  "subjects_rescued_ge_5pp_fraction": 0.2,
  "positive_fold_fraction": 1.0,
  "mean_pairwise_correctness_correlation": 0.6409729624414565,
  "mean_pairwise_correctness_disagreement": 0.12207142857142858,
  "oracle_usage": {
    "B_STRONG_MATCHED_V7": 2,
    "MULTISCALE_TCN_BANK_K4__E0_GENERIC_BLEND50": 1,
    "MULTISCALE_TCN_BANK_K4__E0_HISTORY_RESIDUAL50": 1,
    "MULTISCALE_TCN_BANK_K4__E1_GENERIC_BLEND50": 0,
    "MULTISCALE_TCN_BANK_K4__E1_HISTORY_RESIDUAL50": 0,
    "MULTISCALE_TCN_BANK_K4__E2_GENERIC_BLEND50": 0,
    "MULTISCALE_TCN_BANK_K4__E2_HISTORY_RESIDUAL50": 0,
    "MULTISCALE_TCN_BANK_K4__E3_GENERIC_BLEND50": 1,
    "MULTISCALE_TCN_BANK_K4__E3_HISTORY_RESIDUAL50": 0
  },
  "oracle_assignment_entropy": 1.3321790402101223,
  "headroom_state": "V8_HEADROOM_WEAK",
  "outcome_labels_used_for_headroom_only": true,
  "used_to_train_selector": false,
  "internal_holdout_used": false,
  "OUTER_TEST_USED": false,
  "experts": 4,
  "epochs": 12,
  "tau": 0.08,
  "lambda_mean": 0.4,
  "folds": [
    0
  ],
  "baseline_source_method": "ANCHOR_BLEND__CONFORMER_NORM_FIXED_HEAD",
  "training_objective": "future-session subject-coverage loss plus all-session competence on a multi-scale temporal-spatial TCN",
  "candidate_actions": "generic anchor blend and legal-history feature-head residual"
}
```
