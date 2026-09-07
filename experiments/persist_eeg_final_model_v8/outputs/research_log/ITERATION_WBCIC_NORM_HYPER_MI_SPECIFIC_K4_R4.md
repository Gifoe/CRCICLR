# WBCIC_NORM_HYPER_MI_SPECIFIC_K4_R4

Structural hypothesis: target-history normalization plus a constrained FiLM hypernetwork can correct session shift without generating an unconstrained classifier.

```json
{
  "benchmark": "WBCIC_S1S2_to_S3_authorized_development",
  "family_id": "WBCIC_S1S2_to_S3_authorized_development__NORM_HYPER_MI_SPECIFIC_K4_R4",
  "subjects": 12,
  "baseline_method": "B_STRONG_MATCHED_V7",
  "baseline_BA": 0.8137499999999999,
  "strongest_single_candidate": "NORM_HYPER_MI_SPECIFIC_K4_R4__E0_RESIDUAL50",
  "strongest_single_candidate_BA": 0.8141666666666666,
  "mean_expert_BA": 0.8106249999999999,
  "subject_oracle_BA": 0.8241666666666667,
  "oracle_headroom_pp": 1.0416666666666676,
  "subjects_rescued_ge_2pp_fraction": 0.25,
  "subjects_rescued_ge_5pp_fraction": 0.08333333333333333,
  "positive_fold_fraction": 1.0,
  "mean_pairwise_correctness_correlation": 0.9071124592746055,
  "mean_pairwise_correctness_disagreement": 0.02861111111111111,
  "oracle_usage": {
    "B_STRONG_MATCHED_V7": 8,
    "NORM_HYPER_MI_SPECIFIC_K4_R4__E0_RESIDUAL50": 1,
    "NORM_HYPER_MI_SPECIFIC_K4_R4__E1_RESIDUAL50": 1,
    "NORM_HYPER_MI_SPECIFIC_K4_R4__E2_RESIDUAL50": 2,
    "NORM_HYPER_MI_SPECIFIC_K4_R4__E3_RESIDUAL50": 0
  },
  "oracle_assignment_entropy": 0.9830877585747855,
  "headroom_state": "V8_HEADROOM_WEAK",
  "outcome_labels_used_for_headroom_only": true,
  "used_to_train_selector": false,
  "internal_holdout_used": false,
  "OUTER_TEST_USED": false,
  "feature_family": "MI_SPECIFIC",
  "experts": 4,
  "rank": 4,
  "epochs": 300,
  "tau": 0.08,
  "lambda_mean": 0.35,
  "folds": [
    0,
    1
  ],
  "baseline_source_method": "ANCHOR_BLEND__CONFORMER_NORM_FIXED_HEAD",
  "training_objective": "query-trained history-conditioned normalization/FiLM coefficients with coverage and competence",
  "deployment_transform": "locked strong anchor plus half learned normalization residual"
}
```
