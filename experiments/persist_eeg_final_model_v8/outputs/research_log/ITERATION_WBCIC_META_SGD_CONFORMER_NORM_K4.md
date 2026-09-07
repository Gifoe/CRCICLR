# WBCIC_META_SGD_CONFORMER_NORM_K4

Structural hypothesis: a true future-query-trained Meta-SGD rule can turn legal history gradients into complementary future-session adaptation.

```json
{
  "benchmark": "WBCIC_S1S2_to_S3_authorized_development",
  "family_id": "WBCIC_S1S2_to_S3_authorized_development__META_SGD_CONFORMER_NORM_K4",
  "subjects": 12,
  "baseline_method": "B_STRONG_MATCHED_V7",
  "baseline_BA": 0.8137499999999999,
  "strongest_single_candidate": "META_SGD_CONFORMER_NORM_K4__E0_BLEND50",
  "strongest_single_candidate_BA": 0.8079166666666667,
  "mean_expert_BA": 0.8049999999999999,
  "subject_oracle_BA": 0.8249999999999998,
  "oracle_headroom_pp": 1.1250000000000009,
  "subjects_rescued_ge_2pp_fraction": 0.3333333333333333,
  "subjects_rescued_ge_5pp_fraction": 0.0,
  "positive_fold_fraction": 1.0,
  "mean_pairwise_correctness_correlation": 0.8239900075069125,
  "mean_pairwise_correctness_disagreement": 0.05527777777777779,
  "oracle_usage": {
    "B_STRONG_MATCHED_V7": 5,
    "META_SGD_CONFORMER_NORM_K4__E0_BLEND50": 1,
    "META_SGD_CONFORMER_NORM_K4__E1_BLEND50": 2,
    "META_SGD_CONFORMER_NORM_K4__E2_BLEND50": 2,
    "META_SGD_CONFORMER_NORM_K4__E3_BLEND50": 2
  },
  "oracle_assignment_entropy": 1.4677339293271523,
  "headroom_state": "V8_HEADROOM_WEAK",
  "outcome_labels_used_for_headroom_only": true,
  "used_to_train_selector": false,
  "internal_holdout_used": false,
  "OUTER_TEST_USED": false,
  "feature_family": "CONFORMER_NORM",
  "experts": 4,
  "epochs": 300,
  "tau": 0.08,
  "lambda_mean": 0.35,
  "folds": [
    0,
    1
  ],
  "baseline_source_method": "ANCHOR_BLEND__CONFORMER_NORM_FIXED_HEAD",
  "training_objective": "true differentiable history gradient step with learned signed per-feature step sizes and future-query coverage loss"
}
```
