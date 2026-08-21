# OPENBMI_META_SGD_CONFORMER_NORM_K4

Structural hypothesis: a true future-query-trained Meta-SGD rule can turn legal history gradients into complementary future-session adaptation.

```json
{
  "benchmark": "OpenBMI_MI_S1_to_S2",
  "family_id": "OpenBMI_MI_S1_to_S2__META_SGD_CONFORMER_NORM_K4",
  "subjects": 18,
  "baseline_method": "B_STRONG_MATCHED_V7",
  "baseline_BA": 0.8516666666666668,
  "strongest_single_candidate": "META_SGD_CONFORMER_NORM_K4__E2_BLEND50",
  "strongest_single_candidate_BA": 0.838888888888889,
  "mean_expert_BA": 0.8380555555555556,
  "subject_oracle_BA": 0.8594444444444446,
  "oracle_headroom_pp": 0.7777777777777797,
  "subjects_rescued_ge_2pp_fraction": 0.2222222222222222,
  "subjects_rescued_ge_5pp_fraction": 0.0,
  "positive_fold_fraction": 1.0,
  "mean_pairwise_correctness_correlation": 0.888833834688005,
  "mean_pairwise_correctness_disagreement": 0.030185185185185186,
  "oracle_usage": {
    "B_STRONG_MATCHED_V7": 8,
    "META_SGD_CONFORMER_NORM_K4__E0_BLEND50": 3,
    "META_SGD_CONFORMER_NORM_K4__E1_BLEND50": 2,
    "META_SGD_CONFORMER_NORM_K4__E2_BLEND50": 3,
    "META_SGD_CONFORMER_NORM_K4__E3_BLEND50": 2
  },
  "oracle_assignment_entropy": 1.4459387141357687,
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
