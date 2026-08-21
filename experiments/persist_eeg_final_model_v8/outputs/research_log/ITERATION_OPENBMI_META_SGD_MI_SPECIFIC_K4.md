# OPENBMI_META_SGD_MI_SPECIFIC_K4

Structural hypothesis: a true future-query-trained Meta-SGD rule can turn legal history gradients into complementary future-session adaptation.

```json
{
  "benchmark": "OpenBMI_MI_S1_to_S2",
  "family_id": "OpenBMI_MI_S1_to_S2__META_SGD_MI_SPECIFIC_K4",
  "subjects": 18,
  "baseline_method": "B_STRONG_MATCHED_V7",
  "baseline_BA": 0.8516666666666668,
  "strongest_single_candidate": "META_SGD_MI_SPECIFIC_K4__E2_BLEND50",
  "strongest_single_candidate_BA": 0.846111111111111,
  "mean_expert_BA": 0.8424999999999999,
  "subject_oracle_BA": 0.8688888888888889,
  "oracle_headroom_pp": 1.7222222222222219,
  "subjects_rescued_ge_2pp_fraction": 0.3888888888888889,
  "subjects_rescued_ge_5pp_fraction": 0.16666666666666666,
  "positive_fold_fraction": 1.0,
  "mean_pairwise_correctness_correlation": 0.8202693017674916,
  "mean_pairwise_correctness_disagreement": 0.04777777777777778,
  "oracle_usage": {
    "B_STRONG_MATCHED_V7": 7,
    "META_SGD_MI_SPECIFIC_K4__E0_BLEND50": 2,
    "META_SGD_MI_SPECIFIC_K4__E1_BLEND50": 2,
    "META_SGD_MI_SPECIFIC_K4__E2_BLEND50": 2,
    "META_SGD_MI_SPECIFIC_K4__E3_BLEND50": 5
  },
  "oracle_assignment_entropy": 1.4555137751785332,
  "headroom_state": "V8_HEADROOM_WEAK",
  "outcome_labels_used_for_headroom_only": true,
  "used_to_train_selector": false,
  "internal_holdout_used": false,
  "OUTER_TEST_USED": false,
  "feature_family": "MI_SPECIFIC",
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
