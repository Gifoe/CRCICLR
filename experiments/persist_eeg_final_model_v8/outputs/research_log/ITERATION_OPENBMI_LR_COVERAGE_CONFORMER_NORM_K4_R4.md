# OPENBMI_LR_COVERAGE_CONFORMER_NORM_K4_R4

Structural hypothesis: a query-trained low-rank classifier-adapter bank with a normalized soft-oracle coverage objective will create competent but complementary future-session experts.

```json
{
  "benchmark": "OpenBMI_MI_S1_to_S2",
  "family_id": "OpenBMI_MI_S1_to_S2__LR_COVERAGE_CONFORMER_NORM_K4_R4",
  "subjects": 18,
  "baseline_method": "B_STRONG_MATCHED_V7",
  "baseline_BA": 0.8516666666666668,
  "strongest_single_candidate": "LR_COVERAGE_CONFORMER_NORM_K4_R4__E1_BLEND50",
  "strongest_single_candidate_BA": 0.8222222222222223,
  "mean_expert_BA": 0.8184722222222223,
  "subject_oracle_BA": 0.8583333333333334,
  "oracle_headroom_pp": 0.6666666666666679,
  "subjects_rescued_ge_2pp_fraction": 0.2222222222222222,
  "subjects_rescued_ge_5pp_fraction": 0.0,
  "positive_fold_fraction": 1.0,
  "mean_pairwise_correctness_correlation": 0.8956895040692325,
  "mean_pairwise_correctness_disagreement": 0.031018518518518518,
  "oracle_usage": {
    "B_STRONG_MATCHED_V7": 11,
    "LR_COVERAGE_CONFORMER_NORM_K4_R4__E0_BLEND50": 1,
    "LR_COVERAGE_CONFORMER_NORM_K4_R4__E1_BLEND50": 2,
    "LR_COVERAGE_CONFORMER_NORM_K4_R4__E2_BLEND50": 2,
    "LR_COVERAGE_CONFORMER_NORM_K4_R4__E3_BLEND50": 2
  },
  "oracle_assignment_entropy": 1.1939422532216233,
  "headroom_state": "V8_HEADROOM_WEAK",
  "outcome_labels_used_for_headroom_only": true,
  "used_to_train_selector": false,
  "internal_holdout_used": false,
  "OUTER_TEST_USED": false,
  "feature_family": "CONFORMER_NORM",
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
  "training_objective": "normalized soft-min coverage + mean competence + usage balance + basis diversity"
}
```
