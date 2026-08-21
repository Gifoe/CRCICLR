# WBCIC_METRIC_TRANSPORT_CONFORMER_NORM_K4_D16

Structural hypothesis: learned class-conditional metric transport with legal-history recency can capture future-stable geometry missed by gradient and direct hypernetwork adapters.

```json
{
  "benchmark": "WBCIC_S1S2_to_S3_authorized_development",
  "family_id": "WBCIC_S1S2_to_S3_authorized_development__METRIC_TRANSPORT_CONFORMER_NORM_K4_D16",
  "subjects": 12,
  "baseline_method": "B_STRONG_MATCHED_V7",
  "baseline_BA": 0.8137499999999999,
  "strongest_single_candidate": "METRIC_TRANSPORT_CONFORMER_NORM_K4_D16__E1_RESIDUAL50",
  "strongest_single_candidate_BA": 0.8120833333333333,
  "mean_expert_BA": 0.8107291666666667,
  "subject_oracle_BA": 0.81875,
  "oracle_headroom_pp": 0.5000000000000004,
  "subjects_rescued_ge_2pp_fraction": 0.08333333333333333,
  "subjects_rescued_ge_5pp_fraction": 0.0,
  "positive_fold_fraction": 1.0,
  "mean_pairwise_correctness_correlation": 0.9681178046683803,
  "mean_pairwise_correctness_disagreement": 0.009791666666666666,
  "oracle_usage": {
    "B_STRONG_MATCHED_V7": 5,
    "METRIC_TRANSPORT_CONFORMER_NORM_K4_D16__E0_RESIDUAL50": 2,
    "METRIC_TRANSPORT_CONFORMER_NORM_K4_D16__E1_RESIDUAL50": 1,
    "METRIC_TRANSPORT_CONFORMER_NORM_K4_D16__E2_RESIDUAL50": 1,
    "METRIC_TRANSPORT_CONFORMER_NORM_K4_D16__E3_RESIDUAL50": 3
  },
  "oracle_assignment_entropy": 1.4241299173467734,
  "headroom_state": "V8_HEADROOM_WEAK",
  "outcome_labels_used_for_headroom_only": true,
  "used_to_train_selector": false,
  "internal_holdout_used": false,
  "OUTER_TEST_USED": false,
  "feature_family": "CONFORMER_NORM",
  "experts": 4,
  "metric_dimension": 16,
  "epochs": 300,
  "tau": 0.08,
  "lambda_mean": 0.35,
  "folds": [
    0,
    1
  ],
  "baseline_source_method": "ANCHOR_BLEND__CONFORMER_NORM_FIXED_HEAD",
  "training_objective": "history-class prototype transport through query-trained metrics, learned recency, coverage, competence, and diversity",
  "deployment_transform": "locked strong anchor plus half learned metric residual"
}
```
