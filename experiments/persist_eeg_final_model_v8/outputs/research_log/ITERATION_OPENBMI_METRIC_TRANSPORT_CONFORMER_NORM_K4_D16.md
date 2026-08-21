# OPENBMI_METRIC_TRANSPORT_CONFORMER_NORM_K4_D16

Structural hypothesis: learned class-conditional metric transport with legal-history recency can capture future-stable geometry missed by gradient and direct hypernetwork adapters.

```json
{
  "benchmark": "OpenBMI_MI_S1_to_S2",
  "family_id": "OpenBMI_MI_S1_to_S2__METRIC_TRANSPORT_CONFORMER_NORM_K4_D16",
  "subjects": 18,
  "baseline_method": "B_STRONG_MATCHED_V7",
  "baseline_BA": 0.8516666666666668,
  "strongest_single_candidate": "METRIC_TRANSPORT_CONFORMER_NORM_K4_D16__E1_RESIDUAL50",
  "strongest_single_candidate_BA": 0.8488888888888888,
  "mean_expert_BA": 0.8480555555555556,
  "subject_oracle_BA": 0.8605555555555555,
  "oracle_headroom_pp": 0.8888888888888891,
  "subjects_rescued_ge_2pp_fraction": 0.3333333333333333,
  "subjects_rescued_ge_5pp_fraction": 0.05555555555555555,
  "positive_fold_fraction": 1.0,
  "mean_pairwise_correctness_correlation": 0.9554841746778177,
  "mean_pairwise_correctness_disagreement": 0.011481481481481481,
  "oracle_usage": {
    "B_STRONG_MATCHED_V7": 9,
    "METRIC_TRANSPORT_CONFORMER_NORM_K4_D16__E0_RESIDUAL50": 2,
    "METRIC_TRANSPORT_CONFORMER_NORM_K4_D16__E1_RESIDUAL50": 3,
    "METRIC_TRANSPORT_CONFORMER_NORM_K4_D16__E2_RESIDUAL50": 0,
    "METRIC_TRANSPORT_CONFORMER_NORM_K4_D16__E3_RESIDUAL50": 4
  },
  "oracle_assignment_entropy": 1.223575654138956,
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
