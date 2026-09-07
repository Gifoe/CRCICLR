# OPENBMI_NORM_HYPER_MI_SPECIFIC_K4_R4

Structural hypothesis: target-history normalization plus a constrained FiLM hypernetwork can correct session shift without generating an unconstrained classifier.

```json
{
  "benchmark": "OpenBMI_MI_S1_to_S2",
  "family_id": "OpenBMI_MI_S1_to_S2__NORM_HYPER_MI_SPECIFIC_K4_R4",
  "subjects": 18,
  "baseline_method": "B_STRONG_MATCHED_V7",
  "baseline_BA": 0.8516666666666668,
  "strongest_single_candidate": "NORM_HYPER_MI_SPECIFIC_K4_R4__E2_RESIDUAL50",
  "strongest_single_candidate_BA": 0.8583333333333334,
  "mean_expert_BA": 0.8543055555555554,
  "subject_oracle_BA": 0.8705555555555556,
  "oracle_headroom_pp": 1.8888888888888893,
  "subjects_rescued_ge_2pp_fraction": 0.3888888888888889,
  "subjects_rescued_ge_5pp_fraction": 0.16666666666666666,
  "positive_fold_fraction": 1.0,
  "mean_pairwise_correctness_correlation": 0.8822460009399381,
  "mean_pairwise_correctness_disagreement": 0.029351851851851855,
  "oracle_usage": {
    "B_STRONG_MATCHED_V7": 6,
    "NORM_HYPER_MI_SPECIFIC_K4_R4__E0_RESIDUAL50": 2,
    "NORM_HYPER_MI_SPECIFIC_K4_R4__E1_RESIDUAL50": 1,
    "NORM_HYPER_MI_SPECIFIC_K4_R4__E2_RESIDUAL50": 6,
    "NORM_HYPER_MI_SPECIFIC_K4_R4__E3_RESIDUAL50": 3
  },
  "oracle_assignment_entropy": 1.4357470435705602,
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
