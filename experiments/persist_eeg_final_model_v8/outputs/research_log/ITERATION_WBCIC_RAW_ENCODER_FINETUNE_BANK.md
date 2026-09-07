# WBCIC_RAW_ENCODER_FINETUNE_BANK

Structural diagnostic: test whether normalization, head, tail, or full raw-encoder movement supplies headroom absent from feature adapters.

```json
{
  "benchmark": "WBCIC_S1S2_to_S3_authorized_development",
  "family_id": "WBCIC_S1S2_to_S3_authorized_development__RAW_ENCODER_FINETUNE_BANK",
  "subjects": 12,
  "baseline_method": "B_STRONG_MATCHED_V7",
  "baseline_BA": 0.8137499999999999,
  "strongest_single_candidate": "RAW_ENCODER_FINETUNE_BANK__ADABN_RESIDUAL50",
  "strongest_single_candidate_BA": 0.8133333333333334,
  "mean_expert_BA": 0.8113541666666666,
  "subject_oracle_BA": 0.81875,
  "oracle_headroom_pp": 0.5000000000000004,
  "subjects_rescued_ge_2pp_fraction": 0.0,
  "subjects_rescued_ge_5pp_fraction": 0.0,
  "positive_fold_fraction": 1.0,
  "mean_pairwise_correctness_correlation": 0.9080371560020244,
  "mean_pairwise_correctness_disagreement": 0.028125,
  "oracle_usage": {
    "B_STRONG_MATCHED_V7": 2,
    "RAW_ENCODER_FINETUNE_BANK__ADABN_RESIDUAL50": 2,
    "RAW_ENCODER_FINETUNE_BANK__HEAD_RESIDUAL50": 3,
    "RAW_ENCODER_FINETUNE_BANK__TAIL_RESIDUAL50": 0,
    "RAW_ENCODER_FINETUNE_BANK__FULL_RESIDUAL50": 5
  },
  "oracle_assignment_entropy": 1.308605387253449,
  "headroom_state": "V8_HEADROOM_WEAK",
  "outcome_labels_used_for_headroom_only": true,
  "used_to_train_selector": false,
  "internal_holdout_used": false,
  "OUTER_TEST_USED": false,
  "folds": [
    0,
    1
  ],
  "experts": 4,
  "baseline_source_method": "ANCHOR_BLEND__CONFORMER_NORM_FIXED_HEAD",
  "training_objective": "V6 discovery-query-screened raw encoder rules; V8 uses fixed residual actions",
  "deployment_transform": "locked strong anchor plus half raw adaptation residual",
  "diagnostic_status": "structural raw-adaptation headroom audit, not a newly meta-learned direction bank"
}
```
