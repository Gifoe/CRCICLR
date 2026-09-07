# WBCIC_RAW_FOMAML_MULTISCALE_BANK

Structural hypothesis: raw-signal FOMAML can learn a population initialization whose legal-history head, tail, or full update transfers to a later session.

```json
{
  "benchmark": "WBCIC_S1S2_to_S3_authorized_development",
  "family_id": "WBCIC_S1S2_to_S3_authorized_development__RAW_FOMAML_MULTISCALE_BANK",
  "subjects": 12,
  "baseline_method": "B_STRONG_MATCHED_V7",
  "baseline_BA": 0.8137499999999999,
  "strongest_single_candidate": "RAW_FOMAML_MULTISCALE_BANK__TAIL3_BLEND50",
  "strongest_single_candidate_BA": 0.8133333333333334,
  "mean_expert_BA": 0.8117361111111111,
  "subject_oracle_BA": 0.8295833333333332,
  "oracle_headroom_pp": 1.5833333333333321,
  "subjects_rescued_ge_2pp_fraction": 0.5,
  "subjects_rescued_ge_5pp_fraction": 0.08333333333333333,
  "positive_fold_fraction": 1.0,
  "mean_pairwise_correctness_correlation": 0.7627275835027846,
  "mean_pairwise_correctness_disagreement": 0.07252777777777777,
  "oracle_usage": {
    "B_STRONG_MATCHED_V7": 4,
    "RAW_FOMAML_MULTISCALE_BANK__HEAD4_BLEND50": 2,
    "RAW_FOMAML_MULTISCALE_BANK__HEAD4_DELTA50": 1,
    "RAW_FOMAML_MULTISCALE_BANK__TAIL3_BLEND50": 3,
    "RAW_FOMAML_MULTISCALE_BANK__TAIL3_DELTA50": 1,
    "RAW_FOMAML_MULTISCALE_BANK__FULL1_BLEND50": 1,
    "RAW_FOMAML_MULTISCALE_BANK__FULL1_DELTA50": 0
  },
  "oracle_assignment_entropy": 1.6326309271543518,
  "headroom_state": "V8_HEADROOM_WEAK",
  "outcome_labels_used_for_headroom_only": true,
  "used_to_train_selector": false,
  "internal_holdout_used": false,
  "OUTER_TEST_USED": false,
  "folds": [
    0,
    1
  ],
  "meta_epochs": 4,
  "experts": 3,
  "baseline_source_method": "ANCHOR_BLEND__CONFORMER_NORM_FIXED_HEAD",
  "training_objective": "first-order meta-gradient from future-session balanced query loss after legal-history raw-signal inner adaptation",
  "population_initialization": "competence-first V8_SEARCH-only multiscale temporal-spatial encoder"
}
```
