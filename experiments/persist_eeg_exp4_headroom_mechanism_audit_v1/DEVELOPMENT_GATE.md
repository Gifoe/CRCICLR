# Development gate

```json
{
  "headroom": {
    "H1_oracle_delta_ge_1pp": false,
    "H2_personalization_ge_0_5pp": true,
    "H3_at_least_6_subjects_ge_1pp": true,
    "H4_positive_in_at_least_4_folds": true,
    "H5_actions_constructed_without_future": true
  },
  "mechanism": {
    "M1_exact_Dfinite_validated": true,
    "M2_Dfinite_beats_identity_by_0_03": true,
    "M3_decision_increment_ge_0_03_or_4folds": false,
    "M4_no_future_feature_leakage": true,
    "M5_source_certified_rank": true,
    "strongest_conventional_model": "M_conf",
    "strongest_conventional_AUROC": 0.7275985663082437,
    "strongest_decision_model": "M_PUDfinite",
    "strongest_decision_AUROC": 0.6451612903225807,
    "decision_incremental_AUROC": -0.08243727598566297,
    "decision_fold_positive_count": 2,
    "MECHANISM_SUPPORTED": false
  },
  "Phase_B_authorized": false,
  "terminal_state": "EXP4_STOP_INSUFFICIENT_ACTION_HEADROOM"
}
```
