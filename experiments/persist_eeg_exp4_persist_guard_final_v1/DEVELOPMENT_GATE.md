# Development gate

```json
{
  "terminal_state": "EXP4_PERSIST_GUARD_NOT_SUPPORTED",
  "development_gate_pass": false,
  "selected_development_variant": "V2_PERSIST_FIXED_SHRINK",
  "checks": {
    "A_delta_at_least_0_5pp": false,
    "B_positive_in_4_of_5_folds": false,
    "C_harm_reduction_at_least_35pct": false,
    "D_rescue_at_least_half_harms": false,
    "E_new_harms_at_most_2": true,
    "F_worst_quartile_non_decreasing": true,
    "G_beats_identity_and_confidence": true,
    "H_PERSIST_AUROC_at_least_0_65": false,
    "I_harmed_risk_higher": true,
    "J_all_three_seed_deltas_nonnegative": false
  },
  "checks_passed": 4,
  "checks_total": 10,
  "Strong_Generic_BA": 0.77275,
  "PERSIST_Guard_BA": 0.7737499999999999,
  "delta_vs_generic": 0.000999999999999998,
  "paired_CI95": [
    -0.0012562500000000035,
    0.0034999999999999975
  ],
  "generic_negative_transfer_rate": 0.225,
  "guard_negative_transfer_rate": 0.175,
  "generic_harmed_subjects": 9,
  "rescued_harms": 3,
  "new_harms": 1,
  "PERSIST_AUROC": 0.6129032258064516,
  "confidence_AUROC": 0.7275985663082437,
  "identity_AUROC": 0.4551971326164875,
  "worst_quartile_delta": 0.002000000000000113,
  "fold_positive_count": 2,
  "internal_holdout_used": false,
  "holdout_access_authorized": false,
  "WBCIC_used": false
}
```
