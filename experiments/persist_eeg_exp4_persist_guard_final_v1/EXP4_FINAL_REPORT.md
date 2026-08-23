# Exp4 final report

```json
{
  "final_strongest_clean_Generic_BA": 0.77275,
  "historical_83_775_legal": false,
  "final_PERSIST_Guard_BA": 0.7737499999999999,
  "delta_PERSIST_vs_Generic": 0.000999999999999998,
  "paired_95_CI": [
    -0.0012562500000000035,
    0.0034999999999999975
  ],
  "fold_positive_count": 2,
  "Generic_negative_transfer_rate": 0.225,
  "PERSIST_negative_transfer_rate": 0.175,
  "Generic_harmed_subjects": 9,
  "rescued_by_PERSIST": 3,
  "newly_harmed": 1,
  "PERSIST_risk_AUROC": 0.6129032258064516,
  "confidence_AUROC": 0.7275985663082437,
  "identity_AUROC": 0.4551971326164875,
  "PERSIST_beats_identity_AUROC": true,
  "PERSIST_beats_confidence_update_performance": false,
  "worst_quartile_change": 0.002000000000000113,
  "three_seed_results": [
    {
      "seed": 0,
      "subjects": 40,
      "NoAdapt_BA": 0.75725,
      "Strong_Generic_BA": 0.7655,
      "PERSIST_Guard_BA": 0.76575,
      "delta_vs_generic": 0.00024999999999999746,
      "internal_holdout_used": false
    },
    {
      "seed": 1,
      "subjects": 40,
      "NoAdapt_BA": 0.7394999999999999,
      "Strong_Generic_BA": 0.7422500000000001,
      "PERSIST_Guard_BA": 0.741,
      "delta_vs_generic": -0.0012500000000000011,
      "internal_holdout_used": false
    },
    {
      "seed": 2,
      "subjects": 40,
      "NoAdapt_BA": 0.75225,
      "Strong_Generic_BA": 0.7537499999999999,
      "PERSIST_Guard_BA": 0.755,
      "delta_vs_generic": 0.0012500000000000011,
      "internal_holdout_used": false
    }
  ],
  "three_seed_summary_individual_models": {
    "NoAdapt_BA_mean": 0.7496666666666667,
    "NoAdapt_BA_SD": 0.00915264078467705,
    "Strong_Generic_BA_mean": 0.7538333333333332,
    "Strong_Generic_BA_SD": 0.011625224012178517,
    "PERSIST_Guard_BA_mean": 0.7539166666666667,
    "PERSIST_Guard_BA_SD": 0.012410513016524898,
    "delta_vs_generic_mean": 8.333333333333246e-05,
    "delta_vs_generic_SD": 0.0012583057392117926
  },
  "development_gate": "FAIL",
  "internal_holdout_accessed": false,
  "holdout_Generic_BA": null,
  "holdout_PERSIST_BA": null,
  "holdout_harm_reduction": null,
  "second_backbone_result": "NOT_REACHED",
  "WBCIC_result": "NOT_REACHED",
  "terminal_state": "EXP4_PERSIST_GUARD_NOT_SUPPORTED",
  "strongest_justified_claim": "A bounded, source-only decision-grounded guard was evaluated on the authorized development split; its support is exactly the terminal state and metrics reported here.",
  "strongest_unsupported_claim": "PERSIST-Guard is confirmed on the sealed internal holdout, transfers across backbones, or generalizes to WBCIC."
}
```
