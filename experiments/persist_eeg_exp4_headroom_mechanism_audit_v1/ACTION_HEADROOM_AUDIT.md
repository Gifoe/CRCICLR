# Action headroom audit

```json
{
  "Clean_Strong_Generic_BA": 0.7727499999999999,
  "Binary_Generic_NoAdapt_Oracle_BA": 0.77675,
  "Binary_Oracle_Headroom_vs_Generic": 0.003999999999999998,
  "OracleActionBank_BA": 0.781,
  "OracleHeadroom_vs_Generic": 0.008250000000000002,
  "BestGlobalAction": "A3_Generic_75pct",
  "BestGlobalAction_BA": 0.77475,
  "PersonalizationHeadroom": 0.006249999999999978,
  "subjects_best_action_not_Generic": 36,
  "subjects_alternative_gain_ge_0_5pp": 22,
  "subjects_alternative_gain_ge_1pp": 22,
  "fold_positive_oracle_count": 5,
  "worst_quartile_oracle_improvement": 0.007000000000000006,
  "mean_Strong_Generic_regret": 0.008250000000000002,
  "optimal_action_distribution": {
    "A0_NoAdapt": 10,
    "A1_Generic_25pct": 10,
    "A2_Generic_50pct": 7,
    "A3_Generic_75pct": 9,
    "A4_Strong_Generic": 4
  },
  "per_fold_oracle": [
    {
      "fold": 0,
      "subjects": 9,
      "oracle_BA": 0.7511111111111111,
      "generic_BA": 0.741111111111111,
      "oracle_delta": 0.009999999999999997
    },
    {
      "fold": 1,
      "subjects": 9,
      "oracle_BA": 0.8144444444444444,
      "generic_BA": 0.8055555555555555,
      "oracle_delta": 0.008888888888888898
    },
    {
      "fold": 2,
      "subjects": 6,
      "oracle_BA": 0.7633333333333333,
      "generic_BA": 0.7549999999999999,
      "oracle_delta": 0.00833333333333334
    },
    {
      "fold": 3,
      "subjects": 9,
      "oracle_BA": 0.7922222222222222,
      "generic_BA": 0.7866666666666666,
      "oracle_delta": 0.00555555555555556
    },
    {
      "fold": 4,
      "subjects": 7,
      "oracle_BA": 0.777142857142857,
      "generic_BA": 0.7685714285714287,
      "oracle_delta": 0.008571428571428563
    }
  ],
  "checks": {
    "H1_oracle_delta_ge_1pp": false,
    "H2_personalization_ge_0_5pp": true,
    "H3_at_least_6_subjects_ge_1pp": true,
    "H4_positive_in_at_least_4_folds": true,
    "H5_actions_constructed_without_future": true
  },
  "HEADROOM_SUPPORTED": false,
  "internal_holdout_used": false
}
```
