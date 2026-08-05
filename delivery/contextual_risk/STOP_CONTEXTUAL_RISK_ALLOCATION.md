# STOP_CONTEXTUAL_RISK_ALLOCATION

Both preregistered branches failed at least one mandatory gate on development-only cross-fitting. The contract therefore forbids full-method implementation, formal calibration, internal final evaluation, and CAP evaluation. Negative results are retained without branch switching or post-hoc relaxation.

# Gate evidence

```json
{
  "branch_a": {
    "datasets": {
      "eegmmidb": {
        "alpha20_gain": 0.015567661930890226,
        "clopper_pearson_upper": 0.09370678347465346,
        "constant_mae": 2.606153846153846,
        "gates": {
          "alpha20_direction": true,
          "oracle_ci": true,
          "oracle_gain": true,
          "oracle_positive": true,
          "predictor_mae": true,
          "predictor_rho": true,
          "realized_ci": true,
          "realized_gain": true,
          "reliability": true,
          "sentinel": false,
          "validity": true
        },
        "go": false,
        "n_subjects": 65,
        "oracle_ci": [
          0.9222598178137651,
          1.1697554824561402
        ],
        "oracle_relative_gain": 0.29166147698828715,
        "positive_rate": 0.9846153846153847,
        "predictor_mae": 1.948674550980854,
        "predictor_spearman": 0.6637660525568707,
        "realized_ci": [
          0.19372314439946017,
          0.4223675269905532
        ],
        "realized_relative_gain": 0.09010740014927414,
        "target_reliability_kappa": 0.636351808482956,
        "target_reliability_spearman": 0.8588386831987337,
        "violation_rate": 0.03076923076923077,
        "worst_seed_violation": 0.06153846153846154
      },
      "hmc": {
        "alpha20_gain": -0.18620332789086133,
        "clopper_pearson_upper": 0.051625875005761206,
        "constant_mae": 2.5133333333333336,
        "gates": {
          "alpha20_direction": false,
          "oracle_ci": true,
          "oracle_gain": true,
          "oracle_positive": true,
          "predictor_mae": false,
          "predictor_rho": false,
          "realized_ci": false,
          "realized_gain": false,
          "reliability": true,
          "sentinel": false,
          "validity": true
        },
        "go": false,
        "n_subjects": 90,
        "oracle_ci": [
          1.330786574074074,
          1.678000694444444
        ],
        "oracle_relative_gain": 0.39430743020958925,
        "positive_rate": 0.9222222222222223,
        "predictor_mae": 2.7357332828323817,
        "predictor_spearman": -0.127481832494303,
        "realized_ci": [
          -0.5793537037037038,
          -0.33189606481481476
        ],
        "realized_relative_gain": -0.13352209356915715,
        "target_reliability_kappa": 0.3534409031301671,
        "target_reliability_spearman": 0.5329277487243379,
        "violation_rate": 0.011111111111111112,
        "worst_seed_violation": 0.044444444444444446
      }
    },
    "decision": "A_NO_GO"
  },
  "branch_b": {
    "datasets": {
      "eegmmidb": {
        "alpha20_gain": -0.02629272677264611,
        "clopper_pearson_upper": 0.04504225812301352,
        "gates": {
          "alpha20_direction": false,
          "oracle_ci": true,
          "oracle_gain": true,
          "oracle_positive": true,
          "policy_diversity": true,
          "realized_ci": false,
          "realized_gain": false,
          "selector": false,
          "sentinel": false,
          "validity": true
        },
        "go": false,
        "majority_accuracy": 0.27384615384615385,
        "max_nonfallback_win_rate": 0.2676923076923077,
        "n_subjects": 65,
        "nonfallback_policies_ge_10pct": 3,
        "oracle_ci": [
          0.7217925101214576,
          0.9151313090418354
        ],
        "oracle_relative_gain": 0.22740826918578835,
        "policy_wins": {
          "APS|conservative": 0.04923076923076923,
          "APS|efficient": 0.08307692307692308,
          "APS|moderate": 0.1476923076923077,
          "FULL_SET_FALLBACK": 0.036923076923076927,
          "RAPS-k1-l0.01|conservative": 0.033846153846153845,
          "RAPS-k1-l0.01|efficient": 0.03076923076923077,
          "RAPS-k1-l0.01|moderate": 0.036923076923076927,
          "RAPS-k2-l0.01|conservative": 0.024615384615384615,
          "RAPS-k2-l0.01|moderate": 0.036923076923076927,
          "TPS|conservative": 0.08615384615384615,
          "TPS|efficient": 0.2676923076923077,
          "TPS|moderate": 0.16615384615384615
        },
        "positive_rate": 0.9846153846153847,
        "realized_ci": [
          -0.12854055330634273,
          -0.04705553306342789
        ],
        "realized_relative_gain": -0.02510582424195041,
        "recovered_oracle_fraction": -0.11039978595254782,
        "selector_accuracy": 0.3938461538461538,
        "violation_rate": 0.0,
        "worst_seed_violation": 0.06153846153846154
      },
      "hmc": {
        "alpha20_gain": -0.28527954229969965,
        "clopper_pearson_upper": 0.09882084611586947,
        "gates": {
          "alpha20_direction": false,
          "oracle_ci": true,
          "oracle_gain": true,
          "oracle_positive": true,
          "policy_diversity": true,
          "realized_ci": false,
          "realized_gain": false,
          "selector": false,
          "sentinel": false,
          "validity": true
        },
        "go": false,
        "majority_accuracy": 0.23333333333333334,
        "max_nonfallback_win_rate": 0.2088888888888889,
        "n_subjects": 90,
        "nonfallback_policies_ge_10pct": 3,
        "oracle_ci": [
          1.0543976851851853,
          1.4327159722222225
        ],
        "oracle_relative_gain": 0.32164887756212573,
        "policy_wins": {
          "APS|conservative": 0.02,
          "APS|efficient": 0.2088888888888889,
          "APS|moderate": 0.08666666666666667,
          "FULL_SET_FALLBACK": 0.04888888888888889,
          "RAPS-k1-l0.01|conservative": 0.011111111111111112,
          "RAPS-k1-l0.01|efficient": 0.013333333333333334,
          "RAPS-k1-l0.01|moderate": 0.03333333333333333,
          "RAPS-k2-l0.01|conservative": 0.02,
          "RAPS-k2-l0.01|efficient": 0.006666666666666667,
          "RAPS-k2-l0.01|moderate": 0.013333333333333334,
          "RAPS-k2-l0.05|conservative": 0.006666666666666667,
          "RAPS-k2-l0.05|efficient": 0.013333333333333334,
          "RAPS-k3-l0.01|conservative": 0.07111111111111111,
          "RAPS-k3-l0.01|moderate": 0.0044444444444444444,
          "RAPS-k3-l0.10|conservative": 0.03111111111111111,
          "RAPS-k3-l0.10|moderate": 0.006666666666666667,
          "TPS|conservative": 0.07777777777777778,
          "TPS|efficient": 0.18222222222222223,
          "TPS|moderate": 0.14444444444444443
        },
        "positive_rate": 0.9,
        "realized_ci": [
          -0.3210104166666667,
          -0.13194976851851853
        ],
        "realized_relative_gain": -0.06831608050020524,
        "recovered_oracle_fraction": -0.21239334338112262,
        "selector_accuracy": 0.21999999999999997,
        "violation_rate": 0.044444444444444446,
        "worst_seed_violation": 0.1
      }
    },
    "decision": "B_NO_GO"
  },
  "cap_opened": false,
  "created_utc": "2026-08-05T08:23:27.990237+00:00",
  "decision": "STOP_CONTEXTUAL_RISK_ALLOCATION",
  "formal_calibration_opened": false,
  "internal_final_opened": false,
  "schema_version": "contextual-risk-branch-selection-v1",
  "screening_freeze_hash": "973c38f7c2686f4eba476f9105f59e41399274d8be0a17a4238baa4e1ba8c17f",
  "selection_cohort": "method_development",
  "selection_hash": "d7fd6703b23723909e9bde899bd2221d2e8c6de89fef8065dfdfdc8d830df6ab"
}
```
