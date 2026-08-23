# Dynamic actionability audit

```json
{
  "terminal_state": "EXP4_OPENBMI_DYNAMIC_ACTIONABILITY_NOT_SUPPORTED_FOLD_CORRECT",
  "phase_a_state": "DYNAMIC_ACTIONABILITY_NOT_SUPPORTED",
  "reasons": [
    "dynamic RMSE improvement below 10%",
    "dynamic RMSE improves in 2/5 folds",
    "dynamic Spearman magnitude below 0.25"
  ],
  "gradient_sign_pass": true,
  "gradient_unit_test": {
    "passed": true,
    "max_abs_numeric_gradient_error": 9.412304590207532e-07,
    "dot_task_G": -0.0012605930768227213,
    "actual_delta_G": 1.2606066412956807e-07,
    "sign_convention_ok": true
  },
  "trajectory_first_order_direction_agreement": 1.0,
  "trajectory_first_order_utility_delta_correlation": 0.9984476529953253,
  "overall_prediction": {
    "M0": {
      "RMSE": 0.012742012828232816,
      "Spearman": 0.32030760836222427,
      "n": 40
    },
    "M_static": {
      "RMSE": 0.012331613115384603,
      "Spearman": 0.2367315898332566,
      "n": 40
    },
    "M_dynamic": {
      "RMSE": 0.014467140512462966,
      "Spearman": 0.2482107056312112,
      "n": 40
    },
    "M_gradient": {
      "RMSE": 0.021476771577365692,
      "Spearman": 0.3158770724402067,
      "n": 40
    },
    "M_full": {
      "RMSE": 0.018587172781387776,
      "Spearman": 0.22072124411505678,
      "n": 40
    }
  },
  "dynamic_relative_RMSE_reduction_vs_static": -0.17317502398888385,
  "folds_dynamic_RMSE_improved": 2,
  "fold_improvement_flags": [
    false,
    false,
    true,
    true,
    false
  ],
  "negative_transfer_AUROC_static": 0.5784313725490196,
  "negative_transfer_AUROC_dynamic": 0.6813725490196079,
  "negative_transfer_AUROC_gain": 0.10294117647058831,
  "tail_high_risk_mean_FutureDeltaBA": 0.004000000000000015,
  "tail_low_risk_mean_FutureDeltaBA": 0.007692307692307682,
  "subjects_evaluated": 40,
  "folds": 5,
  "internal_holdout_used": false,
  "outer_test_used": false,
  "wbcic_used": false
}
```
