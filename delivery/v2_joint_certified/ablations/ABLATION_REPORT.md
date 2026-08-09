# HSC-TTA v2 ablation audit

All rows use the same nested subject splits. A2 is explicitly uncertified; A5 and A10 alter the calibration unit and therefore do not inherit the proposed simultaneous post-selection theorem.

## HMC

| Ablation | alpha | Joint validity | CSR | TTA rate | Gain |
|---|---:|---:|---:|---:|---:|
| A10_policy_level_calibration | 0.10 | 0.796 | 0.280 | 0.531 | 0.0014 |
| A10_policy_level_calibration | 0.20 | 0.778 | 0.385 | 0.531 | 0.0017 |
| A1_risk_only | 0.10 | 0.971 | 0.087 | 0.065 | 0.0006 |
| A1_risk_only | 0.20 | 0.960 | 0.145 | 0.131 | 0.0007 |
| A2_utility_only_uncertified | 0.10 | 0.498 | 0.967 | 0.531 | 0.0017 |
| A2_utility_only_uncertified | 0.20 | 0.473 | 1.000 | 0.531 | 0.0017 |
| A3_joint_without_benefit_certificate | 0.10 | 0.982 | 0.087 | 0.033 | 0.0005 |
| A3_joint_without_benefit_certificate | 0.20 | 0.978 | 0.135 | 0.069 | 0.0001 |
| A4_separate_risk_gain_calibration | 0.10 | 0.978 | 0.229 | 0.004 | 0.0006 |
| A4_separate_risk_gain_calibration | 0.20 | 0.993 | 0.320 | 0.004 | 0.0006 |
| A5_pointwise_per_action_calibration | 0.10 | 0.967 | 0.375 | 0.007 | 0.0006 |
| A5_pointwise_per_action_calibration | 0.20 | 0.938 | 0.531 | 0.015 | 0.0002 |
| A6_no_action_specific_diagnostics | 0.10 | 1.000 | 0.000 | 0.000 | 0.0000 |
| A6_no_action_specific_diagnostics | 0.20 | 1.000 | 0.000 | 0.000 | 0.0000 |
| A7_context_set_size_selector | 0.10 | 0.989 | 0.076 | 0.000 | 0.0000 |
| A7_context_set_size_selector | 0.20 | 0.996 | 0.124 | 0.000 | 0.0000 |
| A8_global_critical_index | 0.10 | 0.993 | 0.120 | 0.033 | 0.0005 |
| A8_global_critical_index | 0.20 | 0.975 | 0.440 | 0.069 | 0.0001 |
| A9_no_positive_gain_gate | 0.10 | 0.971 | 0.087 | 0.065 | 0.0007 |
| A9_no_positive_gain_gate | 0.20 | 0.967 | 0.145 | 0.131 | 0.0001 |

## EEGMMIDB

| Ablation | alpha | Joint validity | CSR | TTA rate | Gain |
|---|---:|---:|---:|---:|---:|
| A10_policy_level_calibration | 0.10 | 0.818 | 0.271 | 0.653 | 0.0001 |
| A10_policy_level_calibration | 0.20 | 0.818 | 0.502 | 0.653 | 0.0001 |
| A1_risk_only | 0.10 | 0.987 | 0.284 | 0.249 | 0.0004 |
| A1_risk_only | 0.20 | 0.924 | 0.511 | 0.476 | -0.0006 |
| A2_utility_only_uncertified | 0.10 | 0.640 | 0.996 | 0.653 | 0.0001 |
| A2_utility_only_uncertified | 0.20 | 0.578 | 1.000 | 0.653 | 0.0001 |
| A3_joint_without_benefit_certificate | 0.10 | 0.982 | 0.276 | 0.124 | -0.0001 |
| A3_joint_without_benefit_certificate | 0.20 | 0.951 | 0.507 | 0.280 | -0.0005 |
| A4_separate_risk_gain_calibration | 0.10 | 0.991 | 0.600 | 0.000 | 0.0000 |
| A4_separate_risk_gain_calibration | 0.20 | 0.991 | 0.733 | 0.000 | 0.0000 |
| A5_pointwise_per_action_calibration | 0.10 | 0.978 | 0.671 | 0.013 | 0.0000 |
| A5_pointwise_per_action_calibration | 0.20 | 0.973 | 0.853 | 0.022 | -0.0001 |
| A6_no_action_specific_diagnostics | 0.10 | 1.000 | 0.000 | 0.000 | 0.0000 |
| A6_no_action_specific_diagnostics | 0.20 | 1.000 | 0.000 | 0.000 | 0.0000 |
| A7_context_set_size_selector | 0.10 | 1.000 | 0.271 | 0.000 | 0.0000 |
| A7_context_set_size_selector | 0.20 | 1.000 | 0.498 | 0.000 | 0.0000 |
| A8_global_critical_index | 0.10 | 0.982 | 0.560 | 0.124 | -0.0001 |
| A8_global_critical_index | 0.20 | 0.947 | 1.000 | 0.280 | -0.0005 |
| A9_no_positive_gain_gate | 0.10 | 0.978 | 0.284 | 0.249 | -0.0000 |
| A9_no_positive_gain_gate | 0.20 | 0.924 | 0.511 | 0.476 | -0.0006 |
