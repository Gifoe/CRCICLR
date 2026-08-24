# OpenBMI development gate

- **G1_PERFORMANCE**: `FAIL` — `{"pass": false, "delta_BA": -0.028500000000000192, "threshold": 0.0075}`
- **G2_UNCERTAINTY**: `FAIL` — `{"pass": false, "CI95": [-0.04091875000000003, -0.01625000000000001], "strict_rule": "lower > 0"}`
- **G3_FOLD_CONSISTENCY**: `FAIL` — `{"pass": false, "positive_folds": 0, "required": 4}`
- **G4_SEED_CONSISTENCY**: `FAIL` — `{"pass": false, "positive_seeds": 0, "required": 2}`
- **G5_SAFETY**: `PASS` — `{"pass": true, "FULL_NTR": 0.25, "Generic_NTR": 0.425, "FULL_worst_quartile": -0.008999999999999982, "Generic_worst_quartile": -0.027666666666666635, "worst_quartile_noninferiority_tolerance": 0.0025}`
- **G6_THEORY_SPECIFICITY**: `FAIL` — `{"pass": false, "FULL_minus_controls": {"A7_IDENTITY_PROTECTED": -0.005916666666666681, "A8_RANDOM_PROTECTED": -0.003083333333333438, "A2_DUAL_CONTROL": -0.012416666666666742}}`
- **G7_PROTECTION_NECESSITY**: `PASS` — `{"pass": true, "FULL_minus_all_adapt": 0.0019166666666664556, "preferred_target": 0.003}`
- **G8_MECHANISM**: `PASS` — `{"pass": true, "protected_drift_reduction": 1.0, "FULL_adaptive_update_l2": 0.10366239958143735}`
- **G9_NO_FUTURE_LEAKAGE**: `PASS` — `{"pass": true, "internal_holdout_accessed": false, "outer_test_used": false}`

WBCIC authorized by frozen G1/G6/G9 rule: **NO**.
Current terminal state: **PERSIST_NET_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED**.
