# Iteration 006: I006_CONDITIONAL_ACTION_LOGISTIC

- Failure diagnosis / outcome: residual ranking did not convert into safe positive gain above B6
- Hypothesis: Training only on boundary-cross candidates and separating action families removes trivial eligibility discrimination and may stabilize rescue-versus-harm routing.
- Change: action-specific conditional regularized logistic rescue heads with a finite calibrated action menu
- Grouped OOF Delta BA vs B6: -0.001538
- Subject-bootstrap CI95: [-0.003462, +0.000481]
- Action rate: 0.009038
- Rescue precision: 0.414894
- Harm rate: 0.585106
- Oracle recovery: -0.017897
- Decision: ABANDON

All thresholds and model settings were selected on inner calibration subjects;
outer held-out subjects were not used for selection. This is exploratory OOF
evidence, not an untouched confirmation.

`OUTER_TEST_USED=false`
