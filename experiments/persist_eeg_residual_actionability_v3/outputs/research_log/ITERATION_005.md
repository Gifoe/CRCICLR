# Iteration 005: M5_HIST_GRADIENT_BOOSTING

- Failure diagnosis / outcome: residual ranking did not convert into safe positive gain above B6
- Hypothesis: Small nonlinear interactions improve the legal residual value estimate.
- Change: small depth-controlled HistGradientBoosting heads
- Grouped OOF Delta BA vs B6: -0.000288
- Subject-bootstrap CI95: [-0.000865, +0.000192]
- Action rate: 0.001058
- Rescue precision: 0.363636
- Harm rate: 0.636364
- Oracle recovery: -0.003356
- Decision: ABANDON

All thresholds and model settings were selected on inner calibration subjects;
outer held-out subjects were not used for selection. This is exploratory OOF
evidence, not an untouched confirmation.

`OUTER_TEST_USED=false`
