# Iteration 003: M3_ACTION_MOVEMENT_LOGISTIC

- Failure diagnosis / outcome: residual ranking did not convert into safe positive gain above B6
- Hypothesis: Action movement and ensemble diversity predict rescue versus harm above B6.
- Change: regularized action-movement rescue/harm heads
- Grouped OOF Delta BA vs B6: -0.001154
- Subject-bootstrap CI95: [-0.003654, +0.001250]
- Action rate: 0.013269
- Rescue precision: 0.456522
- Harm rate: 0.543478
- Oracle recovery: -0.013423
- Decision: ABANDON

All thresholds and model settings were selected on inner calibration subjects;
outer held-out subjects were not used for selection. This is exploratory OOF
evidence, not an untouched confirmation.

`OUTER_TEST_USED=false`
