# Iteration 002: M2_ENSEMBLE_DISAGREEMENT_RULE

- Failure diagnosis / outcome: residual ranking did not convert into safe positive gain above B6
- Hypothesis: Run disagreement may expose ensemble errors that an action consensus can repair.
- Change: deterministic disagreement rule
- Grouped OOF Delta BA vs B6: -0.001058
- Subject-bootstrap CI95: [-0.002788, +0.000481]
- Action rate: 0.006442
- Rescue precision: 0.417910
- Harm rate: 0.582090
- Oracle recovery: -0.012304
- Decision: ABANDON

All thresholds and model settings were selected on inner calibration subjects;
outer held-out subjects were not used for selection. This is exploratory OOF
evidence, not an untouched confirmation.

`OUTER_TEST_USED=false`
