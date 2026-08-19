# Iteration 001: M1_ENSEMBLE_CONFIDENCE_RULE

- Failure diagnosis / outcome: residual ranking did not convert into safe positive gain above B6
- Hypothesis: Low B6 margin plus a confident flipping action may isolate residual corrections.
- Change: deterministic confidence rule
- Grouped OOF Delta BA vs B6: -0.000673
- Subject-bootstrap CI95: [-0.002212, +0.000769]
- Action rate: 0.007596
- Rescue precision: 0.455696
- Harm rate: 0.544304
- Oracle recovery: -0.007830
- Decision: ABANDON

All thresholds and model settings were selected on inner calibration subjects;
outer held-out subjects were not used for selection. This is exploratory OOF
evidence, not an untouched confirmation.

`OUTER_TEST_USED=false`
