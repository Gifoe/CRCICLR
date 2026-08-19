# Iteration 007: I007_CONDITIONAL_ACTION_HGB

- Failure diagnosis / outcome: residual ranking did not convert into safe positive gain above B6
- Hypothesis: Small nonlinear action-specific conditional heads may capture the moderate ERASE rescue signal without unrestricted model search.
- Change: action-specific conditional depth-controlled HistGradientBoosting rescue heads with the same finite menu
- Grouped OOF Delta BA vs B6: -0.000673
- Subject-bootstrap CI95: [-0.002308, +0.001058]
- Action rate: 0.007404
- Rescue precision: 0.454545
- Harm rate: 0.545455
- Oracle recovery: -0.007830
- Decision: ABANDON

All thresholds and model settings were selected on inner calibration subjects;
outer held-out subjects were not used for selection. This is exploratory OOF
evidence, not an untouched confirmation.

`OUTER_TEST_USED=false`
