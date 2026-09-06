# SDET Seed-0 Corrected Future-Session Evaluation

Protocol correction: previous TEST scoring pooled history + future sessions; this corrected replay uses future session only.

Frozen model/intervention retrained: NO
Hyperparameters changed: NO
Seed changed: NO
TEST subjects changed: NO
a_pop changed: NO
Normalization reestimated on TEST: NO

OpenBMI:
EEGNet future BA: 0.798571
SDET future BA: 0.799286
Delta BA: +0.071 pp
Macro-F1 delta: +0.000613
Improved / Tie / Harmed: 1 / 12 / 1
Prediction flips: 5 / 1400
Beneficial flips: 3
Harmful flips: 2

WBCIC:
EEGNet future BA: 0.806000
SDET future BA: 0.805500
Delta BA: -0.050 pp
Macro-F1 delta: -0.000381
Improved / Tie / Harmed: 1 / 7 / 2
Prediction flips: 7 / 2000
Beneficial flips: 3
Harmful flips: 4

Both datasets positive: False
At least one dataset negative: True
Subjects with zero prediction change: OpenBMI 11/14; WBCIC 7/10
Probability shift with unchanged argmax: OpenBMI False; WBCIC False
Improvement concentration: inspect the paired subject table; no target-specific adapter was used.
New protocol invalidity: false.

Previous SEED0_DECISION.md is superseded for scientific interpretation because it pooled history and future sessions in TEST scoring.

Terminal state: **SDET_SEED0_FUTURE_MIXED_STOP**
