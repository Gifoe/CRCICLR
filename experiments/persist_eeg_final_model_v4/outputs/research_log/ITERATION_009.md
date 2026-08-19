# Iteration 009: M1_KEEP_RAW_LINEAR

- Diagnosis: gain is not robustly above the static ensemble
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M1_KEEP_RAW_LINEAR`
- Features: `six frozen KEEP tokens + availability + session`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.847115`
- Delta vs B_STRONG: `+0.067 pp`
- Subject bootstrap CI95: `[-0.327, +0.452] pp`
- Macro-F1 / NLL / Brier: `0.845141` / `0.357888` / `0.111415`
- Switch rate: `3.163%`
- Rescue / harm: `168` / `161`
- Worst-subject delta: `-4.000 pp`
- Positive-subject fraction: `0.404`
- Result: `MODIFY`
- `OUTER_TEST_USED=false`
