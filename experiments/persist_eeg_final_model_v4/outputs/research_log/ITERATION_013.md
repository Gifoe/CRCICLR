# Iteration 013: M3_KEEP_ACTION_PERSIST_LINEAR

- Diagnosis: gain is not robustly above the static ensemble
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M3_KEEP_ACTION_PERSIST_LINEAR`
- Features: `KEEP_ACTION_PERSIST`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.845769`
- Delta vs B_STRONG: `-0.067 pp`
- Subject bootstrap CI95: `[-0.423, +0.279] pp`
- Macro-F1 / NLL / Brier: `0.844302` / `0.353676` / `0.110201`
- Switch rate: `2.337%`
- Rescue / harm: `118` / `125`
- Worst-subject delta: `-4.000 pp`
- Positive-subject fraction: `0.385`
- Result: `ABANDON`
- `OUTER_TEST_USED=false`
