# Iteration 019: M2_KEEP_ACTION_HGB

- Diagnosis: gain is not robustly above the static ensemble
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M2_KEEP_ACTION_HGB`
- Features: `KEEP_ACTION`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.842115`
- Delta vs B_STRONG: `-0.433 pp`
- Subject bootstrap CI95: `[-0.846, -0.038] pp`
- Macro-F1 / NLL / Brier: `0.840191` / `0.355476` / `0.110313`
- Switch rate: `3.875%`
- Rescue / harm: `179` / `224`
- Worst-subject delta: `-4.500 pp`
- Positive-subject fraction: `0.288`
- Result: `ABANDON`
- `OUTER_TEST_USED=false`
