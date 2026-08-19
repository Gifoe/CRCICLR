# Iteration 012: M2_KEEP_ACTION_LINEAR

- Diagnosis: gain is not robustly above the static ensemble
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M2_KEEP_ACTION_LINEAR`
- Features: `KEEP_ACTION`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.845962`
- Delta vs B_STRONG: `-0.048 pp`
- Subject bootstrap CI95: `[-0.452, +0.346] pp`
- Macro-F1 / NLL / Brier: `0.844482` / `0.352802` / `0.109765`
- Switch rate: `2.394%`
- Rescue / harm: `122` / `127`
- Worst-subject delta: `-4.500 pp`
- Positive-subject fraction: `0.404`
- Result: `ABANDON`
- `OUTER_TEST_USED=false`
