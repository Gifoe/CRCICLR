# Iteration 016: M1_B6_THRESHOLD_ONLY

- Diagnosis: gain is not robustly above the static ensemble
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M1_B6_THRESHOLD_ONLY`
- Features: `six frozen KEEP tokens + availability + session`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.845288`
- Delta vs B_STRONG: `-0.115 pp`
- Subject bootstrap CI95: `[-0.385, +0.144] pp`
- Macro-F1 / NLL / Brier: `0.843813` / `0.365357` / `0.113901`
- Switch rate: `1.846%`
- Rescue / harm: `90` / `102`
- Worst-subject delta: `-3.000 pp`
- Positive-subject fraction: `0.269`
- Result: `ABANDON`
- `OUTER_TEST_USED=false`
