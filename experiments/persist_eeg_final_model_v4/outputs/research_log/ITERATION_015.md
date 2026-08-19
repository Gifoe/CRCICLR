# Iteration 015: M1_KEEP_SUMMARY_LINEAR

- Diagnosis: gain is not robustly above the static ensemble
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M1_KEEP_SUMMARY_LINEAR`
- Features: `six frozen KEEP tokens + availability + session`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.845481`
- Delta vs B_STRONG: `-0.096 pp`
- Subject bootstrap CI95: `[-0.404, +0.269] pp`
- Macro-F1 / NLL / Brier: `0.844420` / `0.354720` / `0.110472`
- Switch rate: `2.462%`
- Rescue / harm: `123` / `133`
- Worst-subject delta: `-3.500 pp`
- Positive-subject fraction: `0.231`
- Result: `ABANDON`
- `OUTER_TEST_USED=false`
