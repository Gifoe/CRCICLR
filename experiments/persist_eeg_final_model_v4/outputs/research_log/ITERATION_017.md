# Iteration 017: M1_B4_THRESHOLD_ONLY

- Diagnosis: gain is not robustly above the static ensemble
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M1_B4_THRESHOLD_ONLY`
- Features: `six frozen KEEP tokens + availability + session`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.845096`
- Delta vs B_STRONG: `-0.135 pp`
- Subject bootstrap CI95: `[-0.433, +0.144] pp`
- Macro-F1 / NLL / Brier: `0.843632` / `0.370869` / `0.115570`
- Switch rate: `1.981%`
- Rescue / harm: `96` / `110`
- Worst-subject delta: `-3.500 pp`
- Positive-subject fraction: `0.308`
- Result: `ABANDON`
- `OUTER_TEST_USED=false`
