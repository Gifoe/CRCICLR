# Iteration 007: M1_DYNAMIC_KEEP_HGB

- Diagnosis: gain is not robustly above the static ensemble
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M1_DYNAMIC_KEEP_HGB`
- Features: `six frozen KEEP tokens + availability + session`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.848173`
- Delta vs B_STRONG: `+0.173 pp`
- Subject bootstrap CI95: `[-0.163, +0.500] pp`
- Macro-F1 / NLL / Brier: `0.846998` / `0.355008` / `0.109792`
- Switch rate: `3.173%`
- Rescue / harm: `174` / `156`
- Worst-subject delta: `-2.500 pp`
- Positive-subject fraction: `0.462`
- Result: `MODIFY`
- `OUTER_TEST_USED=false`
