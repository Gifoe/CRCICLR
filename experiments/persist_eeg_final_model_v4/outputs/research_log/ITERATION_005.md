# Iteration 005: M1_MASKED_POSITIVE_POOL

- Diagnosis: gain is not robustly above the static ensemble
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M1_MASKED_POSITIVE_POOL`
- Features: `six frozen KEEP tokens + availability + session`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.849135`
- Delta vs B_STRONG: `+0.269 pp`
- Subject bootstrap CI95: `[-0.058, +0.587] pp`
- Macro-F1 / NLL / Brier: `0.847550` / `0.355408` / `0.110161`
- Switch rate: `2.038%`
- Rescue / harm: `120` / `92`
- Worst-subject delta: `-2.500 pp`
- Positive-subject fraction: `0.519`
- Result: `MODIFY`
- `OUTER_TEST_USED=false`
