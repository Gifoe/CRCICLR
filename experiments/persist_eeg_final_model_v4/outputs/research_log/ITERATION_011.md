# Iteration 011: M1_KEEP_NO_SESSION_LINEAR

- Diagnosis: gain is not robustly above the static ensemble
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M1_KEEP_NO_SESSION_LINEAR`
- Features: `six frozen KEEP tokens + availability + session`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.846250`
- Delta vs B_STRONG: `-0.019 pp`
- Subject bootstrap CI95: `[-0.298, +0.269] pp`
- Macro-F1 / NLL / Brier: `0.844421` / `0.350504` / `0.108933`
- Switch rate: `2.962%`
- Rescue / harm: `153` / `155`
- Worst-subject delta: `-2.500 pp`
- Positive-subject fraction: `0.346`
- Result: `ABANDON`
- `OUTER_TEST_USED=false`
