# Iteration 006: M1_DYNAMIC_KEEP_LINEAR

- Diagnosis: gain is not robustly above the static ensemble
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M1_DYNAMIC_KEEP_LINEAR`
- Features: `six frozen KEEP tokens + availability + session`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.848846`
- Delta vs B_STRONG: `+0.240 pp`
- Subject bootstrap CI95: `[-0.058, +0.567] pp`
- Macro-F1 / NLL / Brier: `0.847230` / `0.350493` / `0.108899`
- Switch rate: `2.779%`
- Rescue / harm: `157` / `132`
- Worst-subject delta: `-2.000 pp`
- Positive-subject fraction: `0.462`
- Result: `MODIFY`
- `OUTER_TEST_USED=false`
