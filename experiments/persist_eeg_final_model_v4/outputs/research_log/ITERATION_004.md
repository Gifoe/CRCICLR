# Iteration 004: M1_MASKED_POOL_CONFIG_AVG

- Diagnosis: grouped improvement with positive subject-bootstrap lower bound
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M1_MASKED_POOL_CONFIG_AVG`
- Features: `six frozen KEEP tokens + availability + session`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.849808`
- Delta vs B_STRONG: `+0.337 pp`
- Subject bootstrap CI95: `[+0.183, +0.500] pp`
- Macro-F1 / NLL / Brier: `0.848328` / `0.353331` / `0.109478`
- Switch rate: `0.913%`
- Rescue / harm: `65` / `30`
- Worst-subject delta: `-1.000 pp`
- Positive-subject fraction: `0.519`
- Result: `KEEP`
- `OUTER_TEST_USED=false`
