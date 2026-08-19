# Iteration 001: M1_MASKED_POOL_SHRUNK_THR

- Diagnosis: grouped improvement with positive subject-bootstrap lower bound
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M1_MASKED_POOL_SHRUNK_THR`
- Features: `six frozen KEEP tokens + availability + session`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.850962`
- Delta vs B_STRONG: `+0.452 pp`
- Subject bootstrap CI95: `[+0.202, +0.721] pp`
- Macro-F1 / NLL / Brier: `0.849480` / `0.352939` / `0.109331`
- Switch rate: `1.279%`
- Rescue / harm: `90` / `43`
- Worst-subject delta: `-1.500 pp`
- Positive-subject fraction: `0.500`
- Result: `KEEP`
- `OUTER_TEST_USED=false`
