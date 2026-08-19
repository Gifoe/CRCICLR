# Iteration 002: M1_CONTEXTUAL_POSITIVE_POOL

- Diagnosis: grouped improvement with positive subject-bootstrap lower bound
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M1_CONTEXTUAL_POSITIVE_POOL`
- Features: `six frozen KEEP tokens + availability + session`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.850385`
- Delta vs B_STRONG: `+0.394 pp`
- Subject bootstrap CI95: `[+0.202, +0.596] pp`
- Macro-F1 / NLL / Brier: `0.848933` / `0.352393` / `0.109143`
- Switch rate: `1.452%`
- Rescue / harm: `96` / `55`
- Worst-subject delta: `-1.000 pp`
- Positive-subject fraction: `0.538`
- Result: `KEEP`
- `OUTER_TEST_USED=false`
