# Iteration 003: M1_MASKED_POOL_THR050

- Diagnosis: grouped improvement with positive subject-bootstrap lower bound
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M1_MASKED_POOL_THR050`
- Features: `six frozen KEEP tokens + availability + session`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.849904`
- Delta vs B_STRONG: `+0.346 pp`
- Subject bootstrap CI95: `[+0.058, +0.625] pp`
- Macro-F1 / NLL / Brier: `0.848239` / `0.352501` / `0.109173`
- Switch rate: `1.750%`
- Rescue / harm: `109` / `73`
- Worst-subject delta: `-2.000 pp`
- Positive-subject fraction: `0.519`
- Result: `KEEP`
- `OUTER_TEST_USED=false`
