# Iteration 020: M3_KEEP_ACTION_PERSIST_HGB

- Diagnosis: gain is not robustly above the static ensemble
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M3_KEEP_ACTION_PERSIST_HGB`
- Features: `KEEP_ACTION_PERSIST`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.841923`
- Delta vs B_STRONG: `-0.452 pp`
- Subject bootstrap CI95: `[-0.798, -0.106] pp`
- Macro-F1 / NLL / Brier: `0.839971` / `0.356252` / `0.110755`
- Switch rate: `3.913%`
- Rescue / harm: `180` / `227`
- Worst-subject delta: `-4.000 pp`
- Positive-subject fraction: `0.385`
- Result: `ABANDON`
- `OUTER_TEST_USED=false`
