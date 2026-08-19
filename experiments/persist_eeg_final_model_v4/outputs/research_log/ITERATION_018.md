# Iteration 018: M3_BOUNDED_RESIDUAL_PERSIST

- Diagnosis: gain is not robustly above the static ensemble
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M3_BOUNDED_RESIDUAL_PERSIST`
- Features: `KEEP_ACTION_PERSIST bounded offset`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.843942`
- Delta vs B_STRONG: `-0.250 pp`
- Subject bootstrap CI95: `[-0.692, +0.173] pp`
- Macro-F1 / NLL / Brier: `0.842201` / `0.354419` / `0.110747`
- Switch rate: `3.865%`
- Rescue / harm: `188` / `214`
- Worst-subject delta: `-4.500 pp`
- Positive-subject fraction: `0.423`
- Result: `ABANDON`
- `OUTER_TEST_USED=false`
