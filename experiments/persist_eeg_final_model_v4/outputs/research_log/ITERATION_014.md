# Iteration 014: M2_BOUNDED_RESIDUAL

- Diagnosis: gain is not robustly above the static ensemble
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M2_BOUNDED_RESIDUAL`
- Features: `KEEP_ACTION bounded offset`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.845577`
- Delta vs B_STRONG: `-0.087 pp`
- Subject bootstrap CI95: `[-0.519, +0.327] pp`
- Macro-F1 / NLL / Brier: `0.843856` / `0.351986` / `0.109589`
- Switch rate: `4.029%`
- Rescue / harm: `205` / `214`
- Worst-subject delta: `-4.500 pp`
- Positive-subject fraction: `0.423`
- Result: `ABANDON`
- `OUTER_TEST_USED=false`
