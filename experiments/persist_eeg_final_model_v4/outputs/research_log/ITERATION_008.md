# Iteration 008: M1_DEEPSETS_KEEP

- Diagnosis: gain is not robustly above the static ensemble
- Hypothesis: test whether this information/architecture family converts frozen expert diversity into unseen-subject gain
- Architecture: `M1_DEEPSETS_KEEP`
- Features: `six frozen KEEP tokens + availability + session`
- Validation: five outer subject folds with disjoint model-fit/calibration/heldout subjects
- Mean subject BA: `0.847308`
- Delta vs B_STRONG: `+0.087 pp`
- Subject bootstrap CI95: `[-0.154, +0.317] pp`
- Macro-F1 / NLL / Brier: `0.845647` / `0.355716` / `0.110050`
- Switch rate: `2.163%`
- Rescue / harm: `117` / `108`
- Worst-subject delta: `-2.000 pp`
- Positive-subject fraction: `0.519`
- Result: `MODIFY`
- `OUTER_TEST_USED=false`
