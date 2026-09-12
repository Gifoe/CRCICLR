# Exact LiteBN current-runtime replay control

Terminal label: **LARGE_RUNTIME_TRAJECTORY_DRIFT**.

1. Model: exact recovered historical LiteBN / CompactLite; no architecture change.
2. Initialization: all five expected hashes and full tensor replay checks passed.
3. Manifests: all five original stored OpenBMI-MI manifests matched exactly.
4. Normalizers: all five historical fold-specific normalizers matched.
5. Runtime: Windows, Torch 2.8.0+cu128, CUDA 12.8, GPU NVIDIA GeForce RTX 5090; full metadata is in `CURRENT_RUNTIME_METADATA.json`.
6. Historical Linux LiteBN outer BA: 0.79150.
7. Current-runtime exact LiteBN outer BA: 0.76750.
8. Runtime delta: -2.400 pp.
9. Fold runtime deltas (pp): -0.500, -5.500, -2.250, -1.875, -1.875.
10. Current-runtime internal heldout diagnostic BA: 0.73714.
11. Heldout delta: -0.771 pp; this is not an untouched or final test.
12-14. Same-runtime candidate recalibration:
- HE96: +0.025 pp vs current-runtime LiteBN (COMPATIBLE_CURRENT_RUNTIME).
- TSW: -0.150 pp vs current-runtime LiteBN (COMPATIBLE_CURRENT_RUNTIME).
- StaticScale: +0.025 pp vs current-runtime LiteBN (COMPATIBLE_CURRENT_RUNTIME).
15. Interpretation: LARGE_RUNTIME_TRAJECTORY_DRIFT.
16. Direct use of 79.15 as a current-runtime baseline is not scientifically justified without this matched control.
17. Future current-runtime architecture work should use the measured current-runtime LiteBN control.
18. New sealed test accessed: NO.

```text
MODEL = EXACT_HISTORICAL_LITEBN
TASK = OpenBMI_MI
SEED = 0
FOLDS = 5
ARCHITECTURE_CHANGE = NONE
LOSS_CHANGE = NONE
OPTIMIZER_CHANGE = NONE
CHECKPOINT_RULE_CHANGE = NONE
MANIFEST_CHANGE = NONE
NORMALIZER_CHANGE = NONE
INITIALIZATION_RECOVERY = EXACT
EXECUTION_RUNTIME = CURRENT_WINDOWS_TORCH_CUDA
NEW_SEALED_TEST_ACCESSED = NO
```
