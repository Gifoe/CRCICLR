# Final EEG Conformer / FBCNet report

| Model | Task | Future BA [95% CI] | Future Macro-F1 [95% CI] | WS-BA [95% CI] | Params |
|---|---|---:|---:|---:|---:|
| EEGConformer | OpenBMI_MI | 71.35% [66.08, 77.16] | 70.76% [65.33, 76.72] | 67.43% [63.63, 71.71] | 853,506 |
| EEGConformer | OpenBMI_ERP | 79.47% [74.71, 84.26] | 73.64% [69.16, 78.35] | 76.35% [72.28, 80.68] | 341,506 |
| EEGConformer | OpenBMI_SSVEP | 82.00% [74.25, 89.53] | 81.17% [73.05, 89.06] | 79.92% [72.04, 87.51] | 853,572 |
| EEGConformer | WBCIC_MI | 76.04% [68.80, 82.47] | 75.35% [67.59, 82.10] | 68.11% [59.46, 76.94] | 847,106 |
| FBCNet | OpenBMI_MI | 67.60% [62.06, 73.69] | 65.31% [59.12, 72.09] | 64.60% [59.72, 70.21] | 21,026 |
| FBCNet | OpenBMI_ERP | 78.80% [74.71, 83.03] | 71.54% [68.04, 75.40] | 75.61% [71.65, 79.91] | 21,026 |
| FBCNet | OpenBMI_SSVEP | 80.55% [70.80, 89.27] | 79.57% [69.15, 88.82] | 77.51% [68.30, 85.67] | 23,332 |
| FBCNet | WBCIC_MI | 59.60% [55.06, 64.20] | 57.28% [52.67, 62.19] | 56.47% [53.27, 59.55] | 19,874 |

All eight cells contain 5 folds x 3 seeds (15 selected checkpoints).
Future-session BA/F1 and WS-BA are subject-equal; 95% CIs use 20,000 biological-subject bootstrap draws.
Fold/seed estimates are averaged within a subject/session before any bootstrap; WS-BA is the within-subject minimum over sessions.

Formal SIRE-EEG paired comparison: UNAVAILABLE_FORMAL_SIRE_SOURCE_NOT_PROVIDED. The different local LiteBN is not substituted.

MACs: not fully counted, because the available profiler does not account for the complete fixed filter bank and all attention operations.

Frozen split/normalizer and checkpoint hashes are in RUN_MANIFEST.csv and CHECKPOINT_AUDIT.csv.
Final WBCIC cohort is the true-outer 10, not the V8 internal cohort. No heldout training or checkpoint selection occurred.
