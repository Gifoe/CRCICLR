# SIRE-EEG diagnostic-guided architecture selection: development replay

This is not a strictly prospective independent confirmation. At the user's direction, B0 is the historical frozen Full checkpoint, which the existing multiseed protocol explicitly marks non-matched. Previous outcome exposure cannot be excluded.
The selector and five fold-level choices were frozen and SHA-verified before this program loaded outer-development data. No neural retraining or final-heldout use occurred.
The 40 outer-development subjects were disjoint across folds. The reported subject bootstrap is conditional on five frozen fold-level choices; a five-fold cluster sensitivity is also supplied.
Random is the exact uniform-policy expectation. Oracle selects by fold-mean outer future BA and is a BA upper bound only; its attached F1/WSBA are not necessarily metric-wise upper bounds. Oracle is not deployable.

| Policy | Chosen architectures (fold0–4) | Future BA | Future Macro-F1 | WS-BA |
|---|---|---:|---:|---:|
| Diagnostic-guided | B1_SAME_SCALE_63/B2_SCALE_COLLAPSE/B2_SCALE_COLLAPSE/B2_SCALE_COLLAPSE/B0_FULL_EXISTING | 75.54 | 74.74 | 72.99 |
| Random expectation | uniform/uniform/uniform/uniform/uniform | 76.15 | 75.45 | 73.32 |
| Validation BA | B0_FULL_EXISTING/B0_FULL_EXISTING/B0_FULL_EXISTING/B0_FULL_EXISTING/B0_FULL_EXISTING | 78.62 | 78.02 | 75.72 |
| Fixed Full | B0_FULL_EXISTING/B0_FULL_EXISTING/B0_FULL_EXISTING/B0_FULL_EXISTING/B0_FULL_EXISTING | 78.62 | 78.02 | 75.72 |
| Oracle | B0_FULL_EXISTING/B0_FULL_EXISTING/B0_FULL_EXISTING/B0_FULL_EXISTING/B0_FULL_EXISTING | 78.62 | 78.02 | 75.72 |

## Fold-level decisions

| Fold | Diagnostic choice | Validation-BA choice | Development reason | Outer BA | Outer WS-BA |
|---:|---|---|---|---:|---:|
| 0 | B1_SAME_SCALE_63 | B0_FULL_EXISTING | admissible; highest complement-retained WSBA | 81.12 | 78.79 |
| 1 | B2_SCALE_COLLAPSE | B0_FULL_EXISTING | admissible; highest complement-retained WSBA | 77.71 | 76.38 |
| 2 | B2_SCALE_COLLAPSE | B0_FULL_EXISTING | admissible; highest complement-retained WSBA | 67.67 | 67.12 |
| 3 | B2_SCALE_COLLAPSE | B0_FULL_EXISTING | admissible; highest complement-retained WSBA | 73.42 | 67.92 |
| 4 | B0_FULL_EXISTING | B0_FULL_EXISTING | admissible; highest complement-retained WSBA | 77.79 | 74.75 |

Paired biological-subject 20,000-draw CIs: `POLICY_PAIRED_EFFECTS.csv`. Fold-cluster sensitivity: `POLICY_FOLD_SENSITIVITY.csv`.
Protected PEEH/PSWA were admissibility constraints; complement-retained probe utility chose among admissible candidates. These diagnostics are not a scalar quality ranking and the outcome does not establish a causal mechanism.
A future P4 independent dataset with this unmodified selector is required for genuinely prospective confirmation.
