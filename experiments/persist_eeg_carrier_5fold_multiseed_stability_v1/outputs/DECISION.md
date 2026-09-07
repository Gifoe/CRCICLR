# LiteBN 5-fold x 3-seed stability decision

| Metric | OpenBMI | WBCIC |
|---|---:|---:|
| 5-fold mean delta (pp) | 2.708 | -0.358 |
| median subject delta (pp) | 2.500 | 0.167 |
| bootstrap CI | [+1.325, +4.217] | [-1.724, +0.951] |
| positive folds | 5.000 | 2.000 |
| positive seeds | 3.000 | 1.000 |

Final terminal: **LITEBN_UNSTABLE_MULTIFACTOR**

1. Is LiteBN meaningfully better on OpenBMI? YES
2. Is LiteBN meaningfully better on WBCIC? NO
3. Same-sign across datasets? NO
4. Positive folds: OpenBMI 5/5; WBCIC 2/5
5. Positive seed means: OpenBMI 3/3; WBCIC 1/3
6. Fold vs seed variance: OpenBMI fold SD 1.619 pp, seed SD 0.741 pp; WBCIC fold SD 2.113 pp, seed SD 0.529 pp
7. Consistently negative folds: see SIGN_STABILITY.json.
8. Repeated sign-changing folds: see SIGN_STABILITY.json.
9. Small-subject concentration: see OUTLIER_INFLUENCE.json; primary estimate is unchanged.
10. Difficulty interaction: see SUBJECT_DIFFICULTY_INTERACTION.json.
11. Fixed epoch60 conclusion: see CHECKPOINT_SELECTION_DIAGNOSTIC.json.
12. Inner-val vs outer association: see CHECKPOINT_SELECTION_DIAGNOSTIC.json.
13. Dominant instability source: determined from fold/seed/checkpoint diagnostics above.
14. Historical WBCIC -3.45 pp is evaluated as context only against the fresh split; it is not numerically pooled.

Recommended next scientific action: do not execute another model in this experiment; use the terminal to decide whether carrier development should stop or move to the specified diagnostic direction.
