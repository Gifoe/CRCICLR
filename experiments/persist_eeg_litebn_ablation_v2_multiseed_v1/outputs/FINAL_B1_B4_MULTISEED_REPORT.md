# LiteBN B1--B4 architecture study (seeds 0, 1, 2)

B0 was not trained for any seed by explicit user instruction. Seed-0 B1--B4 checkpoints were reused exactly and seeds 1/2 were newly trained; no heldout labels entered training or selection. The official frozen LiteBN is a non-matched descriptive reference only.

OpenBMI V8 is an internal-heldout diagnostic cohort. The WBCIC true-outer cohort has already been accessed for prior ablation diagnostics and is therefore an exposed benchmark, not untouched final confirmation.

Fold and seed repetitions were averaged within each biological subject and session; WS-BA is the minimum of those session means. The 20,000-draw bootstrap resamples biological subjects.

| Variant | MI BA | ERP BA | SSVEP BA | WBCIC BA | Mean BA | Mean WS-BA |
|---|---:|---:|---:|---:|---:|---:|
| OFFICIAL_FINAL_LITEBN_REFERENCE | 0.7490 | 0.8480 | 0.9102 | 0.8047 | 0.8280 | 0.7895 |
| B1_SAME_SCALE_63 | 0.7263 | 0.8520 | 0.9026 | 0.8017 | 0.8207 | 0.7830 |
| B2_SCALE_COLLAPSE | 0.6994 | 0.8497 | 0.9029 | 0.7964 | 0.8121 | 0.7734 |
| B3_SINGLE_SPATIAL_BASIS | 0.7112 | 0.8504 | 0.9084 | 0.8039 | 0.8185 | 0.7807 |
| B4_ONE_STAGE_BACKEND | 0.7119 | 0.8420 | 0.8654 | 0.7859 | 0.8013 | 0.7611 |

## Paired descriptive effects versus official frozen LiteBN

| Variant | Task | Delta BA pp [95% CI] | Delta Macro-F1 pp [95% CI] | Delta WS-BA pp [95% CI] |
|---|---|---:|---:|---:|
| B1_SAME_SCALE_63 | OpenBMI_MI | -2.271 [-3.671, -1.062] | -2.052 [-3.695, -0.571] | -2.243 [-3.738, -0.938] |
| B2_SCALE_COLLAPSE | OpenBMI_MI | -4.967 [-6.557, -3.557] | -4.995 [-6.859, -3.304] | -4.771 [-6.505, -3.238] |
| B3_SINGLE_SPATIAL_BASIS | OpenBMI_MI | -3.781 [-5.319, -2.390] | -3.703 [-5.446, -2.062] | -3.190 [-5.224, -1.176] |
| B4_ONE_STAGE_BACKEND | OpenBMI_MI | -3.714 [-5.162, -2.419] | -3.871 [-5.393, -2.487] | -3.343 [-4.852, -2.067] |
| B1_SAME_SCALE_63 | OpenBMI_ERP | +0.407 [+0.078, +0.815] | +0.799 [+0.282, +1.571] | +0.664 [+0.232, +1.152] |
| B2_SCALE_COLLAPSE | OpenBMI_ERP | +0.171 [-0.148, +0.516] | +0.365 [-0.088, +0.922] | +0.363 [+0.010, +0.722] |
| B3_SINGLE_SPATIAL_BASIS | OpenBMI_ERP | +0.244 [-0.079, +0.568] | +0.529 [+0.060, +1.112] | +0.495 [+0.013, +1.028] |
| B4_ONE_STAGE_BACKEND | OpenBMI_ERP | -0.594 [-0.911, -0.258] | -0.698 [-1.183, -0.164] | -0.518 [-0.818, -0.233] |
| B1_SAME_SCALE_63 | OpenBMI_SSVEP | -0.767 [-1.395, -0.262] | -0.883 [-1.581, -0.294] | -0.619 [-1.462, +0.210] |
| B2_SCALE_COLLAPSE | OpenBMI_SSVEP | -0.733 [-1.619, -0.119] | -0.693 [-1.604, -0.057] | -0.776 [-1.581, -0.081] |
| B3_SINGLE_SPATIAL_BASIS | OpenBMI_SSVEP | -0.181 [-0.895, +0.329] | -0.230 [-0.955, +0.278] | -0.062 [-0.581, +0.343] |
| B4_ONE_STAGE_BACKEND | OpenBMI_SSVEP | -4.481 [-7.029, -2.195] | -4.807 [-7.559, -2.341] | -5.629 [-8.629, -2.971] |
| B1_SAME_SCALE_63 | WBCIC_MI | -0.300 [-0.990, +0.373] | -0.234 [-0.902, +0.425] | -0.373 [-1.020, +0.233] |
| B2_SCALE_COLLAPSE | WBCIC_MI | -0.837 [-1.643, -0.130] | -0.889 [-1.790, -0.120] | -1.257 [-2.213, -0.507] |
| B3_SINGLE_SPATIAL_BASIS | WBCIC_MI | -0.083 [-0.740, +0.543] | +0.025 [-0.673, +0.700] | -0.730 [-1.903, +0.123] |
| B4_ONE_STAGE_BACKEND | WBCIC_MI | -1.883 [-2.953, -0.963] | -1.847 [-3.002, -0.899] | -1.867 [-2.867, -0.840] |
