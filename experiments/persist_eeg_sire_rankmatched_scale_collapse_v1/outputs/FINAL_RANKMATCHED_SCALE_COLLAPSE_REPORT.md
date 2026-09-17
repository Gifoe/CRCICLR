# Same-checkpoint rank-matched ScaleCollapse control

Frozen official Full SIRE checkpoints and original train-only normalizers; no neural training or adaptation.
ScaleCollapse-fixed is an intervention on Full, not the separately trained B2 architecture.

| Task | Metric | Intact | ScaleCollapse | Random rank-16 | Scale harm pp | Random harm pp | Excess harm pp [95% CI] | Scale harm percentile |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OpenBMI_MI | BA | 74.90 | 63.35 | 62.25 | +11.55 | +12.66 | -1.11 [-2.90, +0.63] | 31.0 |
| OpenBMI_MI | macro_F1 | 74.04 | 59.94 | 57.24 | +14.10 | +16.80 | -2.70 [-5.36, -0.10] | 27.0 |
| OpenBMI_MI | WS_BA | 71.44 | 60.21 | 59.02 | +11.23 | +12.43 | -1.19 [-2.95, +0.45] | 27.0 |
| OpenBMI_ERP | BA | 84.80 | 67.54 | 68.04 | +17.25 | +16.75 | +0.50 [-0.42, +1.35] | 62.0 |
| OpenBMI_ERP | macro_F1 | 80.49 | 68.69 | 69.40 | +11.80 | +11.09 | +0.71 [-0.09, +1.48] | 65.0 |
| OpenBMI_ERP | WS_BA | 81.52 | 65.36 | 65.53 | +16.17 | +16.00 | +0.17 [-0.76, +1.11] | 54.0 |
| OpenBMI_SSVEP | BA | 91.02 | 73.38 | 72.32 | +17.65 | +18.70 | -1.06 [-2.21, +0.19] | 44.0 |
| OpenBMI_SSVEP | macro_F1 | 90.91 | 71.59 | 70.08 | +19.33 | +20.83 | -1.51 [-2.80, -0.13] | 43.0 |
| OpenBMI_SSVEP | WS_BA | 89.96 | 69.59 | 69.35 | +20.37 | +20.61 | -0.23 [-1.65, +1.24] | 51.0 |
| WBCIC_MI | BA | 80.47 | 66.48 | 66.02 | +13.99 | +14.45 | -0.46 [-1.13, +0.22] | 44.0 |
| WBCIC_MI | macro_F1 | 80.23 | 62.28 | 61.34 | +17.95 | +18.89 | -0.95 [-1.84, -0.05] | 43.0 |
| WBCIC_MI | WS_BA | 72.86 | 61.61 | 61.21 | +11.25 | +11.65 | -0.39 [-0.89, +0.10] | 41.0 |

Positive excess harm means the scale-aligned rank-16 collapse damages prediction more than the mean equal-rank random projection under the same frozen decoder.
The empirical percentile is descriptive, not a hypothesis-test p-value.
BA, Macro-F1 and WS-BA are separately reported; WS-BA takes the within-subject session minimum after 15 checkpoint repetitions for each projector.
Previous development replay showed that the diagnostics should not be promoted to an architecture-ranking objective.
PERSIST motivates a general preservation constraint; SIRE instantiates one scale-specific hypothesis.

- OpenBMI_MI: Scale-aligned collapse has lower mean BA harm than generic rank-matched compression; there is no positive scale-specific evidence from this control.
- OpenBMI_ERP: The rank-matched control does not resolve whether collapse harm is specific to scale identity rather than low-rank compression.
- OpenBMI_SSVEP: Scale-aligned collapse has lower mean BA harm than generic rank-matched compression; there is no positive scale-specific evidence from this control.
- WBCIC_MI: Scale-aligned collapse has lower mean BA harm than generic rank-matched compression; there is no positive scale-specific evidence from this control.

Protected-rank details are in PROTECTED_RANK_SUMMARY.md. No selector retuning or manuscript edit was made.
