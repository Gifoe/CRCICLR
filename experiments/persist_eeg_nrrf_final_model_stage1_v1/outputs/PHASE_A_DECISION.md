# NRRF-v1 Phase-A Decision

| Metric | OpenBMI | WBCIC |
|---|---:|---:|
| EEGNet BA | 0.758 | 0.783 |
| LiteBN BA | 0.791 | 0.787 |
| Frozen LOGIT50 BA | 0.800 | 0.794 |
| JOINT-CE BA | 0.799 | 0.746 |
| NRRF BA | 0.797 | 0.738 |
| NRRF gain vs EEGNet (pp) | 3.950 | -4.451 |
| NRRF gain vs LOGIT50 (pp) | -0.275 | -5.582 |
| Median subject gain (pp) | 3.000 | -3.500 |
| Bootstrap CI (pp) | [+2.350, +5.675] | [-7.355, -1.887] |
| Positive folds | 5 | 2 |
| Harm ≤ −1 pp | 0.150 | 0.613 |

WBCIC NRRF − JOINT-CE mean BA: -0.810 pp.
WBCIC harm reduction versus JOINT-CE: -3.226 pp.

## Terminal

**NRRF_CONSTRUCTIVE_FAIL_STOP**

No coefficient, alpha, architecture, batch rule, or epoch budget was changed after outer-development outcomes were produced.
