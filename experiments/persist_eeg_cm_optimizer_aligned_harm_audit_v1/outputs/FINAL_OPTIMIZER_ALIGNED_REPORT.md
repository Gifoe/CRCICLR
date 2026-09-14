# Optimizer-aligned LiteBN C/M harm-audit report

Terminal: `NO_OPTIMIZER_ALIGNED_DUAL_MI_SIGNAL`.

Only canonical inner-training subjects and source sessions were accessed. No model training, development outer, exposed benchmark, heldout, or sealed test access occurred.

## Primary table (K=1/2/4; pass is evaluated only at K=4)

| Dataset | Space | K | same AUROC | AUROC CI | same Spearman | Spearman CI | different AUROC | AUROC advantage | advantage CI | Spearman advantage | advantage CI | Pass |
|---|---|---:|---:|---|---:|---|---:|---:|---|---:|---|---|
| OpenBMI | C | 1 | 0.6618 | [0.5600, 0.7587] | 0.2395 | [0.0499, 0.4117] | 0.6064 | +0.0553 | [-0.0584, +0.1723] | +0.0842 | [-0.1320, +0.2946] | diagnostic |
| OpenBMI | C | 2 | 0.6601 | [0.5576, 0.7584] | 0.2702 | [0.0789, 0.4458] | 0.6509 | +0.0091 | [-0.0837, +0.1019] | +0.0340 | [-0.1313, +0.1944] | diagnostic |
| OpenBMI | C | 4 | 0.7376 | [0.6392, 0.8295] | 0.4191 | [0.2467, 0.5733] | 0.5908 | +0.1468 | [+0.0431, +0.2509] | +0.3173 | [+0.1360, +0.4961] | True |
| OpenBMI | CM | 1 | 0.6635 | [0.5737, 0.7469] | 0.2895 | [0.1069, 0.4587] | 0.5587 | +0.1048 | [-0.0122, +0.2181] | +0.2207 | [+0.0076, +0.4368] | diagnostic |
| OpenBMI | CM | 2 | 0.6780 | [0.5937, 0.7605] | 0.3574 | [0.1831, 0.5188] | 0.5539 | +0.1241 | [+0.0276, +0.2199] | +0.2755 | [+0.0772, +0.4759] | diagnostic |
| OpenBMI | CM | 4 | 0.6525 | [0.5666, 0.7396] | 0.3517 | [0.1994, 0.4993] | 0.5348 | +0.1177 | [+0.0105, +0.2265] | +0.3033 | [+0.1039, +0.4975] | True |
| OpenBMI | M | 1 | 0.6448 | [0.5562, 0.7269] | 0.2856 | [0.1004, 0.4558] | 0.5746 | +0.0701 | [-0.0387, +0.1788] | +0.2094 | [+0.0022, +0.4181] | diagnostic |
| OpenBMI | M | 2 | 0.6810 | [0.5995, 0.7588] | 0.3541 | [0.1768, 0.5135] | 0.5705 | +0.1105 | [+0.0028, +0.2149] | +0.2781 | [+0.0718, +0.4784] | diagnostic |
| OpenBMI | M | 4 | 0.6607 | [0.5781, 0.7372] | 0.3393 | [0.1872, 0.4802] | 0.5500 | +0.1107 | [+0.0011, +0.2200] | +0.2762 | [+0.0729, +0.4733] | True |
| WBCIC | C | 1 | 0.7381 | [0.6418, 0.8324] | 0.4689 | [0.3144, 0.6033] | 0.6171 | +0.1211 | [-0.0043, +0.2443] | +0.2106 | [+0.0290, +0.3968] | diagnostic |
| WBCIC | C | 2 | 0.7334 | [0.6367, 0.8261] | 0.4661 | [0.2641, 0.6399] | 0.6246 | +0.1088 | [-0.0020, +0.2162] | +0.2194 | [+0.0181, +0.4043] | diagnostic |
| WBCIC | C | 4 | 0.7692 | [0.6699, 0.8627] | 0.4941 | [0.2807, 0.6743] | 0.7090 | +0.0602 | [-0.0346, +0.1580] | +0.1082 | [-0.0881, +0.3026] | False |
| WBCIC | CM | 1 | 0.5494 | [0.4363, 0.6627] | 0.1645 | [-0.0617, 0.3738] | 0.5016 | +0.0478 | [-0.0890, +0.1776] | +0.1985 | [-0.0594, +0.4249] | diagnostic |
| WBCIC | CM | 2 | 0.5538 | [0.4380, 0.6615] | 0.1848 | [-0.0547, 0.4056] | 0.4612 | +0.0926 | [-0.0350, +0.2247] | +0.3327 | [+0.0554, +0.5975] | diagnostic |
| WBCIC | CM | 4 | 0.6059 | [0.4935, 0.7128] | 0.2189 | [-0.0043, 0.4240] | 0.4857 | +0.1202 | [-0.0143, +0.2521] | +0.3144 | [+0.0592, +0.5535] | False |
| WBCIC | M | 1 | 0.5552 | [0.4359, 0.6725] | 0.1630 | [-0.0585, 0.3817] | 0.4652 | +0.0901 | [-0.0618, +0.2322] | +0.2506 | [-0.0127, +0.4825] | diagnostic |
| WBCIC | M | 2 | 0.5724 | [0.4613, 0.6790] | 0.1855 | [-0.0456, 0.4004] | 0.4230 | +0.1494 | [+0.0196, +0.2780] | +0.3757 | [+0.1110, +0.6279] | diagnostic |
| WBCIC | M | 4 | 0.6151 | [0.5027, 0.7214] | 0.2094 | [-0.0130, 0.4156] | 0.4188 | +0.1963 | [+0.0642, +0.3236] | +0.3591 | [+0.1101, +0.6007] | False |

## K effect

| Dataset | Space | K1 AUROC | K2 AUROC | K4 AUROC | K1 rho | K2 rho | K4 rho |
|---|---|---:|---:|---:|---:|---:|---:|
| OpenBMI | C | 0.6618 | 0.6601 | 0.7376 | 0.2395 | 0.2702 | 0.4191 |
| OpenBMI | CM | 0.6635 | 0.6780 | 0.6525 | 0.2895 | 0.3574 | 0.3517 |
| OpenBMI | M | 0.6448 | 0.6810 | 0.6607 | 0.2856 | 0.3541 | 0.3393 |
| WBCIC | C | 0.7381 | 0.7334 | 0.7692 | 0.4689 | 0.4661 | 0.4941 |
| WBCIC | CM | 0.5494 | 0.5538 | 0.6059 | 0.1645 | 0.1848 | 0.2189 |
| WBCIC | M | 0.5552 | 0.5724 | 0.6151 | 0.1630 | 0.1855 | 0.2094 |

## Space decision

| Space | OpenBMI K4 pass | WBCIC K4 pass | Worst-dataset AUROC | Worst-dataset AUROC advantage | Dual-MI status |
|---|---|---|---:|---:|---|
| C | True | False | 0.7376 | +0.0602 | NOT_DUAL_MI_SUPPORTED |
| M | True | False | 0.6151 | +0.1107 | NOT_DUAL_MI_SUPPORTED |
| CM | True | False | 0.6059 | +0.1177 | NOT_DUAL_MI_SUPPORTED |

## Required answers

1. The audit tests the proposed mismatch directly; CM K=4 is higher than the old cosine AUROC on both datasets, which is suggestive but not causal proof of a partial mismatch effect.
2. For CM, optimizer-aligned K=4 AUROC exceeds the previous raw-cosine AUROC on both MI datasets; gate status remains `NOT_DUAL_MI_SUPPORTED`.
3. K=4 versus K=1 is descriptive: C=higher on both datasets; M=higher on both datasets; CM=mixed or lower.
4. Same-subject information versus the different-subject control: C=does not establish both positive advantage CIs across datasets; M=both advantage CIs are positive on both datasets; CM=does not establish both positive advantage CIs across datasets.
5. Generic multi-subject consensus: C=has positive K=4 AUROC and Spearman bootstrap support on both datasets; M=does not have that support on both datasets; CM=does not have that support on both datasets. It remains secondary and does not override target-subject gates.
6. The strongest worst-dataset K=4 same-AUROC is `C` (0.7376).
7. M alone passes both MI datasets: `False`.
8. CM passes both MI datasets: `False`.
9. A later guarded residual-training experiment is not justified by this audit; this branch does not train one.
10. The gradient-guard route should be stopped under the locked dual-MI rule.

All confidence intervals are 10,000-draw target-biological-subject cluster bootstrap intervals.
