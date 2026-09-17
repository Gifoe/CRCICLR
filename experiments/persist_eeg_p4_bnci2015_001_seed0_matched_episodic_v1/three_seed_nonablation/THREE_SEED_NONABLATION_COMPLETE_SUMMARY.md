# BNCI2015-001 three-seed non-ablation continuation

Seed 0 is reused unchanged. Seeds 1 and 2 trained only Full EEGNet and Full SIRE-EEG under the final recipe: fixed five subject-disjoint folds and fixed authoritative episode manifests; initialization RNG = seed; training RNG = 100000 + seed. No B1--B4 architecture ablation was run beyond seed 0.

| Seed | EEGNet S2 BA | SIRE S2 BA | EEGNet WS-BA | SIRE WS-BA |
|---:|---:|---:|---:|---:|
| 0 | 0.6350 | 0.6642 | 0.6233 | 0.6333 |
| 1 | 0.6471 | 0.6196 | 0.6142 | 0.5913 |
| 2 | 0.6563 | 0.6250 | 0.6267 | 0.6054 |

After averaging seeds within biological subject, EEGNet has S2 BA 0.6461, S2 Macro-F1 0.6112 and WS-BA 0.6214; SIRE has 0.6363, 0.6005 and 0.6100 respectively. The paired SIRE minus EEGNet contrast is -0.0099 S2 BA, 95% CI [-0.0364, +0.0178], and -0.0114 WS-BA, 95% CI [-0.0372, +0.0139], using 20,000 biological-subject bootstrap draws.

## Frozen PERSIST diagnostics

| Model | PEEH pp [95% CI] | PSWA pp [95% CI] | Nonempty coverage |
|---|---:|---:|---:|
| EEGNet | 5.539 [1.477, 9.846] | 7.263 [3.368, 11.799] | 15/15 |
| SIRE-EEG | 3.112 [0.733, 5.494] | 5.152 [3.430, 6.964] | 15/15 |

PEEH/PSWA are diagnostic quantities, not classifier-quality scores.

## Frozen rank-matched ScaleCollapse control

| Metric | ScaleCollapse harm pp | Random rank-16 harm pp | Excess harm pp [95% CI] |
|---|---:|---:|---:|
| Future S2 BA | 6.208 | 6.992 | -0.783 [-2.344, +0.612] |
| WS-BA | 5.083 | 6.138 | -1.055 [-2.397, +0.153] |

All 72 Full replay cells passed before intervention; the maximum replay metric difference was approximately 1.1e-16. Random projectors remain control conditions, not bootstrap units.
