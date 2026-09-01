# PERSIST-CDE SEED-0 PILOT

Competence-Preserving Counterfactual Decision Ensemble (PERSIST-CDE) was evaluated once with seed=0 under the canonical outer evaluator.

## Primary comparison

| Dataset | Canonical seed0 | CDE seed0 | Delta pp | paired 95% CI |
|---|---:|---:|---:|---|
| OpenBMI | 0.819074 | 0.819074 | +0.000 | [+0.000, +0.000] |
| WBCIC | 0.786299 | 0.786299 | +0.000 | [+0.000, +0.000] |

## Selected alpha per fold

| Dataset | Fold | alpha_inv | alpha_geo | INV discovery BA | GEO discovery BA |
|---|---:|---:|---:|---:|---:|
| OpenBMI | 0 | 0.00 | 0.00 | 0.793333 | 0.790000 |
| OpenBMI | 1 | 0.00 | 0.00 | 0.748889 | 0.754444 |
| OpenBMI | 2 | 0.00 | 0.00 | 0.817778 | 0.814444 |
| OpenBMI | 3 | 0.00 | 0.00 | 0.826667 | 0.827778 |
| OpenBMI | 4 | 0.00 | 0.00 | 0.766667 | 0.771111 |
| WBCIC | 0 | 0.00 | 0.00 | 0.795000 | 0.795625 |
| WBCIC | 1 | 0.00 | 0.00 | 0.783125 | 0.780000 |
| WBCIC | 2 | 0.00 | 0.00 | 0.773355 | 0.774702 |
| WBCIC | 3 | 0.00 | 0.00 | 0.785625 | 0.784375 |
| WBCIC | 4 | 0.00 | 0.00 | 0.812222 | 0.817778 |

The selected-alpha table above records the selected candidate BA; branch BA and all outcome diagnostics are in `BRANCH_DIAGNOSTICS.csv`.

## Outcome mechanism diagnostics

| Dataset | INV BA | GEO BA | INV→base disagreement | GEO→base disagreement | INV→GEO disagreement | CDE rescue | CDE corruption |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenBMI fold 0 | 0.790000 | 0.782727 | 0.0736 | 0.0573 | 0.0527 | 0.0000 | 0.0000 |
| OpenBMI fold 1 | 0.826364 | 0.837273 | 0.0345 | 0.0309 | 0.0382 | 0.0000 | 0.0000 |
| OpenBMI fold 2 | 0.800909 | 0.800909 | 0.0682 | 0.0536 | 0.0455 | 0.0000 | 0.0000 |
| OpenBMI fold 3 | 0.816364 | 0.816364 | 0.0518 | 0.0409 | 0.0473 | 0.0000 | 0.0000 |
| OpenBMI fold 4 | 0.834000 | 0.833000 | 0.0430 | 0.0440 | 0.0270 | 0.0000 | 0.0000 |
| WBCIC fold 0 | 0.802778 | 0.803889 | 0.0544 | 0.0556 | 0.0356 | 0.0000 | 0.0000 |
| WBCIC fold 1 | 0.781875 | 0.778750 | 0.0956 | 0.0712 | 0.0444 | 0.0000 | 0.0000 |
| WBCIC fold 2 | 0.772500 | 0.782500 | 0.0919 | 0.0919 | 0.0500 | 0.0000 | 0.0000 |
| WBCIC fold 3 | 0.768393 | 0.766584 | 0.0721 | 0.0602 | 0.0483 | 0.0000 | 0.0000 |
| WBCIC fold 4 | 0.785000 | 0.786250 | 0.0644 | 0.0694 | 0.0488 | 0.0000 | 0.0000 |

## Direct answers

1. OpenBMI exceeds matched canonical seed0: NO.
2. WBCIC exceeds matched canonical seed0: NO.
3. The two deltas have the same strict direction: NO.
4. Competent auxiliary decision diversity exists: YES.
5. CDE improves over equal conservative fusion on at least one dataset: YES.
6. Any outcome-driven model selection: NO.
7. Sealed outer cohorts untouched: YES.
8. Run seed1/seed2: NO (predeclared rule: only a two-dataset positive seed-0 signal earns YES).

pilot_classification = CDE_SEED0_NEGATIVE_SIGNAL
method_terminal = CDE_SEED0_NEGATIVE_SIGNAL

The seed-0 pilot is not a final multi-seed paper claim. No seed1/seed2 run was started automatically.

terminal = CDE_SEED0_NEGATIVE_SIGNAL
