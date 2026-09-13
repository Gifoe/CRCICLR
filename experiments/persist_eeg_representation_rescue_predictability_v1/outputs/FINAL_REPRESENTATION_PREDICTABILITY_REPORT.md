# Representation rescue predictability audit

EXPERIMENT_TYPE = DEVELOPMENT_ANALYSIS_ONLY
NEW_EEG_MODEL_TRAINED = NO
FINAL_HELDOUT_ACCESSED = NO
INTERNAL_HELDOUT_ACCESSED = NO
DEVELOPMENT_OUTER_ONLY = YES

## Protocol gates

- F0 exact-control reproduction: PASS (maximum AUC difference 0; maximum policy difference 0 pp)
- Frozen representation replay: PASS (120 audited model/fold/seed cells)
- Subject-equal fitting weights: PASS

## Table 1 — XS equal-seed mean

| Task | Feature family | SAFE AUC | CLEAN AUC | Policy delta pp | 95% CI | Nonnegative folds | Worst fold | Switch rate | Intervention precision | Oracle headroom | Oracle fraction recovered |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenBMI_MI | F0 | 0.6131 | 0.6131 | +0.633 | [-0.067, +1.308] | 4/5 | -0.083 | 7.33% | 0.5396 | 8.525 | 0.074 |
| OpenBMI_MI | F1 | 0.5987 | 0.5987 | +0.433 | [-0.383, +1.225] | 4/5 | -1.000 | 6.58% | 0.5309 | 8.525 | 0.051 |
| OpenBMI_MI | F2 | 0.5508 | 0.5508 | -0.158 | [-1.092, +0.717] | 2/5 | -1.417 | 8.39% | 0.4840 | 8.525 | -0.019 |
| OpenBMI_ERP | F0 | 0.6795 | 0.6795 | +1.023 | [+0.510, +1.560] | 5/5 | +0.691 | 5.29% | 0.6910 | 5.482 | 0.187 |
| OpenBMI_ERP | F1 | 0.8236 | 0.8236 | +0.221 | [-0.243, +0.690] | 4/5 | -0.562 | 4.39% | 0.7136 | 5.482 | 0.040 |
| OpenBMI_ERP | F2 | 0.8109 | 0.8109 | +0.218 | [-0.163, +0.610] | 3/5 | -0.467 | 3.01% | 0.8433 | 5.482 | 0.040 |
| OpenBMI_SSVEP | F0 | 0.7683 | 0.7869 | +3.142 | [+1.500, +5.250] | 5/5 | +1.208 | 6.08% | 0.7824 | 5.925 | 0.530 |
| OpenBMI_SSVEP | F1 | 0.7448 | 0.7571 | +3.125 | [+1.375, +5.458] | 5/5 | +0.708 | 6.49% | 0.7656 | 5.925 | 0.527 |
| OpenBMI_SSVEP | F2 | 0.6928 | 0.6695 | +2.717 | [+1.150, +4.925] | 5/5 | +0.958 | 5.89% | 0.7571 | 5.925 | 0.459 |
| WBCIC_MI | F0 | 0.5718 | 0.5718 | +0.915 | [-0.010, +1.769] | 5/5 | +0.056 | 10.63% | 0.5437 | 8.067 | 0.113 |
| WBCIC_MI | F1 | 0.5596 | 0.5596 | +0.704 | [-0.194, +1.532] | 4/5 | -0.056 | 10.72% | 0.5333 | 8.067 | 0.087 |
| WBCIC_MI | F2 | 0.5454 | 0.5454 | +0.478 | [-0.248, +1.160] | 4/5 | -0.111 | 8.12% | 0.5315 | 8.067 | 0.059 |

## Table 2 — feature increments

| Task | SAFE AUC F0 | F1 | F2 | F1-F0 | F2-F1 | Policy F0 | F1 | F2 | F1-F0 | F2-F1 | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OpenBMI_MI | 0.6131 | 0.5987 | 0.5508 | -0.0144 | -0.0479 | +0.633 | +0.433 | -0.158 | -0.200 | -0.592 | HIGH_ORACLE_NOT_OBSERVABLY_ROUTABLE |
| OpenBMI_ERP | 0.6795 | 0.8236 | 0.8109 | +0.1441 | -0.0127 | +1.023 | +0.221 | +0.218 | -0.802 | -0.003 | ROUTING_WEAK_SIGNAL |
| OpenBMI_SSVEP | 0.7683 | 0.7448 | 0.6928 | -0.0235 | -0.0520 | +3.142 | +3.125 | +2.717 | -0.017 | -0.408 | SIGNED_LOGITS_SUFFICIENT |
| WBCIC_MI | 0.5718 | 0.5596 | 0.5454 | -0.0123 | -0.0142 | +0.915 | +0.704 | +0.478 | -0.210 | -0.226 | HIGH_ORACLE_NOT_OBSERVABLY_ROUTABLE |

## Table 3 — XS seed robustness

| Task | Seed | Feature family | SAFE AUC | Policy delta pp | 95% CI | Nonnegative folds | Worst fold |
|---|---|---|---:|---:|---:|---:|---:|
| OpenBMI_MI | seed0 | F1 | 0.6114 | +1.400 | [+0.475, +2.325] | 4/5 | -0.375 |
| OpenBMI_MI | seed0 | F2 | 0.5463 | +1.000 | [-0.150, +2.125] | 3/5 | -0.750 |
| OpenBMI_MI | seed1 | F1 | 0.6168 | -0.225 | [-1.275, +0.750] | 4/5 | -3.625 |
| OpenBMI_MI | seed1 | F2 | 0.5619 | -1.075 | [-2.200, -0.025] | 1/5 | -2.500 |
| OpenBMI_MI | seed2 | F1 | 0.5678 | +0.125 | [-0.900, +1.050] | 3/5 | -1.000 |
| OpenBMI_MI | seed2 | F2 | 0.5442 | -0.400 | [-1.800, +0.850] | 1/5 | -2.250 |
| OpenBMI_ERP | seed0 | F1 | 0.8193 | +0.028 | [-0.560, +0.583] | 2/5 | -0.386 |
| OpenBMI_ERP | seed0 | F2 | 0.8105 | +0.117 | [-0.331, +0.551] | 3/5 | -0.913 |
| OpenBMI_ERP | seed1 | F1 | 0.8260 | +0.491 | [+0.042, +0.983] | 5/5 | +0.027 |
| OpenBMI_ERP | seed1 | F2 | 0.8115 | +0.281 | [-0.094, +0.667] | 3/5 | -0.345 |
| OpenBMI_ERP | seed2 | F1 | 0.8256 | +0.145 | [-0.531, +0.727] | 4/5 | -1.375 |
| OpenBMI_ERP | seed2 | F2 | 0.8106 | +0.258 | [-0.308, +0.817] | 3/5 | -0.841 |
| OpenBMI_SSVEP | seed0 | F1 | 0.7268 | +4.275 | [+2.050, +7.400] | 5/5 | +1.250 |
| OpenBMI_SSVEP | seed0 | F2 | 0.6618 | +3.800 | [+1.425, +7.200] | 5/5 | +1.125 |
| OpenBMI_SSVEP | seed1 | F1 | 0.7497 | +2.325 | [+0.800, +4.250] | 5/5 | +0.625 |
| OpenBMI_SSVEP | seed1 | F2 | 0.6999 | +2.050 | [+0.850, +3.475] | 5/5 | +0.625 |
| OpenBMI_SSVEP | seed2 | F1 | 0.7579 | +2.775 | [+1.075, +5.000] | 5/5 | +0.250 |
| OpenBMI_SSVEP | seed2 | F2 | 0.7168 | +2.300 | [+0.875, +4.325] | 5/5 | +0.750 |
| WBCIC_MI | seed0 | F1 | 0.5555 | +0.517 | [-0.611, +1.565] | 3/5 | -0.828 |
| WBCIC_MI | seed0 | F2 | 0.5432 | +0.274 | [-0.645, +1.209] | 3/5 | -0.918 |
| WBCIC_MI | seed1 | F1 | 0.5570 | +0.710 | [-0.435, +1.677] | 4/5 | -0.750 |
| WBCIC_MI | seed1 | F2 | 0.5640 | +0.498 | [-0.489, +1.369] | 4/5 | -0.083 |
| WBCIC_MI | seed2 | F1 | 0.5662 | +0.886 | [-0.034, +1.854] | 5/5 | +0.083 |
| WBCIC_MI | seed2 | F2 | 0.5291 | +0.662 | [-0.113, +1.468] | 3/5 | -0.417 |

## Final scientific answers

1. No. The previous MI failure was not resolved by retaining class orientation, so class-symmetric compression is not supported as the material bottleneck.
2. No. OpenBMI MI signed features changed SAFE AUC by -0.0144 and policy delta by -0.200 pp.
3. No. WBCIC MI signed features changed SAFE AUC by -0.0123 and policy delta by -0.210 pp.
4. No. Full embeddings reduced SAFE AUC relative to F1 on every task (OpenBMI_MI -0.0479; OpenBMI_ERP -0.0127; OpenBMI_SSVEP -0.0520; WBCIC_MI -0.0142) and no task required F2 to pass the strong gate.
5. SSVEP is preserved as a strong positive control under F1 and F2. ERP is not preserved under the full strong-policy gate: F1 markedly raises AUC but lowers policy gain below +0.30 pp and has a worst fold below -0.50 pp; F2 does not repair it.
6. The intervention signal is robust across all three seeds only for SSVEP. OpenBMI MI is seed-fragile; WBCIC has positive point estimates but weak AUC and CIs crossing zero; ERP does not meet the strong fold/policy gate consistently.
7. Supported labels: OpenBMI_MI=HIGH_ORACLE_BUT_ROUTING_UNRESOLVED; OpenBMI_ERP=NO_STRONG_ROUTING_LABEL; OpenBMI_SSVEP=LOGIT_LEVEL_ROUTING; WBCIC_MI=HIGH_ORACLE_BUT_ROUTING_UNRESOLVED. ERP retains only a weak logit-level signal, not a strong routing label.
8. Broad evidence for a future single-model conditional residual mechanism: NO (1/4 strong tasks; all-task positive best-family policy=True; neither MI high-oracle unresolved=False).

This is a development-only diagnostic conclusion. No EEG network or neural router was trained, no logistic hyperparameter was tuned, and no internal-heldout, final-heldout, final-test, or sealed-test artifact was accessed.
