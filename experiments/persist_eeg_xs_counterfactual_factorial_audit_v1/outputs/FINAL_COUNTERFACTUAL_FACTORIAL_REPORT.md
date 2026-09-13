# XS counterfactual factorial audit

EXPERIMENT_TYPE = COUNTERFACTUAL_ANALYSIS_ONLY
NEW_EEG_MODEL_TRAINED = NO
FINAL_HELDOUT_ACCESSED = NO
INTERNAL_HELDOUT_ACCESSED = NO
DEVELOPMENT_OUTER_ONLY = YES

## Protocol gate

- Full XS exact replay: PASS (60/60 cells)
- Maximum metric reproduction difference: 0
- C0S0M0 is the trained wide XS model with C/S/M disabled; it is not LiteBN.

## Table 1 — full cube, four-task / three-seed aggregate

| State | Rescue retention | Harm reduction | NET vs B0 pp | Delta NET vs Full pp | Tasks improved /4 | Cells improved /12 | Worst task delta | Screen status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C0S0M0 | 0.905 | -0.826 | -4.473 | -4.933 | 0/4 | 0/12 | -6.225 | NO_USEFUL_STRUCTURAL_SIGNAL |
| C1S0M0 | 0.925 | -0.706 | -3.762 | -4.222 | 0/4 | 0/12 | -5.350 | NO_USEFUL_STRUCTURAL_SIGNAL |
| C0S1M0 | 0.913 | -0.805 | -4.339 | -4.799 | 0/4 | 0/12 | -5.986 | NO_USEFUL_STRUCTURAL_SIGNAL |
| C0S0M1 | 0.968 | -0.070 | -0.103 | -0.563 | 0/4 | 0/12 | -0.992 | NO_USEFUL_STRUCTURAL_SIGNAL |
| C1S1M0 | 0.932 | -0.686 | -3.650 | -4.110 | 0/4 | 0/12 | -5.235 | NO_USEFUL_STRUCTURAL_SIGNAL |
| C1S0M1 | 0.994 | -0.003 | +0.397 | -0.063 | 2/4 | 2/12 | -0.192 | NO_USEFUL_STRUCTURAL_SIGNAL |
| C0S1M1 | 0.977 | -0.064 | +0.001 | -0.459 | 0/4 | 1/12 | -0.875 | NO_USEFUL_STRUCTURAL_SIGNAL |
| C1S1M1 | 1.000 | 0.000 | +0.460 | +0.000 | 0/4 | 0/12 | +0.000 | REFERENCE_FULL_XS |

## Table 2 — per task

| Task | State | Rescue pp | Harm pp | NET pp | Delta NET vs Full pp | Rescue retention | Harm reduction | Positive seeds /3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OpenBMI_MI | C0S0M0 | 8.700 | 12.275 | -3.575 | -2.042 | 1.024 | -0.225 | 0/3 |
| OpenBMI_MI | C1S0M0 | 8.883 | 12.008 | -3.125 | -1.592 | 1.046 | -0.197 | 0/3 |
| OpenBMI_MI | C0S1M0 | 8.717 | 12.125 | -3.408 | -1.875 | 1.026 | -0.208 | 0/3 |
| OpenBMI_MI | C0S0M1 | 8.325 | 10.392 | -2.067 | -0.533 | 0.977 | -0.033 | 0/3 |
| OpenBMI_MI | C1S1M0 | 8.892 | 11.942 | -3.050 | -1.517 | 1.047 | -0.189 | 0/3 |
| OpenBMI_MI | C1S0M1 | 8.425 | 10.150 | -1.725 | -0.192 | 0.989 | -0.009 | 0/3 |
| OpenBMI_MI | C0S1M1 | 8.450 | 10.208 | -1.758 | -0.225 | 0.991 | -0.014 | 1/3 |
| OpenBMI_MI | C1S1M1 | 8.525 | 10.058 | -1.533 | +0.000 | 1.000 | 0.000 | 0/3 |
| OpenBMI_ERP | C0S0M0 | 5.049 | 10.653 | -5.604 | -5.952 | 0.921 | -1.080 | 0/3 |
| OpenBMI_ERP | C1S0M0 | 5.149 | 9.513 | -4.364 | -4.712 | 0.939 | -0.859 | 0/3 |
| OpenBMI_ERP | C0S1M0 | 5.095 | 10.732 | -5.638 | -5.986 | 0.930 | -1.094 | 0/3 |
| OpenBMI_ERP | C0S0M1 | 5.392 | 5.507 | -0.114 | -0.462 | 0.984 | -0.073 | 0/3 |
| OpenBMI_ERP | C1S1M0 | 5.182 | 9.582 | -4.400 | -4.748 | 0.946 | -0.871 | 0/3 |
| OpenBMI_ERP | C1S0M1 | 5.493 | 5.090 | +0.403 | +0.055 | 1.002 | 0.007 | 1/3 |
| OpenBMI_ERP | C0S1M1 | 5.433 | 5.552 | -0.119 | -0.467 | 0.991 | -0.081 | 0/3 |
| OpenBMI_ERP | C1S1M1 | 5.482 | 5.134 | +0.348 | +0.000 | 1.000 | 0.000 | 0/3 |
| OpenBMI_SSVEP | C0S0M0 | 3.850 | 7.558 | -3.708 | -6.225 | 0.647 | -1.249 | 0/3 |
| OpenBMI_SSVEP | C1S0M0 | 4.083 | 6.917 | -2.833 | -5.350 | 0.686 | -1.052 | 0/3 |
| OpenBMI_SSVEP | C0S1M0 | 3.992 | 7.275 | -3.283 | -5.800 | 0.671 | -1.167 | 0/3 |
| OpenBMI_SSVEP | C0S0M1 | 5.417 | 3.892 | +1.525 | -0.992 | 0.915 | -0.142 | 0/3 |
| OpenBMI_SSVEP | C1S1M0 | 4.225 | 6.650 | -2.425 | -4.942 | 0.710 | -0.972 | 0/3 |
| OpenBMI_SSVEP | C1S0M1 | 5.833 | 3.433 | +2.400 | -0.117 | 0.985 | -0.010 | 0/3 |
| OpenBMI_SSVEP | C0S1M1 | 5.517 | 3.875 | +1.642 | -0.875 | 0.933 | -0.135 | 0/3 |
| OpenBMI_SSVEP | C1S1M1 | 5.925 | 3.408 | +2.517 | +0.000 | 1.000 | 0.000 | 0/3 |
| WBCIC_MI | C0S0M0 | 8.278 | 13.283 | -5.005 | -5.514 | 1.027 | -0.752 | 0/3 |
| WBCIC_MI | C1S0M0 | 8.288 | 13.014 | -4.726 | -5.235 | 1.027 | -0.715 | 0/3 |
| WBCIC_MI | C0S1M0 | 8.261 | 13.288 | -5.027 | -5.536 | 1.024 | -0.752 | 0/3 |
| WBCIC_MI | C0S0M1 | 8.040 | 7.795 | +0.245 | -0.263 | 0.997 | -0.031 | 0/3 |
| WBCIC_MI | C1S1M0 | 8.267 | 12.992 | -4.726 | -5.235 | 1.025 | -0.712 | 0/3 |
| WBCIC_MI | C1S0M1 | 8.078 | 7.569 | +0.509 | +0.000 | 1.001 | -0.001 | 1/3 |
| WBCIC_MI | C0S1M1 | 8.002 | 7.762 | +0.240 | -0.269 | 0.992 | -0.027 | 0/3 |
| WBCIC_MI | C1S1M1 | 8.067 | 7.558 | +0.509 | +0.000 | 1.000 | 0.000 | 0/3 |

## Table 3 — module attribution

| Scope | Module removed | Harm reversal % | Rescue loss % | Selectivity pp | Delta Rescue pp | Delta Harm pp | Delta NET pp | Positive selectivity cells |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Four-task | C | 7.17 | 5.01 | +2.17 | -0.149 | +0.310 | -0.459 | 7/12 |
| Four-task | M | 31.68 | 29.25 | +2.43 | -0.358 | +3.752 | -4.110 | 8/12 |
| Four-task | S | 3.89 | 2.48 | +1.42 | -0.043 | +0.021 | -0.063 | 8/12 |
| OpenBMI_MI | C | 4.11 | 3.63 | +0.48 | -0.075 | +0.150 | -0.225 | 2/3 |
| OpenBMI_MI | M | 22.50 | 17.08 | +5.42 | +0.367 | +1.883 | -1.517 | 3/3 |
| OpenBMI_MI | S | 1.78 | 2.82 | -1.03 | -0.100 | +0.092 | -0.192 | 0/3 |
| OpenBMI_ERP | C | 19.76 | 5.33 | +14.43 | -0.049 | +0.418 | -0.467 | 3/3 |
| OpenBMI_ERP | M | 52.84 | 29.76 | +23.08 | -0.300 | +4.448 | -4.748 | 3/3 |
| OpenBMI_ERP | S | 8.27 | 4.25 | +4.01 | +0.011 | -0.044 | +0.055 | 3/3 |
| OpenBMI_SSVEP | C | 1.20 | 7.03 | -5.83 | -0.408 | +0.467 | -0.875 | 0/3 |
| OpenBMI_SSVEP | M | 12.40 | 32.37 | -19.97 | -1.700 | +3.242 | -4.942 | 0/3 |
| OpenBMI_SSVEP | S | 4.04 | 2.43 | +1.61 | -0.092 | +0.025 | -0.117 | 2/3 |
| WBCIC_MI | C | 3.62 | 4.04 | -0.42 | -0.065 | +0.204 | -0.269 | 2/3 |
| WBCIC_MI | M | 38.98 | 37.80 | +1.18 | +0.200 | +5.434 | -5.235 | 2/3 |
| WBCIC_MI | S | 1.48 | 0.40 | +1.08 | +0.011 | +0.011 | +0.000 | 3/3 |

## Table 4 — factorial effects

Positive main NET effect means switching the module from OFF to ON increases NET on average across other states.

| Effect | Rescue pp | Harm pp | NET pp |
|---|---:|---:|---:|
| C | +0.143 | -0.447 | +0.590 |
| S | +0.047 | -0.057 | +0.103 |
| M | +0.343 | -3.901 | +4.245 |
| CxS | -0.005 | +0.010 | -0.016 |
| CxM | +0.014 | +0.124 | -0.110 |
| SxM | +0.003 | +0.023 | -0.020 |
| CxSxM | -0.002 | +0.002 | -0.004 |

## Scientific answers

1. The largest positive factorial Rescue main effect is M.
2. None of C/S/M has a positive Harm main effect: turning each ON reduces Harm on average. M is the strongest Harm suppressor; S is merely the least-negative Harm effect (-0.057 pp).
3. No removal reverses substantially more Harm than Rescue while improving NET. Removing M has the largest conditional selectivity (31.68% reversal versus 29.25% loss, +2.43 pp), but its total NET change is -4.110 pp because it also creates new errors.
4. Main-effect signs are consistent for C and M in 12/12 task-seed cells and for S in 9/12; interaction signs are less consistent. Full counts: C 12/12; CxM 8/12; CxS 8/12; CxSxM 6/12; M 12/12; S 9/12; SxM 7/12.
5. Non-additive interactions are not material under the declared descriptive rule (largest absolute interaction NET 0.110 pp; largest absolute main NET 4.245 pp; rule: interaction >=0.10 pp and >=25% of the largest main effect).
6. A reduced state preserving at least 90% Rescue while materially reducing Harm: NO under the full PROMISING screen.
7. A reduced state improving equal-task NET by at least +0.30 pp: NO (best -0.063 pp).
8. Future from-scratch reduced-XS experiment: NO; NO_USEFUL_STRUCTURAL_SIGNAL.

These are post-training counterfactuals, not models trained without the removed mechanisms. Counterfactual BA is not deployable-model accuracy. No EEG model was trained and no heldout or final-test artifact was accessed.
