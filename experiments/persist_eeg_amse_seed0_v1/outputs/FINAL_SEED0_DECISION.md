# AMSE-v1 seed-0 decision

Final terminal: **AMSE_SEED0_FAIL**

| Dataset | EEGNet | LiteBN | REF50 | Stable | Expressive | AMSE Final | Δ vs best ref | Δ vs REF50 | R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenBMI | 75.77% | 79.15% | 80.00% | 73.45% | 77.88% | 78.20% | -1.72 pp | -1.80 pp | -23.000 |
| WBCIC | 78.27% | 78.70% | 79.40% | 75.75% | 77.95% | 78.79% | -1.56 pp | -0.61 pp | NA |

1. Stable path competence: collapse risk
2. Expressive path competence: collapse risk
3. Final over strongest internal path: see `BRANCH_COMPETENCE.csv`.
4. Final over strongest existing single: see Δ vs best ref.
5. Independent fusion gain retention: see `FUSION_GAIN_RETENTION.csv`.
6. OpenBMI/WBCIC direction: not consistent
7. Multi-seed/ERP/SSVEP: STOP and await user decision; no automatic extension.
8. Branch collapse: possible
9. Cost: see `MODEL_COST.json`.
10. This is SEARCH-only; no OpenBMI final holdout or WBCIC true outer data were loaded.
