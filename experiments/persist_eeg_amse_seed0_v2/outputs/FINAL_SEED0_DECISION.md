# AMSE-v2 seed-0 decision

Final terminal: **AMSE_SEED0_MIXED**

| Dataset | EEGNet | LiteBN | Best single | REF50 | Stable | Expressive | V2 Final | Δ vs best ref | Δ vs REF50 | R |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| OpenBMI | 75.77% | 79.15% | LiteBN | 80.00% | 74.55% | 76.88% | 77.68% | -1.47 pp | -2.32 pp | -1.735 |
| WBCIC | 78.27% | 78.70% | LiteBN | 79.40% | 78.53% | 79.35% | 79.41% | +0.71 pp | +0.02 pp | 1.023 |

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
