# PERSIST-EEG P3 Closure Report V2

Decision: `P3_CLOSED_AND_FROZEN`

## Required conclusions

- P2: `P2_PASS_MULTI_SEED_PERSISTENCE_UTILITY`
- Selective compression: `NOT_SUPPORTED`
- Long/Medium independence: `NOT_SUPPORTED`
- `MEDIUM_AS_INDEPENDENT_CORE_SCALE = NOT_SUPPORTED`
- Task-relevant persistence emergence: `EXPLORATORY_SUPPORT`
- P4 formalization: Persistent + Complementary/Fast; do not use `zL + zM + zF`.

## Evidence

- MI epoch0 to best BA change: 0.190222, 95% CI [0.1727773148148148, 0.2074453703703703].
- SSVEP epoch0 to best BA change: 0.649500, 95% CI [0.6344074074074074, 0.6639814814814814].
- Cross-session Long AUROC epoch0 to best change: 0.155701, 95% CI [0.14057923116801696, 0.17081971345124422].
- Seeds showing the preregistered compression direction: 0/5.
- Best-checkpoint Long AUROC: 0.758403.
- Mean U_L rank: epoch0 2.800, best 4.880.
- Mean normalized U_L/PCA overlap: 0.745112.
- Mean variance captured: U_L 0.679194; PCA 0.767749; random same-rank 0.038013.

The trajectory table contains all 5 seeds, 5 folds, 7 locked checkpoint roles, and 3 tasks. U_L erasure was prelocked only for epoch0/early/middle/best. Epoch1/late/final are missing unless their physical checkpoint aliases a prelocked role; no missing intervention was fabricated.

No outer-test data were used to fit U_L, PCA, random controls, or to adapt the P4 method.
