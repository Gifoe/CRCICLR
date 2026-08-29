# PERSIST-EEG P3 Comprehensive Training-Trajectory Report

Decision: `P3_TRAJECTORY_CLAIM_NOT_SUPPORTED`

## Raw event BA trajectory

|   role_order | role   | task   |   balanced_accuracy |
|-------------:|:-------|:-------|--------------------:|
|            0 | epoch0 | erp    |              0.5552 |
|            0 | epoch0 | mi     |              0.5669 |
|            0 | epoch0 | ssvep  |              0.2859 |
|            1 | epoch1 | erp    |              0.5606 |
|            1 | epoch1 | mi     |              0.5779 |
|            1 | epoch1 | ssvep  |              0.633  |
|            2 | early  | erp    |              0.5754 |
|            2 | early  | mi     |              0.6846 |
|            2 | early  | ssvep  |              0.9094 |
|            3 | middle | erp    |              0.6327 |
|            3 | middle | mi     |              0.7488 |
|            3 | middle | ssvep  |              0.9348 |
|            4 | late   | erp    |              0.6506 |
|            4 | late   | mi     |              0.7535 |
|            4 | late   | ssvep  |              0.9366 |
|            5 | final  | erp    |              0.6593 |
|            5 | final  | mi     |              0.7578 |
|            5 | final  | ssvep  |              0.9367 |
|            6 | best   | erp    |              0.652  |
|            6 | best   | mi     |              0.7577 |
|            6 | best   | ssvep  |              0.9353 |

## Cross-session Long trajectory

|   role_order | role   |   auroc |
|-------------:|:-------|--------:|
|            0 | epoch0 |  0.5961 |
|            1 | epoch1 |  0.7165 |
|            2 | early  |  0.7668 |
|            3 | middle |  0.7608 |
|            4 | late   |  0.7573 |
|            5 | final  |  0.7546 |
|            6 | best   |  0.7584 |

## U_L erasure utility trajectory

|   role_order | role   | task   |   U_L_utility |
|-------------:|:-------|:-------|--------------:|
|            0 | epoch0 | erp    |        0.0066 |
|            0 | epoch0 | mi     |        0.0042 |
|            0 | epoch0 | ssvep  |        0.0079 |
|            2 | early  | erp    |       -0.0019 |
|            2 | early  | mi     |       -0.0254 |
|            2 | early  | ssvep  |       -0.0139 |
|            3 | middle | erp    |       -0.0032 |
|            3 | middle | mi     |       -0.0517 |
|            3 | middle | ssvep  |       -0.0122 |
|            6 | best   | erp    |       -0.003  |
|            6 | best   | mi     |       -0.0468 |
|            6 | best   | ssvep  |       -0.0106 |

## Rank and spectrum trajectory

|   role_order | role   |   U_L_rank |   U_M_rank |   U_L_top_fraction |   U_L_cumulative_primary |
|-------------:|:-------|-----------:|-----------:|-------------------:|-------------------------:|
|            0 | epoch0 |       2.8  |       2.52 |             0.7351 |                   0.9364 |
|            1 | epoch1 |       3.44 |       4.28 |             0.7113 |                   0.9168 |
|            2 | early  |       4.6  |       5.84 |             0.6671 |                   0.9176 |
|            3 | middle |       4.88 |       6.04 |             0.566  |                   0.9162 |
|            4 | late   |       4.88 |       6.2  |             0.5664 |                   0.9148 |
|            5 | final  |       4.92 |       6.12 |             0.5606 |                   0.9155 |
|            6 | best   |       4.88 |       6.16 |             0.5724 |                   0.9149 |

## Required conclusions

1. Task BA evolution is summarized below; MI epoch0-to-best=0.1902, SSVEP=0.6495.
2. Broad persistence epoch0-to-best cross-session AUROC change=0.1557; compression clause=FAIL.
3. Mean U_L rank changes from 2.800 to 4.880; low-rank clause=FAIL.
4. Remaining Long directions are task-useful at best only insofar as the independent P2 Gate A/D passed: PASS.
5. Utility emergence is reported at the locked epoch0/early/middle/best roles; no post-hoc checkpoint was selected.
6. MI/ERP/SSVEP trajectories are reported separately; no task was removed.
7. Cross-seed compression direction was negative in 0/5 seeds.
8. Selective-compression decision: P3_TRAJECTORY_CLAIM_NOT_SUPPORTED.

Curves were aggregated across all five seeds and all frozen folds. Non-monotonic points were retained.
