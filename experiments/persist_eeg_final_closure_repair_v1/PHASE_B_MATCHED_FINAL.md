# Phase B exact matched final

| method | BA | Macro-F1 | Δ vs Matched-TaskOnly |
|---|---:|---:|---:|
| Full-Teacher-KD-Aux | 0.773917 | 0.770597 | +0.007917 |
| Identity-Aux | 0.757917 | 0.754875 | -0.008083 |
| Matched-TaskOnly | 0.766000 | 0.762431 | +0.000000 |
| P-only-Aux | 0.769833 | 0.766550 | +0.003833 |
| PUD-Aux | 0.770250 | 0.766894 | +0.004250 |
| Random-Aux | 0.766417 | 0.762979 | +0.000417 |

Primary PUD-Aux−Matched-TaskOnly: +0.004250, median +0.005000, subject-bootstrap 95% CI [-0.000000, +0.008167], positive/negative/tied subjects 27/7/6, positive folds 3/5, positive seeds 3/3.

Historical Vanilla BA: 0.7861667; PUD-Aux−old Vanilla: -0.015917, paired 40-subject CI [-0.028667, -0.004250]. The old pipeline is an external reference, not the matched causal control.

- M1_delta_at_least_0.005: **FAIL**
- M2_subject_CI_lower_positive: **FAIL**
- M3_at_least_4_of_5_folds_positive: **FAIL**
- M4_at_least_2_of_3_seeds_positive: **PASS**
- M5_beats_random_and_identity: **PASS**
- M6_purity_and_integrity: **PASS**

Terminal: **PUD_AUX_MATCHED_NOT_SUPPORTED**.
