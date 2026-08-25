# Failure localization final decision

| hypothesis | prediction/evidence | decision |
|---|---|---|
| H1_teacher_relative_importance_mismatch | task consequence is measurable but does not predict future BA reliably | **PARTIAL** |
| H2_source_future_certificate_mismatch | frozen source directions evaluated on unseen Session 2 | **PARTIAL** |
| H3_hard_factorization_brittle_bottleneck | {'pud_erase_harm': 0.1355833333333333, 'random_erase_harm': 0.02358333333333332, 'identity_erase_harm': 0.010499999999999989, 'R_P_pud': 0.6826850240429242, 'R_P_dual': 0.5026111518343289} | **SUPPORT** |
| H4_optimization_gradient_conflict | gradient cosines are reported without optimizer steps | **PARTIAL** |
| H5_calibration_margin_failure | calibration audit is diagnostic; frozen BA loss is not reduced to calibration alone | **NOT_SUPPORT** |
| H6_adaptation_failure | 0.007416666666666627 | **NOT_SUPPORT** |
| H7_capacity_only_failure | strong single-path and dual controls are available | **NOT_SUPPORT** |

Primary diagnosis: **hard factorization concentrates task-consequential evidence and loses future-session utility; auxiliary/target adaptation can only partially rescue it**.

This is a causal-style localization table, not a claim of biological causality.
