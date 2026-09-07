# P4D Final Report — Mechanism-to-Method Bridge

## Validated terminal

`P4D_METHOD_LEVEL_BRIDGE_NOT_SUPPORTED`

`P4E_MODEL_AUTHORIZATION = NOT_AUTHORIZED`

This is a constrained result. Only DANN passed the frozen source-side manipulation competence gate. Therefore P4D cannot reach strong support, regardless of the numerical interaction, because the prespecified two-method replication gate fails. P4E is independently blocked because P4C was partial rather than strong.

## Required answers

1. **P4C exact safety terminal:** `P4C_SAFETY_BOUNDARY_PARTIAL_SUPPORTED`.
2. **Why P4D was authorized:** conditional authorization; pooled DeltaRegime=+0.004169563, pooled U_high=-0.005007713, S4/S6 directions agree, and purity passes.
3. **P4C validator:** PASS.
4. **P4C low-E actionability:** `LOW_E_SUPPRESSION_NOT_BENEFICIAL`.
5. **Exact R_unsafe:** count(High-I AND High-E) / count(High-I), within frozen P4C ERM setting/fold/seed assignments.
6. **S4/S6 R_unsafe distribution:** shown below; thresholds were source-only frozen.
7. **Historically observed outcomes:** S1/S2/S3 grids and ERM competence outcomes; inventory records every cell.
8. **Trained but sealed before P4D:** all 135 S4 non-ERM cells, 70 partial S5 cells, and then only frozen S6 canonical DANN cells.
9. **Untrained:** remaining S5 and all noncanonical S6 method/lambda cells.
10. **Manipulation-competent methods:** `DANN` only.
11. **Canonical lambdas:** DANN=0.1; MMD and CORAL have no canonical lambda because they failed competence.
12. **Lambda selection purity:** yes; S4 source identity reduction only, no future BA/F1/CE.
13. **S6 training added:** yes, canonical-only.
14. **New training runs:** 15 (not 135).
15. **Exact z_SI:** `within-setting (S_I_abs-median)/(1.4826*MAD+1e-12); fallback setting SD when MAD degenerate`.
16. **beta z_SI:** -0.007172735.
17. **beta R_unsafe:** -0.001244841.
18. **beta interaction:** +0.017615955.
19. **Interaction 95% CI:** [-0.009109571, +0.041107147].
20. **slope_low:** -0.007172735.
21. **slope_high:** +0.001635243.
22. **DeltaSlope_bridge:** -0.008807978, CI [-0.020553573, +0.004554785].
23. **S4 bridge direction:** beta interaction +0.030150816.
24. **S6 bridge direction:** beta interaction +0.004453376.
25. **DANN result:** primary competent method; beta interaction +0.017615955.
26. **MMD result:** identity-manipulation incompetent under the frozen rule; excluded from primary outcome evaluation.
27. **CORAL result:** identity-manipulation incompetent under the frozen rule; excluded from primary outcome evaluation.
28. **Cross-method consistency:** not established; only one competent method, so G5 fails.
29. **Low-unsafe DeltaG_BA:** -0.000014057, CI [-0.009802487, +0.006990717].
30. **High-unsafe DeltaG_BA:** +0.003500970, CI [-0.009116635, +0.012501578].
31. **HeadroomContrast:** -0.003515026, CI [-0.017674600, +0.011994436].
32. **Canonical headroom method:** `DANN`, selected by source identity suppression before outcomes.
33. **P4D terminal:** `P4D_METHOD_LEVEL_BRIDGE_NOT_SUPPORTED`.
34. **P4E model authorization:** `NOT_AUTHORIZED`.
35. **Sealed outer holdouts:** untouched; OpenBMI internal holdout untouched and WBCIC outer 10 unenumerated.
36. **Outcome-driven protocol modification:** none.
37. **P4A 405-grid:** remained paused; no mechanical restart.
38. **Scientific interpretation:** the data test whether run-level task-entangled identity burden moderates global invariance. The result is limited to the competent DANN manipulation and cannot establish a method-general bridge. The exact sign and uncertainty above determine whether even that narrow bridge is partial or unsupported; no selective-invariance model is justified.

## Unsafe burden summary

| setting_id | min | median | mean | max |
| --- | --- | --- | --- | --- |
| S4 | 0.000000000 | 0.333333333 | 0.394444444 | 1.000000000 |
| S6 | 0.000000000 | 0.500000000 | 0.361111111 | 0.750000000 |

## Grid inventory summary

| setting_id | status | cells |
| --- | --- | --- |
| S1 | HISTORICALLY_OBSERVED_TASK_OUTCOME | 150 |
| S2 | HISTORICALLY_OBSERVED_TASK_OUTCOME | 150 |
| S3 | HISTORICALLY_OBSERVED_TASK_OUTCOME | 150 |
| S4 | HISTORICALLY_OBSERVED_TASK_OUTCOME | 15 |
| S4 | TRAINED_BUT_TASK_OUTCOME_SEALED | 135 |
| S5 | HISTORICALLY_OBSERVED_TASK_OUTCOME | 15 |
| S5 | TRAINED_BUT_TASK_OUTCOME_SEALED | 70 |
| S5 | UNTRAINED | 65 |
| S6 | HISTORICALLY_OBSERVED_TASK_OUTCOME | 15 |
| S6 | TRAINED_BUT_TASK_OUTCOME_SEALED | 15 |
| S6 | UNTRAINED | 120 |

## Prospective setting bridge

| setting_id | rows | beta_z_SI | beta_R_unsafe | beta_zSI_x_Runsafe | interaction_CI_lower | interaction_CI_upper | hypothesis_direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S4 | 15 | -0.011330339 | 0.008346036 | 0.030150816 | -0.029558972 | 0.115471832 | False |
| S6 | 15 | -0.001485537 | 0.001061861 | 0.004453376 | -0.036860306 | 0.029570002 | False |

## Prospective method bridge

| method | rows | beta_z_SI | beta_R_unsafe | beta_zSI_x_Runsafe | interaction_CI_lower | interaction_CI_upper | hypothesis_direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DANN | 30 | -0.007172735 | -0.001244841 | 0.017615955 | -0.009485572 | 0.040474860 | False |

## Gate audit

| gate | pass |
| --- | --- |
| G1_manipulation_competence | False |
| G2_primary_interaction | False |
| G3_simple_slope | False |
| G4_setting_consistency | False |
| G5_method_consistency | False |
| G6_purity | True |

## Final limitation

The bridge is scientifically underpowered at the method level because two of three standard methods did not manipulate identity reliably in the frozen S4 audit. Treating their non-effect as usable suppression would be invalid. Expanding S6 training or changing lambdas after seeing outcomes would not repair that identification problem; it would introduce selection bias.
