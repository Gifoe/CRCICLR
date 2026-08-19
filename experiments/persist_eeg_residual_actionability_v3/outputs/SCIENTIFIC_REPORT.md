# PERSIST-EEG residual actionability V3

## Terminal state

`RESIDUAL_ACTION_SIGNAL_BUT_NO_NET_GAIN`

This is exploratory grouped OOF evidence on 52 historical development
subjects. It is not a new confirmation. WBCIC outer was not accessed.

## Direct answers

1. V2.1 B6 reproduced exactly: **true**.
2. Deployment-level B6 mean subject BA: **0.846442**.
3. Global protected-safe action oracle above B6: **+3.538 pp**.
4. Full global action oracle above B6: **+8.596 pp**.
5. Single-expert replacement oracle: protected-safe **+4.981 pp**;
   full **+7.183 pp**.
6. KEEP-only diversity oracle above B6: **+7.663 pp**.
7. Strongest action oracle minus KEEP-only oracle:
   **+0.933 pp**;
   combined action+KEEP minus KEEP-only:
   **+3.038 pp**.
8. Residual oracle distribution: 51/52 positive
   subjects; top-20% concentration 0.426.
9. Unique action rescue is detailed in `ACTION_UNIQUENESS.csv`; ERASE is the
   dominant global source but is not the only source.
10. ERASE is necessary for the strongest oracle, but its unconditional harm
    count is substantially larger than its rescue count.
11. Protected-safe oracle headroom remains nonzero but is smaller than FULL.
12. Residual rescue/harm learnability is reported in
    `RESIDUAL_LEARNABILITY.csv` using held-out subjects only. The decision uses
    boundary-cross candidates only; best conditional AUROC is
    **0.722** and best conditional
    AUPRC/prevalence lift is **1.701**.
13. The largest full-logistic legal features are stored in
    `FEATURE_IMPORTANCE.csv` and `FINAL_DECISION.json`.
14. PERSIST feature increment over movement-only logistic:
    **+0.010 pp**, CI95
    [-0.212,
    +0.240] pp.
15. Best prospective method: `M5_HIST_GRADIENT_BOOSTING`.
16. Delta BA vs B6: **-0.029 pp**,
    grouped subject CI95 [-0.087,
    +0.019] pp.
17. Unique-action oracle recovery:
    **-0.031**.
18. The gain mechanism is a residual correction only if the grouped OOF lower
    bound is positive; otherwise the result remains oracle-only structural
    headroom.
19. Intervention research should continue only under the terminal-state rule
    above and never by further tuning these same OOF subjects.
20. If the policy criterion fails, the constructive next line is B6 ensemble
    compression/distillation while retaining PERSIST for audit and safety.

## Prospective model table

| pool | method_id | subjects | mean_subject_BA | mean_subject_accuracy | mean_subject_macro_f1 | mean_subject_delta_BA_vs_B6 | median_subject_delta_BA_vs_B6 | bootstrap_CI95_L | bootstrap_CI95_U | positive_subject_fraction | nonnegative_subject_fraction | worst_subject_delta_BA_vs_B6 | OUTER_TEST_USED | model_id | action_rate | rescue_count | harm_count | net_correctness_gain | rescue_precision | harm_rate | positive_fold_fraction | NLL | Brier | ECE | action_oracle_recovery_fraction | unique_action_oracle_recovery_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_52_grouped_OOF | M0_B6_KEEP_ENSEMBLE | 52 | 0.8464423076923075 | 0.8464423076923075 | 0.8448742470033863 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | False | M0_B6_KEEP_ENSEMBLE | 0.0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.3653566491315422 | 0.11390110597006627 | 0.05554381811237274 | 0.0 | 0.0 |
| all_52_grouped_OOF | M1_ENSEMBLE_CONFIDENCE_RULE | 52 | 0.8457692307692307 | 0.8457692307692307 | 0.8442074165754163 | -0.0006730769230769323 | 0.0 | -0.0022115384615384636 | 0.0007692307692307549 | 0.15384615384615385 | 0.8076923076923077 | -0.019999999999999907 | False | M1_ENSEMBLE_CONFIDENCE_RULE | 0.007596153846153846 | 36 | 43 | -7 | 0.45569620253164556 | 0.5443037974683544 | 0.0 | 0.3668006148776242 | 0.11451814419169395 | 0.053527936628355706 | -0.007829977628635458 | -0.07216494845360942 |
| all_52_grouped_OOF | M2_ENSEMBLE_DISAGREEMENT_RULE | 52 | 0.8453846153846154 | 0.8453846153846154 | 0.8438324986256021 | -0.0010576923076923085 | 0.0 | -0.002788461538461545 | 0.00048076923076922906 | 0.1346153846153846 | 0.8269230769230769 | -0.030000000000000027 | False | M2_ENSEMBLE_DISAGREEMENT_RULE | 0.0064423076923076925 | 28 | 39 | -11 | 0.417910447761194 | 0.582089552238806 | 0.2 | 0.365995171695009 | 0.1142087046820111 | 0.054200762869017406 | -0.012304250559284132 | -0.11340206185567049 |
| all_52_grouped_OOF | M3_ACTION_MOVEMENT_LOGISTIC | 52 | 0.8452884615384615 | 0.8452884615384615 | 0.8436107299112189 | -0.0011538461538461698 | 0.0 | -0.0036538461538461677 | 0.0012499999999999777 | 0.21153846153846154 | 0.75 | -0.030000000000000027 | False | M3_ACTION_MOVEMENT_LOGISTIC | 0.01326923076923077 | 63 | 75 | -12 | 0.45652173913043476 | 0.5434782608695652 | 0.4 | 0.366437397177243 | 0.11437638697779041 | 0.053546823905877707 | -0.013422818791946501 | -0.12371134020618761 |
| all_52_grouped_OOF | M4_FULL_LEGAL_LOGISTIC | 52 | 0.8453846153846154 | 0.8453846153846154 | 0.8437784689419318 | -0.0010576923076923107 | 0.0 | -0.0029807692307692356 | 0.0 | 0.0 | 0.9423076923076923 | -0.04500000000000004 | False | M4_FULL_LEGAL_LOGISTIC | 0.004519230769230769 | 18 | 29 | -11 | 0.3829787234042553 | 0.6170212765957447 | 0.0 | 0.36602168378498984 | 0.11421000009060367 | 0.054202665532163366 | -0.012304250559284156 | -0.11340206185567073 |
| all_52_grouped_OOF | M5_HIST_GRADIENT_BOOSTING | 52 | 0.846153846153846 | 0.846153846153846 | 0.8446224455129847 | -0.0002884615384615409 | 0.0 | -0.0008653846153846183 | 0.00019230769230769036 | 0.038461538461538464 | 0.9230769230769231 | -0.010000000000000009 | False | M5_HIST_GRADIENT_BOOSTING | 0.0010576923076923077 | 4 | 7 | -3 | 0.36363636363636365 | 0.6363636363636364 | 0.0 | 0.3654062570764864 | 0.11392541274886003 | 0.05521123999439244 | -0.0033557046979866066 | -0.030927835051546733 |
| all_52_grouped_OOF | I006_CONDITIONAL_ACTION_LOGISTIC | 52 | 0.8449038461538462 | 0.8449038461538463 | 0.8432021164501554 | -0.0015384615384615463 | 0.0 | -0.0034615384615384734 | 0.0004807692307692205 | 0.11538461538461539 | 0.6923076923076923 | -0.025000000000000022 | False | I006_CONDITIONAL_ACTION_LOGISTIC | 0.009038461538461539 | 39 | 55 | -16 | 0.4148936170212766 | 0.5851063829787234 | 0.2 | 0.3659585084491151 | 0.11418177241204866 | 0.05354721893873847 | -0.01789709172259518 | -0.1649484536082487 |
| all_52_grouped_OOF | I007_CONDITIONAL_ACTION_HGB | 52 | 0.8457692307692307 | 0.8457692307692307 | 0.8441695726810831 | -0.0006730769230769258 | 0.0 | -0.0023076923076923075 | 0.0010576923076923 | 0.19230769230769232 | 0.7307692307692307 | -0.020000000000000018 | False | I007_CONDITIONAL_ACTION_HGB | 0.007403846153846154 | 35 | 42 | -7 | 0.45454545454545453 | 0.5454545454545454 | 0.4 | 0.3655876026989014 | 0.11401173690863078 | 0.054547858833348915 | -0.007829977628635382 | -0.07216494845360873 |

## Scientific limitation

The model ladder and best-policy selection were explored on the same 52
historical subjects through grouped OOF estimates. I006-I007 were motivated
after auditing M1-M5 conditional learnability, so their CIs are additionally
post-primary-adaptive and descriptive. Even a positive result must be frozen
and tested under a genuinely new independent protocol.

`OUTER_TEST_USED=false`
