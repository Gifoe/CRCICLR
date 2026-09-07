# SCST-DR final report

## Final terminal

`FINAL_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED`

Repair-2 ended at `TRANSPORT_VALIDITY_NOT_SUPPORTED`.  The validator passed with 20/20 units
and 2/4 settings satisfying every frozen gate.  This is a validated scientific
failure, not a runtime failure.  The protocol forbids Repair-3, SCST training,
future-performance inspection, and sealed outer evaluation.

## Four-setting result

| setting_id              |   fraction_alpha_zero |   alpha_mean |   alpha_median |   fraction_alpha_max |   subject_affinity_improvement_mean |   subject_affinity_CI_low |   subject_advantage_over_random_mean |   subject_advantage_over_random_CI_low |   class_accuracy_change |   class_logp_change |   manifold_knn_ratio_to_clean |   scst_off_manifold_rate |   random_off_manifold_rate | all_gates_pass   |
|:------------------------|----------------------:|-------------:|---------------:|---------------------:|------------------------------------:|--------------------------:|-------------------------------------:|---------------------------------------:|------------------------:|--------------------:|------------------------------:|-------------------------:|---------------------------:|:-----------------|
| OPENBMI_MI_EEGNET       |               0.00326 |      0.24855 |        0.25000 |              0.99112 |                             0.12636 |                   0.11697 |                              0.16422 |                                0.15404 |                 0.00000 |            -0.00000 |                       1.16285 |                  0.00000 |                    0.00145 | True             |
| OPENBMI_MI_EEGCONFORMER |               0.02192 |      0.24420 |        0.25000 |              0.97536 |                             0.15324 |                   0.14343 |                              0.18610 |                                0.17496 |                 0.00617 |             0.00627 |                       1.14937 |                  0.00000 |                    0.00145 | True             |
| WBCIC_MI_EEGNET         |               0.04046 |      0.23632 |        0.25000 |              0.91942 |                             0.08492 |                   0.05452 |                              0.13895 |                                0.11756 |                 0.00549 |             0.00369 |                       1.30796 |                  0.01670 |                    0.05888 | False            |
| WBCIC_MI_EEGCONFORMER   |               0.04167 |      0.23698 |        0.25000 |              0.92855 |                             0.09679 |                   0.06777 |                              0.15603 |                                0.13524 |                 0.02353 |             0.01600 |                       1.34080 |                  0.01119 |                    0.05337 | False            |

## Required 31 answers

1. **Frozen before outcomes:** yes; protocol commit `ad86b25`, pre-outcome hash
   freeze commit `2ca4440`.
2. **Sealed resources:** untouched and unenumerated.
3. **Alpha-star distribution:** setting summaries are in the table; all medians
   are 0.25, zero fractions are 0.00326-0.04167, and alpha-max fractions are
   0.91942-0.99112.  Full class/subject strata are preserved in
   `results/STAGE0_REPAIR2_ALPHA_DISTRIBUTION.csv`.
4. **Target-subject affinity:** positive with CI lower above zero in all four.
5. **Versus matched random:** positive with CI lower above zero in all four.
6. **Class fidelity:** passed in all four.
7. **Independent 3NN ratios:** OpenBMI EEGNet 1.16285; OpenBMI EEGConformer
   1.14937; WBCIC EEGNet 1.30796; WBCIC EEGConformer 1.34080.
8. **All four <=1.25:** no; both WBCIC settings failed.
9. **Binary off-manifold rates:** SCST rates were 0, 0, 0.01670, and 0.01119;
   matched-random rates were 0.00145, 0.00145, 0.05888, and 0.05337.
10. **Transport validity supported:** no.
11. **Exact failure:** the two WBCIC 3NN ratios exceeded 1.25 despite every
    other gate passing.
12. **Trainable scope:** not selected; training was prohibited.
13. **Bank staleness:** not applicable because no model was trained.
14. **Matched ERM future BA:** not run.
15. **SCST-DR future BA:** not run.
16. **Delta BA:** not run.
17. **Subject-level CI:** not run.
18. **Positive primary settings:** not evaluated as trained methods.
19. **SCST versus Mixup:** not run.
20. **SCST versus random augmentation:** not run.
21. **Class conditioning:** Stage-0 class compatibility is supported; a training
    contribution is not established.
22. **Decision consistency:** not run.
23. **Subject identity I:** not run after training.
24. **Decision sensitivity D_T:** not run.
25. **I retained / D_T down / G up:** not supported.
26. **Outer authorized:** no.
27. **OpenBMI outer confirmation:** not opened.
28. **WBCIC outer confirmation:** not opened.
29. **Strongest supported claim:** source residual directions are stable and
    constrained transport is subject-faithful, better than norm-matched random,
    and class-compatible, but is not manifold-valid across datasets.
30. **Unsupported stronger claim:** no SCST-DR generalization, mechanism, or
    superiority claim is justified.
31. **Terminal:** Repair-2 `TRANSPORT_VALIDITY_NOT_SUPPORTED`; final
    `FINAL_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED`.

## Most serious limitation

The source-support rule rarely reduced centroid steps: at least 91.9% reached
alpha=0.25 in every setting.  It consequently did not enforce enough support on
the independent WBCIC geometry to satisfy the predeclared 1.25 criterion.  Any
further reduction, radius change, k change, setting exclusion, or outcome-aware
selection would be a new hypothesis, not an implementation repair.
