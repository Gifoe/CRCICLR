# SCST-DR Stage-0 Repair-2 report

## Terminal

`TRANSPORT_VALIDITY_NOT_SUPPORTED`

The protocol and execution hashes were frozen before Repair-2 outcomes.  The
operator kept the original residual direction, used only `final_embedding`,
capped alpha at 0.25, and selected the largest legal value on the fixed 1/64
grid using Session-1-only same-class source support.  Session 2 remained the
independent validity partition.  All 20 setting-by-fold units completed.

## Frozen-gate results

| setting_id              |   fraction_alpha_zero |   alpha_mean |   alpha_median |   fraction_alpha_max |   subject_affinity_improvement_mean |   subject_affinity_CI_low |   subject_advantage_over_random_mean |   subject_advantage_over_random_CI_low |   class_accuracy_change |   class_logp_change |   manifold_knn_ratio_to_clean |   scst_off_manifold_rate |   random_off_manifold_rate | all_gates_pass   |
|:------------------------|----------------------:|-------------:|---------------:|---------------------:|------------------------------------:|--------------------------:|-------------------------------------:|---------------------------------------:|------------------------:|--------------------:|------------------------------:|-------------------------:|---------------------------:|:-----------------|
| OPENBMI_MI_EEGNET       |               0.00326 |      0.24855 |        0.25000 |              0.99112 |                             0.12636 |                   0.11697 |                              0.16422 |                                0.15404 |                 0.00000 |            -0.00000 |                       1.16285 |                  0.00000 |                    0.00145 | True             |
| OPENBMI_MI_EEGCONFORMER |               0.02192 |      0.24420 |        0.25000 |              0.97536 |                             0.15324 |                   0.14343 |                              0.18610 |                                0.17496 |                 0.00617 |             0.00627 |                       1.14937 |                  0.00000 |                    0.00145 | True             |
| WBCIC_MI_EEGNET         |               0.04046 |      0.23632 |        0.25000 |              0.91942 |                             0.08492 |                   0.05452 |                              0.13895 |                                0.11756 |                 0.00549 |             0.00369 |                       1.30796 |                  0.01670 |                    0.05888 | False            |
| WBCIC_MI_EEGCONFORMER   |               0.04167 |      0.23698 |        0.25000 |              0.92855 |                             0.09679 |                   0.06777 |                              0.15603 |                                0.13524 |                 0.02353 |             0.01600 |                       1.34080 |                  0.01119 |                    0.05337 | False            |

Both OpenBMI settings passed every gate.  Both WBCIC settings retained positive
target-subject affinity, positive advantage over norm-matched random, class
fidelity, and the binary off-manifold gate, but failed the unchanged absolute
3NN ratio gate: 1.30796 and 1.34080 versus the maximum 1.25.  Therefore 2/4,
not 4/4, settings passed and transport validity is not supported.

The centroid alpha distributions were not trivial zero operators.  Median alpha
was 0.25 in every setting; the fraction at alpha=0.25 ranged from 0.91942 to
0.99112.  Source support therefore rarely shortened the centroid transport
enough to repair the independent WBCIC manifold distance.

No Repair-3, SCST training, future-performance evaluation, or sealed outer
evaluation is authorized.
