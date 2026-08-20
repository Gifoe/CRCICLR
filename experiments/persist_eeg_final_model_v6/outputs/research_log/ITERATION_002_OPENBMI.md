# Iteration 002 — OPENBMI encoder fine-tuning

- Previous failure: frozen-embedding conditional adapters did not beat matched controls.
- Structural change: adapt the EEG encoder itself from legal target-history trials.
- PERSIST change: parameter-level empirical-Fisher protection with uniform and shuffled controls.

```text
          benchmark                   method_id  subjects  mean_subject_BA  accuracy  macro_f1      NLL    Brier      ECE  exploratory  target_future_labels_used_for_fit  OUTER_TEST_USED  reference_method_id  Delta_BA    CI95_L    CI95_U  median_subject_delta  positive_subject_fraction  nonnegative_subject_fraction  worst_subject_delta  positive_fold_fraction  positive_folds
OpenBMI_MI_S1_to_S2    FT_RANDOM_FISHER_CONTROL        54         0.762778  0.762778  0.746796 0.498160 0.162648 0.039218         True                              False            False B_HISTORY_FUSION_LDA -0.003519 -0.017778  0.011111                -0.005                   0.370370                      0.500000                -0.14                     0.4               2
OpenBMI_MI_S1_to_S2       FT_UNIFORM_L2_CONTROL        54         0.761852  0.761852  0.745843 0.494496 0.161751 0.040865         True                              False            False B_HISTORY_FUSION_LDA -0.004444 -0.019074  0.010741                -0.010                   0.314815                      0.388889                -0.12                     0.4               2
OpenBMI_MI_S1_to_S2 PERSIST_SA_FISHER_PROTECTED        54         0.760926  0.760926  0.744827 0.495306 0.162065 0.037584         True                              False            False B_HISTORY_FUSION_LDA -0.005370 -0.017778  0.006852                -0.010                   0.370370                      0.444444                -0.12                     0.4               2
OpenBMI_MI_S1_to_S2         FT_GENERIC_SELECTED        54         0.760556  0.760556  0.743595 0.499513 0.163702 0.044055         True                              False            False B_HISTORY_FUSION_LDA -0.005741 -0.020000  0.009074                -0.010                   0.351852                      0.462963                -0.13                     0.4               2
OpenBMI_MI_S1_to_S2            FT_FROZEN_EEGNET        54         0.754074  0.754074  0.738428 0.504121 0.167077 0.037623         True                              False            False B_HISTORY_FUSION_LDA -0.012222 -0.025000 -0.000185                -0.010                   0.333333                      0.462963                -0.17                     0.2               1
```
