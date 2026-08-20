# Iteration 001 — OPENBMI conditional adapter

- Failure addressed: target-local last-layer/prototype controls did not substantially beat the population model.
- Structural hypothesis: cross-subject history-to-future episodes can learn a bilinear adaptation rule for unseen subjects.
- PERSIST intervention: stable model-fit cross-session dimensions are continuously protected; a shuffled equal-capacity control is retained.
- Result:

```text
          benchmark                     method_id  subjects  mean_subject_BA  accuracy  macro_f1      NLL    Brier      ECE  exploratory  target_future_labels_used_for_fit  OUTER_TEST_USED  reference_method_id  Delta_BA    CI95_L    CI95_U  median_subject_delta  positive_subject_fraction  nonnegative_subject_fraction  worst_subject_delta  positive_fold_fraction  positive_folds
OpenBMI_MI_S1_to_S2            A_GENERIC_BILINEAR        54         0.764815  0.764815  0.753508 0.509257 0.165675 0.062917         True                              False            False B_HISTORY_FUSION_LDA -0.001481 -0.011667  0.008519                -0.005                   0.407407                      0.481481                -0.10                     0.4               2
OpenBMI_MI_S1_to_S2              A_GENERIC_AFFINE        54         0.761852  0.761852  0.750480 0.508501 0.165733 0.061882         True                              False            False B_HISTORY_FUSION_LDA -0.004444 -0.014630  0.005741                -0.010                   0.388889                      0.462963                -0.09                     0.4               2
OpenBMI_MI_S1_to_S2 PERSIST_SA_PROTECTED_BILINEAR        54         0.756296  0.756296  0.741054 0.535272 0.174181 0.088248         True                              False            False B_HISTORY_FUSION_LDA -0.010000 -0.021852  0.001667                -0.010                   0.314815                      0.444444                -0.15                     0.2               1
OpenBMI_MI_S1_to_S2    A_RANDOM_PROTECTED_CONTROL        54         0.751852  0.751852  0.738214 0.592635 0.179369 0.098085         True                              False            False B_HISTORY_FUSION_LDA -0.014444 -0.025185 -0.004074                -0.010                   0.296296                      0.388889                -0.12                     0.4               2
```
