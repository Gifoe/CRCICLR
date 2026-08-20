# Iteration 005 — MI representation selective history head

A low-capacity subject head is selected and safety-gated entirely within S1, then applied once to S2.

```text
          benchmark                         method_id  subjects  mean_subject_BA  accuracy  macro_f1      NLL    Brier      ECE  exploratory  target_future_labels_used_for_fit  OUTER_TEST_USED  reference_method_id  Delta_BA   CI95_L   CI95_U  median_subject_delta  positive_subject_fraction  nonnegative_subject_fraction  worst_subject_delta  positive_fold_fraction  positive_folds
OpenBMI_MI_S1_to_S2                MI_BACKBONE_FROZEN        54         0.822593  0.822593  0.817602 0.424139 0.129586 0.064534         True                              False            False B_HISTORY_FUSION_LDA  0.056296 0.032222 0.080926                 0.055                   0.685185                      0.740741                -0.16                     1.0               5
OpenBMI_MI_S1_to_S2 PERSIST_MI_SELECTIVE_HISTORY_HEAD        54         0.820185  0.820185  0.816452 0.437194 0.131067 0.053159         True                              False            False B_HISTORY_FUSION_LDA  0.053889 0.028704 0.080000                 0.050                   0.703704                      0.759259                -0.16                     1.0               5
OpenBMI_MI_S1_to_S2           MI_GENERIC_HISTORY_HEAD        54         0.818889  0.818889  0.815920 0.427874 0.131500 0.043733         True                              False            False B_HISTORY_FUSION_LDA  0.052593 0.027222 0.078704                 0.050                   0.685185                      0.740741                -0.14                     1.0               5
```
