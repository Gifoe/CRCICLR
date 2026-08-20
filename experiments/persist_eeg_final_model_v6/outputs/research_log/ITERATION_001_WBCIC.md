# Iteration 001 — WBCIC conditional adapter

- Failure addressed: target-local last-layer/prototype controls did not substantially beat the population model.
- Structural hypothesis: cross-subject history-to-future episodes can learn a bilinear adaptation rule for unseen subjects.
- PERSIST intervention: stable model-fit cross-session dimensions are continuously protected; a shuffled equal-capacity control is retained.
- Result:

```text
                              benchmark                     method_id  subjects  mean_subject_BA  accuracy  macro_f1      NLL    Brier      ECE  exploratory  target_future_labels_used_for_fit  OUTER_TEST_USED  reference_method_id  Delta_BA    CI95_L    CI95_U  median_subject_delta  positive_subject_fraction  nonnegative_subject_fraction  worst_subject_delta  positive_fold_fraction  positive_folds
WBCIC_S1S2_to_S3_authorized_development              A_GENERIC_AFFINE        41         0.793501  0.793411  0.792269 0.411004 0.134252 0.027612         True                              False            False B_HISTORY_FUSION_LDA -0.003660 -0.009028  0.001707                -0.005                   0.365854                      0.487805               -0.050                     0.2               1
WBCIC_S1S2_to_S3_authorized_development    A_RANDOM_PROTECTED_CONTROL        41         0.792282  0.792190  0.788346 0.406951 0.134386 0.018276         True                              False            False B_HISTORY_FUSION_LDA -0.004879 -0.014148  0.004024                 0.000                   0.414634                      0.512195               -0.100                     0.0               0
WBCIC_S1S2_to_S3_authorized_development PERSIST_SA_PROTECTED_BILINEAR        41         0.790087  0.789994  0.786162 0.405097 0.133927 0.018749         True                              False            False B_HISTORY_FUSION_LDA -0.007074 -0.014757 -0.000367                 0.000                   0.463415                      0.536585               -0.105                     0.0               0
WBCIC_S1S2_to_S3_authorized_development            A_GENERIC_BILINEAR        41         0.784721  0.784625  0.782832 0.449567 0.142176 0.021040         True                              False            False B_HISTORY_FUSION_LDA -0.012440 -0.024637 -0.001466                -0.005                   0.341463                      0.439024               -0.155                     0.2               1
```
