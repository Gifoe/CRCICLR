# Iteration 004 — low-rank history hypernetwork

Eight globally shared feature-space bases are mixed from legal history context. META-GENERIC and PERSIST-Meta are capacity matched; PERSIST alone receives P/U/D/G/R.

## OpenBMI

```text
          benchmark                      method_id  subjects  mean_subject_BA  mean_subject_NLL          reference_method_id  Delta_BA  positive_subject_fraction  nonnegative_subject_fraction  worst_subject_delta  target_future_labels_used_for_fit  OUTER_TEST_USED
OpenBMI_MI_S1_to_S2   MI_SPECIFIC_BACKBONE_ADAPTED        54         0.832037          0.413667 MI_SPECIFIC_BACKBONE_ADAPTED  0.000000                   0.000000                      1.000000                 0.00                              False            False
OpenBMI_MI_S1_to_S2 ANCHOR_PLUS_META_GENERIC_HYPER        54         0.829074          0.462807 MI_SPECIFIC_BACKBONE_ADAPTED -0.002963                   0.500000                      0.574074                -0.15                              False            False
OpenBMI_MI_S1_to_S2 ANCHOR_PLUS_PERSIST_META_HYPER        54         0.828704          0.453924 MI_SPECIFIC_BACKBONE_ADAPTED -0.003333                   0.444444                      0.592593                -0.14                              False            False
OpenBMI_MI_S1_to_S2        META_GENERIC_HYPER_HALF        54         0.793519          0.593437 MI_SPECIFIC_BACKBONE_ADAPTED -0.038519                   0.222222                      0.370370                -0.36                              False            False
OpenBMI_MI_S1_to_S2        PERSIST_META_HYPER_HALF        54         0.792037          0.582100 MI_SPECIFIC_BACKBONE_ADAPTED -0.040000                   0.296296                      0.333333                -0.37                              False            False
OpenBMI_MI_S1_to_S2             PERSIST_META_HYPER        54         0.782593          0.748045 MI_SPECIFIC_BACKBONE_ADAPTED -0.049444                   0.203704                      0.222222                -0.37                              False            False
OpenBMI_MI_S1_to_S2             META_GENERIC_HYPER        54         0.781296          0.774343 MI_SPECIFIC_BACKBONE_ADAPTED -0.050741                   0.203704                      0.277778                -0.37                              False            False
```

## WBCIC

```text
                              benchmark                                             method_id  subjects  mean_subject_BA  mean_subject_NLL                                   reference_method_id  Delta_BA  positive_subject_fraction  nonnegative_subject_fraction  worst_subject_delta  target_future_labels_used_for_fit  OUTER_TEST_USED
WBCIC_S1S2_to_S3_authorized_development V5_FIXED_LOGIT_BLEND__A_FUTURE_SESSION_TARGET_ADAPTED        41         0.820817          0.349142 V5_FIXED_LOGIT_BLEND__A_FUTURE_SESSION_TARGET_ADAPTED  0.000000                   0.000000                      1.000000                0.000                              False            False
WBCIC_S1S2_to_S3_authorized_development                        ANCHOR_PLUS_PERSIST_META_HYPER        41         0.813878          0.361188 V5_FIXED_LOGIT_BLEND__A_FUTURE_SESSION_TARGET_ADAPTED -0.006939                   0.243902                      0.341463               -0.050                              False            False
WBCIC_S1S2_to_S3_authorized_development                        ANCHOR_PLUS_META_GENERIC_HYPER        41         0.812041          0.359476 V5_FIXED_LOGIT_BLEND__A_FUTURE_SESSION_TARGET_ADAPTED -0.008776                   0.170732                      0.365854               -0.050                              False            False
WBCIC_S1S2_to_S3_authorized_development                               META_GENERIC_HYPER_HALF        41         0.799713          0.417789 V5_FIXED_LOGIT_BLEND__A_FUTURE_SESSION_TARGET_ADAPTED -0.021104                   0.219512                      0.268293               -0.160                              False            False
WBCIC_S1S2_to_S3_authorized_development                               PERSIST_META_HYPER_HALF        41         0.798250          0.419365 V5_FIXED_LOGIT_BLEND__A_FUTURE_SESSION_TARGET_ADAPTED -0.022567                   0.170732                      0.195122               -0.140                              False            False
WBCIC_S1S2_to_S3_authorized_development                                    PERSIST_META_HYPER        41         0.790811          0.465821 V5_FIXED_LOGIT_BLEND__A_FUTURE_SESSION_TARGET_ADAPTED -0.030006                   0.146341                      0.219512               -0.145                              False            False
WBCIC_S1S2_to_S3_authorized_development                                    META_GENERIC_HYPER        41         0.787382          0.464449 V5_FIXED_LOGIT_BLEND__A_FUTURE_SESSION_TARGET_ADAPTED -0.033435                   0.121951                      0.146341               -0.145                              False            False
```
