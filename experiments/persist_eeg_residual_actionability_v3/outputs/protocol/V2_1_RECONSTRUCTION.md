# Exact V2.1 reconstruction

`V2_1_RECONSTRUCTION_PASS`

- Historical V2.1 state: `ENSEMBLE_EXPLAINS_V2_GAIN`
- Frozen constructive reference: `B6_ALL_RUN_LOGIT_MEAN`
- Target-run rows: `40800`
- Unique deployment trials: `10400`
- Subjects: `52`
- Numerical tolerance: `1e-14`
- WBCIC outer accessed: `false`

B0, B4, B6, B7, FULL and protected-safe were rebuilt from the frozen cache.
B6 target-run and unique-trial subject metrics were compared to the historical
V2.1 tables. The unique B6 prediction hash is stored in the JSON artifact.

| check | passed | absolute_difference |
| --- | --- | --- |
| historical_V2_1_reconstruction_status | True | 0.0 |
| V2_policy_lock_hash | True | 0.0 |
| cache_sha256:OOF_BASE_LOGITS.parquet | True | 0.0 |
| cache_sha256:OOF_COUNTERFACTUAL_LOGITS.parquet | True | 0.0 |
| cache_sha256:OOF_GEOMETRY_FEATURES.parquet | True | 0.0 |
| cache_sha256:OOF_ROUTER_FEATURES.parquet | True | 0.0 |
| exploration:B0_TARGET_KEEP:mean_subject_BA | True | 0.0 |
| exploration:B0_TARGET_KEEP:mean_subject_delta_BA | True | 0.0 |
| exploration:B4_ALL_RUN_PROBABILITY_MEAN:mean_subject_BA | True | 0.0 |
| exploration:B4_ALL_RUN_PROBABILITY_MEAN:mean_subject_delta_BA | True | 2.0816681711721685e-17 |
| exploration:B6_ALL_RUN_LOGIT_MEAN:mean_subject_BA | True | 0.0 |
| exploration:B6_ALL_RUN_LOGIT_MEAN:mean_subject_delta_BA | True | 4.85722573273506e-17 |
| exploration:B7_CONFIDENCE_WEIGHTED_KEEP_ENSEMBLE:mean_subject_BA | True | 0.0 |
| exploration:B7_CONFIDENCE_WEIGHTED_KEEP_ENSEMBLE:mean_subject_delta_BA | True | 3.469446951953614e-17 |
| exploration:I003_CROSS_RUN_FULL:mean_subject_BA | True | 0.0 |
| exploration:I003_CROSS_RUN_FULL:mean_subject_delta_BA | True | 2.949029909160572e-17 |
| exploration:I003_CROSS_RUN_PROTECTED_SAFE:mean_subject_BA | True | 0.0 |
| exploration:I003_CROSS_RUN_PROTECTED_SAFE:mean_subject_delta_BA | True | 1.5525775109992423e-16 |
| exploration:B6:deployment_mean_subject_BA | True | 0.0 |
| holdout:B0_TARGET_KEEP:mean_subject_BA | True | 0.0 |
| holdout:B0_TARGET_KEEP:mean_subject_delta_BA | True | 0.0 |
| holdout:B4_ALL_RUN_PROBABILITY_MEAN:mean_subject_BA | True | 0.0 |
| holdout:B4_ALL_RUN_PROBABILITY_MEAN:mean_subject_delta_BA | True | 1.1796119636642288e-16 |
| holdout:B6_ALL_RUN_LOGIT_MEAN:mean_subject_BA | True | 0.0 |
| holdout:B6_ALL_RUN_LOGIT_MEAN:mean_subject_delta_BA | True | 3.8163916471489756e-17 |
| holdout:B7_CONFIDENCE_WEIGHTED_KEEP_ENSEMBLE:mean_subject_BA | True | 0.0 |
| holdout:B7_CONFIDENCE_WEIGHTED_KEEP_ENSEMBLE:mean_subject_delta_BA | True | 3.469446951953614e-17 |
| holdout:I003_CROSS_RUN_FULL:mean_subject_BA | True | 0.0 |
| holdout:I003_CROSS_RUN_FULL:mean_subject_delta_BA | True | 2.0816681711721685e-16 |
| holdout:I003_CROSS_RUN_PROTECTED_SAFE:mean_subject_BA | True | 0.0 |
| holdout:I003_CROSS_RUN_PROTECTED_SAFE:mean_subject_delta_BA | True | 8.673617379884035e-17 |
| holdout:B6:deployment_mean_subject_BA | True | 0.0 |
