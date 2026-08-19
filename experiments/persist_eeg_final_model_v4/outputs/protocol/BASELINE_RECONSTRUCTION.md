# V4 baseline reconstruction

`BASELINE_RECONSTRUCTION_PASS`

- Frozen source commit: `ee9e280e350073f4cc30f8e3cad8c27cd1347bec`
- B_STRONG: `B6_ALL_RUN_LOGIT_MEAN`
- B_STRONG mean subject BA: `0.846442307692`
- KEEP-only oracle: `0.923076923077` (+7.663 pp)
- Global action oracle: `0.932403846154` (+8.596 pp)
- Complete KEEP+ACTION oracle: `0.953461538462` (+10.702 pp)
- Historical V1/V2/V2.1/V3 files modified: `false`
- WBCIC outer used: `false`

V3 prospective methods are provenance-reconstructed by exact artifact hash;
B6, the unique-trial expert space, and all three oracle ladders are rebuilt
numerically from the frozen parquet cache in the new V4 directory.

| check | passed | absolute_difference |
| --- | --- | --- |
| openbmi_cache_sha256:OOF_BASE_LOGITS.parquet | True | 0.0 |
| openbmi_cache_sha256:OOF_COUNTERFACTUAL_LOGITS.parquet | True | 0.0 |
| openbmi_cache_sha256:OOF_GEOMETRY_FEATURES.parquet | True | 0.0 |
| openbmi_cache_sha256:OOF_ROUTER_FEATURES.parquet | True | 0.0 |
| openbmi_unique_trials | True | 0.0 |
| openbmi_subjects | True | 0.0 |
| openbmi_manifest_identity_sha256 | True | 0.0 |
| B6_unique_margin_max_abs_difference | True | 3.5762786865234375e-07 |
| B6_unique_prediction_exact | True | 0.0 |
| B6_mean_subject_BA | True | 0.0 |
| KEEP_only_oracle_mean_subject_BA | True | 0.0 |
| KEEP_plus_global_ACTION_oracle_mean_subject_BA | True | 0.0 |
| complete_KEEP_plus_ACTION_oracle_mean_subject_BA | True | 1.1102230246251565e-16 |
| V3_prospective_table_sha256 | True | 0.0 |
| V3_prospective_method_roster | True | 0.0 |
