# Exact V2 reconstruction

`V2_RECONSTRUCTION_PASS`

- Frozen lock: `e679c7a955ccf3745bb35ce6c86a61c57705557f3eed8917b724b0e5613b5fd4`
- Split hash: `f033c6ce4b6c79dcbba60581423b812ed32a921595c740396aac62163edac944`
- Numerical tolerance: `1e-14`
- WBCIC outer accessed: `false`

The frozen V2 implementation was imported without modification. Subject,
run, action, rescue/harm, and summary tables were recomputed from the hashed
router caches. Identity and final prediction hashes cover every manifest/run
row and are stored in the JSON artifact.

| check | passed | max_abs_difference | actual |
| --- | --- | --- | --- |
| V2_POLICY_LOCK_HASH | True | 0.0 | e679c7a955ccf3745bb35ce6c86a61c57705557f3eed8917b724b0e5613b5fd4 |
| source_cache_sha256:OOF_ROUTER_FEATURES.parquet | True | 0.0 | 93b32654f67baa653d1d89a4e76b3241256cec5f5f5977b2ea08d4b166da0fbd |
| source_cache_sha256:OOF_BASE_LOGITS.parquet | True | 0.0 | 9072912ee0b1b4513fd5ec1034727d755b118be20d7c5f8355f95c1aaa1cc8a9 |
| source_cache_sha256:OOF_COUNTERFACTUAL_LOGITS.parquet | True | 0.0 | 1ebbe70ac854f9abb957875c49ba81fb782dc5be88a53a832ff70498db14ba2e |
| source_cache_sha256:OOF_GEOMETRY_FEATURES.parquet | True | 0.0 | 77b2234deae19c18f3c7cf9217077ce44d1a8181acf400c6b96cfdf980c6e76d |
| exploration:subject_delta | True | 9.71445146547012e-17 | numerically identical |
| exploration:run_delta | True | 8.153200337090993e-17 | numerically identical |
| exploration:action_counts | True | 9.71445146547012e-17 | numerically identical |
| exploration:summary | True | 7.112366251504909e-17 | numerically identical |
| exploration:oracle_summary | True | 0.0 | numerically identical |
| holdout:subject_delta | True | 9.71445146547012e-17 | numerically identical |
| holdout:run_delta | True | 8.326672684688674e-17 | numerically identical |
| holdout:action_counts | True | 1.1102230246251565e-16 | numerically identical |
| holdout:summary | True | 9.8879238130678e-17 | numerically identical |
