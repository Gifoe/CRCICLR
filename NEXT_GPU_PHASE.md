# Next GPU phase — do not execute during CPU phase

1. Freeze and archive CPU manifests, five-seed splits, episodes, preprocessing hashes, and calibration provenance.
2. Download the public CBraMod checkpoint only after GPU-stage authorization.
3. Implement a frozen `BackboneAdapter`; fit task heads only on the declared `task_head_train` subjects and meta-risk predictors only on `meta_risk_train` subjects.
4. Generate context features and full action surfaces using the frozen schemas. Real `entropy_adapter` resets per subject and sees only `U_s`.
5. Fit residual quantiles only on the declared calibration subjects; evaluate final-test subjects once.
6. Validate every Parquet file, check action–lambda completeness and U/V leakage, then run subject-level metrics and bootstrap confidence intervals.

Embedding storage should be estimated after the observed window counts and embedding dimension are known: `subjects × windows × dimension × bytes`, plus 20% metadata/temporary headroom. No estimate is fabricated in the CPU phase.

