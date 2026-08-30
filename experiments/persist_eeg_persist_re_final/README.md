# PERSIST-RE

Prospective Decision-Level Random-Effects Learning for future-session EEG
generalization.  The primary predictor is population-only at inference:
`z = LayerNorm(E(x))`, `p = g(z)`.  Training-only subject random effects are
centered exactly and are never used with a test subject ID.

The source screen uses the authorized OpenBMI and WBCIC development archives,
five folds and three seeds.  The bounded recipe grid is rank `{1,2,4}`,
`lambda_R` `{1e-3,1e-2}`, and `lambda_P` `{0.5,1.0}`.  Outer WBCIC subjects and
the OpenBMI sealed holdout are not opened.

Run the unit tests with `pytest -q tests`.  On the server, use the pinned
`Benchmark_TTA_Win` environment and `code/run_source.py`; runtime fits and
archives are intentionally ignored by Git.

