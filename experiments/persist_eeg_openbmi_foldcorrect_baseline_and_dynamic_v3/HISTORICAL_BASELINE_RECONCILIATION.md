# Historical baseline reconciliation

Historical inferred EEGNet reference: **75.297%**.

Fresh fold-correct seed-0 and three-seed values are recorded in `results/VANILLA_EEGNET_SEED_RESULTS.csv`; the primary three-seed mean is **0.749667** and the S1-only sensitivity mean is **0.714500**.

Difference between fresh primary mean and historical inferred reference: **-0.330 pp**. This was not tuned to the historical aggregate. Any discrepancy above 2 pp is investigated in the protocol audit (subject subset, source sessions, crop, normalization, architecture, epoch rule, and metric aggregation); the fresh legal result is not modified to force agreement.
