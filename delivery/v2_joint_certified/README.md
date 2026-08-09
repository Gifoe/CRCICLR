# HSC-TTA v2 delivery

This directory mirrors the small reports, CSV summaries, and provenance JSON produced on the AutoDL server. It deliberately excludes EEG data, subject-level parquet artifacts, token caches, checkpoints, serialized predictors, action states, and logs. Recreate those artifacts with `scripts/run_v2_full_development.sh`.
