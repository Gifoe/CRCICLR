# Frozen GPU interface

The GPU phase must consume subject episodes and write three Parquet tables. It must not change subject roles or episode membership.

`subject_context_features.parquet` contains dataset, seed, subject_id, split_role, backbone, episode_id, n_context, embedding mean/std columns, entropy and max-probability quantiles, predicted-class proportions, instability, missing-channel rate, signal-quality features, and action-specific context features. Every value is computable from `U_s` alone.

`subject_action_surface.parquet` contains dataset, seed, subject_id, split_role, episode_id, action, lambda, predicted risk, within-subject empirical risk/margin/upper risk, certified bound, future risk, classification and set metrics, context/future/block counts, and status. Context features, adaptation, probabilities, and predicted risk use `U_s`; future risk and offline classification metrics may use `V_s` only after the decision inputs are frozen.

`subject_decisions.parquet` contains dataset, seed, subject_id, alpha, selected action/lambda, predicted and certified risk, offline true risk, certificate flags, utilities, baseline/selected errors, harm flag, status, and reason.

Pydantic validators enforce ranges and required columns. Leakage audit verifies that context APIs accept no future inputs, episode indices/runs do not overlap, test subjects are absent from fitting/calibration, and every subject-action has the configured lambda grid exactly once.

