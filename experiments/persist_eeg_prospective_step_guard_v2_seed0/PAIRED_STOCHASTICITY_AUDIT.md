# Paired stochasticity audit

All candidates use the same canonical checkpoint, deterministic A/B/probe schedule and candidate-independent RNG keys. Dropout keys contain only dataset, fold, seed, epoch, step and role. CAP_ZERO_IDENTITY compares the full trajectory to TASK_ONLY_MATCHED with tolerances 1e-7.

pass = True
