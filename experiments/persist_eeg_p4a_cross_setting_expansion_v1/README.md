# PERSIST-EEG Phase 4A — Cross-Setting Expansion

This experiment freezes a harmonized six-setting source evidence cube across dataset, task, and backbone. S1–S3 are read-only historical settings; S4–S6 are new frozen-protocol settings.

P4A computes source-side subject identifiability, persistence, finite decision dependence, source consequence, and task-subspace overlap. New-setting direction-level future utility and invariance outcome deltas remain sealed for P4B.

Execution order:

1. `python code/preflight.py`
2. commit the protocol freeze before any outcome competence evaluation
3. `python code/train.py --tier all`
4. `python code/aggregate.py`
5. `python code/validate_p4a.py`

The runtime cache, embeddings, and checkpoints are intentionally excluded from Git; their paths and hashes are recorded in compact committed artifacts.
