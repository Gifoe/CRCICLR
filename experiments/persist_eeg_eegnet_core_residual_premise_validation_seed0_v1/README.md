# EEGNet Core–Residual premise validation, seed 0

Frozen-mechanism audit of canonical EEGNet / OpenBMI_MI folds 0–4 at the
64-dimensional embedding. The immutable scientific sources are commits
`8a8b708b30e19a20a83902d272fbe53fe8279f12` and
`d5184c5423806693a3b218f2d0329cd6b29d5d9a`.

This experiment asks whether Protected G is specifically a reproducible
cross-subject task relation while information outside G can improve held
prediction. It does not train or update the EEGNet, implement a new loss,
or use formal final-heldout EEG.

Subspaces and decoder/stacker fits use `TRAIN_GEOMETRY` only. All five
construction audits must pass before any `OUTER_DEVELOPMENT` extraction.
`CHECKPOINT_VALIDATION` is descriptive and historically selection-exposed.
The nondeployable oracle union is labeled as an upper bound and never enters
the deployable model decision. The 1-D Fisher direction is not treated as a
rank-matched alternative to G.

The compact outputs and exact file hashes are under `outputs/`. Runtime
activations, checkpoints and raw EEG are excluded from Git.

The successful server evaluation was the version-forwarded `evaluate_all_v3`;
the successful decision aggregation was `aggregate_v2`. Earlier fail-closed
attempts, their causes and log hashes are disclosed in
`protocol/EXECUTION_DISCLOSURE.json`. The final Case C interpretation is not
evidence for a Core–Residual architecture: the equal-rank supervised decision
subspace exceeded G on relation consistency and BA in every fold. These OUTER
results are a frozen follow-up to source work that already inspected OUTER, so
they are descriptive rather than an untouched confirmatory test.
