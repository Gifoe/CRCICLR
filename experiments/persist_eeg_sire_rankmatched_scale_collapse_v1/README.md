# SIRE-EEG frozen rank-matched ScaleCollapse control

This reviewer control asks whether the previously observed B2 ScaleCollapse
loss can be attributed entirely to its reduction of a 48-channel representation
to effective rank 16. It is not an architecture-selection experiment and does
not retrain any model.

The intervention uses the 60 official Full SIRE checkpoints (four tasks, five
folds, three seeds) and their original fold-specific train-only normalizers.
At the exact post-branch-dropout/pre-backend 48-channel tensor, it compares
identity, `(J_3 / 3) ⊗ I_16`, and 100 pre-frozen Haar-like Gaussian-QR rank-16
orthogonal projectors. The same projectors are used for every checkpoint and
task. The original decoder remains fixed. These outcomes are not the trained
B2 architecture's outcomes.

`code/run_rankmatched_projection_control.py` implements three ordered stages:
`prepare` freezes sources/projectors and audits the projection algebra;
`replay` runs the unmodified Full forward on every evaluation session and
requires exact agreement with the official multiseed result artifact; and
`intervene` evaluates all fixed projectors only after replay passes. Inference
cells are restartable under `runtime/cells` and are not the source of a new
checkpoint selection.

`code/summarize_rankmatched_projection_control.py` first averages all 15
fold/seed metric values inside each biological subject and session. For WS-BA,
it then takes the subject's minimum session BA separately for each condition
and each random projector. It compares ScaleCollapse harm to the average harm
of 100 projectors and uses 20,000 paired biological-subject bootstrap draws;
folds, seeds and projectors are never bootstrap sampling units. The empirical
percentile of ScaleCollapse among fixed random controls is descriptive only.

OpenBMI evaluation is on the previously exposed 14-person internal-heldout
diagnostic cohort; WBCIC evaluation is on the 10-person true-outer cohort.
The historical Full/B1 Protected-rank audit is read from the previous bridge
`CELL_RESULTS.csv`, without rerunning decomposition. The prior failed
architecture selector is not retuned.
