# PERSIST-EEG Experiment 3 V1.1 — Block-wise Matched Identity-Removal Causal Test

V1 ended honestly at a train-only feasibility failure because unioning two
Protected blocks produced rank 8 while the non-Protected supported pool also
had rank 8.  V1.1 keeps V1 intact and changes only the causal unit: each frozen
Signed-V3.1 Protected block is tested separately.  The primary question remains
whether Protected deletion causes more task harm than a structurally matched
Non-Protected persistent control at the same held-out subject-identity
reduction.

The experiment is restricted to OpenBMI MI, folds 0–2 and seeds 0–1.  It uses
the V3.1 EEGNet checkpoints and canonical spectra as immutable upstream
artifacts.  Controls are exact-rank subsets of non-Protected,
persistence-supported coordinates, certified on train-only persistence before
structural matching.  Outer subjects are never materialized or enumerated.

Run on the server (the data root is intentionally external to Git):

```text
python experiments/persist_eeg_matched_identity_causal_v1_1/code/experiment3.py phase0
python experiments/persist_eeg_matched_identity_causal_v1_1/code/experiment3.py train_only
python experiments/persist_eeg_matched_identity_causal_v1_1/code/experiment3.py freeze
python experiments/persist_eeg_matched_identity_causal_v1_1/code/experiment3.py final
python experiments/persist_eeg_matched_identity_causal_v1_1/code/experiment3.py finalize
```

`freeze` must be executed after train-only and before `final`.  If the
block-wise coverage rule fails, use the explicit `terminal` phase; it will not
run validation task outcomes.

`finalize` (or `terminal`) is deterministic and should be run twice; the
reproducibility audit checks lightweight output hashes.

The experiment can terminate with a negative scientific state.  No validation
BA is used to change matching, dose, alpha, metric, or gate definitions.

The V1.1 result is intentionally not known until the train-only feasibility
phase and the frozen final evaluation are run.  No validation outcome may alter
the control, dose, metric or gate definitions.
