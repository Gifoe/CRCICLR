# PERSIST-EEG Experiment 3 — Matched Identity-Removal Causal Test

This directory contains the prospectively frozen Experiment-3 closure.  The
primary question is whether the Signed-V3.1 Protected persistent structure
causes more task harm than a structurally matched Non-Protected persistent
control at the same held-out subject-identity reduction.

The experiment is restricted to OpenBMI MI, folds 0–2 and seeds 0–1.  It uses
the V3.1 EEGNet checkpoints and canonical spectra as immutable upstream
artifacts.  Development subjects are used for all design and calibration;
validation subjects are used only for the final held-out persistence,
identity-manipulation and task-consequence measurements.  Outer subjects are
never materialized or enumerated.

Run on the server (the data root is intentionally external to Git):

```text
python experiments/persist_eeg_matched_identity_causal_v1/code/experiment3.py phase0
python experiments/persist_eeg_matched_identity_causal_v1/code/experiment3.py train_only
python experiments/persist_eeg_matched_identity_causal_v1/code/experiment3.py freeze
python experiments/persist_eeg_matched_identity_causal_v1/code/experiment3.py final
python experiments/persist_eeg_matched_identity_causal_v1/code/experiment3.py finalize
```

`final` runs the frozen G0–G3, dose-response and random-control diagnostics.
`finalize` is deterministic and should be run twice; the reproducibility audit
checks lightweight output hashes.

The experiment can terminate with a negative scientific state.  No validation
BA is used to change matching, dose, alpha, metric, or gate definitions.
