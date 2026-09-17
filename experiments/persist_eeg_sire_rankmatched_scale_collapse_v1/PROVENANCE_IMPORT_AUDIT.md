# Provenance import audit

This directory was independently verified and imported from the requested
server on 2026-09-17.  It is a frozen same-checkpoint intervention, not a
neural-training run and not an architecture-selection result.

## Execution source

- Execution tree: `/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK`
- Execution-tree commit: `60d053f0cf1c6d8f77a0fd337f0ca8ad03224843`
- The experiment directory was untracked in that execution tree, so this
  branch is the first Git-tracked delivery of these artifacts.
- The source files were copied byte-for-byte from that tree; their SHA-256
  values are `4a603ce9b892994c5e40ce7644e1060c3009f3e980be708600cfaedb42c2cf4e`
  for `code/run_rankmatched_projection_control.py` and
  `e8191d229e88347893f9cdbb2254d621f2110cdd5a063fb099e06a62e31cb93a`
  for `code/summarize_rankmatched_projection_control.py`.

## Authoritative SIRE chain

The actual model factory imports
`experiments/persist_eeg_carrier_dualdataset_screen_v1/code/run_carrier_screen.py::CompactLite`
through the locked four-task `litebn_x.py` factory.  The carrier source SHA-256
on the execution server is
`920af131aabc272317da128f42be9961d5592619ce85ca99192029d1181f126f`,
matching the final-heldout source.  The only task-specific alteration is the
official classifier-head output width (2 for MI/ERP/WBCIC and 4 for SSVEP);
all 60 state dictionaries strictly load into this topology.

The resulting audited parameter counts are 47,978 for OpenBMI MI/ERP, 48,108
for OpenBMI SSVEP, and 47,786 for WBCIC MI.  The per-checkpoint paths,
checkpoint hashes, normalizer hashes, counts, folds, and seeds are frozen in
`protocol/CHECKPOINT_SOURCE_AUDIT.csv`.

## Independent import checks

- All 60 checkpoint and normalizer SHA-256 values were rechecked on the
  requested server before import.
- `FULL_REPLAY_AUDIT.csv` has 12/12 passing Full metric replays; its largest
  discrepancy is `2.220446049250313e-16`.
- `IDENTITY_WRAPPER_CELLS.csv` has 60/60 passing cells with a maximum logit
  difference of exactly 0.0.
- `P_scale` is symmetric and idempotent with rank 16; the synthetic B2
  equivalence error is `1.1920928955078125e-07`.
- All 100 frozen random controls have rank 16.  Their maximum stored
  idempotence residual is within the recorded numerical tolerance.
- The raw output has the complete subject/session cardinality for all
  60 checkpoints and 102 conditions; aggregation uses 20,000 deterministic
  biological-subject bootstrap draws.

No manuscript was edited, no model was retrained, no normalizer/BN state was
adapted, and no selector was retuned during this import.
