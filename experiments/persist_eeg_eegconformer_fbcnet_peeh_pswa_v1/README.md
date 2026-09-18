# EEGConformer and FBCNet PEEH/PSWA extension

This is a frozen, seed-0 analysis extension for the 20 previously selected
EEGConformer/FBCNet source checkpoints (two models, four tasks, five folds).

- No model is trained, tuned, or reselected.
- PEEH uses the same capped trial protocol, train-only standardizer, ridge
  alpha, persistence null, matched random erasures, and biological-subject
  bootstrap as the earlier cross-backbone audit.
- PSWA consumes only the completed PEEH protected-coordinate assignments and
  deterministically reconstructs the canonical transform.  It does not rerun
  PEEH selection or redraw a new protected assignment.
- FBCNet's fixed nine-band preprocessing is applied in bounded batches.  This
  is numerically identical to the original FBCNet adapter and avoids a
  multi-gigabyte fold-wide intermediate allocation.

Runtime artifacts, representation caches, and checkpoint binaries remain
outside version control.  The committed deliverables are the protocol lock and
compact CSV/JSON/Markdown reports.
