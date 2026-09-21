# Interim selected completed-cell snapshot

This is an intentionally partial, compact status/result snapshot requested
before the full emergence-routing experiment completes.  It contains exactly
21 `COMPLETE` cell artifacts:

- EEGNet: OpenBMI MI and OpenBMI SSVEP, all five folds each (10 cells).
- EEGConformer: OpenBMI MI and OpenBMI SSVEP, all five folds each (10 cells).
- FBCNet: OpenBMI MI fold 2 only (one completed cell).

Excluded by request: all FBCNet failed-closed artifacts, its
`EMPTY_PROTECTED` artifact, and all FBCNet OpenBMI SSVEP artifacts while their
technical recovery is in progress.  This directory is therefore **not** a
30-cell aggregate and must not be interpreted as a complete model comparison.

All included artifacts record `final_heldout_accessed: false`.  No raw EEG,
cache, model checkpoint, or final-heldout data is included here.
