# Native Protected utilization (seed 0)

Frozen native-head intervention audit for EEGNet, EEGConformer and FBCNet on
OpenBMI MI and SSVEP.  The audit uses only each fold's outer-development
subjects; final/true-heldout subjects are excluded.  It neither trains a
backbone nor refits a classifier/probe.

Run `code/run_native_protected_utilization.py lock` before any cell.  Cells
fail closed if the frozen checkpoint, TRAIN normalizer, current PERSIST
Protected assignment, canonical basis, or outer-development split cannot be
matched exactly.
