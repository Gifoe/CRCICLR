# PU vs U-only explanatory analysis

Status: `COMPLETE_REPLAY_UNVERIFIED`. No neural network was trained; selectors and frozen checkpoint provenance were unchanged.

## Required interpretation

The CSV outputs contain the per-cell, subject-unit, and biological-subject bootstrap summaries required to distinguish overlap/persistence rediscovery from distinct subspaces. `UTILITY_MATCHED_TASK_SUMMARY.csv` is the primary conditional persistence test; block rows are aggregated within biological subject before bootstrap.

Do not interpret P-only as a success criterion; it is retained only as the Experiment 1 negative control.
