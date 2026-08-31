# R1 engineering repair rationale

The previous CGR-Fuse run had identifiable implementation/protocol failures: complete-case filtering removed historical OpenBMI subjects, WBCIC inference evaluated samples with models trained on the same subject, the calibrated KEEP baseline did not fit a temperature, preprocessing was not nested, and soft action mass was confused with actual decision changes. R1 repairs only those issues and replaces the unstable MLP/ranking objective with the preregistered nested scalar-calibrated simplex stacks and monotone consensus gate. No new expert, dataset, search dimension, or target-session resource is introduced.

The same R1 engineering pass also fixed a post-bank reporting omission: the
concatenated held-subject table is now carried into the report writer, so a
completed legal-bank run cannot fail with a non-scientific aggregation
KeyError. Existing banks were reused only to resume this deterministic
post-processing step; no predictions, labels, gates, or thresholds were
changed.
