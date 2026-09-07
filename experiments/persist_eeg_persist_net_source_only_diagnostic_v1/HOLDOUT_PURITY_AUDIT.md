# Holdout purity audit

`internal_holdout_accessed = false`

`WBCIC_outer_accessed = false`

Only the existing OpenBMI `V8_SEARCH` cache (40 subjects) was opened. Session identifiers for that cache were read, while labels were predicate-filtered to Session 2 before materialization. Target Session-1 labels were not materialized or used. No OpenBMI 14-subject internal-holdout EEG, labels, predictions, embeddings, or metrics were accessed. No WBCIC path or sealed-outer artifact was accessed.
