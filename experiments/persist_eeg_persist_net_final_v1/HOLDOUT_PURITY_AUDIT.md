# Holdout purity audit

## OpenBMI

Development is restricted to the 40 IDs in the frozen `V8_SEARCH` split.  The
14 internal-holdout IDs may be read only from the split JSON so they can be
excluded.  The raw-manifest read uses parquet predicate pushdown on
`subject_id in V8_SEARCH` and `paradigm == mi`; holdout EEG rows, labels,
tensors, embeddings, normalization statistics, predictions, and metrics are
not materialized.  Runtime asserts exactly 8,000 rows, 40 subjects, two
sessions, two labels, and 50 trials per subject/session/label.

The OpenBMI internal holdout is not considered project-level independent due
to prior historical access and will remain unopened in this experiment.

## WBCIC

The 41 authorized development subjects are eligible only after OpenBMI G1,
G6, and G9 pass.  The sealed outer 10 are not enumerated, loaded, featurized,
or scored before a complete model lock.  If prior outer contamination is
found, outer confirmation is cancelled.

## Runtime status

The executable writes `protocol/HOLDOUT_RUNTIME_AUDIT.json` before training and
updates it after finalization.  Absence of that artifact or any failed
assertion is a G9 failure.
