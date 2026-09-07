# Fold-cache legality audit

Strict subject-level matching passed for **40/40** subjects: original outer fold = evaluation fold = representation cache fold. All five feature/logit/metadata hashes and checkpoint provenance are in `protocol/FOLD_CACHE_LEGALITY.json`. A mismatch would have terminated the run with `OPENBMI_FOLD_CACHE_LEGALITY_FAILED`.
