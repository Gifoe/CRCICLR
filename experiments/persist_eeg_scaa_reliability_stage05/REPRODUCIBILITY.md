# Reproducibility

The experiment is anchored to validated Stage-0 commit `46b8ecf2c39b0e32045cad9d78ca12327f0a3f0d`. Protocol locks contain exact subjects, folds, seeds, backbone names, feature formulas, model definitions, success gates, and SHA-256 hashes of Stage-0 provenance and frozen Stage-0.5 code.

The authorized WBCIC development cache and existing out-of-fold checkpoints remain server-local. `extract_features.py` opens the signal as a memory map and reads only rows whose metadata session is 0 or 1. It records an execution receipt and recomputes Delta2 to verify exact agreement with Stage-0. S3 outcomes are later merged from the committed compact Stage-0 table.

All cross-validation follows the frozen five subject-disjoint folds. Both backbone rows are resampled together in 10,000 subject-bootstrap draws. Runtime files, raw EEG, checkpoints, and caches are not committed.

